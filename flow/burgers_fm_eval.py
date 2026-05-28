"""
burgers_fm_eval.py — standalone evaluator for FM Burgers checkpoints.

SELF-CONTAINED (imports burgers_fm_train + stable repo modules only).

Loads a trained joint (+ optional prior) checkpoint, samples on the HELD-OUT
TEST split (first-N samples, matching paper get_target(0..N-1) convention), runs
a γ prior-reweighting sweep, computes J/E via burgers_metric, and prints a
comparison table including the paper's published DiffPhyCon FOPC J = 0.00037.

Example (compare vanilla vs OT after training):
    python flow/burgers_fm_eval.py \
        --dataset free_u_f_paper_fopc --n_test 50 --dim 128 \
        --vanilla_joint checkpoints/fm_vanilla_joint_paper.pt \
        --vanilla_prior checkpoints/fm_vanilla_prior_paper.pt \
        --ot_joint      checkpoints/fm_ot_joint_paper.pt \
        --ot_prior      checkpoints/fm_ot_prior_paper.pt \
        --out_dir flow/results/paper_fopc
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flow.burgers_fm_train import (
    BurgersVectorField, GaussianConditionalProbabilityPath,
    LinearAlpha, LinearBeta, load_burgers, T_IDX, RESCALER,
)
from utils import burgers_metric
from diffusion.diffusion_1d_burgers import sigmoid_schedule_flip

# paper Table 1 (PDF): DiffPhyCon FO-PC J_actual
PAPER_FOPC_J = 0.00037


# ---- conditioning / sampling primitives (standalone) ----

def inpaint_overwrite(x, c):
    x = x.clone()
    x[:, 0, 0, :]     = c[:, 0]
    x[:, 0, T_IDX, :] = c[:, 1]
    return x


def w_scheduler_fm(tau):
    """FM-time sigmoid_flip: small at τ=0 (noise), large at τ=1 (clean)."""
    dev = tau.device
    t_ddpm = ((1.0 - tau) * 999).round().long().clamp(min=0, max=999).cpu()
    return sigmoid_schedule_flip(t_ddpm).float().to(dev)


class ReweightedVectorField:
    """γ-reweighted velocity:  v_joint + (γ−1)·η(τ)·[v_prior − (1/τ)·[0,w]]."""
    def __init__(self, net_joint, net_prior, gamma=1.0, use_scheduler=True, tau_min=1e-3):
        self.net_joint, self.net_prior = net_joint, net_prior
        self.gamma, self.use_scheduler, self.tau_min = gamma, use_scheduler, tau_min

    def __call__(self, x, t, c):
        v_joint = self.net_joint(x, t, c)
        if self.gamma == 1.0 or self.net_prior is None:
            return v_joint
        x_prior = x.clone(); x_prior[:, 0] = 0
        v_prior = self.net_prior(x_prior, t, c)
        x_w = x.clone(); x_w[:, 0] = 0
        b_t = (1.0 / t.clamp(min=self.tau_min)).view(-1, 1, 1, 1)
        eta = w_scheduler_fm(t.view(-1)) if self.use_scheduler else torch.ones_like(t.view(-1))
        eta = eta.view(-1, 1, 1, 1).to(x.dtype)
        return v_joint + (self.gamma - 1) * eta * (v_prior - b_t * x_w)


@torch.no_grad()
def sample(vf, c, n_steps=100, tau_min=1e-3, shape=(2, 16, 128)):
    b, device = c.shape[0], c.device
    dtau = (1.0 - 2 * tau_min) / n_steps
    x = torch.randn(b, *shape, device=device)
    x = inpaint_overwrite(x, c)
    for i in range(n_steps):
        tau = tau_min + i * dtau
        t = torch.full((b,), tau, device=device)
        x = x + vf(x, t, c) * dtau
        x = inpaint_overwrite(x, c)
    return x


def compute_J_E(x_pred, c, rescaler=RESCALER):
    x_pred = x_pred.detach().cpu() * rescaler
    c_un = c.detach().cpu() * rescaler
    b, Nx = x_pred.shape[0], c_un.shape[-1]
    u_target = torch.zeros(b, 11, Nx)
    u_target[:, 0] = c_un[:, 0]
    u_target[:, 10] = c_un[:, 1]
    w = x_pred[:, 1, :10, :]
    J, E = burgers_metric(u_target, w, target="final_u", partial_control="front_rear_quarter")
    return float(J.mean()), float(E.mean())


# ---- helpers ----

def load_net(path, dim, dim_mults, device):
    if path is None or not os.path.isfile(path):
        return None
    net = BurgersVectorField(dim=dim, dim_mults=tuple(dim_mults)).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    net.eval()
    return net


def make_test_batch(dataset, n, device):
    """First n samples of the TEST split (deterministic, matches paper get_target)."""
    ds = load_burgers(dataset, split="test", device="cpu")
    zs = torch.stack([ds[i] for i in range(min(n, len(ds)))], dim=0).to(device)
    c = torch.stack([zs[:, 0, 0, :], zs[:, 0, T_IDX, :]], dim=1)
    return c


def sweep_variant(net_joint, net_prior, c, gammas, n_steps, seed):
    rows = []
    for g in gammas:
        torch.manual_seed(seed)
        vf = ReweightedVectorField(net_joint, net_prior, gamma=g)
        x = sample(vf, c, n_steps=n_steps)
        J, E = compute_J_E(x, c)
        rows.append({"gamma": g, "J": J, "E": E})
    return rows


def main():
    p = argparse.ArgumentParser(description="Eval FM Burgers checkpoints (FOPC)")
    p.add_argument("--dataset", type=str, default="free_u_f_paper_fopc")
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.5])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--vanilla_joint", type=str, default=None)
    p.add_argument("--vanilla_prior", type=str, default=None)
    p.add_argument("--ot_joint", type=str, default=None)
    p.add_argument("--ot_prior", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="flow/results/paper_fopc")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[warn] cuda unavailable → {args.device}")

    c = make_test_batch(args.dataset, args.n_test, args.device)
    print(f"held-out test batch: {tuple(c.shape)} (first {args.n_test} of {args.dataset} test)")

    variants = {}
    for name, vj, vp in [("vanilla", args.vanilla_joint, args.vanilla_prior),
                         ("OT-CFM",  args.ot_joint,      args.ot_prior)]:
        nj = load_net(vj, args.dim, args.dim_mults, args.device)
        if nj is None:
            continue
        npr = load_net(vp, args.dim, args.dim_mults, args.device)
        print(f"\n=== {name} (joint={'ok' if nj else 'none'}, prior={'ok' if npr else 'none'}) ===")
        rows = sweep_variant(nj, npr, c, args.gammas, args.n_steps, args.seed)
        variants[name] = rows
        for r in rows:
            print(f"  γ={r['gamma']:<4} J={r['J']:.5f}  E={r['E']:.1f}")

    # comparison table
    print("\n" + "=" * 64)
    print(f"{'γ':>5} | " + " | ".join(f"{n} J" for n in variants) +
          f" | paper FOPC")
    print("-" * 64)
    for i, g in enumerate(args.gammas):
        cells = " | ".join(f"{variants[n][i]['J']:.5f}" for n in variants)
        print(f"{g:>5} | {cells} | {PAPER_FOPC_J} (γ-free)")
    print("=" * 64)
    print(f"paper DiffPhyCon FO-PC J_actual = {PAPER_FOPC_J} (Table 1).")
    print("→ FM matches paper if best J ~0.0004-0.001; OT wins if OT J < vanilla J consistently.")

    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    try:
        import pandas as pd
        recs = [{"variant": n, **r} for n, rows in variants.items() for r in rows]
        df = pd.DataFrame(recs)
        csv = os.path.join(ROOT, args.out_dir, "paper_fopc_sweep.csv")
        df.to_csv(csv, index=False)
        print(f"💾 saved {csv}")
    except ImportError:
        print("(pandas not installed — skipped csv)")


if __name__ == "__main__":
    main()
