"""
flow/lab_four_explore.py — shared helper module for Burgers FM explore notebooks.

Used by (downstream consumers):
- flow/lab_four_ood.ipynb   (Q3 OOD test)
- flow/lab_four_jgrad.ipynb (Q1 J-gradient guidance)
- flow/lab_four_popc.ipynb  (Q2 POPC variant)

Provides:
- `load_fm(name)`                — one-liner checkpoint loader (registry-based)
- `LiveLossTrainer`              — Trainer with matplotlib live-update loss plot
- `infer(net_joint, c, ...)`     — unified inference (joint + optional prior + optional J-grad)
- `sweep(configs, infer_fn)`     — runs configs, returns pandas DataFrame
- `JGradEulerSampler`            — Euler sampler with J-grad guidance (uses FM Tweedie)
- `plot_trajectory_grid(...)`    — N-row × 4-panel comparison plot
- `savefig(fig, workstream, name)` — auto-save to flow/results/<workstream>/<ts>_<name>.png

This file IMPORTS lab_four primitives but never modifies them.
"""

from __future__ import annotations
import math
import os
import sys
from datetime import datetime
from typing import Callable, Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# --- path setup (works as .py or in Jupyter cell) ---
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.path.abspath(os.getcwd())
    if os.path.basename(HERE) != "flow":
        HERE = os.path.join(HERE, "flow")
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================================
# Re-exports from lab_four_solved (auto-generated from lab_four.ipynb)
# ============================================================================
#
# lab_four.py is the FILL-IN scaffold (with raise NotImplementedError).
# lab_four.ipynb is where the user fills in the answers — that's the source
# of truth for working code.
#
# To make working code importable, we auto-regenerate `flow/lab_four_solved.py`
# from `flow/lab_four.ipynb` whenever the .ipynb is newer.

def _is_safe_assign(node) -> bool:
    """True if an assignment's RHS won't trigger expensive user code on import.

    Safe:   literals, names, binops, and calls to stdlib modules
            (os.*, sys.*, torch.*, np.*, math.*) — e.g. HERE = os.path.dirname(...)
    Unsafe: a bare-Name call like `net_joint = train_joint_for_part5(...)` —
            importing that would re-run training!
    """
    import ast
    val = node.value
    if not isinstance(val, ast.Call):
        return True
    func = val.func
    if isinstance(func, ast.Attribute):
        root = func
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in ("os", "sys", "torch", "np", "math", "F", "nn"):
            return True
    return False  # bare user-function call → unsafe


def _ensure_lab_four_solved():
    """Regenerate flow/lab_four_solved.py from flow/lab_four.ipynb if stale.

    Keeps ONLY definitions (def / class / import / safe-assign / if-guards).
    Drops all top-level EXECUTION (bare calls, loops, train_*(), sanity_check_*())
    so importing the generated module never re-runs the user's training cells.
    """
    import ast
    import nbformat

    ipynb_path = os.path.join(HERE, "lab_four.ipynb")
    solved_path = os.path.join(HERE, "lab_four_solved.py")
    if not os.path.isfile(ipynb_path):
        raise FileNotFoundError(
            f"{ipynb_path} not found. This explore module needs the user's "
            f"filled-in lab_four.ipynb (not lab_four.py — that's the scaffold)."
        )
    ipynb_mtime = os.path.getmtime(ipynb_path)
    solved_mtime = os.path.getmtime(solved_path) if os.path.isfile(solved_path) else 0
    if solved_mtime >= ipynb_mtime:
        return  # already fresh

    nb = nbformat.read(ipynb_path, as_version=4)
    # Concatenate code cells, stripping jupyter magics (%/!) which break ast.parse
    sources = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        lines = [ln for ln in cell.source.split("\n")
                 if not ln.lstrip().startswith(("%", "!"))]
        sources.append("\n".join(lines))
    full = "\n\n".join(sources)

    tree = ast.parse(full)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.Try)):
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if _is_safe_assign(node):
                keep.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            keep.append(node)  # module/section docstring
        # else: drop (bare Expr(Call), For, While, With, unsafe Assign)

    new_module = ast.Module(body=keep, type_ignores=[])
    body_source = ast.unparse(new_module)
    header = (
        "# AUTO-GENERATED from flow/lab_four.ipynb by lab_four_explore._ensure_lab_four_solved()\n"
        "# DO NOT EDIT BY HAND — regenerated whenever the .ipynb is newer.\n"
        "# Only DEFINITIONS are kept; top-level execution (train_*, sanity_check_*) is dropped.\n\n"
    )
    with open(solved_path, "w") as f:
        f.write(header + body_source + "\n")
    print(f"  🔄 regenerated {solved_path} (definitions only, {len(keep)} top-level nodes)")


_ensure_lab_four_solved()


from flow.lab_four_solved import (
    # Core FM scaffolding
    GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta,
    VectorFieldNet, Trainer, EMA, finetune_with_ema,
    # Burgers-specific
    BurgersDataset, BurgersFlowTrainer, BurgersVectorField, BurgersEulerSampler,
    BurgersPriorDataset, BurgersPriorTrainer, ReweightedVectorField,
    inpaint_overwrite, w_scheduler_fm,
    # Evaluation
    compute_J_and_energy, simulate_with_predicted_w,
    visualize_trajectory_with_simulation,
    # Data loading
    load_burgers_train, load_burgers_test,
    BURGERS_DATASET_NAME, T_IDX,
)


# ============================================================================
# Device + checkpoint registry
# ============================================================================

# Checkpoints live in flow/flow/checkpoints/ (relative double-flow because
# training was launched with CWD=flow/ so save_path was 'flow/checkpoints/').
CHECKPOINTS_DIR = os.path.join(HERE, "flow", "checkpoints")

CHECKPOINT_REGISTRY: Dict[str, str] = {
    "joint":             "fm_joint_step25000.pt",
    "joint_ema":         "fm_joint_ema.pt",
    "joint_ema_large":   "fm_joint_ema_large.pt",
    "prior":             "fm_prior_step6250.pt",
    "prior_ema":         "fm_prior_ema.pt",
    "prior_ema_large":   "fm_prior_ema_large.pt",
}


def get_device(device: str = "auto") -> str:
    """Resolve 'auto' → mps if available, else cuda, else cpu."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_fm(
    name: str,
    device: str = "auto",
    dim: int = 64,
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
    verbose: bool = True,
) -> nn.Module:
    """One-liner FM checkpoint load.

    Args:
        name: key in CHECKPOINT_REGISTRY  (e.g. 'joint_ema', 'prior_ema')
        device: 'auto' / 'mps' / 'cuda' / 'cpu'
        dim, dim_mults: must match the architecture the ckpt was trained with
                        (default = paper baseline FOPC: dim=64, dim_mults=(1,2,4,8))

    Returns: nn.Module in .eval() mode on the chosen device.
    """
    if name not in CHECKPOINT_REGISTRY:
        raise KeyError(f"Unknown checkpoint '{name}'. Available: {list(CHECKPOINT_REGISTRY)}")
    device = get_device(device)
    path = os.path.join(CHECKPOINTS_DIR, CHECKPOINT_REGISTRY[name])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Expected layout: {CHECKPOINTS_DIR}/<file>.pt — see CHECKPOINT_REGISTRY."
        )
    net = BurgersVectorField(dim=dim, dim_mults=dim_mults).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(sd)
    net.eval()
    if verbose:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  ✅ load_fm('{name}'): {os.path.basename(path)}  on {device}  ({n_params:,} params)")
    return net


# ============================================================================
# LiveLossTrainer — Trainer with matplotlib live-update loss plot
# ============================================================================


class _LivePlotMixin:
    """Mixin that overrides `train()` to draw a live matplotlib loss curve in a
    Jupyter cell, updating every `plot_every` steps. Mix into any Trainer
    subclass that defines `get_train_loss`, `opt`, `net`, `loss_history`.
    """
    _prior_plot = False  # subclasses set True for prior-model title

    def train(self, num_steps: int, batch_size: int = 64,
              print_every: int = 100, plot_every: int = 200):
        from IPython.display import clear_output, display
        import matplotlib.pyplot as plt
        from tqdm.auto import tqdm

        fig, ax = plt.subplots(figsize=(9, 3.5))
        self.net.train()
        desc = "train (prior)" if self._prior_plot else "train"
        pbar = tqdm(range(num_steps), desc=desc, leave=True)
        for step in pbar:
            self.opt.zero_grad()
            loss = self.get_train_loss(batch_size)
            loss.backward()
            self.opt.step()
            self.loss_history.append(loss.item())
            window = min(print_every, len(self.loss_history))
            avg = float(np.mean(self.loss_history[-window:]))
            pbar.set_postfix({"loss": f"{loss.item():.4f}", f"avg{window}": f"{avg:.4f}"})
            if (step + 1) % plot_every == 0 or step == num_steps - 1:
                ax.clear()
                _plot_loss_curve(ax, self.loss_history, num_steps, step + 1,
                                 prior=self._prior_plot)
                clear_output(wait=True)
                display(fig)
        return self.loss_history


class LiveLossTrainer(_LivePlotMixin, BurgersFlowTrainer):
    """BurgersFlowTrainer with a live loss plot. Same constructor signature.

    Example:
        trainer = LiveLossTrainer(net, path, ds, lr=1e-4)
        trainer.train(num_steps=25000, batch_size=64, plot_every=200)
    """
    pass


class LiveLossTrainerPrior(_LivePlotMixin, BurgersPriorTrainer):
    """Live-loss-plot prior trainer."""
    _prior_plot = True


def _plot_loss_curve(ax, history: list, total_steps: int, current_step: int,
                     prior: bool = False):
    """Draw raw + smoothed loss curve."""
    ax.semilogy(history, alpha=0.3, color="C0", label="raw")
    if len(history) >= 100:
        window = np.convolve(history, np.ones(100) / 100, mode="valid")
        ax.semilogy(np.arange(99, len(history)), window, color="C1", label="smoothed (100)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    title_prefix = "prior FM" if prior else "joint FM"
    ax.set_title(f"{title_prefix} training: step {current_step:,} / {total_steps:,}")


# ============================================================================
# J-gradient sampler (used by Phase B / Q1)
# ============================================================================


def _get_jgrad_scheduler(name: str) -> Callable[[float], float]:
    """Returns f(τ) ∈ [0, 1] used as a multiplier on the J-gradient term.

    For FM: τ=0 = noise (no guidance, model is far from clean basin),
            τ=1 = clean (full guidance, push toward minimum J).
    So we want sched(0) ≈ 0 and sched(1) ≈ 1 (monotone increasing).
    """
    if name == "constant":
        return lambda tau: 1.0
    elif name == "linear":
        return lambda tau: float(tau)
    elif name == "cosine":
        return lambda tau: 0.5 * (1.0 - math.cos(math.pi * float(tau)))
    elif name == "off":
        return lambda tau: 0.0
    else:
        raise ValueError(f"Unknown J-grad scheduler '{name}'. "
                         f"Available: constant / linear / cosine / off")


class JGradEulerSampler(BurgersEulerSampler):
    """Euler ODE sampler with J-gradient guidance term.

    Per-step update:
        x_τ        ← inpaint_overwrite(x_τ, c)
        v_FM       ← net(x_τ, t, c)
        z_est      ← x_τ + (1-τ)·v_FM           (FM Tweedie for CondOT path)
        ∇_x J      ← autograd(J(z_est, c), x_τ)
        v_total    ← v_FM - ∇_x J · sched(τ)    (sign: minus to MINIMIZE J)
        x_{τ+dτ}  ← x_τ + v_total · dτ

    The `wfs` (control energy weight) and `wu` (boundary-match weight) are
    baked INTO the loss function — they scale the gradient. The `sched(τ)`
    is an additional time-dependent ramp.

    Reduces to plain BurgersEulerSampler when wfs == wu == 0.
    """
    def __init__(self, net, n_steps: int = 100, tau_min: float = 1e-3,
                 wfs: float = 0.0, wu: float = 0.0,
                 j_scheduler_name: str = "cosine",
                 partially_observed: Optional[str] = None,
                 rescaler: float = 10.0):
        super().__init__(net, n_steps=n_steps, tau_min=tau_min)
        self.wfs = float(wfs)
        self.wu = float(wu)
        self.sched = _get_jgrad_scheduler(j_scheduler_name)
        self.partially_observed = partially_observed
        self.rescaler = float(rescaler)

    def _build_loss_fn(self, c: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """Construct J(x) for THIS batch's c. Returns scalar tensor (sum over batch).

        Adapts paper's `ddpm_guidance_loss`: J = wu·u_endpoint_match + wfs·||f||²
        Both `c` and `x` are NORMALIZED (rescaler-divided). The loss compares
        normalized values throughout — no unscaling needed inside loss_fn.
        """
        from utils import ddpm_guidance_loss

        b = c.shape[0]
        Nx = c.shape[-1]
        # u_target shape (b, 11, 128) — only row 0 (u_0) and row 10 (u_T*) matter.
        # Pass NORMALIZED c (no *rescaler) because loss_fn compares to NORMALIZED x.
        u_target = torch.zeros(b, 11, Nx, device=c.device, dtype=c.dtype)
        u_target[:, 0]  = c[:, 0]
        u_target[:, 10] = c[:, 1]

        wfs, wu = self.wfs, self.wu
        po = self.partially_observed

        def J(x: torch.Tensor) -> torch.Tensor:
            # x: (b, 2, 16, 128) NORMALIZED — same scale as u_target
            return ddpm_guidance_loss(
                u_target,
                x[:, 0, :11, :],
                x[:, 1, :10, :],
                wu=wu, wf=wfs,
                partially_observed=po,
            )
        return J

    def sample(self, c: torch.Tensor,
               shape: Tuple[int, ...] = (2, 16, 128)) -> torch.Tensor:
        b = c.shape[0]
        device = c.device
        dtau = (1.0 - 2 * self.tau_min) / self.n_steps

        loss_fn = self._build_loss_fn(c) if (self.wfs > 0 or self.wu > 0) else None

        x = torch.randn(b, *shape, device=device)
        for i in range(self.n_steps):
            tau = self.tau_min + i * dtau
            x = inpaint_overwrite(x, c)
            t_batch = torch.full((b,), tau, device=device)

            if loss_fn is None:
                with torch.no_grad():
                    v_fm = self.net(x, t_batch, c)
                v_total = v_fm
            else:
                # autograd through Tweedie
                x_grad = x.detach().requires_grad_(True)
                with torch.enable_grad():
                    v_fm = self.net(x_grad, t_batch, c)
                    z_est = x_grad + (1.0 - tau) * v_fm   # FM Tweedie for α=τ, β=1-τ
                    J_val = loss_fn(z_est)
                    if J_val.dim() > 0:
                        J_scalar = J_val.sum()
                    else:
                        J_scalar = J_val
                    nabla_J = torch.autograd.grad(J_scalar, x_grad,
                                                  retain_graph=False)[0]
                # minus sign → push x toward LOWER J
                v_total = v_fm.detach() - nabla_J.detach() * self.sched(tau)

            x = (x + v_total * dtau).detach()
        x = inpaint_overwrite(x, c)
        return x


# ============================================================================
# Unified inference interface
# ============================================================================


def infer(
    net_joint: nn.Module,
    c: torch.Tensor,
    net_prior: Optional[nn.Module] = None,
    gamma: float = 1.0,
    n_steps: int = 100,
    seed: Optional[int] = 42,
    # J-grad options
    wfs: float = 0.0,
    wu: float = 0.0,
    j_scheduler_name: str = "cosine",
    # POPC option
    partially_observed: Optional[str] = None,
    # sampler override (e.g. BurgersPOPCEulerSampler for POPC inference)
    sampler_cls: Optional[type] = None,
) -> dict:
    """Build sampler + run inference + compute J/E. Returns a results dict.

    Args:
        net_joint: the joint FM model (required)
        c: (b, 2, 128) boundary condition, NORMALIZED
        net_prior: optional prior FM model (for γ != 1)
        gamma: prior reweighting strength. 1.0 = pure joint, > 1.0 = prior boost
        n_steps: Euler ODE steps
        seed: torch.manual_seed before sampling (for reproducibility)
        wfs, wu: J-gradient weights. If both 0, uses plain BurgersEulerSampler.
                 If > 0, uses JGradEulerSampler.
        j_scheduler_name: 'cosine' / 'linear' / 'constant' / 'off'
        partially_observed: 'front_rear_quarter' for POPC, None for FOPC

    Returns:
        dict with keys: x_pred (cpu tensor), c (cpu), J, E, gamma, n_steps,
                        seed, wfs, wu, use_jgrad, partially_observed
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Build velocity field
    if net_prior is None or gamma == 1.0:
        net = net_joint
    else:
        net = ReweightedVectorField(net_joint, net_prior, gamma=gamma)

    # Build sampler
    use_jgrad = (wfs > 0 or wu > 0)
    if sampler_cls is not None:
        # explicit override (e.g. BurgersPOPCEulerSampler). J-grad not combined here.
        sampler = sampler_cls(net, n_steps=n_steps)
    elif use_jgrad:
        sampler = JGradEulerSampler(
            net, n_steps=n_steps, wfs=wfs, wu=wu,
            j_scheduler_name=j_scheduler_name,
            partially_observed=partially_observed,
        )
    else:
        sampler = BurgersEulerSampler(net, n_steps=n_steps)

    x_pred = sampler.sample(c)
    J, E = compute_J_and_energy(x_pred, c)

    return {
        "x_pred": x_pred.detach().cpu(),
        "c": c.detach().cpu(),
        "J": float(J),
        "E": float(E),
        "gamma": float(gamma),
        "n_steps": int(n_steps),
        "seed": seed,
        "wfs": float(wfs),
        "wu": float(wu),
        "use_jgrad": bool(use_jgrad),
        "partially_observed": partially_observed,
    }


# ============================================================================
# Sweep runner → pandas DataFrame
# ============================================================================


def sweep(
    configs: List[Dict[str, Any]],
    infer_fn: Callable[..., dict] = infer,
    verbose: bool = True,
    fixed_kwargs: Optional[Dict[str, Any]] = None,
) -> "pd.DataFrame":
    """Run infer_fn for each config, return pandas DataFrame.

    Args:
        configs: list of dicts. Each dict is **kwargs passed to infer_fn.
                 e.g. [{"gamma": 1.0, "wfs": 0}, {"gamma": 2.5, "wfs": 1.0}]
        infer_fn: defaults to `infer`. Must return dict with J, E keys.
        fixed_kwargs: additional kwargs added to EVERY config (e.g. net_joint, c)

    Returns:
        DataFrame with one row per config. Columns = config keys + J + E.
    """
    import pandas as pd
    fixed_kwargs = fixed_kwargs or {}
    rows = []
    for i, cfg in enumerate(configs):
        merged = {**fixed_kwargs, **cfg}
        result = infer_fn(**merged)
        # row = config keys (not fixed) + J + E
        row = {**cfg, "J": result["J"], "E": result["E"]}
        rows.append(row)
        if verbose:
            kvs = " | ".join(f"{k}={v}" for k, v in cfg.items())
            print(f"  [{i+1}/{len(configs)}] {kvs}  →  J={result['J']:.5f}  E={result['E']:.1f}")
    return pd.DataFrame(rows)


# ============================================================================
# Plot helpers
# ============================================================================


def savefig(fig, workstream: str, name: str, dpi: int = 100) -> str:
    """Save fig to flow/results/<workstream>/<timestamp>_<name>.png. Returns path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "flow", "results", workstream)
    os.makedirs(out_dir, exist_ok=True)
    safe_name = name.replace("/", "_").replace(" ", "_")
    path = os.path.join(out_dir, f"{ts}_{safe_name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  💾 saved {path}")
    return path


def plot_trajectory_grid(
    results: List[dict],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_sample_idx: int = 0,
):
    """N rows × 4 panels for N inference results. Each row shows one result.

    4 panels per row:
        [0] predicted u(t, x) heatmap
        [1] predicted w(t, x) heatmap
        [2] simulated u(t, x) heatmap (PDE solve from u_0 + predicted w)
        [3] terminal-state comparison: u_T* vs sim u(T) vs pred u(T)

    Each result is a dict from `infer(...)`. show_sample_idx picks which
    sample to plot when results have batch > 1.
    """
    import matplotlib.pyplot as plt

    N = len(results)
    titles = titles or [f"row {i}" for i in range(N)]
    fig, axes = plt.subplots(N, 4, figsize=(20, 4 * N), squeeze=False)

    rescaler = 10.0
    for r, (result, title) in enumerate(zip(results, titles)):
        x_pred = result["x_pred"][show_sample_idx : show_sample_idx + 1]
        c      = result["c"]    [show_sample_idx : show_sample_idx + 1]

        # PDE simulate from predicted w
        u_sim_full, w_used = simulate_with_predicted_w(x_pred, c, rescaler=rescaler)
        # shapes: u_sim_full (1, 11, 128); w_used (1, 10, 128) in PHYSICAL units

        u_pred = (x_pred[0, 0, :11, :] * rescaler).numpy()  # (11, 128)
        w_pred = (x_pred[0, 1, :10, :] * rescaler).numpy()  # (10, 128)
        u_sim  = u_sim_full[0].numpy()                       # (11, 128)
        u0     = (c[0, 0] * rescaler).numpy()                # (128,)
        uT_star = (c[0, 1] * rescaler).numpy()               # (128,)

        # Panel 0: predicted u
        im0 = axes[r, 0].imshow(u_pred, aspect="auto", cmap="RdBu_r")
        axes[r, 0].set_title(f"{title}\npred u(t,x)", fontsize=10)
        axes[r, 0].set_xlabel("x"); axes[r, 0].set_ylabel("t")
        plt.colorbar(im0, ax=axes[r, 0], fraction=0.046)

        # Panel 1: predicted w
        im1 = axes[r, 1].imshow(w_pred, aspect="auto", cmap="RdBu_r")
        axes[r, 1].set_title("pred w(t,x)", fontsize=10)
        axes[r, 1].set_xlabel("x"); axes[r, 1].set_ylabel("t")
        plt.colorbar(im1, ax=axes[r, 1], fraction=0.046)

        # Panel 2: simulated u (from w_pred via real PDE)
        im2 = axes[r, 2].imshow(u_sim, aspect="auto", cmap="RdBu_r")
        axes[r, 2].set_title("sim u(t,x)  (PDE solve)", fontsize=10)
        axes[r, 2].set_xlabel("x"); axes[r, 2].set_ylabel("t")
        plt.colorbar(im2, ax=axes[r, 2], fraction=0.046)

        # Panel 3: terminal comparison
        x_axis = np.arange(128)
        axes[r, 3].plot(x_axis, uT_star, "k-",  lw=2, label="target u_T*")
        axes[r, 3].plot(x_axis, u_sim[-1], "b--", lw=1.5, label="sim u(T)")
        axes[r, 3].plot(x_axis, u_pred[10], "o", color="orange", ms=3, label="pred u(T)  [inpainted]")
        axes[r, 3].plot(x_axis, u0, color="gray", lw=0.5, alpha=0.7, label="u_0")
        axes[r, 3].set_title(f"terminal:  J={result['J']:.4f}  E={result['E']:.1f}", fontsize=10)
        axes[r, 3].legend(fontsize=7, loc="best"); axes[r, 3].grid(alpha=0.3)
        axes[r, 3].set_xlabel("x")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"  💾 saved {save_path}")
    return fig


def build_c_from_u0_uT(u0: np.ndarray, uT_star: np.ndarray,
                       device: str = "cpu", rescaler: float = 10.0) -> torch.Tensor:
    """Build c tensor of shape (1, 2, 128) from custom u_0 and u_T* arrays.

    Args:
        u0:      (128,) numpy array in PHYSICAL units (not normalized)
        uT_star: (128,) numpy array in PHYSICAL units
        device:  target device
        rescaler: dataset normalization factor (default 10.0 for Burgers)

    Returns: c tensor (1, 2, 128) on `device`, NORMALIZED (divided by rescaler).
    """
    assert u0.shape == (128,), f"u0 must be shape (128,), got {u0.shape}"
    assert uT_star.shape == (128,), f"uT_star must be shape (128,), got {uT_star.shape}"
    c_np = np.stack([u0, uT_star], axis=0)[None]  # (1, 2, 128)
    c = torch.from_numpy(c_np).float().to(device) / rescaler
    return c


# ============================================================================
# Canonical held-out evaluation batch
# ============================================================================
#
# ⚠️ The FM models were trained on `free_u_f_1e4_front_rear_quarter` (8k train /
# 2k test). For an HONEST, paper-aligned comparison we must evaluate on the
# *held-out TEST split* of THAT dataset — NOT on training samples, and NOT on the
# tiny `free_u_f_1e5` set (only 160/40) that `BURGERS_DATASET_NAME` defaults to.
#
# The paper's DDPM baseline (`run_gamma_sweep_FOPC_paper.sh`) used
# `--dataset free_u_f_1e4 --n_test_samples 8`, and `get_target` takes the FIRST 8
# test samples (indices 0..7). So to compare against those exact numbers we take
# the SAME first-N test samples deterministically (not a random seed).

EVAL_DATASET = "free_u_f_1e4_front_rear_quarter"   # dataset the FM was trained on


def make_eval_batch(n: int = 8, dataset: str = EVAL_DATASET,
                    split: str = "test", device: str = "auto",
                    partially_observed: Optional[str] = None) -> torch.Tensor:
    """Return c (n, 2, 128) for the FIRST `n` samples of the held-out test split.

    Deterministic (indices 0..n-1) so it matches the paper's `get_target(0..n-1)`
    convention — letting us compare directly against the DDPM baseline numbers.

    Returns NORMALIZED c (the FM model's input space; ÷rescaler already applied
    by the dataset loader).

    partially_observed: if 'front_rear_quarter' (POPC), zero the unobserved middle
        50% of c so only the front/rear quarter boundary is given.
    """
    device = get_device(device)
    if split == "test":
        ds_raw = load_burgers_test(device=device, dataset=dataset)
    else:
        ds_raw = load_burgers_train(device=device, dataset=dataset)
    ds = BurgersDataset(ds_raw, device=device)
    # first n samples in file order (NOT random) → matches get_target(0..n-1)
    z = ds.all_z[:n]                                   # (n, 2, 16, 128)
    c = torch.stack([z[:, 0, 0, :], z[:, 0, T_IDX, :]], dim=1)   # (n, 2, 128)
    if partially_observed == "front_rear_quarter":
        Nx = c.shape[-1]
        c = c.clone()
        c[:, :, Nx // 4 : (3 * Nx) // 4] = 0.0          # zero unobserved middle 50%
    elif partially_observed not in (None, "full"):
        raise ValueError(f"unknown partially_observed: {partially_observed}")
    return c


# ============================================================================
# Sanity check (run at import time? no, separate function)
# ============================================================================


def selftest(device: str = "auto"):
    """Quick smoke test: load joint EMA + 1 inference + 1 OOD inference."""
    print("=== lab_four_explore selftest ===")
    device = get_device(device)
    print(f"device: {device}")

    # Load
    net = load_fm("joint_ema", device=device)

    # In-distribution: 1 train sample
    ds = load_burgers_train(device=device)
    ds_wrap = BurgersDataset(ds, device=device)
    _, c = ds_wrap.sample(1)
    result = infer(net, c, gamma=1.0, n_steps=20, seed=42)
    print(f"  in-dist 20-step: J={result['J']:.5f}  E={result['E']:.1f}")

    # Quick OOD: random-ish u_0
    rng = np.random.RandomState(0)
    u0 = 1.5 * np.sin(2 * np.pi * np.linspace(0, 1, 128))
    uT = 0.5 * u0
    c_ood = build_c_from_u0_uT(u0, uT, device=device)
    result_ood = infer(net, c_ood, gamma=1.0, n_steps=20, seed=42)
    print(f"  OOD (sin) 20-step: J={result_ood['J']:.5f}  E={result_ood['E']:.1f}")

    # J-grad smoke (wfs=0 must equal no J-grad)
    r_jg_0 = infer(net, c, gamma=1.0, n_steps=20, seed=42, wfs=0.0)
    r_plain = infer(net, c, gamma=1.0, n_steps=20, seed=42)
    diff = abs(r_jg_0["J"] - r_plain["J"])
    assert diff < 1e-6, f"wfs=0 path should match plain sampler exactly, diff={diff}"
    print(f"  wfs=0 ≡ plain sampler ✓ (J diff = {diff:.2e})")

    print("=== selftest passed ===")


if __name__ == "__main__":
    selftest()
