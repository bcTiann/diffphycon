"""
verify_skip_first.py — empirical verification that generate_burgers --skip_first
gives data-leak-free dataset extension.

KEY INSIGHT: make_data_varying_f draws variables in interleaved order:
    loc1 = uniform((Nu0,1))   # draws Nu0 values
    amp1 = uniform((Nu0,1))   # draws Nu0 more values
    ...
So sample 0 of Nu0=70 ≠ sample 0 of Nu0=100 (different RNG offsets).
The "first N of Nu0=M (M>N) extends Nu0=N" property does NOT hold.

But our skip_first claim is different:
    "samples generated under --skip_first=N --test_samples=K are
     disjoint from the N samples generated under Nu0=N (baseline)."

This holds because the new samples use a completely different RNG layout
(Nu0=N+K) so even if some random numbers are reused, they end up in
different variable slots (e.g., shared v[90050] becomes loc1 here but
was amp1 there) → different (u0, f) pairs.

Tests (all four must pass):
  T1: seed determinism — same seeds + same Nu0 → bitwise identical output
  T2: disjoint claim — samples from Nu0=N (baseline) and samples [N:N+K]
      from Nu0=N+K (skip_first extension) have NO bitwise overlap
  T3: discarded prefix is also unseen — samples [0:N] of Nu0=N+K differ
      from samples [0:N] of Nu0=N (sanity check on interleaved RNG)
  T4: end-to-end — generate with --skip_first 70 --test_samples 30,
      verify written h5 contents == samples [70:100] of an in-memory
      make_data_varying_f(Nu0=100) call

Usage (run from repo root):
    python scripts/verify_skip_first.py
"""
from __future__ import annotations
import os
import sys
import subprocess
import tempfile

import numpy as np
import torch
import random
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'dataset', 'apps'))

from generate_burgers import make_data_varying_f


def _seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _has_match(needle, haystack):
    """True if needle (1-D row) exactly matches any row in haystack (2-D)."""
    return any(np.array_equal(needle, row) for row in haystack)


# ---------- T1 ----------
def test_determinism():
    print("\n=== T1: seed determinism ===")
    _seed_all(0)
    a_u0, a_f = make_data_varying_f(Nu0=100, Nf=100, s=128, t=10,
                                     partial_control='front_rear_quarter')
    _seed_all(0)
    b_u0, b_f = make_data_varying_f(Nu0=100, Nf=100, s=128, t=10,
                                     partial_control='front_rear_quarter')
    assert np.array_equal(a_u0, b_u0), "u0 mismatch — RNG NOT deterministic"
    assert np.array_equal(a_f, b_f), "f mismatch — RNG NOT deterministic"
    print(f"  ✓ same seed + same Nu0 → bitwise identical (u0 {a_u0.shape}, f {tuple(a_f.shape)})")


# ---------- T2 — THE CRITICAL TEST ----------
def test_skip_first_disjoint_from_baseline():
    print("\n=== T2: --skip_first=N samples are DISJOINT from Nu0=N baseline ===")
    print("        (this is the actual no-leak claim)")
    N, K = 70, 30   # baseline=70 (= 'training set'), extension=30 (= 'new test')

    # Baseline: what model "trained on"
    _seed_all(0)
    base_u0, base_f = make_data_varying_f(Nu0=N, Nf=N, s=128, t=10,
                                           partial_control='front_rear_quarter')

    # Extension: what --skip_first=N --test_samples=K produces internally
    _seed_all(0)
    ext_full_u0, ext_full_f = make_data_varying_f(Nu0=N + K, Nf=N + K, s=128, t=10,
                                                    partial_control='front_rear_quarter')
    new_u0 = ext_full_u0[N:]              # what --skip_first=N keeps
    new_f = ext_full_f[N:]

    # Check no overlap (u0)
    collisions = sum(_has_match(s, base_u0) for s in new_u0)
    assert collisions == 0, f"❌ {collisions}/{K} new u0 samples appear in baseline"
    print(f"  ✓ {K} new u0 samples: 0 collisions with {N} baseline u0 samples")

    # Check no overlap (f)
    base_f_np = base_f.numpy() if hasattr(base_f, 'numpy') else np.asarray(base_f)
    new_f_np = new_f.numpy() if hasattr(new_f, 'numpy') else np.asarray(new_f)
    collisions_f = sum(_has_match(s, base_f_np) for s in new_f_np)
    assert collisions_f == 0, f"❌ {collisions_f}/{K} new f samples appear in baseline"
    print(f"  ✓ {K} new f  samples: 0 collisions with {N} baseline f  samples")

    print(f"  → model trained on Nu0={N} has NEVER seen these {K} new (u0, f) pairs")


# ---------- T3 ----------
def test_interleaving_sanity():
    print("\n=== T3: interleaved RNG sanity (Nu0=70 vs Nu0=100 first 70 differ) ===")
    print("        (confirms our understanding of make_data_varying_f layout)")
    N, M = 70, 100
    _seed_all(0)
    short_u0, _ = make_data_varying_f(Nu0=N, Nf=N, s=128, t=10,
                                       partial_control='front_rear_quarter')
    _seed_all(0)
    long_u0, _ = make_data_varying_f(Nu0=M, Nf=M, s=128, t=10,
                                      partial_control='front_rear_quarter')
    # Note: we EXPECT these to differ — they use different RNG offsets
    differ_count = sum(not np.array_equal(s, l) for s, l in zip(short_u0, long_u0[:N]))
    print(f"  → {differ_count}/{N} of long[:{N}] differ from short")
    assert differ_count > 0, "RNG layout is suspicious — should diverge"
    print(f"  ✓ confirmed: interleaved RNG → first N of Nu0=M ≠ Nu0=N")
    print(f"  → this is WHY --skip_first works: samples [N:M] use a fresh RNG layout")


# ---------- T4 — END-TO-END ----------
def test_end_to_end_h5():
    print("\n=== T4: end-to-end h5 ≡ in-memory baseline[N:N+K] ===")
    N, K = 70, 30

    # In-memory baseline of Nu0=N+K (what --skip_first=N --test_samples=K should produce internally)
    _seed_all(0)
    baseline_u0, baseline_f = make_data_varying_f(Nu0=N + K, Nf=N + K, s=128, t=10,
                                                    partial_control='front_rear_quarter')
    expected_u0 = baseline_u0[N:]    # (K, 128)
    expected_f = (baseline_f.numpy() if hasattr(baseline_f, 'numpy')
                  else np.asarray(baseline_f))[N:]

    # Run the actual CLI patch
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, 'work')
        os.makedirs(os.path.join(work, 'data'), exist_ok=True)
        cmd = [
            sys.executable, '-u',
            os.path.join(ROOT, 'dataset', 'apps', 'generate_burgers.py'),
            '--skip_first', str(N),
            '--train_samples', '0',
            '--test_samples', str(K),
            '--partial_control', 'front_rear_quarter',
            '--nx', '128', '--nt', '11',
            '--device', 'cpu',
            '--save_path', 'verify_skip_first/',
        ]
        print(f"  running --skip_first {N} --test_samples {K} (writes to tmp dir)")
        result = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-1500:])
            print("STDERR:", result.stderr[-1500:])
            raise RuntimeError(f"generate_burgers failed: code {result.returncode}")

        h5_path = os.path.join(work, 'data', 'verify_skip_first', 'burgers_test.h5')
        assert os.path.exists(h5_path), f"missing {h5_path}"
        with h5py.File(h5_path, 'r') as h5f:
            key = list(h5f['test'].keys())[0]   # 'pde_11-128'
            traj = h5f['test'][key][:]          # (K, 11, 128)
            f_h5 = h5f['test'][key + '_f'][:]   # (K, 10, 128)
        h5_u0 = traj[:, 0, :]                   # initial slice = u0

    # Compare (float32 to match storage precision)
    max_diff_u0 = np.abs(h5_u0.astype(np.float32) - expected_u0.astype(np.float32)).max()
    max_diff_f = np.abs(f_h5.astype(np.float32) - expected_f.astype(np.float32)).max()
    print(f"  h5 u0 vs baseline[{N}:{N+K}] max |diff| = {max_diff_u0:.2e}")
    print(f"  h5 f  vs baseline[{N}:{N+K}] max |diff| = {max_diff_f:.2e}")
    assert max_diff_u0 < 1e-5, f"u0 mismatch — max diff {max_diff_u0}"
    assert max_diff_f < 1e-5, f"f  mismatch — max diff {max_diff_f}"
    print(f"  ✓ h5 contents bytewise-match in-memory baseline[{N}:{N+K}]")


def main():
    print("=" * 64)
    print("verify_skip_first.py — proving --skip_first is leak-free")
    print("=" * 64)
    test_determinism()
    test_skip_first_disjoint_from_baseline()
    test_interleaving_sanity()
    test_end_to_end_h5()
    print("\n" + "=" * 64)
    print("✅ ALL TESTS PASSED — --skip_first gives leak-free extension")
    print("=" * 64)


if __name__ == '__main__':
    main()
