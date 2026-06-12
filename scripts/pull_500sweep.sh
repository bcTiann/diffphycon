#!/bin/bash
# Pull ALL sweep results from AutoDL.
#
# Three sweep dirs:
#   sweep_500fresh/             — original 500-sample sweep (γ=1.0 + γ-sweep)
#   sweep_500fresh_jellyfish/   — paper jellyfish β schedule sweep
#   sweep_ushape_diag/          — factorial U-shape diagnostic
#
# Plus paper inference npz (for per-sample J recompute).
#
# Usage:
#   bash /Users/baochen/diffphycon/pull_500sweep.sh

set -e

H=root@region-9.autodl.pro
P=50713
LOCAL=/Users/baochen/Desktop/diffphycon_results_500fresh
mkdir -p $LOCAL

echo "── (1) FM sweep dirs (3 sweeps under flow/results/) ──"
for SWEEP in sweep_500fresh sweep_500fresh_jellyfish sweep_ushape_diag; do
    echo ""
    echo "  pulling $SWEEP/"
    rsync -av --delete -e "ssh -p $P" \
        $H:/root/autodl-tmp/diffphycon/flow/results/$SWEEP/ \
        $LOCAL/$SWEEP/ || echo "  (skip: $SWEEP not on remote)"
done

echo ""
echo "── (2) paper inference trajectories npz ──"
mkdir -p $LOCAL/paper_npz
rsync -av -e "ssh -p $P" \
    $H:/root/autodl-tmp/diffphycon/outputs/trajectories/inference_trajectories_*.npz \
    $LOCAL/paper_npz/ || true

echo ""
echo "── (3) Summary ──"
echo ""
echo "▶ sweep_500fresh (original γ-sweep, sigmoid_flip):"
for n in 1 4 8 100 500 1000; do
    cnt=$(ls $LOCAL/sweep_500fresh/fm_n${n}/per_sample_J_*.npy 2>/dev/null | wc -l | tr -d ' ')
    echo "  fm_n${n}: $cnt npy files"
done
echo "  total: $(find $LOCAL/sweep_500fresh -name 'per_sample_J_*.npy' 2>/dev/null | wc -l | tr -d ' ')"

echo ""
echo "▶ sweep_500fresh_jellyfish (paper β schedule, 7γ × 6n_steps = 42 expected):"
echo "  total: $(find $LOCAL/sweep_500fresh_jellyfish -name 'per_sample_J_*.npy' 2>/dev/null | wc -l | tr -d ' ')"

echo ""
echo "▶ sweep_ushape_diag (factorial U-shape diag, 60 expected):"
echo "  cap_τ variants:"
for ct in 0.7 0.85 0.9 0.95; do
    cnt=$(find $LOCAL/sweep_ushape_diag/cap_tau_${ct} -name 'per_sample_J_*.npy' 2>/dev/null | wc -l | tr -d ' ')
    echo "    cap_tau_${ct}/: $cnt"
done
echo "  reinpaint: $(find $LOCAL/sweep_ushape_diag/reinpaint -name 'per_sample_J_*.npy' 2>/dev/null | wc -l | tr -d ' ')"
echo "  cap_τ + reinpaint variants:"
for ct in 0.7 0.85 0.9 0.95; do
    cnt=$(find $LOCAL/sweep_ushape_diag/cap_tau_${ct}_reinpaint -name 'per_sample_J_*.npy' 2>/dev/null | wc -l | tr -d ' ')
    echo "    cap_tau_${ct}_reinpaint/: $cnt"
done
echo "  rk4: $(find $LOCAL/sweep_ushape_diag/rk4 -name 'per_sample_J_*.npy' 2>/dev/null | wc -l | tr -d ' ')"
echo "  total: $(find $LOCAL/sweep_ushape_diag -name 'per_sample_J_*.npy' 2>/dev/null | wc -l | tr -d ' ')"

echo ""
echo "▶ paper_npz: $(ls $LOCAL/paper_npz/*.npz 2>/dev/null | wc -l | tr -d ' ') files"
