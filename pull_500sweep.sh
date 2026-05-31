#!/bin/bash
# Pull everything from 500-fresh sweep + paper npz + logs.
#
# Uses rsync (not scp -r) to avoid double-nesting if dest already exists.
# Cleans local target dir first to ensure clean snapshot.
#
# Usage:
#   bash /Users/baochen/diffphycon/pull_500sweep.sh

set -e

H=root@region-9.autodl.pro
P=50713
LOCAL=/Users/baochen/Desktop/diffphycon_results_500fresh
mkdir -p $LOCAL

echo "── (1) FM sweep results + per-sample J npy ──"
# rsync handles dest-exists correctly (replaces contents, no nesting)
rsync -av --delete -e "ssh -p $P" \
    $H:/root/autodl-tmp/diffphycon/flow/results/sweep_500fresh/ \
    $LOCAL/sweep_500fresh/

echo ""
echo "── (2) paper inference trajectories npz ──"
mkdir -p $LOCAL/paper_npz
rsync -av -e "ssh -p $P" \
    $H:/root/autodl-tmp/diffphycon/outputs/trajectories/inference_trajectories_*.npz \
    $LOCAL/paper_npz/ || true

echo ""
echo "── (3) Summary ──"
echo "FM npy count per n_steps:"
for n in 1 4 8 100 500 1000; do
    cnt=$(ls $LOCAL/sweep_500fresh/fm_n${n}/per_sample_J_*.npy 2>/dev/null | wc -l | tr -d ' ')
    echo "  fm_n${n}: $cnt npy files"
done

echo ""
echo "Total per_sample_J npy files:"
find $LOCAL/sweep_500fresh -name "per_sample_J_*.npy" | wc -l

echo ""
echo "Paper npz:"
ls -la $LOCAL/paper_npz/
