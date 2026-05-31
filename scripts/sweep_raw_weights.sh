#!/bin/bash
# sweep_raw_weights.sh — re-run baseline with raw (non-EMA) weights.
# Tests if EMA smoothing contributes to FM U-shape.
#
# 6 n_steps × γ=1.0 = 6 configs. ~30 sec on A100.
#
# Usage: bash scripts/sweep_raw_weights.sh

cd /root/autodl-tmp/diffphycon
FM_TMP=/tmp/fm_step170k
[ ! -e $FM_TMP/vanilla_joint.pt ] && { echo "ERROR: run sweep_500_fresh.sh first"; exit 1; }

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda; echo "🚀 GPU"
else
    DEVICE=cpu; export CUDA_VISIBLE_DEVICES=""; echo "🐢 CPU"
fi

STEPS=${STEPS:-"1 4 8 100 500 1000"}
N_TEST=${N_TEST:-500}
OUT=flow/results/sweep_raw_weights
mkdir -p $OUT

for S in $STEPS; do
    echo ""
    echo "----- FM n=$S (raw weights, no EMA) -----"
    python -u flow/burgers_fm_eval_v2.py \
        --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
        --out_dir $OUT/fm_n${S} \
        --n_test $N_TEST --n_steps $S --gammas 1.0 --variants vanilla \
        --no_ema \
        --device $DEVICE 2>&1 | tee $OUT/log_fm_n${S}.log
done

echo ""
echo "✓ Output: $OUT/fm_n*/per_sample_J_vanilla_g1.00_n*.npy"
echo "Total: $(find $OUT -name 'per_sample_J_*.npy' | wc -l) (expected: 6)"
