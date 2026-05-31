#!/bin/bash
# sweep_baseline_local.sh — local baseline (no dense_jump) for fair comparison
# Same setup as sweep_dense_jump_local.sh: 180k ckpt, 100 samples, MPS, sequential
# Pair with sweep_dense_jump_local.sh outputs to compare DJ vs baseline.

cd /Users/baochen/diffphycon
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate diffphycon

FM_TMP=/Users/baochen/diffphycon/flow/checkpoints/local
[ ! -e $FM_TMP/vanilla_joint.pt ] && { echo "ERROR: missing $FM_TMP/vanilla_joint.pt"; exit 1; }

STEPS=${STEPS:-"8 100 500 1000"}
N_TEST=${N_TEST:-100}
ROOT=flow/results/sweep_baseline_local
mkdir -p $ROOT

echo "🖥 Mac MPS local baseline (no dense_jump, no cap_tau)"
echo "STEPS=[$STEPS]  N_TEST=$N_TEST"

for S in $STEPS; do
    OUT=$ROOT/fm_n${S}
    mkdir -p $OUT
    echo ""
    echo "----- baseline n_steps=$S -----"
    python -u flow/burgers_fm_eval_v2.py \
        --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
        --out_dir $OUT \
        --n_test $N_TEST --n_steps $S --gammas 1.0 --variants vanilla \
        --device mps 2>&1 | tee $OUT/log.log | grep -E "γ=|sched\("
done

echo ""
echo "########## ✅ baseline DONE ##########"
for S in $STEPS; do
    F=$ROOT/fm_n${S}/per_sample_J_vanilla_g1.00_n${S}.npy
    if [ -f $F ]; then
        J=$(python -c "import numpy as np; print(f'{np.load(\"$F\").mean():.6f}')")
        echo "  baseline n=$S: J=$J"
    fi
done
