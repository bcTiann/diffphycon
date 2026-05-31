#!/bin/bash
# sweep_dense_jump_local.sh — local Mac MPS dense-jump sweep
#
# Uses local 180k joint ckpt + just-pulled 500-sample test h5.
# Sequential (Mac MPS doesn't benefit from parallel processes — single GPU).
#
# Time estimate (Mac MPS):
#   n_steps=8:   ~10s
#   n_steps=100: ~1 min
#   n_steps=500: ~5 min
#   n_steps=1000: ~10 min
# Total for default 8 cells: ~30 min
#
# Usage:
#   bash scripts/sweep_dense_jump_local.sh
# Faster (skip n=500/1000):
#   STEPS="8 100" bash scripts/sweep_dense_jump_local.sh

cd /Users/baochen/diffphycon

# Activate conda env (needed for scipy/torch)
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate diffphycon

FM_TMP=/Users/baochen/diffphycon/flow/checkpoints/local   # project dir, not /tmp (which macOS clears)
[ ! -e $FM_TMP/vanilla_joint.pt ] && { echo "ERROR: missing $FM_TMP/vanilla_joint.pt"; exit 1; }

STEPS=${STEPS:-"8 100 500 1000"}
TAUS=${TAUS:-"0.5 0.875"}
N_TEST=${N_TEST:-100}    # Reduced from 500 for local memory budget. Still enough samples for U-shape pattern.
ROOT=flow/results/sweep_dense_jump_local
mkdir -p $ROOT
# Don't delete existing (preserve any cell already finished)

echo "🖥  Mac MPS, sequential"
echo "TAUS=[$TAUS]  STEPS=[$STEPS]  N_TEST=$N_TEST"

for TAU in $TAUS; do
    for S in $STEPS; do
        OUT=$ROOT/tau_$TAU
        mkdir -p $OUT/fm_n${S}
        echo ""
        echo "----- tau=$TAU  n_steps=$S -----"
        python -u flow/burgers_fm_eval_v2.py \
            --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
            --out_dir $OUT/fm_n${S} \
            --n_test $N_TEST --n_steps $S --gammas 1.0 --variants vanilla \
            --dense_jump_tau $TAU \
            --device mps 2>&1 | tee $OUT/log_fm_n${S}.log | grep -E "γ=|sched\("
    done
done

echo ""
echo "########## ✅ ALL DONE ##########"
echo ""
echo "Per-cell J:"
for TAU in $TAUS; do
    for S in $STEPS; do
        F=$ROOT/tau_$TAU/fm_n${S}/per_sample_J_vanilla_g1.00_n${S}.npy
        if [ -f $F ]; then
            J=$(python -c "import numpy as np; print(f'{np.load(\"$F\").mean():.6f}')")
            echo "  tau=$TAU n=$S:  DJ J=$J"
        fi
    done
done
