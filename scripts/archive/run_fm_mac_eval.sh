#!/bin/bash
# Run FM eval + 4-panel plots on Mac MPS using step 170k ckpt + paper-faithful data.
# Mirrors AutoDL FM sweep results for cross-verification.

set -e

cd /Users/baochen/diffphycon

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate diffphycon

CKPT_DIR=/tmp/fm_ckpts_170k
OUT_DIR=/tmp/fm_mac_eval_170k
mkdir -p $OUT_DIR

if [ ! -f $CKPT_DIR/vanilla_joint.pt ]; then
    echo "❌ FM ckpt missing — run: bash pull_fm_ckpt.sh first"
    exit 1
fi

echo "========== STAGE 1: J sweep at n_steps {1, 8, 1000} =========="
for STEPS in 1 8 1000; do
    echo ""
    echo ">>> FM n_steps=$STEPS"
    PERSTEP=$OUT_DIR/n${STEPS}
    mkdir -p $PERSTEP
    python flow/burgers_fm_eval_v2.py \
        --ckpt_dir $CKPT_DIR \
        --dataset free_u_f_paper_fopc \
        --out_dir $PERSTEP \
        --n_test 50 \
        --n_steps $STEPS \
        --gammas 1.0 \
        --variants vanilla \
        --device mps
done

echo ""
echo "========== STAGE 2: 4-panel plots =========="
for STEPS in 1 8 1000; do
    echo ""
    echo ">>> Plot n_steps=$STEPS"
    PLOT_DIR=$OUT_DIR/plots_n${STEPS}
    mkdir -p $PLOT_DIR
    python flow/plot_trajectories.py \
        --ckpt_dir $CKPT_DIR \
        --dataset free_u_f_paper_fopc \
        --out_dir $PLOT_DIR \
        --n_samples 5 \
        --n_steps $STEPS \
        --gammas 1.0 \
        --variants vanilla \
        --device mps
done

echo ""
echo "========== DONE =========="
echo "CSV results:"
for STEPS in 1 8 1000; do
    echo ""
    echo "--- n_steps=$STEPS ---"
    cat $OUT_DIR/n${STEPS}/eval_table.csv
done

echo ""
echo "4-panel plots: $OUT_DIR/plots_n{1,8,1000}/trajectories_vanilla_g1.0.png"
echo ""
echo "Open them: open $OUT_DIR/plots_n*/trajectories_vanilla_g1.0.png"
