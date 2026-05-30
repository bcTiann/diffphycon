#!/bin/bash
# sweep_fm_nsteps_autodl.sh — sweep our FM at BOTH step 170k and 190k joint.
#
# For each training step ckpt ∈ {170000, 190000}:
#   sweep n_steps ∈ {1, 4, 8, 16, 50, 100, 1000}
#   generate 4-panel trajectory plot at n_steps ∈ {1, 8, 1000}
#
# Uses paper-faithful eval (n_test=50, paper data).
#
# Why both step ckpts?
#   - step 170k: head-to-head fair comparison with paper's release (also 170k)
#   - step 190k: paper Table 5's "target" — see if extra training matters
#
# Usage:
#   bash scripts/sweep_fm_nsteps_autodl.sh
#
# Time: ~30-40 min on A100 (shared with prior training if still running).

set -e

cd /root/autodl-tmp/diffphycon

# Source ckpts in repo
CKPT_170=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_joint_step170000.pt
CKPT_190=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_joint.pt

OUT=/root/autodl-tmp/diffphycon/flow/results/fm_nsteps_sweep
mkdir -p $OUT
COMBINED=$OUT/fm_nsteps_sweep.csv
echo "train_step,n_steps,J_mean,J_std,E_mean" > $COMBINED

# Helper: sweep one ckpt across n_steps + generate plots
sweep_one_ckpt () {
    local TRAIN_STEP=$1
    local CKPT_PATH=$2

    if [ ! -f "$CKPT_PATH" ]; then
        echo "########## SKIP train_step=$TRAIN_STEP (missing $CKPT_PATH) ##########"
        return
    fi

    echo ""
    echo "########################################################"
    echo "########## TRAIN STEP = $TRAIN_STEP ##########"
    echo "########################################################"

    # Symlink ckpt under expected name for eval scripts
    local CKPT_TMP=/tmp/fm_step${TRAIN_STEP}
    mkdir -p $CKPT_TMP
    ln -sf $CKPT_PATH $CKPT_TMP/vanilla_joint.pt

    # STAGE 1: J vs n_steps sweep
    echo ""
    echo "########## STAGE 1: J vs n_steps sweep (train_step=$TRAIN_STEP) ##########"
    for STEPS in 1 4 8 16 50 100 1000; do
        echo ""
        echo ">>> FM n_steps=$STEPS (train_step=$TRAIN_STEP)"
        local PERSTEP_OUT=$OUT/train${TRAIN_STEP}/n${STEPS}
        mkdir -p $PERSTEP_OUT
        python flow/burgers_fm_eval_v2.py \
            --ckpt_dir $CKPT_TMP \
            --dataset free_u_f_paper_fopc \
            --out_dir $PERSTEP_OUT \
            --n_test 50 \
            --n_steps $STEPS \
            --gammas 1.0 \
            --variants vanilla \
            --device cuda 2>&1 | tail -3

        tail -n +2 $PERSTEP_OUT/eval_table.csv \
            | awk -F',' -v ts=$TRAIN_STEP -v ns=$STEPS \
                  'BEGIN{OFS=","} {print ts,ns,$3,$4,$5}' \
            >> $COMBINED
    done

    # STAGE 2: 4-panel plots at key n_steps
    echo ""
    echo "########## STAGE 2: 4-panel plots (train_step=$TRAIN_STEP) ##########"
    for STEPS in 1 8 1000; do
        echo ""
        echo ">>> Plot n_steps=$STEPS (train_step=$TRAIN_STEP)"
        local PLOT_OUT=$OUT/train${TRAIN_STEP}/plots_n${STEPS}
        mkdir -p $PLOT_OUT
        python flow/plot_trajectories.py \
            --ckpt_dir $CKPT_TMP \
            --dataset free_u_f_paper_fopc \
            --out_dir $PLOT_OUT \
            --n_samples 5 \
            --n_steps $STEPS \
            --gammas 1.0 \
            --variants vanilla \
            --device cuda
    done
}

# Run for both ckpts
sweep_one_ckpt 170000 $CKPT_170
sweep_one_ckpt 190000 $CKPT_190

echo ""
echo "########## ✅ ALL DONE ##########"
echo ""
echo "Combined CSV: $COMBINED"
cat $COMBINED
echo ""
echo "Per-train-step output dirs:"
ls -la $OUT/
