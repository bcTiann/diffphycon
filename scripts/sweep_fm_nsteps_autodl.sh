#!/bin/bash
# sweep_fm_nsteps_autodl.sh — sweep our FM (step 190k joint) at various n_steps.
#
# Tests J convergence with n_steps in {1, 4, 8, 16, 50, 100, 1000}.
# Uses paper-faithful eval (n_test=50, paper data).
#
# Also generates 4-panel trajectory plots at n_steps {1, 8, 1000}.
#
# Usage:
#   bash scripts/sweep_fm_nsteps_autodl.sh
#
# Time: ~15-20 min on A100 (shared with prior training if still running).

set -e

cd /root/autodl-tmp/diffphycon

# Use step 190k (vanilla_joint.pt = final)
CKPT_DIR=/tmp/fm_step190k
mkdir -p $CKPT_DIR
ln -sf /root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_joint.pt $CKPT_DIR/vanilla_joint.pt

OUT=/root/autodl-tmp/diffphycon/flow/results/fm_nsteps_sweep
mkdir -p $OUT

# CSV header
COMBINED=$OUT/fm_nsteps_sweep.csv
echo "n_steps,J_mean,J_std,E_mean" > $COMBINED

echo "########## STAGE 1: J vs n_steps sweep ##########"
for STEPS in 1 4 8 16 50 100 1000; do
    echo ""
    echo "########## FM n_steps=$STEPS ##########"
    PERSTEP_OUT=$OUT/n${STEPS}
    mkdir -p $PERSTEP_OUT
    python flow/burgers_fm_eval_v2.py \
        --ckpt_dir $CKPT_DIR \
        --dataset free_u_f_paper_fopc \
        --out_dir $PERSTEP_OUT \
        --n_test 50 \
        --n_steps $STEPS \
        --gammas 1.0 \
        --variants vanilla \
        --device cuda 2>&1 | tail -3

    # Append to combined CSV: n_steps,J_mean,J_std,E_mean
    tail -n +2 $PERSTEP_OUT/eval_table.csv \
        | awk -F',' -v n=$STEPS 'BEGIN{OFS=","} {print n,$3,$4,$5}' \
        >> $COMBINED
done

echo ""
echo "########## STAGE 2: 4-panel plots at key n_steps ##########"
for STEPS in 1 8 1000; do
    echo ""
    echo "########## Plot at n_steps=$STEPS ##########"
    PLOT_OUT=$OUT/plots_n${STEPS}
    mkdir -p $PLOT_OUT
    python flow/plot_trajectories.py \
        --ckpt_dir $CKPT_DIR \
        --dataset free_u_f_paper_fopc \
        --out_dir $PLOT_OUT \
        --n_samples 5 \
        --n_steps $STEPS \
        --gammas 1.0 \
        --variants vanilla \
        --device cuda
done

echo ""
echo "########## ✅ ALL DONE ##########"
echo ""
echo "Combined CSV: $COMBINED"
cat $COMBINED
echo ""
echo "Plots in:"
ls -la $OUT/plots_*/ 2>/dev/null
