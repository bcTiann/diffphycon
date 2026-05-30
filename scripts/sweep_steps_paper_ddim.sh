#!/bin/bash
# sweep_steps_paper_ddim.sh — run paper's DDPM/DDIM at multiple step counts on Mac MPS.
#
# Compares Paper DDPM (1000 step) vs DDIM (1000, 100, 50, 8, 1 step).
# Each run uses 50 test samples (paper-faithful) on MPS.
#
# Output: prints J / Energy table to terminal + saves .yaml per run.
#
# Usage:
#   bash scripts/sweep_steps_paper_ddim.sh
#
# Estimated time: ~20-30 min on Mac MPS (most time in DDIM 1000 step + DDPM 1000 step)

set -e

cd /Users/baochen/diffphycon

# Activate conda env
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate diffphycon

OUT=/tmp/paper_ddim_sweep
mkdir -p $OUT

# Common args for all runs
COMMON_ARGS=(
    --exp_id FOPC
    --dataset free_u_f_paper_fopc   # full paper-scale data (10000 test, take first 50)
    --is_condition_u0 True
    --is_condition_uT True
    --J_scheduler cosine
    --dim 128
    --dim_muls 1 2 4
    --partial_control front_rear_quarter
    --partially_observed None
    --train_on_partially_observed None
    --set_unobserved_to_zero_during_sampling False
    --checkpoint_interval 1000
    --checkpoint 170
    --n_test_samples 50   # paper D.1 = 50 test samples
)

SKIP_1000=${SKIP_1000:-0}   # set SKIP_1000=1 to skip the slow 1000-step runs (already done)

if [ "$SKIP_1000" != "1" ]; then
    echo "########## DDPM 1000 (baseline, paper default) ##########"
    python inference/inference_1d_burgers.py "${COMMON_ARGS[@]}" \
        --save_file $OUT/ddpm_1000.yaml 2>&1 | tail -3

    echo ""
    echo "########## DDIM 1000 step ##########"
    python inference/inference_1d_burgers.py "${COMMON_ARGS[@]}" \
        --using_ddim True \
        --ddim_sampling_steps 1000 \
        --ddim_eta 0. \
        --save_file $OUT/ddim_1000.yaml 2>&1 | tail -3
fi

for STEPS in 100 50 8 1; do
    echo ""
    echo "########## DDIM ${STEPS} step ##########"
    python inference/inference_1d_burgers.py "${COMMON_ARGS[@]}" \
        --using_ddim True \
        --ddim_sampling_steps $STEPS \
        --ddim_eta 0. \
        --save_file $OUT/ddim_${STEPS}.yaml 2>&1 | tail -3
done

echo ""
echo "########## ✅ ALL DONE ##########"
echo "Results saved to $OUT/"
ls -la $OUT/
