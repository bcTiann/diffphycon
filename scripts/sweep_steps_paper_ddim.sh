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
    --dataset free_u_f_1e5_front_rear_quarter
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
    --n_test_samples 50
)

echo "########## DDPM 1000 (baseline, paper default) ##########"
python inference/inference_1d_burgers.py "${COMMON_ARGS[@]}" \
    --save_file $OUT/ddpm_1000.yaml 2>&1 | tail -3

for STEPS in 1000 100 50 8 1; do
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
