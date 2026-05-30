#!/bin/bash
# sweep_paper_ddim_mac.sh — sweep paper DDIM on Mac MPS at various step counts,
# + generate 4-panel plots at key step counts.
#
# Uses paper's released joint ckpt (trained_models/burgers/FOPC/cos10000-model-170.pt
# at step 170k). Uses our local paper-scale data (free_u_f_paper_fopc with 10000 test).
#
# Usage:
#   bash scripts/sweep_paper_ddim_mac.sh

set -e

cd /Users/baochen/diffphycon

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate diffphycon

OUT=/tmp/paper_ddim_sweep
mkdir -p $OUT outputs/trajectories
PLOT_OUT=/tmp/paper_ddim_plots
mkdir -p $PLOT_OUT

COMMON_ARGS=(
    --exp_id FOPC
    --dataset free_u_f_paper_fopc
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

echo "########## STAGE 1: J sweep across n_steps ##########"

# DDPM 1000 (baseline; DDPM only supports 1000)
echo ""
echo "########## DDPM 1000 ##########"
python inference/inference_1d_burgers.py "${COMMON_ARGS[@]}" \
    --save_file $OUT/ddpm_1000.yaml \
    --save_tag ddpm_1000 2>&1 | tail -3

# DDIM at various step counts
for STEPS in 1 4 8 16 50 100 1000; do
    echo ""
    echo "########## DDIM ${STEPS} step ##########"
    python inference/inference_1d_burgers.py "${COMMON_ARGS[@]}" \
        --using_ddim True \
        --ddim_sampling_steps $STEPS \
        --ddim_eta 0. \
        --save_file $OUT/ddim_${STEPS}.yaml \
        --save_tag ddim_${STEPS} 2>&1 | tail -3
done

echo ""
echo "########## STAGE 2: 4-panel plots at key step counts ##########"
# Plot for: DDPM 1000, DDIM 8, DDIM 1
for TAG_TITLE in "ddpm_1000:Paper DDPM 1000 step" "ddim_1000:Paper DDIM 1000 step" "ddim_100:Paper DDIM 100 step" "ddim_8:Paper DDIM 8 step (paper fast)" "ddim_1:Paper DDIM 1 step"; do
    TAG="${TAG_TITLE%%:*}"
    TITLE="${TAG_TITLE##*:}"
    NPZ=outputs/trajectories/inference_trajectories_${TAG}.npz
    if [ -f "$NPZ" ]; then
        python flow/plot_paper_trajectories.py \
            --npz "$NPZ" \
            --out_png "$PLOT_OUT/${TAG}.png" \
            --title "$TITLE" \
            --n_samples 5
    else
        echo "⚠️  missing $NPZ — skip plot"
    fi
done

echo ""
echo "########## ✅ ALL DONE ##########"
echo ""
echo "YAML results: $OUT/"
ls -la $OUT/
echo ""
echo "4-panel plots: $PLOT_OUT/"
ls -la $PLOT_OUT/
