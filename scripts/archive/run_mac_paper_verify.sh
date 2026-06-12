#!/bin/bash
# Run paper DDPM 1000 step on Mac with the freshly-synced AutoDL data (50 samples).
# Expected: J ≈ 0.00123 (matching AutoDL, since same 50 samples now).
# Previous Mac result (on different 50 samples): J = 0.000629
#
# Usage:
#   bash /Users/baochen/diffphycon/run_mac_paper_verify.sh

set -e

cd /Users/baochen/diffphycon

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate diffphycon

# Verify data is synced (50 test samples = ~1MB, not 10000 = ~215MB)
TEST_H5=data/free_u_f_paper_fopc/burgers_test.h5
if [ ! -f $TEST_H5 ]; then
    echo "❌ test h5 missing — run pull_autodl_data.sh first"
    exit 1
fi
SIZE_MB=$(du -m $TEST_H5 | cut -f1)
if [ $SIZE_MB -gt 50 ]; then
    echo "⚠️  test h5 is ${SIZE_MB}MB — looks like Mac's old 10000-sample version, NOT AutoDL synced!"
    echo "   run: bash pull_autodl_data.sh"
    exit 1
fi
echo "✓ test h5 is ${SIZE_MB}MB (~50 samples, AutoDL synced)"

echo ""
echo "Running paper DDPM 1000 step (~6-7 min on MPS)..."

python inference/inference_1d_burgers.py \
    --exp_id FOPC \
    --dataset free_u_f_paper_fopc \
    --is_condition_u0 True \
    --is_condition_uT True \
    --J_scheduler cosine \
    --dim 128 \
    --dim_muls 1 2 4 \
    --partial_control front_rear_quarter \
    --partially_observed None \
    --train_on_partially_observed None \
    --set_unobserved_to_zero_during_sampling False \
    --checkpoint_interval 1000 \
    --checkpoint 170 \
    --n_test_samples 50 \
    --save_file /tmp/verify_mac_synced.yaml \
    --save_tag verify_mac_synced 2>&1 | grep -E "J_actual|Energy|Sampling time"

echo ""
echo "========== INTERPRETATION =========="
echo "Expected: J ≈ 0.00123 (matching AutoDL on same 50 samples)"
echo "If matches → confirms J differences across platforms are due to sample selection, not numerical precision"
echo "Previous Mac result on different 50 samples: J = 0.000629"
