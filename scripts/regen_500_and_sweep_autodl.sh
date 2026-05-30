#!/bin/bash
# regen_500_and_sweep_autodl.sh — regenerate test set with 500 samples + run all 4 methods.
#
# Steps:
#   1. Backup current 50-sample data
#   2. Regenerate as 90000+500 (deterministic, paper-faithful style)
#   3. Run paper DDPM 1000, paper DDIM 8, FM n=8, FM n=1000 on 500 samples
#   4. Save per-sample J for each method
#
# Total: ~20-25 min on A100
# Usage:
#   bash scripts/regen_500_and_sweep_autodl.sh

set -e

cd /root/autodl-tmp/diffphycon

DATA_DIR=data/free_u_f_paper_fopc
BACKUP=data/free_u_f_paper_fopc_50sample_backup

OUT=flow/results/sweep_500samples
mkdir -p $OUT

# --- Step 1: backup 50-sample data ---
echo "########## Step 1: backup 50-sample data ##########"
if [ -d $BACKUP ]; then
    echo "  backup already exists, skipping"
else
    cp -r $DATA_DIR $BACKUP
    echo "  ✓ backed up to $BACKUP"
fi

# --- Step 2: regenerate with 500 test ---
echo ""
echo "########## Step 2: regenerate 90000+500 ##########"
rm -rf $DATA_DIR
python dataset/apps/generate_burgers.py \
    --train_samples 90000 --test_samples 500 \
    --partial_control front_rear_quarter \
    --nx 128 --nt 11 --device cuda:0 \
    --save_path free_u_f_paper_fopc/

# verify
python -c "import h5py; f=h5py.File('$DATA_DIR/burgers_test.h5','r'); ds=f['test']['pde_11-128']; print('test shape:', ds.shape); assert ds.shape[0]==500"

# --- Step 3: run all 4 methods, save per-sample J ---
COMMON_PAPER=(
    --exp_id FOPC
    --dataset free_u_f_paper_fopc
    --is_condition_u0 True --is_condition_uT True
    --J_scheduler cosine
    --dim 128 --dim_muls 1 2 4
    --partial_control front_rear_quarter
    --partially_observed None
    --train_on_partially_observed None
    --set_unobserved_to_zero_during_sampling False
    --checkpoint_interval 1000 --checkpoint 170
    --n_test_samples 500
)

echo ""
echo "########## Step 3a: Paper DDPM 1000 ##########"
python inference/inference_1d_burgers.py "${COMMON_PAPER[@]}" \
    --save_file $OUT/ddpm_1000.yaml \
    --save_tag ddpm_1000 2>&1 | grep -E "J_actual|Energy|Sampling time"

echo ""
echo "########## Step 3b: Paper DDIM 8 ##########"
python inference/inference_1d_burgers.py "${COMMON_PAPER[@]}" \
    --using_ddim True --ddim_sampling_steps 8 --ddim_eta 0. \
    --save_file $OUT/ddim_8.yaml \
    --save_tag ddim_8 2>&1 | grep -E "J_actual|Energy|Sampling time"

# Setup FM ckpt symlink under canonical name
FM_TMP=/tmp/fm_step170k
mkdir -p $FM_TMP
ln -sf /root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_joint_step170000.pt $FM_TMP/vanilla_joint.pt

echo ""
echo "########## Step 3c: FM n=8 ##########"
python flow/burgers_fm_eval_v2.py \
    --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
    --out_dir $OUT/fm_n8 \
    --n_test 500 --n_steps 8 --gammas 1.0 --variants vanilla \
    --device cuda

echo ""
echo "########## Step 3d: FM n=1000 ##########"
python flow/burgers_fm_eval_v2.py \
    --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
    --out_dir $OUT/fm_n1000 \
    --n_test 500 --n_steps 1000 --gammas 1.0 --variants vanilla \
    --device cuda

echo ""
echo "########## ✅ DONE ##########"
echo "Output dir: $OUT/"
ls -la $OUT/

echo ""
echo "Paper trajectories npz (for per-sample J recompute):"
ls -la outputs/trajectories/inference_trajectories_ddpm_1000.npz \
       outputs/trajectories/inference_trajectories_ddim_8.npz 2>/dev/null
