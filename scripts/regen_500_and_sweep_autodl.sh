#!/bin/bash
# regen_500_and_sweep_autodl.sh — regenerate test set + run all 4 methods.
#
# Usage (GPU mode, default — full 500-sample run, ~20 min on A100):
#   bash scripts/regen_500_and_sweep_autodl.sh
#
# Usage (CPU mode — AutoDL no-card mode, smoke test only with tiny N):
#   MODE=cpu N_TEST=2 bash scripts/regen_500_and_sweep_autodl.sh
#   (paper DDPM 1000 step ≈ 30 min/sample on CPU, do NOT run 500 on CPU)
#
# Override sample count:
#   N_TEST=100 bash scripts/regen_500_and_sweep_autodl.sh
#
# Note: no `set -e`, full output streams to terminal AND log files

cd /root/autodl-tmp/diffphycon

# --- Auto-detect CUDA. Set MODE=cpu to force CPU. Set MODE=gpu to force GPU ---
N_TEST=${N_TEST:-500}
N_TRAIN=${N_TRAIN:-90000}

if [ -z "$MODE" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        MODE=gpu
    else
        MODE=cpu
        echo "ℹ️  no CUDA detected, auto-switching to CPU mode"
    fi
fi

if [ "$MODE" = "cpu" ]; then
    echo "🐢 CPU mode — slow! auto-reducing sizes for smoke test"
    export CUDA_VISIBLE_DEVICES=""
    GEN_DEVICE=cpu
    if [ $N_TEST -gt 5 ]; then
        echo "   N_TEST was $N_TEST → reducing to 2"
        N_TEST=2
    fi
    if [ $N_TRAIN -gt 100 ]; then
        echo "   N_TRAIN was $N_TRAIN → reducing to 50 (CPU data gen is slow)"
        N_TRAIN=50
    fi
else
    echo "🚀 GPU mode"
    GEN_DEVICE=cuda:0
fi
echo "N_TRAIN = $N_TRAIN, N_TEST = $N_TEST"

DATA_DIR=data/free_u_f_paper_fopc
BACKUP=data/free_u_f_paper_fopc_50sample_backup

OUT=flow/results/sweep_${N_TEST}samples
mkdir -p $OUT

# --- Step 1: backup current data ---
echo ""
echo "########## Step 1: backup current data ##########"
if [ -d $BACKUP ]; then
    echo "  backup already exists, skipping"
else
    if [ -d $DATA_DIR ]; then
        cp -r $DATA_DIR $BACKUP
        echo "  ✓ backed up to $BACKUP"
    else
        echo "  no existing $DATA_DIR to back up"
    fi
fi

# --- Step 2: regenerate ---
echo ""
echo "########## Step 2: regenerate ${N_TRAIN}+${N_TEST} ##########"
rm -rf $DATA_DIR
python -u dataset/apps/generate_burgers.py \
    --train_samples $N_TRAIN --test_samples $N_TEST \
    --partial_control front_rear_quarter \
    --nx 128 --nt 11 --device $GEN_DEVICE \
    --save_path free_u_f_paper_fopc/ 2>&1 | tee $OUT/log_generate.log

# verify
python -u -c "
import h5py
f = h5py.File('$DATA_DIR/burgers_test.h5', 'r')
ds = f['test']['pde_11-128']
print('test shape:', ds.shape)
assert ds.shape[0] == $N_TEST, f'expected $N_TEST, got {ds.shape[0]}'
print('✓ test set has $N_TEST samples')
"

# --- Step 3: run all 4 methods ---
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
    --n_test_samples $N_TEST
)

echo ""
echo "########## Step 3a: Paper DDPM 1000 ##########"
python -u inference/inference_1d_burgers.py "${COMMON_PAPER[@]}" \
    --save_file $OUT/ddpm_1000.yaml \
    --save_tag ddpm_1000 2>&1 | tee $OUT/log_ddpm_1000.log

echo ""
echo "########## Step 3b: Paper DDIM 8 ##########"
python -u inference/inference_1d_burgers.py "${COMMON_PAPER[@]}" \
    --using_ddim True --ddim_sampling_steps 8 --ddim_eta 0. \
    --save_file $OUT/ddim_8.yaml \
    --save_tag ddim_8 2>&1 | tee $OUT/log_ddim_8.log

# Setup FM ckpt symlink under canonical name expected by burgers_fm_eval_v2.py
FM_TMP=/tmp/fm_step170k
mkdir -p $FM_TMP
FM_SRC=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_joint_step170000.pt
if [ -f "$FM_SRC" ]; then
    ln -sf "$FM_SRC" $FM_TMP/vanilla_joint.pt
    echo "✓ FM ckpt linked: $FM_SRC"
else
    echo "❌ FM ckpt missing: $FM_SRC — skip FM stages"
    FM_TMP=""
fi

if [ -n "$FM_TMP" ]; then
    FM_DEVICE=$([ "$MODE" = "cpu" ] && echo cpu || echo cuda)

    echo ""
    echo "########## Step 3c: FM n=8 ##########"
    python -u flow/burgers_fm_eval_v2.py \
        --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
        --out_dir $OUT/fm_n8 \
        --n_test $N_TEST --n_steps 8 --gammas 1.0 --variants vanilla \
        --device $FM_DEVICE 2>&1 | tee $OUT/log_fm_n8.log

    echo ""
    echo "########## Step 3d: FM n=1000 ##########"
    python -u flow/burgers_fm_eval_v2.py \
        --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
        --out_dir $OUT/fm_n1000 \
        --n_test $N_TEST --n_steps 1000 --gammas 1.0 --variants vanilla \
        --device $FM_DEVICE 2>&1 | tee $OUT/log_fm_n1000.log
fi

echo ""
echo "########## ✅ DONE ##########"
echo "Output dir: $OUT/"
ls -la $OUT/ 2>/dev/null

echo ""
echo "Paper trajectories npz (per-sample J recompute source):"
ls -la outputs/trajectories/inference_trajectories_ddpm_1000.npz \
       outputs/trajectories/inference_trajectories_ddim_8.npz 2>/dev/null
