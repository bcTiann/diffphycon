#!/bin/bash
# sweep_500_fresh.sh — generate 500 FRESH (data-leak-free) test samples and
# sweep paper DDPM/DDIM + FM at the same n_steps set [1, 4, 8, 100, 500, 1000].
#
# Critical: uses --skip_first 90050 in generate_burgers so the new 500 test
# samples occupy RNG positions [90050, 90549] — guaranteed disjoint from the
# original 90050-sample dataset the model was trained on. No leak.
#
# Usage (GPU mode, default — ~25 min on A100):
#   bash scripts/sweep_500_fresh.sh
#
# Skip paper portion (paper already done, just rerun FM with prior + γ-sweep):
#   SKIP_PAPER=1 bash scripts/sweep_500_fresh.sh
#   (~5 min: only Step 5 FM sweep runs. Steps 1-4 skipped. Existing test h5 reused.)
#
# Force CPU smoke (auto-shrinks N_TEST to 2):
#   MODE=cpu bash scripts/sweep_500_fresh.sh
#
# Override sample count:
#   N_TEST=200 bash scripts/sweep_500_fresh.sh

cd /root/autodl-tmp/diffphycon

# --- Auto-detect CUDA. Override with MODE=cpu or MODE=gpu ---
N_TEST=${N_TEST:-500}
SKIP_FIRST=${SKIP_FIRST:-90050}  # = original train(90000) + original test(50)
STEPS=${STEPS:-"1 4 8 100 500 1000"}

if [ -z "$MODE" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        MODE=gpu
    else
        MODE=cpu
        echo "ℹ️  no CUDA detected, auto-switching to CPU mode"
    fi
fi

if [ "$MODE" = "cpu" ]; then
    echo "🐢 CPU mode — auto-reducing for smoke test"
    export CUDA_VISIBLE_DEVICES=""
    GEN_DEVICE=cpu
    FM_DEVICE=cpu
    if [ $N_TEST -gt 5 ]; then
        echo "   N_TEST $N_TEST → 2"
        N_TEST=2
    fi
    STEPS="1 8 100"  # cut the long ones in CPU smoke
    echo "   STEPS reduced to: $STEPS"
else
    echo "🚀 GPU mode"
    GEN_DEVICE=cuda:0
    FM_DEVICE=cuda
fi
echo "N_TEST=$N_TEST  SKIP_FIRST=$SKIP_FIRST  STEPS='$STEPS'"

DATA_DIR=data/free_u_f_paper_fopc
TEST_H5=$DATA_DIR/burgers_test.h5
TRAIN_H5=$DATA_DIR/burgers_train.h5
TEST_BAK=$DATA_DIR/burgers_test_orig50.h5.bak
NPZ_DIR=outputs/trajectories
NPZ_BAK=outputs/trajectories_orig50_bak

OUT=flow/results/sweep_${N_TEST}fresh
mkdir -p $OUT $NPZ_DIR

# --- Pre-flight checks ---
PAPER_CKPT=trained_models/burgers/FOPC/cos10000-model-170.pt
FM_SRC=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_joint_step170000.pt
for f in "$PAPER_CKPT" "$FM_SRC" "$TRAIN_H5"; do
    if [ ! -f "$f" ]; then
        echo "❌ missing: $f"
        echo "   train h5 is needed as a placeholder DataLoader by paper inference"
        exit 1
    fi
done
echo "✅ pre-flight OK: paper ckpt, FM ckpt, train h5 all present"

if [ -n "$SKIP_PAPER" ]; then
    echo ""
    echo "########## SKIP_PAPER=1 — skipping Steps 1-4 (backup, regen, DDPM, DDIM) ##########"
    echo "Will reuse existing test h5 + paper npz, jumping straight to Step 5 (FM γ-sweep)"
    if [ ! -f "$TEST_H5" ]; then
        echo "❌ SKIP_PAPER requires existing $TEST_H5 — abort"
        exit 1
    fi
    echo "✓ existing test h5 found: $(du -h $TEST_H5 | cut -f1)"
else

# --- Step 1: backup ORIGINAL 50-sample test + paper npz (idempotent) ---
echo ""
echo "########## Step 1: backup original test h5 + npz ##########"
if [ -f "$TEST_H5" ] && [ ! -f "$TEST_BAK" ]; then
    mv "$TEST_H5" "$TEST_BAK"
    echo "  ✓ moved $TEST_H5 → $TEST_BAK"
elif [ -f "$TEST_BAK" ]; then
    echo "  backup already exists at $TEST_BAK (skipping)"
    [ -f "$TEST_H5" ] && rm "$TEST_H5"
fi
if [ -d "$NPZ_DIR" ] && [ -n "$(ls -A $NPZ_DIR 2>/dev/null)" ] && [ ! -d "$NPZ_BAK" ]; then
    mv "$NPZ_DIR" "$NPZ_BAK"
    mkdir -p "$NPZ_DIR"
    echo "  ✓ moved $NPZ_DIR/*.npz → $NPZ_BAK/"
elif [ -d "$NPZ_BAK" ]; then
    echo "  npz backup already exists at $NPZ_BAK (skipping)"
fi

# --- Step 2: generate fresh test ---
echo ""
echo "########## Step 2: generate $N_TEST FRESH test samples (skip_first=$SKIP_FIRST) ##########"
python -u dataset/apps/generate_burgers.py \
    --skip_first $SKIP_FIRST \
    --train_samples 0 --test_samples $N_TEST \
    --partial_control front_rear_quarter \
    --nx 128 --nt 11 --device $GEN_DEVICE \
    --save_path free_u_f_paper_fopc/ 2>&1 | tee $OUT/log_generate.log

# Verify
python -u -c "
import h5py
f = h5py.File('$TEST_H5', 'r')
ds = f['test']['pde_11-128']
print('✓ test shape:', ds.shape)
assert ds.shape[0] == $N_TEST, f'expected $N_TEST, got {ds.shape[0]}'
"
if [ $? -ne 0 ]; then
    echo "❌ test h5 verification failed — abort"
    exit 1
fi

# --- Step 3: Paper DDPM 1000 (baseline) ---
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
echo "########## Step 3: Paper DDPM 1000 (baseline) ##########"
python -u inference/inference_1d_burgers.py "${COMMON_PAPER[@]}" \
    --save_file $OUT/ddpm_1000.yaml --save_tag ddpm_1000 \
    2>&1 | tee $OUT/log_ddpm_1000.log

# --- Step 4: Paper DDIM sweep ---
echo ""
echo "########## Step 4: Paper DDIM sweep [$STEPS] ##########"
for S in $STEPS; do
    echo ""
    echo "----- DDIM $S -----"
    python -u inference/inference_1d_burgers.py "${COMMON_PAPER[@]}" \
        --using_ddim True --ddim_sampling_steps $S --ddim_eta 0. \
        --save_file $OUT/ddim_${S}.yaml --save_tag ddim_${S} \
        2>&1 | tee $OUT/log_ddim_${S}.log
done

fi   # end of SKIP_PAPER guard (closes the else from Step 1)

# --- Step 5: FM sweep (same n_steps set, with prior + γ-sweep) ---
FM_TMP=/tmp/fm_step170k
mkdir -p $FM_TMP
ln -sf "$FM_SRC" $FM_TMP/vanilla_joint.pt

# Link prior (paper Table 5 prior, same 170k step as joint for fair comparison)
FM_PRIOR_SRC=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_prior_step170000.pt
if [ -f "$FM_PRIOR_SRC" ]; then
    ln -sf "$FM_PRIOR_SRC" $FM_TMP/vanilla_prior.pt
    echo "✓ FM prior linked: $FM_PRIOR_SRC"
    GAMMAS="0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0"
else
    echo "⚠️  FM prior missing at $FM_PRIOR_SRC — falling back to γ=1.0 only"
    GAMMAS="1.0"
fi

echo ""
echo "########## Step 5: FM sweep [$STEPS] × γ-sweep [$GAMMAS] ##########"
for S in $STEPS; do
    echo ""
    echo "----- FM n=$S (all γ) -----"
    python -u flow/burgers_fm_eval_v2.py \
        --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
        --out_dir $OUT/fm_n${S} \
        --n_test $N_TEST --n_steps $S --gammas $GAMMAS --variants vanilla \
        --device $FM_DEVICE 2>&1 | tee $OUT/log_fm_n${S}.log
done

# --- Step 6: summary table ---
echo ""
echo "########## Step 6: summary table ##########"
SUMMARY=$OUT/summary.txt
{
    echo "Fresh $N_TEST-sample sweep (skip_first=$SKIP_FIRST, no train leak)"
    echo "======================================================================"
    printf "%-15s | %-12s | %-12s | %-12s\n" "tag" "J" "Energy" "time(s)"
    echo "----------------------------------------------------------------------"
    # paper DDPM + DDIM (from logs)
    for LOG in $OUT/log_ddpm_*.log $OUT/log_ddim_*.log; do
        [ -f "$LOG" ] || continue
        TAG=$(basename "$LOG" .log | sed 's/^log_//')
        J=$(grep "J_actual" "$LOG" | tail -1 | awk '{print $2}')
        E=$(grep "Energy" "$LOG" | tail -1 | awk '{print $2}')
        T=$(grep "Sampling time" "$LOG" | tail -1 | awk '{print $3}')
        printf "%-15s | %-12s | %-12s | %-12s\n" "$TAG" "$J" "$E" "$T"
    done
    # FM (from eval_table.csv)
    for S in $STEPS; do
        CSV=$OUT/fm_n${S}/eval_table.csv
        if [ -f "$CSV" ]; then
            ROW=$(tail -n 1 "$CSV")
            J=$(echo "$ROW" | awk -F, '{print $3}')
            E=$(echo "$ROW" | awk -F, '{print $5}')
            T=$(echo "$ROW" | awk -F, '{print $6}')
            printf "%-15s | %-12s | %-12s | %-12s\n" "fm_n${S}" "$J" "$E" "$T"
        fi
    done
} | tee $SUMMARY

echo ""
echo "########## ✅ DONE ##########"
echo "Output: $OUT/"
echo ""
echo "Per-sample J npy files (for distribution plots):"
ls $OUT/fm_n*/per_sample_J_*.npy 2>/dev/null | head
echo ""
echo "Paper per-sample J: recompute via outputs/trajectories/inference_trajectories_*.npz"
ls outputs/trajectories/*.npz 2>/dev/null
echo ""
echo "Restore original 50-sample setup if needed:"
echo "  mv $TEST_BAK $TEST_H5"
echo "  rm -rf $NPZ_DIR && mv $NPZ_BAK $NPZ_DIR"
