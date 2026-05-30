#!/bin/bash
# sweep_paper_ddim_autodl.sh — sweep paper DDPM/DDIM on AutoDL.
#
# Tests J vs n_steps for paper DDIM, + paper DDPM at 1000 step (baseline).
# Generates 4-panel plots at key step counts.
#
# Usage (GPU mode, default):
#   bash scripts/sweep_paper_ddim_autodl.sh
#
# Usage (CPU mode — AutoDL no-card mode for quick smoke test):
#   MODE=cpu bash scripts/sweep_paper_ddim_autodl.sh
#   (CPU is SLOW: paper DDPM 1000 step ≈ 30 min/sample. Use N_SAMPLES=2 to smoke-test)
#
# Override sample count:
#   N_SAMPLES=5 bash scripts/sweep_paper_ddim_autodl.sh
#
# Note: removed `set -e` and grep filter — full output streams to terminal AND log file.

cd /root/autodl-tmp/diffphycon

# --- Auto-detect CUDA. Set MODE=cpu to force CPU. Set MODE=gpu to force GPU (and fail if no CUDA) ---
N_SAMPLES=${N_SAMPLES:-50}

if [ -z "$MODE" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        MODE=gpu
    else
        MODE=cpu
        echo "ℹ️  no CUDA detected, auto-switching to CPU mode"
    fi
fi

if [ "$MODE" = "cpu" ]; then
    echo "🐢 CPU mode — slow! auto-reducing N_SAMPLES if > 5"
    export CUDA_VISIBLE_DEVICES=""
    if [ $N_SAMPLES -gt 5 ]; then
        echo "   N_SAMPLES was $N_SAMPLES → reducing to 2 for smoke test"
        echo "   (override with: N_SAMPLES=<n> MODE=cpu bash ...)"
        N_SAMPLES=2
    fi
else
    echo "🚀 GPU mode"
fi
echo "n_test_samples = $N_SAMPLES"

# Verify paper ckpt is present
PAPER_CKPT=trained_models/burgers/FOPC/cos10000-model-170.pt
if [ ! -f "$PAPER_CKPT" ]; then
    echo "❌ paper ckpt missing: $PAPER_CKPT"
    echo "   scp from Mac first:"
    echo "   scp -P <port> /Users/baochen/Downloads/fopc_cp170.pt root@<host>:$PWD/$PAPER_CKPT"
    exit 1
fi
echo "✅ paper ckpt found: $(ls -la $PAPER_CKPT)"

OUT=/root/autodl-tmp/diffphycon/flow/results/paper_ddim_sweep
mkdir -p $OUT outputs/trajectories
PLOT_OUT=/root/autodl-tmp/diffphycon/flow/results/paper_ddim_plots
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
    --n_test_samples $N_SAMPLES
)

echo ""
echo "########## STAGE 1: J sweep across n_steps ##########"

# DDPM 1000 (paper baseline)
echo ""
echo "########## DDPM 1000 ##########"
python -u inference/inference_1d_burgers.py "${COMMON_ARGS[@]}" \
    --save_file $OUT/ddpm_1000.yaml \
    --save_tag ddpm_1000 2>&1 | tee "$OUT/log_ddpm_1000.log"

# DDIM at various step counts
for STEPS in 1 4 8 16 50 100 1000; do
    echo ""
    echo "########## DDIM ${STEPS} step ##########"
    python -u inference/inference_1d_burgers.py "${COMMON_ARGS[@]}" \
        --using_ddim True \
        --ddim_sampling_steps $STEPS \
        --ddim_eta 0. \
        --save_file $OUT/ddim_${STEPS}.yaml \
        --save_tag ddim_${STEPS} 2>&1 | tee "$OUT/log_ddim_${STEPS}.log"
done

echo ""
echo "########## STAGE 2: 4-panel plots ##########"
for TAG_TITLE in "ddpm_1000:Paper DDPM 1000 step" \
                 "ddim_1000:Paper DDIM 1000 step" \
                 "ddim_100:Paper DDIM 100 step" \
                 "ddim_8:Paper DDIM 8 step (paper fast)" \
                 "ddim_1:Paper DDIM 1 step"; do
    TAG="${TAG_TITLE%%:*}"
    TITLE="${TAG_TITLE##*:}"
    NPZ=outputs/trajectories/inference_trajectories_${TAG}.npz
    if [ -f "$NPZ" ]; then
        python -u flow/plot_paper_trajectories.py \
            --npz "$NPZ" \
            --out_png "$PLOT_OUT/${TAG}.png" \
            --title "$TITLE" \
            --n_samples 5
    else
        echo "⚠️  missing $NPZ — skip plot"
    fi
done

echo ""
echo "########## STAGE 3: build summary table ##########"
SUMMARY=$OUT/summary.txt
{
    echo "Paper inference J + Energy + timing (auto-extracted from log files)"
    echo "===================================================================="
    printf "%-15s | %-12s | %-12s | %-12s\n" "tag" "J_actual" "Energy" "time(s)"
    echo "--------------------------------------------------------------------"
    for LOG in $OUT/log_*.log; do
        TAG=$(basename "$LOG" .log | sed 's/^log_//')
        J=$(grep "J_actual" "$LOG" | tail -1 | awk '{print $2}')
        E=$(grep "Energy" "$LOG" | tail -1 | awk '{print $2}')
        T=$(grep "Sampling time" "$LOG" | tail -1 | awk '{print $3}')
        printf "%-15s | %-12s | %-12s | %-12s\n" "$TAG" "$J" "$E" "$T"
    done
} | tee $SUMMARY

echo ""
echo "########## ✅ ALL DONE ##########"
echo ""
echo "Log files (full output per run): $OUT/log_*.log"
echo "Summary table:                   $SUMMARY"
echo "YAML results dir:                $OUT/"
ls -la $OUT/
echo ""
echo "4-panel plots: $PLOT_OUT/"
ls -la $PLOT_OUT/
