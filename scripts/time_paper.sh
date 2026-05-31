#!/bin/bash
# time_paper.sh — measure paper DDPM/DDIM inference timing with proper warmup + repeats.
#
# Strategy: run each config N_REPEATS times via paper's inference script.
# Discard first run (warmup), average the rest.
# Parses "Sampling time:" line from logs.
#
# Usage:
#   bash scripts/time_paper.sh
#
# Time: ~10-15 min on A100 (8 configs × 6 repeats).

set -e

cd /root/autodl-tmp/diffphycon

PAPER_CKPT=trained_models/burgers/FOPC/cos10000-model-170.pt
if [ ! -f "$PAPER_CKPT" ]; then
    echo "❌ paper ckpt missing: $PAPER_CKPT"
    exit 1
fi

OUT=/root/autodl-tmp/diffphycon/flow/results/paper_timing
mkdir -p $OUT

N_REPEATS=${N_REPEATS:-6}   # first one = warmup, rest averaged

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

time_config () {
    local TAG=$1
    shift
    local LOG=$OUT/log_${TAG}.log
    : > $LOG
    for ((i=1; i<=N_REPEATS; i++)); do
        echo "  [$TAG repeat $i/$N_REPEATS]"
        python inference/inference_1d_burgers.py "${COMMON_ARGS[@]}" "$@" \
            --save_file /tmp/_throwaway.yaml --save_tag _time \
            2>&1 | grep "Sampling time" | tee -a $LOG
    done
}

echo "########## Paper inference timing (warmup + ${N_REPEATS}-run mean) ##########"
echo ""

echo ">>> DDPM 1000"
time_config ddpm_1000

echo ""
for STEPS in 1 4 8 16 50 100 1000; do
    echo ">>> DDIM $STEPS"
    time_config ddim_$STEPS --using_ddim True --ddim_sampling_steps $STEPS --ddim_eta 0.
done

echo ""
echo "########## Build summary ##########"
SUMMARY=$OUT/timing_summary.txt
{
    echo "Paper inference timing (warmup discarded, mean of N=$((N_REPEATS-1)) runs)"
    echo "============================================================================"
    printf "%-15s | %-10s | %-10s | %-10s | %-14s\n" "tag" "warmup_s" "mean_s" "std_s" "per_sample_ms"
    echo "----------------------------------------------------------------------------"
    for LOG in $OUT/log_*.log; do
        TAG=$(basename "$LOG" .log | sed 's/^log_//')
        python -c "
import numpy as np
with open('$LOG') as f:
    ts = [float(line.split()[2]) for line in f if 'Sampling time' in line]
if not ts:
    print(f\"%-15s | (no times)\" % '$TAG')
else:
    warmup = ts[0]
    rest = ts[1:] if len(ts) > 1 else ts
    m = np.mean(rest); s = np.std(rest)
    per_sample_ms = m / 50 * 1000
    print(f\"%-15s | %-10.4f | %-10.4f | %-10.4f | %-14.3f\" % ('$TAG', warmup, m, s, per_sample_ms))
"
    done
} | tee $SUMMARY

echo ""
echo "########## ✅ DONE — $SUMMARY ##########"
