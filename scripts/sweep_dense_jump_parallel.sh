#!/bin/bash
# sweep_dense_jump_parallel.sh — clean parallel sweep using & + wait (no xargs).
#
# Previously had xargs+bash -c quoting issue. Now uses simple background jobs.
#
# Default: TAUS="0.5 0.875" STEPS="8 100 500 1000" J=4 → 8 jobs in batches of 4.

cd /root/autodl-tmp/diffphycon
FM_TMP=/tmp/fm_step170k
[ ! -e $FM_TMP/vanilla_joint.pt ] && { echo "ERROR: run sweep_500_fresh.sh first"; exit 1; }

DEVICE=cuda
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "ERROR: parallel needs GPU"
    exit 1
fi

STEPS=${STEPS:-"8 100 500 1000"}
TAUS=${TAUS:-"0.5 0.875"}
N_TEST=${N_TEST:-500}
J=${J:-4}
ROOT=flow/results/sweep_dense_jump
mkdir -p $ROOT

# Delete old (potentially stale) outputs to avoid confusion
echo "Cleaning old outputs in $ROOT..."
rm -rf $ROOT/tau_*

echo "🚀 GPU, max $J parallel jobs"
echo "TAUS=[$TAUS]  STEPS=[$STEPS]"

# Build list of (TAU, STEP) tuples
declare -a JOBS
for TAU in $TAUS; do
    for S in $STEPS; do
        JOBS+=("$TAU $S")
    done
done

total=${#JOBS[@]}
echo "Total: $total jobs, ~$(( (total + J - 1) / J )) batches"

# Process in batches of J
i=0
batch=0
while [ $i -lt $total ]; do
    batch=$((batch + 1))
    echo ""
    echo "──── Batch $batch ────"
    # Launch up to J jobs in background
    pids=()
    for ((b=0; b<J && i<total; b++, i++)); do
        TAU=$(echo "${JOBS[$i]}" | cut -d' ' -f1)
        S=$(echo "${JOBS[$i]}" | cut -d' ' -f2)
        OUT=$ROOT/tau_$TAU
        mkdir -p $OUT/fm_n${S}
        echo "  launch tau=$TAU n=$S (job $((i+1))/$total)"
        python -u flow/burgers_fm_eval_v2.py \
            --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
            --out_dir $OUT/fm_n${S} \
            --n_test $N_TEST --n_steps $S --gammas 1.0 --variants vanilla \
            --dense_jump_tau $TAU \
            --device $DEVICE > $OUT/log_fm_n${S}.log 2>&1 &
        pids+=($!)
    done
    # Wait for this batch
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo "  ✓ batch $batch done"
done

echo ""
echo "########## ✅ ALL DONE ##########"

# Print per-cell J for quick check
echo ""
echo "Per-cell J (compare to baseline at each n_steps):"
for TAU in $TAUS; do
    for S in $STEPS; do
        F=$ROOT/tau_$TAU/fm_n${S}/per_sample_J_vanilla_g1.00_n${S}.npy
        BASE=flow/results/sweep_500fresh/fm_n${S}/per_sample_J_vanilla_g1.00_n${S}.npy
        if [ -f $F ] && [ -f $BASE ]; then
            J_DJ=$(python -c "import numpy as np; print(f'{np.load(\"$F\").mean():.6f}')")
            J_B=$(python -c "import numpy as np; print(f'{np.load(\"$BASE\").mean():.6f}')")
            DIFF=$(python -c "import numpy as np; a=np.load('$F'); b=np.load('$BASE'); print('DIFF' if not np.array_equal(a, b) else 'SAME')")
            echo "  tau=$TAU n=$S:  DJ J=$J_DJ  baseline J=$J_B  [$DIFF]"
        fi
    done
done
