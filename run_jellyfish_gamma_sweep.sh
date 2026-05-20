#!/bin/bash
# Jellyfish gamma sweep using paper checkpoints + small sampling_timesteps.
# Each run produces 4 samples in a fresh timestamped dir.
# After all runs, evaluates v_bar / J via the force surrogate model.
set -e

# raise file descriptor limit for the DataLoader worker pool
ulimit -n 4096

export PYTHONUNBUFFERED=1
export PYTHONPATH=/Users/baochen/diffphycon

GAMMAS=(1.0 0.7 0.5 0.3)

# log result dir per gamma
RESULTS_LOG="/tmp/jellyfish_gamma_results.txt"
: > "$RESULTS_LOG"

for G in "${GAMMAS[@]}"; do
    echo "============================================================"
    echo ">>> Running w_prob_exp=${G}"
    echo "============================================================"
    LOG_TMP=$(mktemp)
    python -u inference/inference_2d_jellyfish.py \
        --num_batches 1 --batch_size 4 \
        --w_prob_exp ${G} \
        --sampling_timesteps 1000 2>&1 | tee "$LOG_TMP"
    OUTPUT=$(cat "$LOG_TMP")
    rm -f "$LOG_TMP"
    # extract result dir from "id N saved at /path/..." lines
    RDIR=$(echo "$OUTPUT" | grep "id 0 saved at" | head -1 | sed 's/.*saved at \(.*\):.*/\1/')
    echo "gamma=${G} -> ${RDIR}" | tee -a "$RESULTS_LOG"
done

echo "============================================================"
echo "All inference runs done. Evaluating J for each:"
echo "============================================================"
while IFS= read -r line; do
    G=$(echo "$line" | sed 's/gamma=\([^ ]*\).*/\1/')
    RDIR=$(echo "$line" | sed 's/.* -> //')
    echo ""
    echo "--- gamma=${G} ---"
    python eval_jellyfish.py "$RDIR"
done < "$RESULTS_LOG"
