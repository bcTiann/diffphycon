#!/bin/bash
# Memory probe: track peak RSS of one inference run

PY="/opt/homebrew/Caskroom/miniconda/base/envs/diffphycon/bin/python"
cd /Users/baochen/diffphycon

BATCH=${1:-1}
STEPS=${2:-100}

echo "===================================="
echo "Memory probe: batch=$BATCH steps=$STEPS"
echo "Start: $(date)"
echo "===================================="

# Start inference in background
PYTHONPATH=. "$PY" inference/inference_2d_jellyfish.py \
  --sampling_timesteps $STEPS \
  --coeff_ratio_w 0.0 \
  --batch_size $BATCH \
  --num_batches 1 \
  --diffusion_w_checkpoint 50 \
  --diffusion_joint_checkpoint 100 \
  > /tmp/mem_probe.log 2>&1 &

PID=$!
echo "Inference PID = $PID"
echo ""

# Sample RSS in a tight loop
MAX_RSS=0
SAMPLE_COUNT=0
PHASE_LOG=""

while kill -0 $PID 2>/dev/null; do
  # Sum RSS across main process + all descendants (catches dataloader workers)
  CHILD_PIDS=$(pgrep -P $PID 2>/dev/null | tr '\n' ' ')
  ALL_PIDS="$PID $CHILD_PIDS"
  RSS=$(ps -o rss= -p $ALL_PIDS 2>/dev/null | awk '{s+=$1} END {print s}')
  N_PROC=$(echo "$ALL_PIDS" | wc -w | tr -d ' ')
  if [ -n "$RSS" ] && [ "$RSS" -gt 0 ]; then
    [ "$RSS" -gt "$MAX_RSS" ] && MAX_RSS=$RSS
    RSS_MB=$((RSS / 1024))
    MAX_MB=$((MAX_RSS / 1024))
    SAMPLE_COUNT=$((SAMPLE_COUNT + 1))

    # Show every 3rd sample so output is digestible
    if [ $((SAMPLE_COUNT % 3)) -eq 0 ]; then
      LAST_LINE=$(tail -1 /tmp/mem_probe.log 2>/dev/null | head -c 60)
      echo "[$(date +%H:%M:%S)] rss=${RSS_MB} MB  peak=${MAX_MB} MB  procs=${N_PROC}  |  $LAST_LINE"
    fi
  fi
  sleep 2
done

EXIT_CODE=$?
echo ""
echo "===================================="
echo "Process exited. End: $(date)"
echo "PEAK RSS = $((MAX_RSS / 1024)) MB"
echo "===================================="
echo ""
echo "Last 20 lines of inference output:"
echo "------------------------------------"
tail -20 /tmp/mem_probe.log
