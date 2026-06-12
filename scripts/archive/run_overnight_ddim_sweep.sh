#!/bin/bash
# Jellyfish DDIM step-count sweep on Mac MPS
#
# Question: can we use fewer than 1000 DDPM steps for jellyfish inference
# without breaking θ output? Memory says 8-step DDIM is broken (θ → 1020°).
# Find threshold between 8 (broken) and 1000 (known good).
#
# Configs: 3 step counts × 3 ξ values × 5 samples each = 9 inference runs

set -e  # exit on first error so smoke test failure stops the rest

cd /Users/baochen/diffphycon

# Use diffphycon conda env's python (has einops + torch with MPS)
PY="/opt/homebrew/Caskroom/miniconda/base/envs/diffphycon/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: diffphycon conda env python not found at $PY"
  exit 1
fi

LOG_DIR="overnight_ddim_results"
mkdir -p $LOG_DIR

STEPS=(100 500 1000)
XIS=(0.4 0.0 -0.5)

echo "===================================================="
echo "Starting jellyfish DDIM step sweep on Mac MPS"
echo "Date: $(date)"
echo "Configs: 3 steps × 3 ξ = 9 runs, batch=5 each"
echo "===================================================="
echo ""

# --- Smoke test FIRST (fastest config: 100 steps × ξ=0.0)
echo ">>> SMOKE TEST: 100 steps × ξ=0.0, expect ~5-10 min..."
PYTHONPATH=. "$PY" inference/inference_2d_jellyfish.py \
  --sampling_timesteps 100 \
  --coeff_ratio_w 0.0 \
  --batch_size 5 \
  --num_batches 1 \
  --diffusion_w_checkpoint 50 \
  --diffusion_joint_checkpoint 100 \
  2>&1 | tee "$LOG_DIR/smoke_steps100_xi0.0.log"

SMOKE_EXIT=${PIPESTATUS[0]}
if [ $SMOKE_EXIT -ne 0 ]; then
  echo ""
  echo "!!! SMOKE FAILED (exit=$SMOKE_EXIT). Check $LOG_DIR/smoke_*.log"
  echo "!!! NOT proceeding with full sweep."
  exit 1
fi

echo ""
echo ">>> Smoke OK. Proceeding to full 9-config sweep..."
echo ""

# --- Full sweep
for STEP in "${STEPS[@]}"; do
  for XI in "${XIS[@]}"; do
    XI_TAG=$(echo $XI | tr '.-' '__')
    TAG="steps${STEP}_xi${XI_TAG}"
    LOG_FILE="$LOG_DIR/$TAG.log"

    echo ">>> Running: steps=$STEP xi=$XI  →  $LOG_FILE"
    echo "    Started: $(date)"

    PYTHONPATH=. "$PY" inference/inference_2d_jellyfish.py \
      --sampling_timesteps $STEP \
      --coeff_ratio_w $XI \
      --batch_size 5 \
      --num_batches 1 \
      --diffusion_w_checkpoint 50 \
      --diffusion_joint_checkpoint 100 \
      2>&1 | tee "$LOG_FILE"

    EXIT=${PIPESTATUS[0]}
    if [ $EXIT -ne 0 ]; then
      echo "!!! FAILED at steps=$STEP xi=$XI (exit=$EXIT). Continuing to next config."
      echo "FAILED" > "$LOG_DIR/$TAG.FAILED"
    fi

    echo "    Done:    $(date)"
    echo ""
  done
done

echo "===================================================="
echo "Sweep complete: $(date)"
echo "Logs in $LOG_DIR/"
echo "===================================================="

# --- Summary: extract theta range from each run for quick diagnosis
echo ""
echo "=== Quick θ range summary (look for 1020° outliers) ==="
for LOG in $LOG_DIR/steps*.log; do
  TAG=$(basename $LOG .log)
  # try to extract any "theta" range printed; this is approximate
  THETA_INFO=$(grep -iE "theta|θ" $LOG 2>/dev/null | head -3 || echo "(no theta info found in log)")
  echo "$TAG:"
  echo "$THETA_INFO" | sed 's/^/  /'
done
