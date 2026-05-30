#!/bin/bash
# train_fast_autodl.sh — Fast mode training on AutoDL (NO auto-shutdown)
#
# Trains vanilla joint+prior at BATCH=64, NUM_STEPS=50000 (~2.5 hr on A100, ~¥8)
# Output: J estimated 0.0008-0.002 (2-5× paper 0.00037)
#
# NO auto-shutdown — manual shutdown via AutoDL console after checking results.
#
# Usage:
#   tmux new -s train
#   bash scripts/train_fast_autodl.sh
#   (Ctrl+B D to detach, close SSH, sleep)
#   After waking up: check results, then power off via console.

set -e

cd /root/autodl-tmp/diffphycon

LOG=/root/autodl-tmp/diffphycon/train_fast_$(date +%Y%m%d_%H%M%S).log

echo "########## FAST MODE: vanilla joint+prior, BATCH=64, NUM_STEPS=50000 ##########"
echo "Log file: $LOG"
echo "Expected time: ~2.5 hr on A100"
echo "🔔 NO auto-shutdown — manually power off after checking results."
echo ""

BATCH=64 NUM_STEPS=50000 SKIP_OT=1 \
    bash run_paper_fopc_v2.sh 2>&1 | tee "$LOG"

echo ""
echo "########## ✅ TRAINING DONE ##########"
echo "Results at: /root/autodl-tmp/diffphycon/flow/results/paper_fopc_v2/"
echo ""
echo "Now MANUALLY power off via AutoDL console to stop billing."
