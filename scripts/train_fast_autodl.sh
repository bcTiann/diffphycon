#!/bin/bash
# train_fast_autodl.sh — Fast mode training on AutoDL with auto-shutdown
#
# Trains vanilla joint+prior at BATCH=64, NUM_STEPS=50000 (~2.5 hr on A100, ~¥8)
# Output: J estimated 0.0008-0.002 (2-5× paper 0.00037)
#
# Includes:
#   - log redirect to data disk (survives shutdown)
#   - automatic shutdown after training (stops billing)
#
# Usage:
#   tmux new -s train
#   bash scripts/train_fast_autodl.sh
#   (Ctrl+B D to detach, close SSH, sleep)

set -e

cd /root/autodl-tmp/diffphycon

LOG=/root/autodl-tmp/diffphycon/train_fast_$(date +%Y%m%d_%H%M%S).log

echo "########## FAST MODE: vanilla joint+prior, BATCH=64, NUM_STEPS=50000 ##########"
echo "Log file: $LOG"
echo "Expected time: ~2.5 hr on A100"
echo "Will auto-shutdown when done."
echo ""

BATCH=64 NUM_STEPS=50000 SKIP_OT=1 \
    bash run_paper_fopc_v2.sh 2>&1 | tee "$LOG"

echo ""
echo "########## TRAINING DONE. Shutting down in 60s... ##########"
sleep 60
/usr/bin/shutdown
