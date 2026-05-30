#!/bin/bash
# train_paper_autodl.sh — STRICT paper-faithful training on AutoDL (NO auto-shutdown)
#
# Trains vanilla joint+prior at BATCH=16, NUM_STEPS=190000 (Table 5 exact)
# Expected time: ~10-12 hr on A100, ~¥35
# Expected J: 0.0004-0.0005 (1.1-1.4× paper 0.00037, near-perfect)
#
# NO auto-shutdown — manual shutdown via AutoDL console after checking results.
# (Previous version's hard-shutdown corrupted h5 files; manual is safer.)
#
# Usage:
#   tmux new -s train
#   bash scripts/train_paper_autodl.sh
#   (Ctrl+B D to detach, close SSH, sleep)
#   After waking up: check results, then power off via console.

set -e

cd /root/autodl-tmp/diffphycon

LOG=/root/autodl-tmp/diffphycon/train_paper_$(date +%Y%m%d_%H%M%S).log

echo "########## STRICT PAPER MODE: BATCH=16, NUM_STEPS=190000 ##########"
echo "Log file: $LOG"
echo "Expected time: ~10-12 hr on A100"
echo "🔔 NO auto-shutdown — manually power off after checking results."
echo ""

SKIP_OT=1 bash run_paper_fopc_v2.sh 2>&1 | tee "$LOG"

echo ""
echo "########## ✅ TRAINING DONE ##########"
echo "Results at: /root/autodl-tmp/diffphycon/flow/results/paper_fopc_v2/"
echo ""
echo "Now MANUALLY power off via AutoDL console to stop billing."
echo "Or use 无卡模式 (¥0.1/h) to inspect results cheaply."
