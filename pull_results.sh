#!/bin/bash
# Pull all results from AutoDL to Mac. Run from anywhere on Mac.
#
# Usage:
#   bash /Users/baochen/diffphycon/pull_results.sh

set -e

H=root@region-9.autodl.pro
P=50713
REMOTE=/root/autodl-tmp/diffphycon/flow/results
LOCAL=/Users/baochen/Desktop/diffphycon_results_new

echo "Pulling $REMOTE → $LOCAL"
scp -P $P -r $H:$REMOTE $LOCAL

echo ""
echo "Done. Listing what came down:"
ls -la $LOCAL

echo ""
echo "FM sweep CSV:"
cat $LOCAL/fm_nsteps_sweep/fm_nsteps_sweep.csv 2>/dev/null || echo "(missing)"

echo ""
echo "Open all 4-panel plots? (uncomment below)"
echo "  open $LOCAL/paper_ddim_plots/*.png"
echo "  open $LOCAL/fm_nsteps_sweep/train170000/plots_n*/trajectories_vanilla_g1.0.png"
echo "  open $LOCAL/fm_nsteps_sweep/train190000/plots_n*/trajectories_vanilla_g1.0.png"
