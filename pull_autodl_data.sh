#!/bin/bash
# Pull AutoDL's paper-faithful data (90000 train + 50 test) to Mac,
# overwriting Mac's 90000+10000 version so both platforms use SAME samples.

set -e

H=root@region-9.autodl.pro
P=50713
REMOTE=/root/autodl-tmp/diffphycon/data/free_u_f_paper_fopc
LOCAL=/Users/baochen/diffphycon/data/free_u_f_paper_fopc

echo "Pulling $REMOTE → $LOCAL (overwriting)"
echo "(WARNING: this replaces Mac's 90000+10000 test data with AutoDL's 90000+50)"

# Backup Mac's existing test h5 (in case you want it back)
if [ -f $LOCAL/burgers_test.h5 ]; then
    BACKUP=$LOCAL/burgers_test_mac10000.h5.bak
    mv $LOCAL/burgers_test.h5 $BACKUP
    echo "  backed up Mac old test → $BACKUP"
fi

# Pull both train and test from AutoDL (deterministic, same seed but different N)
scp -P $P $H:$REMOTE/burgers_train.h5 $LOCAL/burgers_train.h5
scp -P $P $H:$REMOTE/burgers_test.h5 $LOCAL/burgers_test.h5

# Also wipe PyG cache so it doesn't compare against stale shapes
rm -rf $LOCAL/processed $LOCAL/pre_filter.pt $LOCAL/pre_transform.pt
rm -rf $LOCAL/raw

echo ""
echo "Done. Mac now uses SAME data as AutoDL:"
ls -la $LOCAL/

echo ""
echo "Now re-run paper inference on Mac (should give J ≈ AutoDL's 0.00123, not 0.00063):"
echo "  bash scripts/run_paper_ddpm_mac.sh   # or your existing command"
