#!/bin/bash
# Pull our FM step 170k joint ckpt to Mac (for head-to-head with paper's 170k DDPM).

set -e

H=root@region-9.autodl.pro
P=50713
REMOTE=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_joint_step170000.pt
LOCAL_DIR=/tmp/fm_ckpts_170k

mkdir -p $LOCAL_DIR

echo "Pulling FM step 170k ckpt (~619MB)..."
scp -P $P $H:$REMOTE $LOCAL_DIR/vanilla_joint.pt

echo ""
echo "Done. Mac now has FM step 170k at $LOCAL_DIR/vanilla_joint.pt"
ls -la $LOCAL_DIR/

echo ""
echo "Next: bash /Users/baochen/diffphycon/run_fm_mac_eval.sh"
