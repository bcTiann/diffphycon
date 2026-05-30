#!/bin/bash
# sweep_joint_steps_autodl.sh — sweep eval over multiple joint step ckpts.
#
# Compares J for joint at step {50k, 100k, 150k, 170k, 180k, 190k} at γ=1.
# Uses paper-faithful eval (n_test=50, n_steps=1000).
#
# Designed to run while prior training is still active on the SAME A100.
# Each eval ≈ 1-3 min on A100, total sweep ≈ 15-20 min.
# Uses tmp symlinks so vanilla_joint.pt (final) is preserved for later γ-sweep.
#
# Usage:
#   bash scripts/sweep_joint_steps_autodl.sh

set -e

cd /root/autodl-tmp/diffphycon

CKPT_DIR=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2
OUT=/root/autodl-tmp/diffphycon/flow/results/joint_step_sweep
mkdir -p $OUT

# Steps to test (use the final ckpt for 190k)
# (intermediate ckpts are vanilla_joint_step*.pt; final is vanilla_joint.pt)
declare -A CKPT_PATHS=(
    [50000]=$CKPT_DIR/vanilla_joint_step50000.pt
    [100000]=$CKPT_DIR/vanilla_joint_step100000.pt
    [150000]=$CKPT_DIR/vanilla_joint_step150000.pt
    [170000]=$CKPT_DIR/vanilla_joint_step170000.pt
    [180000]=$CKPT_DIR/vanilla_joint_step180000.pt
    [190000]=$CKPT_DIR/vanilla_joint.pt
)

# Sort steps numerically
STEPS=(50000 100000 150000 170000 180000 190000)

# Combined CSV
COMBINED=$OUT/joint_step_sweep.csv
echo "step,J_mean,J_std,E_mean" > $COMBINED

for S in "${STEPS[@]}"; do
    CKPT="${CKPT_PATHS[$S]}"
    if [ ! -f "$CKPT" ]; then
        echo "########## SKIP step=$S (missing: $CKPT) ##########"
        continue
    fi

    echo ""
    echo "########## Joint step=$S ##########"

    # Set up tmp ckpt dir with vanilla_joint.pt → this specific step
    TMPDIR=/tmp/sweep_ckpt_step$S
    mkdir -p $TMPDIR
    ln -sf $CKPT $TMPDIR/vanilla_joint.pt

    PERSTEP_OUT=$OUT/step$S
    mkdir -p $PERSTEP_OUT

    python flow/burgers_fm_eval_v2.py \
        --ckpt_dir $TMPDIR \
        --dataset free_u_f_paper_fopc \
        --out_dir $PERSTEP_OUT \
        --n_test 50 \
        --n_steps 1000 \
        --gammas 1.0 \
        --variants vanilla \
        --device cuda 2>&1 | tail -3

    # Append to combined CSV
    # eval CSV columns: variant,gamma,J_mean,J_std,E_mean
    # we want: step,J_mean,J_std,E_mean
    tail -n +2 $PERSTEP_OUT/eval_table.csv | awk -F',' -v s=$S 'BEGIN{OFS=","} {print s,$3,$4,$5}' >> $COMBINED

    # Cleanup tmp
    rm -rf $TMPDIR
done

echo ""
echo "########## ✅ ALL DONE ##########"
echo ""
echo "Combined results: $COMBINED"
cat $COMBINED
