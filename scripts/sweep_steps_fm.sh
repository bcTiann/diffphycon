#!/bin/bash
# sweep_steps_fm.sh — run our FM model at multiple step counts on Mac MPS.
#
# Compares our FM eval at n_steps = 1, 4, 16, 64, 256, 1000.
# Uses the same /tmp/ckpts_180k/vanilla_joint.pt from earlier eval.
#
# Output: eval_table_fm_sweep.csv with rows for each n_steps.
#
# Usage:
#   bash scripts/sweep_steps_fm.sh
#
# Estimated time: ~15-20 min on Mac MPS (most time in 1000 step run)

set -e

cd /Users/baochen/diffphycon

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate diffphycon

OUT=/tmp/fm_step_sweep
mkdir -p $OUT

# Final combined CSV
COMBINED=$OUT/eval_table_fm_sweep.csv
echo "variant,gamma,n_steps,J_mean,J_std,E_mean" > $COMBINED

for STEPS in 1 4 16 64 256 1000; do
    echo ""
    echo "########## FM n_steps=${STEPS} ##########"
    PERSTEP_OUT=$OUT/n${STEPS}
    mkdir -p $PERSTEP_OUT
    python flow/burgers_fm_eval_v2.py \
        --ckpt_dir /tmp/ckpts_180k \
        --dataset free_u_f_paper_fopc \
        --out_dir $PERSTEP_OUT \
        --n_test 50 \
        --n_steps $STEPS \
        --gammas 1.0 \
        --variants vanilla \
        --device mps 2>&1 | tail -3
    # Append to combined CSV (skip header line of per-step CSV)
    tail -n +2 $PERSTEP_OUT/eval_table.csv | awk -F',' -v n=$STEPS '{print $1","$2","n","$3","$4","$5}' >> $COMBINED
done

echo ""
echo "########## ✅ ALL DONE ##########"
echo ""
echo "Combined results:"
cat $COMBINED
