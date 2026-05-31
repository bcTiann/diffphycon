#!/bin/bash
# sweep_dense_jump.sh — test Dense-Jump (paper 2509.13574) on FM U-shape.
#
# Hypothesis: U-shape comes from non-Lipschitz velocity at τ→1.
# Dense-Jump fix: N-1 small Euler steps in [0, t_jump], single big jump to τ=1.
# Reuses existing /tmp/fm_step170k symlinks. ~2 min on A100.
#
# Sweep: 3 dense_jump_tau values × 6 n_steps = 18 configs.
#
# Usage: bash scripts/sweep_dense_jump.sh

cd /root/autodl-tmp/diffphycon
FM_TMP=/tmp/fm_step170k
[ ! -e $FM_TMP/vanilla_joint.pt ] && { echo "ERROR: run sweep_500_fresh.sh first"; exit 1; }

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda; echo "🚀 GPU"
else
    DEVICE=cpu; export CUDA_VISIBLE_DEVICES=""; echo "🐢 CPU"
fi

STEPS=${STEPS:-"1 4 8 100 500 1000"}
TAUS=${TAUS:-"0.5 0.7 0.875"}  # paper default 0.5 + middle + our n=8 match
N_TEST=${N_TEST:-500}
ROOT=flow/results/sweep_dense_jump
mkdir -p $ROOT

for TAU in $TAUS; do
    OUT=$ROOT/tau_$TAU
    for S in $STEPS; do
        echo ""
        echo "----- dense_jump_tau=$TAU  n_steps=$S -----"
        python -u flow/burgers_fm_eval_v2.py \
            --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
            --out_dir $OUT/fm_n${S} \
            --n_test $N_TEST --n_steps $S --gammas 1.0 --variants vanilla \
            --dense_jump_tau $TAU \
            --device $DEVICE 2>&1 | tee $OUT/log_fm_n${S}.log
    done
done

echo ""
echo "✓ Output: $ROOT/tau_*/fm_n*/per_sample_J_*.npy"
echo "Total: $(find $ROOT -name 'per_sample_J_*.npy' | wc -l) (expected: 18)"
