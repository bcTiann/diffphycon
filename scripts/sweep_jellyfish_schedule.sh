#!/bin/bash
# sweep_jellyfish_schedule.sh — paper-faithful γ-sweep using jellyfish β-schedule.
#
# Reuses existing test h5 + /tmp/fm_step170k symlinks from sweep_500_fresh.sh.
# No data regen, no paper-inference. Only adds jellyfish-schedule FM npy files
# alongside our existing sigmoid_flip npy files (different output dir).
#
# γ in {0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2} with PAPER MAGNITUDE
# (γ is treated like paper ξ; peak prior coef ≈ (γ-1) × 0.02).
#
# Usage:
#   bash scripts/sweep_jellyfish_schedule.sh
#
# Override:
#   GAMMAS="0.0 0.3 0.5 0.7 1.0" STEPS="8 100" bash scripts/sweep_jellyfish_schedule.sh

cd /root/autodl-tmp/diffphycon

FM_TMP=/tmp/fm_step170k

# Pre-flight: symlinks must exist (created by sweep_500_fresh.sh Step 5)
if [ ! -L "$FM_TMP/vanilla_joint.pt" ] && [ ! -f "$FM_TMP/vanilla_joint.pt" ]; then
    echo "❌ $FM_TMP/vanilla_joint.pt missing — run sweep_500_fresh.sh first to set up symlinks"
    exit 1
fi
if [ ! -L "$FM_TMP/vanilla_prior.pt" ] && [ ! -f "$FM_TMP/vanilla_prior.pt" ]; then
    echo "❌ $FM_TMP/vanilla_prior.pt missing — run sweep_500_fresh.sh first (with prior link)"
    exit 1
fi
echo "✓ joint + prior symlinks present in $FM_TMP"

# Auto-detect GPU
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    FM_DEVICE=cuda
    echo "🚀 GPU mode"
else
    FM_DEVICE=cpu
    export CUDA_VISIBLE_DEVICES=""
    echo "🐢 CPU mode — slow"
fi

STEPS=${STEPS:-"1 4 8 100 500 1000"}
GAMMAS=${GAMMAS:-"0.5 0.7 0.8 0.9 1.0 1.1 1.2"}
N_TEST=${N_TEST:-500}
OUT=flow/results/sweep_500fresh_jellyfish
mkdir -p $OUT
echo "STEPS=[$STEPS]  GAMMAS=[$GAMMAS]  N_TEST=$N_TEST"
echo "OUT=$OUT"

for S in $STEPS; do
    echo ""
    echo "########## FM n=$S (γ ∈ $GAMMAS, schedule=jellyfish_beta) ##########"
    python -u flow/burgers_fm_eval_v2.py \
        --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
        --out_dir $OUT/fm_n${S} \
        --n_test $N_TEST --n_steps $S --gammas $GAMMAS --variants vanilla \
        --schedule jellyfish_beta \
        --device $FM_DEVICE 2>&1 | tee $OUT/log_fm_n${S}.log
done

echo ""
echo "########## ✅ DONE ##########"
echo "Per-n_steps npy count:"
for S in $STEPS; do
    cnt=$(ls $OUT/fm_n${S}/per_sample_J_*.npy 2>/dev/null | wc -l | tr -d ' ')
    echo "  fm_n${S}: $cnt npy files"
done
echo ""
echo "Total npy:"
find $OUT -name 'per_sample_J_*.npy' | wc -l
