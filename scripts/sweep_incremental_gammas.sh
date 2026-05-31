#!/bin/bash
# sweep_incremental_gammas.sh — add MISSING gammas to existing sweep_500fresh.
#
# Use this when sweep_500_fresh.sh has already been run with a partial γ set
# and you want to add more γ values WITHOUT redoing the existing ones.
#
# Default: adds γ ∈ {0.1, 0.3, 0.7} to a sweep that already has {0.5, 1.0, 1.5, 2.0, 3.0}.
# Total: 3 γ × 6 n_steps = 18 new npy files in ~3 min on A100.
#
# Usage:
#   bash scripts/sweep_incremental_gammas.sh
#
# Override:
#   GAMMAS_NEW="0.05 0.2" STEPS="8 100" bash scripts/sweep_incremental_gammas.sh

cd /root/autodl-tmp/diffphycon

GAMMAS_NEW=${GAMMAS_NEW:-"0.1 0.3 0.7"}
STEPS=${STEPS:-"1 4 8 100 500 1000"}
N_TEST=${N_TEST:-500}

# Auto-detect CUDA
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    FM_DEVICE=cuda
    echo "🚀 GPU mode"
else
    FM_DEVICE=cpu
    export CUDA_VISIBLE_DEVICES=""
    echo "🐢 CPU mode — will be slow"
fi

# Setup ckpt symlinks (same as sweep_500_fresh.sh Step 5)
FM_TMP=/tmp/fm_step170k
mkdir -p $FM_TMP
FM_JOINT=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_joint_step170000.pt
FM_PRIOR=/root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_prior_step170000.pt
ln -sf "$FM_JOINT" $FM_TMP/vanilla_joint.pt
ln -sf "$FM_PRIOR" $FM_TMP/vanilla_prior.pt
echo "✓ joint + prior linked at $FM_TMP/"

OUT=flow/results/sweep_500fresh

# Pre-flight: confirm sweep_500fresh dir + existing npys
if [ ! -d "$OUT" ]; then
    echo "❌ $OUT does not exist — run sweep_500_fresh.sh first"
    exit 1
fi
echo "✓ existing γ npys (sanity check):"
ls $OUT/fm_n8/per_sample_J_*.npy 2>/dev/null | head -5 || echo "  (none found at fm_n8/, may have other layout)"

echo ""
echo "########## Adding γ = [$GAMMAS_NEW] to existing sweep ##########"
echo "n_steps to sweep: [$STEPS]"
echo "N_TEST: $N_TEST"

for S in $STEPS; do
    echo ""
    echo "----- FM n=$S (γ = $GAMMAS_NEW) -----"
    python -u flow/burgers_fm_eval_v2.py \
        --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
        --out_dir $OUT/fm_n${S} \
        --n_test $N_TEST --n_steps $S --gammas $GAMMAS_NEW --variants vanilla \
        --device $FM_DEVICE 2>&1 | tee -a $OUT/log_fm_n${S}.log
done

echo ""
echo "########## ✅ DONE ##########"
echo "Total npy files now in $OUT/fm_n*/:"
ls $OUT/fm_n*/per_sample_J_*.npy 2>/dev/null | wc -l
echo ""
echo "Per-step listing:"
for S in $STEPS; do
    cnt=$(ls $OUT/fm_n${S}/per_sample_J_*.npy 2>/dev/null | wc -l)
    echo "  fm_n${S}: $cnt npy files"
done
