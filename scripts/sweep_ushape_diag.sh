#!/bin/bash
# sweep_ushape_diag.sh — full factorial diagnostic for FM U-shape.
#
# Cell design (4 cells + RK4 control = 5 conditions):
#   Cell A:  cap_τ ∈ {0.7, 0.85, 0.9, 0.95}, reinpaint=OFF  → E1 only (4 × 6 = 24)
#   Cell B:  cap_τ=1.0, reinpaint=ON                        → E3 only      (1 × 6 = 6)
#   Cell C:  cap_τ ∈ {0.7, 0.85, 0.9, 0.95}, reinpaint=ON   → E1+E3        (4 × 6 = 24)
#   Cell D:  cap_τ=1.0, reinpaint=OFF, integrator=rk4       → E4 only      (1 × 6 = 6)
#
# Baseline (cap_τ=1.0, reinpaint=OFF, euler) skipped — already in
# sweep_500fresh/fm_n*/per_sample_J_*_g1.00_n*.npy
#
# Total: 60 configs. ~13 min on A100.
#
# All use γ=1.0 (no prior contribution; pure joint dynamics).
#
# Usage: bash scripts/sweep_ushape_diag.sh

cd /root/autodl-tmp/diffphycon
FM_TMP=/tmp/fm_step170k
[ ! -e $FM_TMP/vanilla_joint.pt ] && { echo "ERROR: run sweep_500_fresh.sh first"; exit 1; }
[ ! -e $FM_TMP/vanilla_prior.pt ] && { echo "ERROR: prior missing"; exit 1; }
echo "✓ joint + prior symlinks present"

# Auto-detect GPU
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda; echo "🚀 GPU"
else
    DEVICE=cpu; export CUDA_VISIBLE_DEVICES=""; echo "🐢 CPU (slow)"
fi

STEPS=${STEPS:-"1 4 8 100 500 1000"}
N_TEST=${N_TEST:-500}
CAP_TAUS=${CAP_TAUS:-"0.7 0.85 0.9 0.95"}
ROOT=flow/results/sweep_ushape_diag
mkdir -p $ROOT

# Run a single config
run_one() {
    local out=$1; local n=$2; local extra="$3"
    mkdir -p $out
    python -u flow/burgers_fm_eval_v2.py \
        --ckpt_dir $FM_TMP --dataset free_u_f_paper_fopc \
        --out_dir $out/fm_n${n} \
        --n_test $N_TEST --n_steps $n --gammas 1.0 --variants vanilla \
        --device $DEVICE $extra 2>&1 | tee $out/log_fm_n${n}.log
}

# ─── Cell A: E1 only (cap_τ × reinpaint=off × Euler) ───
echo ""
echo "########## Cell A: E1 — cap_τ ∈ [$CAP_TAUS], reinpaint=OFF ##########"
for ct in $CAP_TAUS; do
    OUT=$ROOT/cap_tau_${ct}
    for S in $STEPS; do
        echo "----- cap_τ=$ct  n_steps=$S -----"
        run_one $OUT $S "--cap_tau $ct"
    done
done

# ─── Cell B: E3 only (reinpaint=on, cap_τ=1.0) ───
echo ""
echo "########## Cell B: E3 — reinpaint=ON (no cap_τ) ##########"
OUT=$ROOT/reinpaint
for S in $STEPS; do
    echo "----- reinpaint=on  n_steps=$S -----"
    run_one $OUT $S "--reinpaint_boundary"
done

# ─── Cell C: E1+E3 combined (cap_τ × reinpaint=on × Euler) ───
echo ""
echo "########## Cell C: E1+E3 — cap_τ ∈ [$CAP_TAUS] WITH reinpaint=ON ##########"
for ct in $CAP_TAUS; do
    OUT=$ROOT/cap_tau_${ct}_reinpaint
    for S in $STEPS; do
        echo "----- cap_τ=$ct reinpaint=on  n_steps=$S -----"
        run_one $OUT $S "--cap_tau $ct --reinpaint_boundary"
    done
done

# ─── Cell D: E4 only (RK4 integrator, cap_τ=1.0, reinpaint=off) ───
echo ""
echo "########## Cell D: E4 — integrator=rk4 (Euler control) ##########"
OUT=$ROOT/rk4
for S in $STEPS; do
    echo "----- rk4  n_steps=$S -----"
    run_one $OUT $S "--integrator rk4"
done

echo ""
echo "########## ✅ ALL DONE ##########"
echo "Output: $ROOT/{cap_tau_*,reinpaint,cap_tau_*_reinpaint,rk4}/fm_n*/per_sample_J_vanilla_g1.00_n*.npy"
echo ""
echo "Total npy:"
find $ROOT -name 'per_sample_J_*.npy' | wc -l
echo "(expected: 60 = 24 + 6 + 24 + 6)"
