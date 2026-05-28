#!/bin/bash
# Paper-scale FM training (FOPC): vanilla vs OT-CFM, on a single GPU (AutoDL 4090).
# Run AFTER setup (see AUTODL_SETUP.md). Run from repo root.
#
# Stages: (1) smoke  (2) generate 100k data  (3) train 4 models  (4) eval
# Use STAGE env var to run a single stage, e.g.:  STAGE=smoke bash run_autodl_fopc_paper.sh
set -e

DATASET=free_u_f_paper_fopc
DIM=128
DIM_MULTS="1 2 4 8"
JOINT_STEPS=200000
PRIOR_STEPS=50000
BATCH=16
CKPT=checkpoints/paper_fopc
mkdir -p $CKPT
STAGE=${STAGE:-all}

# ---------------------------------------------------------------- smoke
if [ "$STAGE" = "smoke" ] || [ "$STAGE" = "all" ]; then
  echo "########## STAGE 1: SMOKE (tiny data + 500 steps) ##########"
  python dataset/apps/generate_burgers.py --train_samples 1000 --test_samples 200 \
      --partial_control front_rear_quarter --nx 128 --nt 11 --device cuda:0 \
      --save_path data/free_u_f_smoke_fopc
  python flow/burgers_fm_train.py --variant vanilla --model joint --dim $DIM --dim_mults $DIM_MULTS \
      --num_steps 500 --batch_size $BATCH --dataset free_u_f_smoke_fopc --device cuda \
      --save_path $CKPT/smoke_vanilla_joint.pt --ckpt_every 0 --print_every 100
  echo ">>> SMOKE OK. Note the s/step above; if 200k×0.X s is acceptable, run STAGE=all."
  [ "$STAGE" = "smoke" ] && exit 0
fi

# ---------------------------------------------------------------- data
if [ "$STAGE" = "data" ] || [ "$STAGE" = "all" ]; then
  echo "########## STAGE 2: GENERATE 100k DATA ##########"
  python dataset/apps/generate_burgers.py --train_samples 90000 --test_samples 10000 \
      --partial_control front_rear_quarter --nx 128 --nt 11 --device cuda:0 \
      --save_path data/$DATASET
fi

# ---------------------------------------------------------------- train
if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ]; then
  echo "########## STAGE 3: TRAIN 4 MODELS ##########"
  for V in vanilla ot; do
    echo ">>> $V joint ($JOINT_STEPS steps)"
    python flow/burgers_fm_train.py --variant $V --model joint --dim $DIM --dim_mults $DIM_MULTS \
        --num_steps $JOINT_STEPS --batch_size $BATCH --lr 1e-4 --dataset $DATASET --device cuda \
        --save_path $CKPT/fm_${V}_joint.pt --ckpt_every 25000
    echo ">>> $V prior ($PRIOR_STEPS steps)"
    python flow/burgers_fm_train.py --variant $V --model prior --dim $DIM --dim_mults $DIM_MULTS \
        --num_steps $PRIOR_STEPS --batch_size $BATCH --lr 1e-4 --dataset $DATASET --device cuda \
        --save_path $CKPT/fm_${V}_prior.pt --ckpt_every 25000
  done
fi

# ---------------------------------------------------------------- eval
if [ "$STAGE" = "eval" ] || [ "$STAGE" = "all" ]; then
  echo "########## STAGE 4: EVAL (vanilla vs OT vs paper 0.00037) ##########"
  python flow/burgers_fm_eval.py --dataset $DATASET --n_test 50 --dim $DIM --dim_mults $DIM_MULTS \
      --n_steps 100 --gammas 0.5 1.0 1.5 2.5 --device cuda \
      --vanilla_joint $CKPT/fm_vanilla_joint.pt --vanilla_prior $CKPT/fm_vanilla_prior.pt \
      --ot_joint      $CKPT/fm_ot_joint.pt      --ot_prior      $CKPT/fm_ot_prior.pt \
      --out_dir flow/results/paper_fopc
fi

echo "########## DONE. Download $CKPT and flow/results/paper_fopc back to your Mac. ##########"
