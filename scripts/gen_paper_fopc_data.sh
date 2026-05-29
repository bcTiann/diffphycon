#!/bin/bash
# gen_paper_fopc_data.sh — clean re-generate 90k train + 50 test FO-PC data
# Usage:  bash scripts/gen_paper_fopc_data.sh

set -e

cd /root/autodl-tmp/diffphycon

echo "########## STEP 1: remove old data dir ##########"
rm -rf data/free_u_f_paper_fopc
echo "✓ deleted"

echo ""
echo "########## STEP 2: generate 90000 train + 50 test (~3-6 min on A100) ##########"
python dataset/apps/generate_burgers.py \
    --train_samples 90000 \
    --test_samples 50 \
    --partial_control front_rear_quarter \
    --nx 128 \
    --nt 11 \
    --device cuda:0 \
    --save_path free_u_f_paper_fopc/

echo ""
echo "########## STEP 3: verify shape ##########"
python -c "import h5py; f=h5py.File('data/free_u_f_paper_fopc/burgers_train.h5','r'); print('train shape:', f['train']['pde_11-128'].shape); f.close(); g=h5py.File('data/free_u_f_paper_fopc/burgers_test.h5','r'); print('test shape:', g['test']['pde_11-128'].shape); g.close()"

echo ""
echo "########## STEP 4: file sizes ##########"
ls -la data/free_u_f_paper_fopc/

echo ""
echo "########## DONE ##########"
echo "Expected:"
echo "  train shape: (90000, 11, 128)   ← ~1.9 GB"
echo "  test  shape: (50,    11, 128)   ← ~1 MB"
