#!/bin/bash
# fix_h5.sh — fix h5 attribute corruption on both train + test files,
# then verify by reading nt attribute.

set -e

cd /root/autodl-tmp/diffphycon

DATASET=data/free_u_f_paper_fopc

echo "########## STEP 1: cancel pending shutdown (if any) ##########"
/usr/bin/shutdown -c 2>&1 || true

echo ""
echo "########## STEP 2: repair train ##########"
python scripts/fix_h5_attrs.py $DATASET/burgers_train.h5

echo ""
echo "########## STEP 3: repair test ##########"
python scripts/fix_h5_attrs.py $DATASET/burgers_test.h5

echo ""
echo "########## STEP 4: verify by reading nt attr (should print 11) ##########"
python -c "import h5py; f=h5py.File('$DATASET/burgers_train.h5','r'); ds=f['train']['pde_11-128']; print('train nt:', ds.attrs['nt'], '  shape:', ds.shape); f.close()"
python -c "import h5py; f=h5py.File('$DATASET/burgers_test.h5','r'); ds=f['test']['pde_11-128'];  print('test  nt:', ds.attrs['nt'], '  shape:', ds.shape); f.close()"

echo ""
echo "✅ DONE. Now you can resume training:"
echo "  export TERM=xterm-256color"
echo "  tmux new -s train"
echo "  bash scripts/train_paper_autodl.sh"
