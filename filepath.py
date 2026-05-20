import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

JELLYFISH_DATA_PATH = "/Users/baochen/diffphycon/data/jellyfish/"
JELLYFISH_RESULTS_PATH = "/Users/baochen/diffphycon/data/jellyfish/"
SMOKE_DATA_PATH = "/data/smoke/" # save the training / testing data of the 2D smoke control task
SMOKE_RESULTS_PATH  = "/data/smoke/" # save the results (training log, trained checkpoints, inference results) of the 2D smoke control task