import numpy as np
import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import os
from pathlib import Path
from utils import update_static_masks, compute_pressForce
from tqdm.auto import tqdm
import time 
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Polygon
from shapely.ops import unary_union
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


root = "/Users/weilong/data/control/fixed_jellyfish64_gentle_test/"


start = 0
n_files = 4
m = 20 # number of points on each ellipse wing of jellyfish
steps = 40 # time steps in each example
num_threads = 1
dirname = "train_data/" # construct training data folder
# dirname = "test_data/" # construct test data folder

device = torch.device("cuda:0")


##############################################################
#Step 1: generate boundary mask and offset for boundary points
#############################################################
if not os.path.exists(os.path.join(root, dirname, "bdry_mask_offsets")):
    os.makedirs(os.path.join(root, dirname, "bdry_mask_offsets"))   

def process_one_bd(sim_id):
    # print("sim_id: ", sim_id)
    bd_points = torch.FloatTensor(np.load(os.path.join(root, dirname, "bdry/sim_{:06d}.npy".format(sim_id)))).to(device)
    mask_offsets = []
    for i in range(steps):
        bd1 = bd_points[0,i]
        bd2 = bd_points[1,i]
        mask1, offset1 = update_static_masks(bd1, n_p=20, res=128)
        mask2, offset2 = update_static_masks(bd2, n_p=20, res=128)
        mask_offset1 = torch.cat([mask1.unsqueeze(-1), offset1], dim=-1)
        mask_offset2 = torch.cat([mask2.unsqueeze(-1), offset2], dim=-1)
        mask_offset = torch.cat(
            [
                mask_offset1.unsqueeze(0), 
                mask_offset2.unsqueeze(0)
            ], 
            dim=0
        ).unsqueeze(0)
        mask_offsets.append(mask_offset)    

    mask_offsets = torch.cat(mask_offsets, dim=0).cpu().numpy()

    np.savez_compressed(os.path.join(root, dirname, 'bdry_mask_offsets/sim_{:06d}'.format(sim_id)), a=mask_offsets) # very sparse

for batch_id in tqdm(range(0, (n_files - start)// num_threads + 1)):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_one_bd, [sim_id for sim_id in range(
            start + batch_id * num_threads, 
            min(start + (batch_id + 1) * num_threads, n_files)
        )]) 


########################################
#Step 2: merge two boundaries into one
########################################


if not os.path.exists(os.path.join(root, dirname, "bdry_merged_mask_offsets")):
    os.makedirs(os.path.join(root, dirname, "bdry_merged_mask_offsets"))

for sim_id in tqdm(range(start, n_files)):
    if not os.path.exists(os.path.join(root, dirname, "bdry/sim_{:06d}.npy".format(sim_id))):
        print("sim_id: ", sim_id)
        assert False
    bd_points = torch.FloatTensor(np.load(os.path.join(root, dirname, "bdry/sim_{:06d}.npy".format(sim_id))))
    mask_offsets = []
    head = bd_points[0,0,m//2]
    xh, yh = head
    for i in range(steps):
        bd1 = bd_points[0,i].tolist()
        bd2 = bd_points[1,i].tolist()
        x1, y1 = bd1[0] # tail point of the upper wing
        x2, y2 = bd2[0] # tail point of the lower wing
        poly1 = Polygon(bd1)
        poly2 = Polygon(bd2)
        if not poly1.intersects(poly2):
            print("not valid data: ", bd1, bd2)
            assert False
        intersection = poly1.intersection(poly2)
        if intersection.geom_type == "Point":
            mask1, offset1 = update_static_masks(bd_points[0,i].to(device), n_p=m, res=128)
            mask2, offset2 = update_static_masks(bd_points[1,i].to(device), n_p=m, res=128)
            mask_offset1 = torch.cat([mask1.unsqueeze(-1), offset1], dim=-1)
            mask_offset2 = torch.cat([mask2.unsqueeze(-1), offset2], dim=-1)
            mask_offset = (mask_offset1 + mask_offset2).unsqueeze(0)
            mask_offset[0,:,:,0] = mask_offset[0,:,:,0].clamp(0., 1.) # clamp mask to (0,1) after summation
        else:
            union_poly = poly1.union(poly2)
            bd_point = union_poly.exterior.coords[:]
            mask, offset = update_static_masks(torch.FloatTensor(bd_point).to(device), n_p=len(bd_point), res=128)
            mask_offset = torch.cat([mask.unsqueeze(-1), offset], dim=-1).unsqueeze(0)
        mask_offsets.append(mask_offset)     
            
    mask_offsets = torch.cat(mask_offsets, dim=0).cpu().numpy()
    np.savez_compressed(os.path.join(root, dirname, 'bdry_merged_mask_offsets/sim_{:06d}'.format(sim_id)), a=mask_offsets) # very sparse

    
##############################################################
# Step 3: compute head point and theta of upper wing of each time
##############################################################
EPS = 1e-3
if not os.path.exists(os.path.join(root, dirname, "bdry_head_thetas")):
    os.makedirs(os.path.join(root, dirname, "bdry_head_thetas"))
for sim_id in tqdm(range(start, n_files)):
    # print("sim_id: {}".format(sim_id))
    if not os.path.exists(os.path.join(root, dirname, "bdry/sim_{:06d}.npy".format(sim_id))):
        print("sim_id: ", sim_id)
        assert False
    bd_points = np.load(os.path.join(root, dirname, "bdry/sim_{:06d}.npy".format(sim_id)))
    head = bd_points[0, 0, m//2]
    xh, yh = head
    # for i in range(steps):
    #     bd1 = bd_points[0,i].tolist()
    #     bd2 = bd_points[1,i].tolist()
    #     x1, y1 = bd1[0]
    #     x2, y2 = bd2[0]
    #     assert x1 > xh and x2 > xh and y1 > yh and y2< yh and abs(x1 - x2) < EPS and abs(y1 + y2 - 2 * yh) < EPS
    
    x_tail, y_tail = bd_points[0, :, 0, 0], bd_points[0, :, 0, 1]
    thetas = np.arctan((y_tail - yh) / (x_tail - xh))
    # thetas_degree = thetas * 180 / np.pi
    with open(os.path.join(root, dirname, 'bdry_head_thetas/sim_{:06d}.npz'.format(sim_id)), 'wb') as f:
        np.savez(f, head=head, thetas=thetas)

x_min, x_max, y_min, y_max = 128, 0, 128, 0
for sim_id in tqdm(range(start, n_files)):
    # print("loading sim_id: {}...".format(sim_id))
    with np.load(os.path.join(root, dirname, 'bdry_head_thetas/sim_{:06d}.npz'.format(sim_id))) as data:
        head = data['head']
        thetas = data['thetas']
        xh, yh = head
        x_min = min(x_min, xh)
        x_max = max(x_max, xh)
        y_min = min(y_min, yh)
        y_max = max(y_max, yh)
print(x_min, x_max, y_min, y_max)


##############################################################
# Step 4: down sampling state, boundary and offset data
##############################################################
source_dir = "/root/user/jellyfish64/"
target_dir = "/root/user/jellyfish64/"

# Step 4.1 downsample state data
dirname = "train_data"
source_state_dir = os.path.join(source_dir, dirname, "states")
target_state_dir = os.path.join(target_dir, dirname, "states")
state_files = os.listdir(source_state_dir)
state_files.sort()
n_sim = 50000
for sim_id in tqdm(range(5)):
    state = np.load(os.path.join(source_state_dir,"sim_{:06d}.npz".format(sim_id)))["a"] # [40, 3, 126, 126]
    state_pad = np.zeros([40, 3, 128, 128], dtype=state.dtype)
    state_pad[:,:,1:-1,1:-1] = state
    state_down = state_pad[:, :, ::2, ::2] # down sample by taking every alternate pixel
    np.savez_compressed(os.path.join(target_state_dir, "sim_{:06d}".format(sim_id)), a=state_down)


# Step 4.2 downsample boundary mask and offset
root = "/data/user/pde_ctrl/jellyfish64/"
dirname = "train_data"
source_bd_dir = os.path.join(root, dirname, "bdry128")
target_bd_dir = os.path.join(root, dirname, "bdry64")
target_mask_offset_dir = os.path.join(root, dirname, "bdry_merged_mask_offsets")
state_files = os.listdir(source_bd_dir)
state_files.sort()
n_sim = 40000
res = 64
for sim_id in tqdm(range(0, n_sim)):
    if not os.path.exists(os.path.join(source_bd_dir, "sim_{:06d}.npy".format(sim_id))):
        print("sim_id: ", source_bd_dir, sim_id)
        assert False
    original_bd_points = torch.FloatTensor(np.load(os.path.join(source_bd_dir, "sim_{:06d}.npy".format(sim_id))))
    # bd_mask_offset128 = torch.FloatTensor(np.load(os.path.join(root, dirname, "bdry_merged_mask_offsets/sim_{:06d}.npz".format(sim_id)))["a"])
    ######### down sampling: devide by 2  #######
    bd_points = original_bd_points / 2
    # save down sampled boundary
    np.save(os.path.join(target_bd_dir, "sim_{:06d}.npy".format(sim_id)), bd_points.cpu().numpy())
    mask_offsets = []
    head = bd_points[0,0,m//2]
    xh, yh = head
    for i in range(steps):
        bd1 = bd_points[0,i].tolist()
        bd2 = bd_points[1,i].tolist()
        x1, y1 = bd1[0] # tail point of the upper wing
        x2, y2 = bd2[0] # tail point of the lower wing
        poly1 = Polygon(bd1)
        poly2 = Polygon(bd2)
        if not poly1.intersects(poly2):
            print("not valid data: ", bd1, bd2)
            assert False
        intersection = poly1.intersection(poly2)
        if intersection.geom_type == "Point":
            mask1, offset1 = update_static_masks(bd_points[0,i].to(device), n_p=m, res=res)
            mask2, offset2 = update_static_masks(bd_points[1,i].to(device), n_p=m, res=res)
            mask_offset1 = torch.cat([mask1.unsqueeze(-1), offset1], dim=-1)
            mask_offset2 = torch.cat([mask2.unsqueeze(-1), offset2], dim=-1)
            mask_offset = (mask_offset1 + mask_offset2).unsqueeze(0)
            mask_offset[0,:,:,0] = mask_offset[0,:,:,0].clamp(0., 1.) # clamp mask to (0,1) after summation
        else:
            union_poly = poly1.union(poly2)
            bd_point = union_poly.exterior.coords[:]
            # mask, offset = update_static_masks(torch.FloatTensor(bd_point).to(device), n_p=len(bd_point), res=res)
            mask, offset = update_static_masks(torch.FloatTensor(bd_point), n_p=len(bd_point), res=res)
            mask_offset = torch.cat([mask.unsqueeze(-1), offset], dim=-1).unsqueeze(0)
        mask_offsets.append(mask_offset)     
    mask_offsets = torch.cat(mask_offsets, dim=0).cpu().numpy()
    np.savez_compressed(os.path.join(target_mask_offset_dir, 'sim_{:06d}'.format(sim_id)), a=mask_offsets) # very sparse, easy to compress
