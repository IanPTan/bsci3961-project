import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torchvision.transforms as transforms
from PIL import Image

from dataset import Patcher

# -------------------------
# Config
# -------------------------
HIDDEN_DIM = 2048
HIDDEN_LOOPS = 5

VAL_H5 = f"{HIDDEN_DIM}_{HIDDEN_LOOPS}_val.h5"
PATHS_H5 = "vae_paths.h5"
POS_WEIGHTS_PT = f"{HIDDEN_DIM}_{HIDDEN_LOOPS}_pos_weights.pt"
IMAGE_PATH = "frieren.png"

# Analysis Params
K_LOOPS = 4    # Number of best-performing loops to analyze
K_NEURONS = 10  # Number of top neurons to visualize per loop

# Patcher Params (to match gen_vae_paths.py)
PATCH_SIZE = 64
PATCH_STRIDE = 1

def get_grid_dims():
    if not os.path.exists(IMAGE_PATH):
        print(f"Warning: {IMAGE_PATH} not found, falling back to 961x961")
        return 961, 961
    img = transforms.ToTensor()(transforms.Resize(1024)(Image.open(IMAGE_PATH).convert("RGB")))
    patcher = Patcher(img, PATCH_SIZE, PATCH_STRIDE)
    return patcher.shape # (n_h, n_w)

GRID_H, GRID_W = get_grid_dims()

def probe_grid_cells():
    if not os.path.exists(POS_WEIGHTS_PT):
        print(f"Error: {POS_WEIGHTS_PT} not found. Run learn_pos.py first.")
        return
    if not os.path.exists(VAL_H5):
        print(f"Error: {VAL_H5} not found. Run eval_rnn.py first.")
        return

    print(f"Loading position model weights and stats from {POS_WEIGHTS_PT}...")
    checkpoint = torch.load(POS_WEIGHTS_PT, map_location="cpu")
    mses = checkpoint["mses"]
    state_dicts = checkpoint["state_dicts"]
    
    # 1. Identify "Bottom K" MSE loops (best performance)
    # mses is a dict: {"loop_0_mse": val, ...}
    sorted_loops = sorted(mses.items(), key=lambda x: x[1])
    best_loops = [k.replace("_mse", "") for k, v in sorted_loops[:K_LOOPS]]
    
    print(f"Top {K_LOOPS} Loops by MSE:")
    for loop_id in best_loops:
        print(f"  {loop_id}: MSE = {mses[loop_id + '_mse']:.6f}")

    # 2. Process each best loop
    os.makedirs("figs/probes", exist_ok=True)
    
    # Geometric Center for Skew calculation
    CENTER = torch.tensor([GRID_H / 2.0, GRID_W / 2.0])
    MAX_DIST = torch.norm(CENTER) # Dist from center to (0,0)
    COVERAGE_THRESHOLD = 0.3

    with h5py.File(VAL_H5, "r") as f_val, h5py.File(PATHS_H5, "r") as f_paths:
        # Load ground truth coordinates (skip first step to match h_all)
        coords_np = f_paths["val"]["coords"][:, 1:].reshape(-1, 2)
        coords = torch.from_numpy(coords_np).to(torch.float32)
        
        for loop_name in best_loops:
            loop_idx = int(loop_name.split("_")[1])
            print(f"\nAnalyzing Loop {loop_idx}...")
            
            # 1. Saliency from MLP Weights
            weights = state_dicts[loop_name]["net.0.weight"] # [1024, 2048]
            saliency = torch.sum(torch.abs(weights), dim=0) # [2048]

            # 2. Spatial Metrics (Coverage & Skew)
            print(f"  Calculating spatial metrics for all {HIDDEN_DIM} neurons...")
            H = torch.from_numpy(f_val["h_all"][:, :, loop_idx, :]).view(-1, HIDDEN_DIM)
            
            # Normalize H per neuron (0 to 1)
            H_min = H.min(dim=0).values
            H_max = H.max(dim=0).values
            H_norm = (H - H_min) / (H_max - H_min + 1e-8)

            # Coverage: % of space occupied
            coverage = (H_norm > COVERAGE_THRESHOLD).float().mean(dim=0)

            # Centroid: Weighted average of coordinates
            sum_act = H_norm.sum(dim=0) + 1e-8
            centroid = (H_norm.T @ coords) / sum_act.unsqueeze(-1) # [2048, 2]

            # Skew: Normalized distance from center (0 = center, 1 = extreme)
            skew_dist = torch.norm(centroid - CENTER, dim=1)
            skew_norm = (skew_dist / MAX_DIST).clamp(0, 1)

            # 3. Global Coordinate Score
            # High saliency * High coverage * Low skew
            global_score = saliency * coverage * (1.0 - skew_norm)
            
            top_indices = torch.topk(global_score, K_NEURONS).indices.tolist()
            
            print(f"  Top {K_NEURONS} neurons by Global Score:")
            for idx in top_indices:
                print(f"    Neuron {idx:4d}: Score={global_score[idx]:.2f}, Sal={saliency[idx]:.2f}, Cov={coverage[idx]:.2f}, Skew={skew_norm[idx]:.2f}")

            # 4. Generate heatmaps
            for neuron_idx in top_indices:
                print(f"    Generating heatmap for Neuron {neuron_idx}...")
                h_neuron = H[:, neuron_idx].numpy()
                
                # Interpolate heatmap
                grid_y, grid_x = np.mgrid[0:GRID_H, 0:GRID_W]
                heatmap = griddata(coords_np, h_neuron, (grid_y, grid_x), method='linear')
                
                if np.isnan(heatmap).any():
                    heatmap_nearest = griddata(coords_np, h_neuron, (grid_y, grid_x), method='nearest')
                    heatmap[np.isnan(heatmap)] = heatmap_nearest[np.isnan(heatmap)]

                plt.figure(figsize=(8, 6))
                plt.title(f"Loop {loop_idx} | Neuron {neuron_idx} | Score: {global_score[neuron_idx]:.2f}\n"
                          f"Sal: {saliency[neuron_idx]:.2f} | Cov: {coverage[neuron_idx]:.2f} | Skew: {skew_norm[neuron_idx]:.2f}")
                
                im = plt.imshow(heatmap, cmap='magma', origin='upper')
                plt.colorbar(im, label='Activation')
                
                # Mark Centroid
                plt.scatter(centroid[neuron_idx, 1], centroid[neuron_idx, 0], color='red', marker='x', s=100, label='Centroid')
                
                plt.xlabel("X (Patcher Column)")
                plt.ylabel("Y (Patcher Row)")
                plt.legend()
                
                save_path = f"figs/probes/loop{loop_idx}_neuron{neuron_idx}.png"
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()

    print("\nProbing complete. Heatmaps saved to figs/probes/")

if __name__ == "__main__":
    probe_grid_cells()
