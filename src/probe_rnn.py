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
K_LOOPS = 2    # Number of best-performing loops to analyze
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
    
    with h5py.File(VAL_H5, "r") as f_val, h5py.File(PATHS_H5, "r") as f_paths:
        # Load ground truth coordinates (skip first step to match h_all)
        coords = f_paths["val"]["coords"][:, 1:].reshape(-1, 2)
        
        for loop_name in best_loops:
            loop_idx = int(loop_name.split("_")[1])
            print(f"\nAnalyzing {loop_name} (Loop {loop_idx})...")
            
            # Extract weights from the first layer of the MLP
            # net.0 is the first Linear layer (input_dim -> hidden_dim1)
            weights = state_dicts[loop_name]["net.0.weight"] # [1024, 2048]
            
            # 3. Identify "Top K" absolute value input neurons
            # We sum absolute weights across all outputs of the first layer for each input neuron
            saliency = torch.sum(torch.abs(weights), dim=0) # [2048]
            top_neuron_indices = torch.topk(saliency, K_NEURONS).indices.tolist()
            
            print(f"  Top {K_NEURONS} neurons in Loop {loop_idx}: {top_neuron_indices}")

            # 4. Generate heatmaps for each top neuron
            for neuron_idx in top_neuron_indices:
                print(f"    Generating heatmap for Neuron {neuron_idx}...")
                
                # Load activations for ONLY this neuron across all samples/timesteps
                # h_all shape: (num_samples, seq_len, internal_loops, hidden_dim)
                h_neuron = f_val["h_all"][:, :, loop_idx, neuron_idx].reshape(-1)
                
                # Interpolate heatmap
                grid_y, grid_x = np.mgrid[0:GRID_H, 0:GRID_W]
                points = coords # (N, 2) where [:, 0] is y, [:, 1] is x
                values = h_neuron
                
                heatmap = griddata(points, values, (grid_y, grid_x), method='linear')
                
                # Fill nans
                if np.isnan(heatmap).any():
                    heatmap_nearest = griddata(points, values, (grid_y, grid_x), method='nearest')
                    heatmap[np.isnan(heatmap)] = heatmap_nearest[np.isnan(heatmap)]

                # Plot
                plt.figure(figsize=(8, 6))
                plt.title(f"Loop {loop_idx} | Neuron {neuron_idx} Activation Map\n(Saliency: {saliency[neuron_idx]:.2f})")
                im = plt.imshow(heatmap, cmap='viridis', origin='upper')
                plt.colorbar(im, label='Activation Value')
                plt.xlabel("X (Patcher Column)")
                plt.ylabel("Y (Patcher Row)")
                
                save_path = f"figs/probes/loop{loop_idx}_neuron{neuron_idx}.png"
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()

    print("\nProbing complete. Heatmaps saved to figs/probes/")

if __name__ == "__main__":
    probe_grid_cells()
