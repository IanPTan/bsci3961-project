import os
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# -------------------------
# Config
# -------------------------
HIDDEN_DIM = 2048
HIDDEN_LOOPS = 3

VAL_H5 = f"{HIDDEN_DIM}_{HIDDEN_LOOPS}_val.h5"
PATHS_H5 = "vae_paths.h5"
PATCHES_H5 = "vae_patches.h5"
IMAGE_PATH = "frieren.png"

# Determine grid size from the number of unique patches (assumes square grid)
if os.path.exists(PATCHES_H5):
    with h5py.File(PATCHES_H5, "r") as f:
        num_patches = len(f["coords"])
        grid_size = int(np.sqrt(num_patches))
        GRID_H, GRID_W = grid_size, grid_size
        print(f"Detected {GRID_H}x{GRID_W} grid from {PATCHES_H5}")
else:
    # Fallback to default
    GRID_H, GRID_W = 961, 961


def create_heatmap():
    if not os.path.exists(VAL_H5):
        print(f"Error: {VAL_H5} not found. Run eval_rnn.py first.")
        return

    print(f"Loading data from {VAL_H5} and {PATHS_H5}...")
    
    with h5py.File(VAL_H5, "r") as f_val, h5py.File(PATHS_H5, "r") as f_paths:
        # y: (num_samples, seq_len, output_dim)
        y_pred = f_val["y"][:]
        
        # Ground truth patches from the 'val' group in vae_paths.h5
        # patches: (num_samples, seq_len_patches, enc_dim)
        # eval_rnn.py saves out_patches which are patches[1:]
        gt_patches = f_paths["val"]["patches"][:, 1:] 
        
        # Coords for these patches
        # coords: (num_samples, seq_len_patches, 2)
        coords = f_paths["val"]["coords"][:, 1:]

        num_samples, seq_len, _ = y_pred.shape

        print(f"Processing {num_samples} samples...")
        
        # Flatten everything to get (N_total, 2) coords and (N_total,) mses
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        gt_patches_flat = gt_patches.reshape(-1, gt_patches.shape[-1])
        coords_flat = coords.reshape(-1, 2)

        # Calculate MSE
        mses = np.mean((y_pred_flat - gt_patches_flat)**2, axis=1)

        # Calculate Statistics
        min_mse = np.min(mses)
        max_mse = np.max(mses)
        mean_mse = np.mean(mses)
        std_mse = np.std(mses)

        print(f"MSE Stats:")
        print(f"  Min:  {min_mse:.6f}")
        print(f"  Max:  {max_mse:.6f}")
        print(f"  Mean: {mean_mse:.6f}")
        print(f"  Std:  {std_mse:.6f}")

    # -------------------------
    # Interpolate Heatmap
    # -------------------------
    print("Interpolating heatmap...")
    
    # Grid coordinates (y, x)
    grid_y, grid_x = np.mgrid[0:GRID_H, 0:GRID_W]
    
    # coords_flat is (y, x)
    points = coords_flat
    values = mses

    # Interpolate
    # 'linear' or 'cubic' for smoothness
    heatmap = griddata(points, values, (grid_y, grid_x), method='linear')
    
    # Fill nans (areas not covered by any interpolation triangle) with the nearest value or 0
    if np.isnan(heatmap).any():
        heatmap_nearest = griddata(points, values, (grid_y, grid_x), method='nearest')
        heatmap[np.isnan(heatmap)] = heatmap_nearest[np.isnan(heatmap)]

    # -------------------------
    # Plot
    # -------------------------
    os.makedirs("figs", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.title(f"RNN Prediction MSE Heatmap ({HIDDEN_DIM}_{HIDDEN_LOOPS})")
    
    # Use log scale for better visualization if the range is large
    im = plt.imshow(heatmap, cmap='inferno', origin='upper')
    plt.colorbar(im, label='MSE Loss')
    
    # Add stats to the plot
    stats_text = (f"Min MSE: {min_mse:.6f}\n"
                  f"Max MSE: {max_mse:.6f}\n"
                  f"Mean MSE: {mean_mse:.6f}\n"
                  f"Std MSE: {std_mse:.6f}")
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel("X (Patcher Column)")
    plt.ylabel("Y (Patcher Row)")
    
    save_path = "figs/mse_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    create_heatmap()
