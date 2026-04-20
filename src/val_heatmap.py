import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# -------------------------
# Config
# -------------------------
PATHS_H5 = "vae_paths.h5"
# Fixed 100x100 grid for better visualization
BINS_W, BINS_H = 100, 100
# The actual extent of the patcher coordinates
GRID_W, GRID_H = 961, 961

def create_val_heatmap():
    if not os.path.exists(PATHS_H5):
        print(f"Error: {PATHS_H5} not found.")
        return

    print(f"Loading coordinates from {PATHS_H5}...")
    with h5py.File(PATHS_H5, "r") as f:
        if "val" not in f:
            print("Error: 'val' group not found in H5.")
            return
        coords = f["val"]["coords"][:]
    
    # Flatten to (N, 2)
    coords_flat = coords.reshape(-1, 2)
    rows = coords_flat[:, 0]
    cols = coords_flat[:, 1]

    print(f"Bining {len(coords_flat)} points into {BINS_W}x{BINS_H} grid...")

    # Create 2D histogram for density
    heatmap, xedges, yedges = np.histogram2d(
        cols, rows, 
        bins=[BINS_W, BINS_H], 
        range=[[0, GRID_W], [0, GRID_H]]
    )

    # Transpose because histogram2d returns H[x, y] but imshow expects H[row, col]
    heatmap = heatmap.T

    # Apply Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=1.0)

    # -------------------------
    # Plot
    # -------------------------
    os.makedirs("figs", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.title("Validation Path Coverage Density (100x100 Bins)")
    
    # Use inferno for high contrast, no hardcoded vmax
    im = plt.imshow(heatmap, cmap='inferno', origin='upper', extent=[0, GRID_W, 0, GRID_H])
    plt.colorbar(im, label='Density (Smoothed Hits)')
    
    plt.xlabel("X (Patcher Column)")
    plt.ylabel("Y (Patcher Row)")
    plt.gca().invert_yaxis() # Top-left origin
    
    save_path = "figs/val_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Coverage heatmap saved to {save_path}")

if __name__ == "__main__":
    create_val_heatmap()
