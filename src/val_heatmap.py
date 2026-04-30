import os
import h5py
import numpy as np
import torch as pt
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse
import tomllib
from pathlib import Path

from dataset import Patcher

def get_latest_exp_dir(prefix="rnn_"):
    exp_root = Path("experiments")
    if not exp_root.exists():
        return None
        
    max_num = 0
    latest_dir = None
    for d in exp_root.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            try:
                num = int(d.name.split("_")[1])
                if num > max_num:
                    max_num = num
                    latest_dir = d
            except ValueError:
                pass
    return latest_dir

def get_grid_dims(image_path, patch_size, patch_stride):
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found, falling back to 961x961")
        return 961, 961
    
    img = transforms.ToTensor()(transforms.Resize(1024)(Image.open(image_path).convert("RGB")))
    patcher = Patcher(img, patch_size, patch_stride)
    n_h, n_w = patcher.shape
    print(f"Detected grid dimensions: {n_w}x{n_h} from {image_path}")
    return n_w, n_h

def load_defaults(path="experiments/paths_defaults.toml"):
    if not os.path.exists(path):
        print(f"Error: Default config not found at {path}")
        exit(1)
    with open(path, "rb") as f:
        return tomllib.load(f)

def create_val_heatmap():
    parser = argparse.ArgumentParser(description="Generate validation path coverage density heatmap")
    parser.add_argument("--exp-dir", type=str, help="Path to experiment directory (optional, for output location)")
    args = parser.parse_args()
    
    # Defaults
    defaults = load_defaults()
    paths_h5_path = Path(defaults.get("data_path", "data/vae_paths.h5"))
    image_path = "data/frieren.png"
    patch_size = defaults.get("patch_size", 64)
    patch_stride = defaults.get("patch_stride", 1)
    
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = get_latest_exp_dir("rnn_")
        
    if exp_dir:
        config_path = exp_dir / "config.toml"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
                paths_h5_path = Path(config.get("data_path", paths_h5_path))
                image_path = config.get("image_path", image_path)
                patch_size = config.get("patch_size", patch_size)
                patch_stride = config.get("patch_stride", patch_stride)

    grid_w, grid_h = get_grid_dims(image_path, patch_size, patch_stride)

    if not paths_h5_path.exists():
        print(f"Error: {paths_h5_path} not found.")
        return

    print(f"Loading coordinates from {paths_h5_path}...")
    with h5py.File(paths_h5_path, "r") as f:
        if "val" not in f:
            print("Error: 'val' group not found in H5.")
            return
        coords = f["val"]["coords"][:]
    
    # Flatten to (N, 2)
    coords_flat = coords.reshape(-1, 2)
    rows = coords_flat[:, 0]
    cols = coords_flat[:, 1]

    # Fixed bins for visualization density
    BINS_W, BINS_H = 100, 100
    print(f"Bining {len(coords_flat)} points into {BINS_W}x{BINS_H} grid...")

    # Create 2D histogram for density
    heatmap, xedges, yedges = np.histogram2d(
        cols, rows, 
        bins=[BINS_W, BINS_H], 
        range=[[0, grid_w], [0, grid_h]]
    )

    # Transpose because histogram2d returns H[x, y] but imshow expects H[row, col]
    heatmap = heatmap.T

    # Apply Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=1.0)

    # -------------------------
    # Plot
    # -------------------------
    if exp_dir:
        figs_dir = exp_dir / "figs"
    else:
        figs_dir = Path("figs")
        
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.title("Validation Path Coverage Density (100x100 Bins)")
    
    im = plt.imshow(heatmap, cmap='inferno', origin='upper', extent=[0, grid_w, 0, grid_h])
    plt.colorbar(im, label='Density (Smoothed Hits)')
    
    plt.xlabel("X (Patcher Column)")
    plt.ylabel("Y (Patcher Row)")
    plt.gca().invert_yaxis() # Top-left origin
    
    save_path = figs_dir / "val_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Coverage heatmap saved to {save_path}")

if __name__ == "__main__":
    create_val_heatmap()
