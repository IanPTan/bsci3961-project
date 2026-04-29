import os
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
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
    return n_h, n_w

def create_heatmap():
    parser = argparse.ArgumentParser(description="Generate MSE heatmap for RNN predictions")
    parser.add_argument("--exp-dir", type=str, help="Path to experiment directory")
    args = parser.parse_args()
    
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = get_latest_exp_dir("rnn_")
        if not exp_dir:
            print("Error: No RNN experiment found in experiments/")
            return
            
    print(f"Using experiment directory: {exp_dir}")
    
    config_path = exp_dir / "config.toml"
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
        
    val_h5_path = exp_dir / "val.h5"
    paths_h5_path = Path(config["data_path"])
    image_path = config.get("image_path", "data/frieren.png")
    patch_size = config.get("patch_size", 64)
    patch_stride = config.get("patch_stride", 1)
    hidden_dim = config["hidden_dim"]
    hidden_loops = config["hidden_loops"]

    if not val_h5_path.exists():
        print(f"Error: {val_h5_path} not found. Run eval_rnn.py first.")
        return

    grid_h, grid_w = get_grid_dims(image_path, patch_size, patch_stride)

    print(f"Loading data from {val_h5_path} and {paths_h5_path}...")
    
    with h5py.File(val_h5_path, "r") as f_val, h5py.File(paths_h5_path, "r") as f_paths:
        # y: (num_samples, seq_len, output_dim)
        y_pred = f_val["y"][:]
        
        # Ground truth patches from the 'val' group in vae_paths.h5
        if "val" not in f_paths:
            print(f"Error: 'val' group not found in {paths_h5_path}.")
            return
            
        gt_patches = f_paths["val"]["patches"][:, 1:] 
        coords = f_paths["val"]["coords"][:, 1:]

        num_samples, seq_len, _ = y_pred.shape

        print(f"Processing {num_samples} samples...")
        
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
    grid_y, grid_x = np.mgrid[0:grid_h, 0:grid_w]
    points = coords_flat
    values = mses

    heatmap = griddata(points, values, (grid_y, grid_x), method='linear')
    
    if np.isnan(heatmap).any():
        heatmap_nearest = griddata(points, values, (grid_y, grid_x), method='nearest')
        heatmap[np.isnan(heatmap)] = heatmap_nearest[np.isnan(heatmap)]

    # -------------------------
    # Plot
    # -------------------------
    figs_dir = exp_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.title(f"RNN Prediction MSE Heatmap ({exp_dir.name})")
    
    im = plt.imshow(heatmap, cmap='inferno', origin='upper')
    plt.colorbar(im, label='MSE Loss')
    
    stats_text = (f"Min MSE: {min_mse:.6f}\n"
                  f"Max MSE: {max_mse:.6f}\n"
                  f"Mean MSE: {mean_mse:.6f}\n"
                  f"Std MSE: {std_mse:.6f}")
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel("X (Patcher Column)")
    plt.ylabel("Y (Patcher Row)")
    
    save_path = figs_dir / "mse_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    create_heatmap()
