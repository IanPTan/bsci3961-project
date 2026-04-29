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
    return patcher.shape # (n_h, n_w)

def probe_grid_cells():
    parser = argparse.ArgumentParser(description="Probe RNN hidden states for spatial information")
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
    pos_weights_path = exp_dir / "pos_weights.pt"
    
    # These should ideally be in config, adding defaults if not present
    image_path = config.get("image_path", "data/frieren.png")
    patch_size = config.get("patch_size", 64)
    patch_stride = config.get("patch_stride", 1)
    hidden_dim = config["hidden_dim"]

    if not pos_weights_path.exists():
        print(f"Error: {pos_weights_path} not found. Run learn_pos.py first.")
        return
    if not val_h5_path.exists():
        print(f"Error: {val_h5_path} not found. Run eval_rnn.py first.")
        return

    grid_h, grid_w = get_grid_dims(image_path, patch_size, patch_stride)
    print(f"Grid dimensions: {grid_w}x{grid_h}")

    # Analysis Params
    K_LOOPS = 4    # Number of best-performing loops to analyze
    K_NEURONS = 30 # Number of top neurons to visualize per loop (after filtering)
    COVERAGE_THRESHOLD = 0.3 # Activation threshold to count as 'occupied' space

    # Score Gains
    SALIENCY_GAIN = 1
    COVERAGE_GAIN = 2
    SKEW_GAIN = 1

    print(f"Loading position model weights and stats from {pos_weights_path}...")
    checkpoint = torch.load(pos_weights_path, map_location="cpu")
    mses = checkpoint["mses"]
    state_dicts = checkpoint["state_dicts"]
    
    sorted_loops = sorted(mses.items(), key=lambda x: x[1])
    best_loops = [k.replace("_mse", "") for k, v in sorted_loops[:K_LOOPS]]
    
    print(f"Top {K_LOOPS} Loops by MSE:")
    for loop_id in best_loops:
        print(f"  {loop_id}: MSE = {mses[loop_id + '_mse']:.6f}")

    figs_dir = exp_dir / "figs" / "probes"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    CENTER = torch.tensor([grid_h / 2.0, grid_w / 2.0])
    MAX_DIST = torch.norm(CENTER) 

    with h5py.File(val_h5_path, "r") as f_val, h5py.File(paths_h5_path, "r") as f_paths:
        coords_np = f_paths["val"]["coords"][:, 1:].reshape(-1, 2)
        coords = torch.from_numpy(coords_np).to(torch.float32)
        
        for loop_name in best_loops:
            loop_idx = int(loop_name.split("_")[1])
            print(f"\nAnalyzing Loop {loop_idx}...")
            
            weights = state_dicts[loop_name]["net.0.weight"] 
            saliency = torch.sum(torch.abs(weights), dim=0) 

            print(f"  Calculating spatial metrics for all {hidden_dim} neurons...")
            H = torch.from_numpy(f_val["h_all"][:, :, loop_idx, :]).view(-1, hidden_dim)
            
            H_min = H.min(dim=0).values
            H_max = H.max(dim=0).values
            H_norm = (H - H_min) / (H_max - H_min + 1e-8)

            coverage = (H_norm > COVERAGE_THRESHOLD).float().mean(dim=0)
            sum_act = H_norm.sum(dim=0) + 1e-8
            centroid = (H_norm.T @ coords) / sum_act.unsqueeze(-1) 

            skew_dist = torch.norm(centroid - CENTER, dim=1)
            skew_norm = (skew_dist / MAX_DIST).clamp(0, 1)

            global_score = (saliency ** SALIENCY_GAIN) * \
                           (coverage ** COVERAGE_GAIN) * \
                           ((1.0 - skew_norm) ** SKEW_GAIN)
            
            top_indices = torch.topk(global_score, K_NEURONS).indices.tolist()
            
            print(f"  Top {K_NEURONS} neurons by Global Score:")
            for idx in top_indices:
                print(f"    Neuron {idx:4d}: Score={global_score[idx]:.2f}, Sal={saliency[idx]:.2f}, Cov={coverage[idx]:.2f}, Skew={skew_norm[idx]:.2f}")

            for neuron_idx in top_indices:
                print(f"    Generating heatmap for Neuron {neuron_idx}...")
                h_neuron = H[:, neuron_idx].numpy()
                
                grid_y, grid_x = np.mgrid[0:grid_h, 0:grid_w]
                heatmap = griddata(coords_np, h_neuron, (grid_y, grid_x), method='linear')
                
                if np.isnan(heatmap).any():
                    heatmap_nearest = griddata(coords_np, h_neuron, (grid_y, grid_x), method='nearest')
                    heatmap[np.isnan(heatmap)] = heatmap_nearest[np.isnan(heatmap)]

                plt.figure(figsize=(8, 6))
                plt.title(f"Loop {loop_idx} | Neuron {neuron_idx} | Score: {global_score[neuron_idx]:.2f}\n"
                          f"Sal: {saliency[neuron_idx]:.2f} | Cov: {coverage[neuron_idx]:.2f} | Skew: {skew_norm[neuron_idx]:.2f}")
                
                im = plt.imshow(heatmap, cmap='magma', origin='upper')
                plt.colorbar(im, label='Activation')
                
                plt.scatter(centroid[neuron_idx, 1], centroid[neuron_idx, 0], color='red', marker='x', s=100, label='Centroid')
                
                plt.xlabel("X (Patcher Column)")
                plt.ylabel("Y (Patcher Row)")
                plt.legend()
                
                save_path = figs_dir / f"loop{loop_idx}_neuron{neuron_idx}.png"
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()

    print(f"\nProbing complete. Heatmaps saved to {figs_dir}")

if __name__ == "__main__":
    probe_grid_cells()
