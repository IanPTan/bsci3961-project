import os
import torch
import cv2
import h5py
import numpy as np
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import tomllib
from pathlib import Path

from rnn import RNN
from vae import VAE
from dataset import VAEPathDataset

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

def decode_latents(vae, z):
    original_shape = z.shape[:-1]
    latent_dim = z.shape[-1]
    z_flat = z.reshape(-1, latent_dim)
    x_recon = vae.generate(z_flat)
    x_recon = x_recon.reshape(*original_shape, *x_recon.shape[1:])
    return x_recon

def patch_to_numpy_bgr(patch, size=512):
    patch = patch.detach().cpu().numpy()
    if patch.ndim == 3:
        if patch.shape[0] == 1:
            patch = patch[0]
        else:
            patch = np.transpose(patch, (1, 2, 0))
    patch = (patch * 255).clip(0, 255).astype(np.uint8)
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    else:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)
    return patch

def main():
    parser = argparse.ArgumentParser(description="Visualize RNN predictions")
    parser.add_argument("--exp-dir", type=str, help="Path to RNN experiment directory")
    parser.add_argument("--vae-dir", type=str, help="Path to VAE experiment directory")
    parser.add_argument("--indices", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Sequence indices to visualize")
    parser.add_argument("--fps", type=int, default=2, help="Video frame rate")
    parser.add_argument("--size", type=int, default=512, help="Output patch size")
    args = parser.parse_args()
    
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = get_latest_exp_dir("rnn_")
        if not exp_dir:
            print("Error: No RNN experiment found.")
            return
            
    if args.vae_dir:
        vae_dir = Path(args.vae_dir)
    else:
        vae_dir = get_latest_exp_dir("vae_")
        if not vae_dir:
            vae_dir = Path("data")
            
    print(f"Using RNN: {exp_dir} | VAE: {vae_dir}")
    
    with open(exp_dir / "config.toml", "rb") as f:
        config = tomllib.load(f)
        
    vae_config_path = vae_dir / "config.toml"
    if vae_config_path.exists():
        with open(vae_config_path, "rb") as f:
            vae_config = tomllib.load(f)
    else:
        vae_config = {"conv_channels": [32, 64, 128, 256, 512, 512, 1024, 1024], "linear_features": [512, 128, 64]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_h5_path = exp_dir / "val.h5"
    paths_h5_path = Path(config["data_path"])
    
    if not val_h5_path.exists():
        print(f"Error: {val_h5_path} not found. Run eval_rnn.py first.")
        return

    dataset = VAEPathDataset(paths_h5_path, split="val", mask_prob=0, mask_start_idx=config["mask_start_idx"])

    with h5py.File(val_h5_path, "r") as f:
        pred_out_patches = torch.from_numpy(f["y"][args.indices]).to(device)
        masks_batch = torch.from_numpy(f["mask"][args.indices]).to(device) if "mask" in f else torch.ones((len(args.indices), pred_out_patches.shape[1])).to(device)

    out_patches = torch.stack([dataset[i][2] for i in args.indices]).to(device)

    vae = VAE(conv_channels=vae_config["conv_channels"], linear_features=vae_config["linear_features"]).to(device)
    vae_weights_path = vae_dir / f"{vae_dir.name}.pt"
    if not vae_weights_path.exists(): vae_weights_path = vae_dir / "vae.pt"
    if not vae_weights_path.exists(): vae_weights_path = vae_dir / "checkpoint.pt"
    
    sd = torch.load(vae_weights_path, map_location=device)
    vae.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
    vae.eval()

    true_imgs_batch = decode_latents(vae, out_patches)
    pred_imgs_batch = decode_latents(vae, pred_out_patches)

    all_mses_batch = torch.nn.functional.mse_loss(pred_out_patches, out_patches, reduction='none').mean(dim=-1).cpu().numpy()

    figs_dir = exp_dir / "figs" / "paths"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss Plots
    for i, seq_idx in enumerate(args.indices):
        plt.figure(figsize=(8, 4))
        plt.plot(all_mses_batch[i])
        plt.title(f'MSE Loss over Time - Sequence {seq_idx}')
        plt.savefig(figs_dir / f"mse_{seq_idx:03d}.png")
        plt.close()

    # Videos
    for i, seq_idx in enumerate(args.indices):
        true_latents, pred_latents = out_patches[i], pred_out_patches[i]
        true_imgs, pred_imgs = true_imgs_batch[i], pred_imgs_batch[i]
        masks = masks_batch[i]

        final_path = figs_dir / f"path_{seq_idx:03d}.mp4"
        temp_path = figs_dir / f"temp_{seq_idx:03d}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(temp_path), fourcc, args.fps, (args.size * 2, args.size))

        for t in range(true_imgs.shape[0]):
            tp = patch_to_numpy_bgr(true_imgs[t], size=args.size)
            pp = patch_to_numpy_bgr(pred_imgs[t], size=args.size)
            mse = torch.nn.functional.mse_loss(true_latents[t], pred_latents[t]).item()
            is_masked = masks[t].item() == 0

            cv2.putText(tp, "Ground Truth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(pp, "Prediction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(pp, f"MSE: {mse:.4f}", (args.size - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if is_masked: cv2.putText(pp, "MASKED", (args.size - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            writer.write(np.hstack([tp, pp]))
        writer.release()

        subprocess.run(["ffmpeg", "-y", "-i", str(temp_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(final_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_path)

    print(f"Visualization complete. Results in {figs_dir}")

if __name__ == "__main__":
    main()
