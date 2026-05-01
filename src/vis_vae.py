import argparse
import tomllib
from pathlib import Path
import h5py
import torch as pt
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from vae import VAE
from dataset import PatchDataset

def get_latest_exp_dir(prefix="vae_"):
    exp_root = Path("experiments")
    if not exp_root.exists(): return None
    max_num = 0
    latest_dir = None
    for d in exp_root.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            try:
                num = int(d.name.split("_")[1])
                if num > max_num:
                    max_num = num
                    latest_dir = d
            except ValueError: pass
    return latest_dir

def main():
    parser = argparse.ArgumentParser(description="Visualize VAE results")
    parser.add_argument("--exp-dir", type=str, help="Path to VAE experiment directory")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Indices for comparison plots")
    parser.add_argument("--no-loss", action="store_true", help="Skip loss plotting")
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir) if args.exp_dir else get_latest_exp_dir("vae_")
    if not exp_dir:
        print("Error: No VAE experiment found.")
        return
        
    print(f"Using experiment directory: {exp_dir}")
    
    config_path = exp_dir / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
        
    # Merge config vis indices if args.indices is default
    vis_config = config.get("vis", {})
    if args.indices == [0, 1, 2, 3, 4] and "indices" in vis_config:
        args.indices = vis_config["indices"]
    
    figs_dir = exp_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    
    # 1. Plot Losses
    if not args.no_loss:
        loss_path = exp_dir / "loss.h5"
        if loss_path.exists():
            with h5py.File(loss_path, "r") as f:
                train_losses = f["train_losses"][:]
                val_losses = f["val_losses"][:]
                lrs = f["lrs"][:]
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
            ax1.plot(train_losses, label='Train Loss')
            ax1.plot(val_losses, label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            ax2.plot(lrs, color='orange')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Learning Rate')
            
            plt.tight_layout()
            plt.savefig(figs_dir / "vae_loss.png")
            plt.close()
            print(f"Saved loss plot to {figs_dir / 'vae_loss.png'}")

    # 2. Reconstructions
    device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
    vae_model = VAE(
        enc_channels=config["enc_channels"],
        enc_inner_channels=config["enc_inner_channels"],
        enc_scales=config["enc_scales"],
        enc_linears=config["enc_linears"],
        dec_channels=config["dec_channels"],
        dec_inner_channels=config["dec_inner_channels"],
        dec_scales=config["dec_scales"],
        dec_linears=config["dec_linears"],
        latent_dim=config["latent_dim"],
        image_size=config["patch_size"]
    ).to(device)
    
    weights_path = exp_dir / f"{exp_dir.name}.pt"
    if not weights_path.exists(): weights_path = exp_dir / "checkpoint.pt"
    
    checkpoint = pt.load(weights_path, map_location=device)
    vae_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    vae_model.eval()

    IMG = transforms.ToTensor()(transforms.Resize(config["image_size"])(Image.open(config["image_path"]).convert("RGB")))
    dataset = PatchDataset(IMG, config["patch_size"], config["patch_stride"])

    for idx in args.indices:
        if idx >= len(dataset): continue
        img, _ = dataset[idx]
        img = img.unsqueeze(0).to(device)
        
        recon = vae_model.reconstruct(img)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img[0].detach().cpu().permute(1, 2, 0).clamp(0, 1))
        axes[0].set_title(f"Original (Idx {idx})")
        axes[0].axis('off')
        
        axes[1].imshow(recon[0].detach().cpu().permute(1, 2, 0).clamp(0, 1))
        axes[1].set_title("Reconstruction")
        axes[1].axis('off')
        
        plt.savefig(figs_dir / f"vae_comp_{idx}.png")
        plt.close()
    print(f"Saved {len(args.indices)} comparison plots to {figs_dir}")

if __name__ == "__main__":
    main()
