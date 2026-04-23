import argparse
import os
import tomllib
from pathlib import Path
import h5py
import numpy as np

import torch as pt
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from vae import VAE
from dataset import PatchDataset

DEFAULT_CONFIG = {
    "image_path": "data/frieren.png",
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "batch_size": 16,
    "conv_channels": [32, 64, 128, 256, 512, 512, 1024, 1024],
    "linear_features": [512, 128, 64],
    "better_by": 1.0,
    "patch_size": 64,
    "patch_stride": 1,
    "val_ratio": 0.2
}

def write_toml(config, path):
    with open(path, "w") as f:
        for k, v in config.items():
            if isinstance(v, str):
                f.write(f'{k} = "{v}"\n')
            elif isinstance(v, list):
                f.write(f'{k} = {v}\n')
            else:
                f.write(f'{k} = {v}\n')

def get_exp_dir(args_exp_dir):
    exp_root = Path("experiments")
    exp_root.mkdir(exist_ok=True)
    
    if args_exp_dir:
        exp_dir = Path(args_exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
        
    # Find highest vae_#
    max_num = 0
    for d in exp_root.iterdir():
        if d.is_dir() and d.name.startswith("vae_"):
            try:
                num = int(d.name.split("_")[1])
                max_num = max(max_num, num)
            except ValueError:
                pass
                
    if max_num == 0:
        new_dir = exp_root / "vae_1"
    else:
        new_dir = exp_root / f"vae_{max_num}"
        
    new_dir.mkdir(exist_ok=True)
    return new_dir

def main():
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--exp-dir", type=str, help="Path to experiment directory")
    args = parser.parse_args()
    
    exp_dir = get_exp_dir(args.exp_dir)
    print(f"Using experiment directory: {exp_dir}")
    
    config_path = exp_dir / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
            # Merge with defaults
            for k, v in DEFAULT_CONFIG.items():
                if k not in config:
                    config[k] = v
    else:
        config = DEFAULT_CONFIG.copy()
        write_toml(config, config_path)
        
    # Set up figs dir
    figs_dir = exp_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    
    # Load Image
    image_path = config["image_path"]
    print(f"Using image path: {image_path}")
    IMG = transforms.ToTensor()(transforms.Resize(1024)(Image.open(image_path)))
    
    device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vae_model = VAE(conv_channels=config["conv_channels"], linear_features=config["linear_features"]).to(device)
    optimizer = pt.optim.AdamW(vae_model.parameters(), lr=config["learning_rate"])
    scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    dataset = PatchDataset(IMG, config["patch_size"], config["patch_stride"])
    
    val_ratio = config["val_ratio"]
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use fixed generator for reproducibility of splits across resuming
    gen = pt.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    start_epoch = 0
    best_val = float('inf')
    
    # Resume from checkpoint if it exists
    checkpoint_path = exp_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        checkpoint = pt.load(checkpoint_path, map_location=device, weights_only=False)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint.get('best_val_loss', float('inf'))
        print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch+1} (best val loss: {best_val:.4f}).")
    
    train_losses = []
    val_losses = []
    lrs = []
    
    # Load existing losses from h5 if we are resuming
    loss_h5_path = exp_dir / "loss.h5"
    if loss_h5_path.exists():
        with h5py.File(loss_h5_path, "r") as f:
            if "train_losses" in f:
                train_losses = list(f["train_losses"][:])
            if "val_losses" in f:
                val_losses = list(f["val_losses"][:])
            if "lrs" in f:
                lrs = list(f["lrs"][:])
                
    num_epochs = config["num_epochs"]
    better_by = config["better_by"]
    
    for epoch in range(start_epoch, num_epochs):
        # --- Training Phase ---
        vae_model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, _) in enumerate(train_pbar):
            images = images.to(device)
            
            recon_images, mu, logvar = vae_model(images)
            loss = vae_model.vae_loss(recon_images, images, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix(L=f"{loss.item() / len(images):.4f}")
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        
        # --- Validation Phase ---
        vae_model.eval()
        total_val_loss = 0
        
        with pt.no_grad():
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                recon_images, mu, logvar = vae_model(images)
                loss = vae_model.vae_loss(recon_images, images, mu, logvar)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        
        lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        lrs.append(lr)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {lr}')
        
        # Save to H5
        with h5py.File(loss_h5_path, "w") as f:
            f.create_dataset("train_losses", data=np.array(train_losses))
            f.create_dataset("val_losses", data=np.array(val_losses))
            f.create_dataset("lrs", data=np.array(lrs))
            
        # Checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': vae_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val
        }
        pt.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if best_val == float('inf') or (best_val - avg_val_loss > better_by):
            best_val = avg_val_loss
            # Save vae_#.pt
            vae_pt_name = f"{exp_dir.name}.pt"
            best_model_path = exp_dir / vae_pt_name
            pt.save(vae_model.state_dict(), best_model_path)
            print(f"--> Validation loss improved. Saved {vae_pt_name}")

    print("VAE training complete.")
    
    # Generate final plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(lrs, color='orange')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(figs_dir / "vae_loss.png")
    plt.close(fig)
    
    # Generate comparison images
    vae_model.eval()
    # Get a batch from val_loader
    val_iter = iter(val_loader)
    images, _ = next(val_iter)
    images = images.to(device)
    with pt.no_grad():
        recon_images, _, _ = vae_model(images)
        
    for i in range(min(len(images), 5)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(images[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        axes[1].imshow(recon_images[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
        axes[1].set_title("Reconstruction")
        axes[1].axis('off')
        
        plt.savefig(figs_dir / f"vae_comp_{i}.png")
        plt.close(fig)

if __name__ == "__main__":
    main()
