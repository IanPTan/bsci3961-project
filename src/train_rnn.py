import argparse
import os
import tomllib
from pathlib import Path
import h5py
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from pytorch_optimizer import create_optimizer

from rnn import RNN
from dataset import VAEPathDataset

def load_defaults(path="experiments/rnn_defaults.toml"):
    if not os.path.exists(path):
        print(f"Error: Default config not found at {path}")
        exit(1)
    with open(path, "rb") as f:
        return tomllib.load(f)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def write_toml(config, path):
    with open(path, "w") as f:
        for k, v in config.items():
            if isinstance(v, str):
                f.write(f'{k} = "{v}"\n')
            elif isinstance(v, list):
                f.write(f'{k} = {v}\n')
            elif isinstance(v, bool):
                f.write(f'{k} = {"true" if v else "false"}\n')
            else:
                f.write(f'{k} = {v}\n')

def get_exp_dir(args_exp_dir):
    exp_root = Path("experiments")
    exp_root.mkdir(exist_ok=True)
    
    if args_exp_dir:
        exp_dir = Path(args_exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
        
    # Find highest rnn_#
    max_num = 0
    for d in exp_root.iterdir():
        if d.is_dir() and d.name.startswith("rnn_"):
            try:
                num = int(d.name.split("_")[1])
                max_num = max(max_num, num)
            except ValueError:
                pass
                
    if max_num == 0:
        new_dir = exp_root / "rnn_1"
    else:
        new_dir = exp_root / f"rnn_{max_num}"
        
    new_dir.mkdir(exist_ok=True)
    return new_dir

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training...", unit="batch")

    for moves, in_patches, out_patches, _, _ in pbar:
        moves = moves.to(device)
        in_patches = in_patches.to(device)
        out_patches = out_patches.to(device)

        # CONCATENATE PATCH FEATURES + MOVE FEATURES
        x = torch.cat([in_patches, moves], dim=-1)

        optimizer.zero_grad()

        y_pred, h_seq, _ = model(x)
        loss = criterion(y_pred, out_patches)

        loss.backward()
        optimizer.step()
        
        # Step the scheduler (per-batch)
        if scheduler is not None:
            scheduler.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        pbar.set_postfix(mse=f"{batch_loss:.6f}")

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for moves, in_patches, out_patches, _, _ in tqdm(loader, desc="Evaluating...", unit="batch"):
        moves = moves.to(device)
        in_patches = in_patches.to(device)
        out_patches = out_patches.to(device)

        x = torch.cat([in_patches, moves], dim=-1)

        y_pred, h_seq, _ = model(x)
        loss = criterion(y_pred, out_patches)

        total_loss += loss.item()

    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="Train RNN")
    parser.add_argument("--exp-dir", type=str, help="Path to experiment directory")
    args = parser.parse_args()
    
    exp_dir = get_exp_dir(args.exp_dir)
    print(f"Using experiment directory: {exp_dir}")
    
    defaults = load_defaults()
    config_path = exp_dir / "config.toml"
    config = defaults.copy()
    if config_path.exists():
        with open(config_path, "rb") as f:
            loaded_config = tomllib.load(f)
            config.update(loaded_config)
    else:
        write_toml(config, config_path)
        
    # Apply global seed
    seed_everything(config["rng_seed"])
    
    print("\n--- Configuration ---")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("----------------------\n")
    
    # Set up figs dir
    figs_dir = exp_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # LOAD DATASET
    train_dataset = VAEPathDataset(
        config["data_path"], 
        split="train", 
        mask_prob=config["mask_prob"], 
        mask_start_idx=config["mask_start_idx"]
    )
    test_dataset = VAEPathDataset(
        config["data_path"], 
        split="test", 
        mask_prob=config["mask_prob"], 
        mask_start_idx=config["mask_start_idx"]
    )
    
    # INFER DIMENSIONS FROM ONE SAMPLE
    sample_moves, sample_in_patches, sample_out_patches, _, _ = train_dataset[0]
    
    seq_len = sample_in_patches.shape[0]
    patch_dim = sample_in_patches.shape[-1]
    move_dim = sample_moves.shape[-1]
    input_dim = patch_dim + move_dim
    output_dim = sample_out_patches.shape[-1]
    
    assert patch_dim == config["vae_embedding_size"], f"Dataset patch dim ({patch_dim}) does not match vae_embedding_size ({config['vae_embedding_size']})"
    
    print(f"Sequence length: {seq_len}")
    print(f"Patch dim:       {patch_dim}")
    print(f"Move dim:        {move_dim}")
    print(f"Input dim:       {input_dim}")
    print(f"Output dim:      {output_dim}")
    
    # Use fixed generator for dataloader shuffling reproducibility
    gen = torch.Generator().manual_seed(config["rng_seed"])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        generator=gen
    )
    val_loader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(test_dataset)}")
    
    # MODEL
    model = RNN(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        output_dim=output_dim,
        hidden_loops=config["hidden_loops"],
        k=config["k"],
    ).to(device)
    
    # OPTIMIZER
    optimizer = create_optimizer(
        model,
        optimizer_name='adamw',
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config["adamw_betas"]),
        use_orthograd=config["use_orthograd"],
        eps=config["orthograd_eps"]
    )
    
    # SCHEDULER
    total_batches = len(train_loader)
    total_steps = config["num_epochs"] * total_batches
    warmup_steps = int(total_steps * config["warmup_proportion"])
    
    print(f"Total steps:  {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    warmup_sch = LinearLR(optimizer, start_factor=0.0001, end_factor=1.0, total_iters=warmup_steps)
    main_sch = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sch, main_sch],
        milestones=[warmup_steps]
    )
    
    criterion = nn.MSELoss()
    
    start_epoch = 0
    best_val_loss = float("inf")
    
    # Resume from checkpoint if it exists
    checkpoint_path = exp_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch+1} (best val loss: {best_val_loss:.6f}).")
    
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
    
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(current_lr)
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )
        
        # Stop training if NaN occurs
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            print("!!! NaN/Inf loss detected. Stopping training to prevent weight corruption.")
            break
            
        # Save to H5
        with h5py.File(loss_h5_path, "w") as f:
            f.create_dataset("train_losses", data=np.array(train_losses))
            f.create_dataset("val_losses", data=np.array(val_losses))
            f.create_dataset("lrs", data=np.array(lrs))
            
        # Checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if np.isfinite(val_loss) and (best_val_loss == float('inf') or val_loss < best_val_loss):
            best_val_loss = val_loss
            rnn_pt_name = f"{exp_dir.name}.pt"
            best_model_path = exp_dir / rnn_pt_name
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Validation loss improved. Saved {rnn_pt_name}")
            
    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Generate final plots
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        
        ax2.plot(lrs, color='orange')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(figs_dir / "rnn_loss.png")
        plt.close(fig)
        print(f"Saved loss plots to {figs_dir / 'rnn_loss.png'}")
    except Exception as e:
        print(f"Could not generate loss plot: {e}")

if __name__ == "__main__":
    main()
