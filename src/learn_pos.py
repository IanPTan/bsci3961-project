import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import argparse
import tomllib
from pathlib import Path

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

class PositionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=1024, hidden_dim2=512, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser(description="Learn position from hidden states")
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
    if not val_h5_path.exists():
        print(f"Error: {val_h5_path} not found. Run eval_rnn.py first.")
        return
        
    paths_h5_path = Path(config["data_path"])
    weights_out_path = exp_dir / "pos_weights.pt"
    
    # Training Hyperparameters
    BATCH_SIZE = 1024
    LR = 1e-4
    EPOCHS = 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")
    print(f"Opening {val_h5_path} and {paths_h5_path}...")

    with h5py.File(val_h5_path, "r") as f_val, h5py.File(paths_h5_path, "r") as f_paths:
        # Metadata
        num_samples, seq_len, num_loops, hidden_dim = f_val["h_all"].shape
        
        # Load coordinates once (small compared to activations)
        if "val" not in f_paths:
            print(f"Error: 'val' group not found in {paths_h5_path}.")
            return
        
        # eval_rnn.py saves out_patches which are patches[1:], so we skip the first coord
        coords = torch.from_numpy(f_paths["val"]["coords"][:, 1:]).to(torch.float32)
        coords_flat = coords.reshape(-1, 2)
        N = coords_flat.shape[0]

        print(f"Detected {num_loops} loops, {num_samples} samples.")

        all_models = {}
        results = {}

        for l in range(num_loops):
            print(f"\nLoop {l:2d}: Training MLP...")
            # Load ONLY this loop's activations into CPU memory first
            H = torch.from_numpy(f_val["h_all"][:, :, l, :]).view(-1, hidden_dim)
            
            # Check for NaNs
            nan_count = torch.isnan(H).sum().item()
            if nan_count > 0:
                print(f"Loop {l:2d}: Found {nan_count} NaNs. Skipping.")
                results[f"loop_{l}_mse"] = float('nan')
                continue

            # Prepare DataLoader
            dataset = TensorDataset(H, coords_flat)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Initialize model
            model = PositionMLP(input_dim=hidden_dim).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=LR)
            criterion = nn.MSELoss()

            # Training loop
            model.train()
            pbar = tqdm(range(EPOCHS), desc=f"Loop {l}")
            for epoch in pbar:
                epoch_loss = 0.0
                for batch_h, batch_c in loader:
                    batch_h, batch_c = batch_h.to(DEVICE), batch_c.to(DEVICE)
                    
                    optimizer.zero_grad()
                    pred_c = model(batch_h)
                    loss = criterion(pred_c, batch_c)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                pbar.set_postfix(mse=epoch_loss/len(loader))

            # Final Evaluation
            model.eval()
            with torch.no_grad():
                # Batch evaluation to avoid memory issues
                total_mse = 0.0
                for i in range(0, N, BATCH_SIZE):
                    batch_h = H[i : i + BATCH_SIZE].to(DEVICE)
                    batch_c = coords_flat[i : i + BATCH_SIZE].to(DEVICE)
                    pred_c = model(batch_h)
                    total_mse += criterion(pred_c, batch_c).item() * batch_h.size(0)
                mse = total_mse / N
            
            print(f"Loop {l:2d}: Final MSE = {mse:10.6f}")
            
            all_models[f"loop_{l}"] = model.state_dict()
            results[f"loop_{l}_mse"] = mse

    # Save models and stats
    print(f"Saving MLP models and stats to {weights_out_path}...")
    save_dict = {
        "state_dicts": all_models,
        "mses": results,
        "metadata": {
            "num_samples": N,
            "hidden_dim": hidden_dim,
            "num_loops": num_loops,
            "hidden_dim1": 1024,
            "hidden_dim2": 512,
            "epochs": EPOCHS,
            "lr": LR
        }
    }
    torch.save(save_dict, weights_out_path)

    print("Done.")

if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    main()
