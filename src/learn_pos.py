import os
import h5py
import torch
import numpy as np

# -------------------------
# Config
# -------------------------
HIDDEN_DIM = 2048
HIDDEN_LOOPS = 9 # Updated for 10 loop data

VAL_H5 = f"{HIDDEN_DIM}_{HIDDEN_LOOPS}_val.h5"
PATHS_H5 = "vae_paths.h5"
WEIGHTS_OUT = f"{HIDDEN_DIM}_{HIDDEN_LOOPS}_pos_weights.h5"

if not os.path.exists(VAL_H5):
    print(f"Error: {VAL_H5} not found. Run eval_rnn.py first.")
    exit(1)

print(f"Opening {VAL_H5} and {PATHS_H5}...")
with h5py.File(VAL_H5, "r") as f_val, h5py.File(PATHS_H5, "r") as f_paths:
    # Metadata
    num_samples, seq_len, num_loops, hidden_dim = f_val["h_all"].shape
    
    # Load coordinates once (small compared to activations)
    if "val" not in f_paths:
        print("Error: 'val' group not found in vae_paths.h5.")
        exit(1)
    
    coords = torch.from_numpy(f_paths["val"]["coords"][:, 1:]).to(torch.float32)
    coords_flat = coords.reshape(-1, 2)
    N = coords_flat.shape[0]

    print(f"Detected {num_loops} loops, {num_samples} samples. Fitting on CPU...")

    results = {}

    for l in range(num_loops):
        # Load ONLY this loop's activations into CPU memory
        H = torch.from_numpy(f_val["h_all"][:, :, l, :]).view(-1, hidden_dim)
        
        # Check for NaNs
        nan_count = torch.isnan(H).sum().item()
        if nan_count > 0:
            print(f"Loop {l:2d}: Found {nan_count} NaNs. Skipping.")
            results[f"loop_{l}_weights"] = np.zeros((hidden_dim + 1, 2))
            results[f"loop_{l}_mse"] = float('nan')
            continue

        # Add bias
        ones = torch.ones(N, 1)
        H_aug = torch.cat([H, ones], dim=1)
        
        # Solve W = (H_aug^T H_aug)^-1 H_aug^T C
        # Note: CPU linalg.lstsq is very stable
        sol = torch.linalg.lstsq(H_aug, coords_flat).solution
        
        coords_pred = H_aug @ sol
        mse = torch.nn.functional.mse_loss(coords_pred, coords_flat).item()
        
        print(f"Loop {l:2d}: MSE = {mse:10.6f}")
        
        results[f"loop_{l}_weights"] = sol.numpy()
        results[f"loop_{l}_mse"] = mse

# Save weights and stats
print(f"Saving linear transforms to {WEIGHTS_OUT}...")
with h5py.File(WEIGHTS_OUT, "w") as f:
    for k, v in results.items():
        f.create_dataset(k, data=v)
    f.attrs["num_samples"] = N
    f.attrs["hidden_dim"] = hidden_dim
    f.attrs["num_loops"] = num_loops

print("Done. Used loop-by-loop CPU fitting to preserve memory.")
