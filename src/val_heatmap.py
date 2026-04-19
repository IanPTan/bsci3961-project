import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import VAEPathDataset


# -------------------------
# Config
# -------------------------
PATHS_H5 = "vae_paths.h5"
VAL_OUTPUT_H5 = "2048_9_val.h5"
SPLIT = "val"

# If coords for each sample are a full path with shape (T, 2):
#   False -> use only the first coordinate of the path for that sample's MSE
#   True  -> write that same sample MSE onto every coordinate visited in the path
USE_ALL_PATH_COORDS = False

# Output files
HEATMAP_PNG = "2048_9_val_mse_heatmap.png"
COUNT_PNG = "2048_9_val_count_heatmap.png"
OUTPUT_H5 = "2048_9_val_heatmap_data.h5"


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_coords(coords):
    """
    Normalize coords into one of:
      (2,)     -> single (row, col)
      (T, 2)   -> path of coordinates

    Accepts tensors, lists, tuples, numpy arrays.
    """
    coords = to_numpy(coords)
    coords = np.asarray(coords)

    if coords.ndim == 1:
        if coords.shape[0] != 2:
            raise ValueError(f"Expected coords shape (2,), got {coords.shape}")
        return coords

    if coords.ndim == 2:
        if coords.shape[-1] != 2:
            raise ValueError(f"Expected coords shape (T, 2), got {coords.shape}")
        return coords

    raise ValueError(f"Unsupported coords shape: {coords.shape}")


# -------------------------
# Load dataset and saved predictions
# -------------------------
val_dataset = VAEPathDataset(PATHS_H5, split=SPLIT)
num_samples = len(val_dataset)
print(f"Validation samples: {num_samples}")

with h5py.File(VAL_OUTPUT_H5, "r") as pred_f:
    if "y" not in pred_f:
        raise KeyError(f"Could not find dataset 'y' in {VAL_OUTPUT_H5}")

    y_ds = pred_f["y"]

    if len(y_ds) != num_samples:
        raise ValueError(
            f"Mismatch between saved predictions and dataset length: "
            f"len(y)={len(y_ds)} vs len(val_dataset)={num_samples}"
        )

    sample_mse = np.zeros(num_samples, dtype=np.float32)
    rep_rows = np.zeros(num_samples, dtype=np.int64)
    rep_cols = np.zeros(num_samples, dtype=np.int64)

    coord_sets = []
    max_row = -1
    max_col = -1

    for i in tqdm(range(num_samples), desc="Computing per-sample MSE"):
        _, _, out_patches, coords = val_dataset[i]

        y_pred = to_numpy(y_ds[i]).astype(np.float32)
        y_true = to_numpy(out_patches).astype(np.float32)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch at sample {i}: y_pred {y_pred.shape} vs out_patches {y_true.shape}"
            )

        mse = np.mean((y_pred - y_true) ** 2, dtype=np.float64)
        sample_mse[i] = mse

        coords = normalize_coords(coords)

        if coords.ndim == 1:
            rc = coords.astype(np.int64)
            coord_sets.append(rc[None, :])
            rep_rows[i] = rc[0]
            rep_cols[i] = rc[1]
            max_row = max(max_row, int(rc[0]))
            max_col = max(max_col, int(rc[1]))
        else:
            rc = coords.astype(np.int64)
            coord_sets.append(rc)
            rep_rows[i] = rc[0, 0]
            rep_cols[i] = rc[0, 1]
            max_row = max(max_row, int(rc[:, 0].max()))
            max_col = max(max_col, int(rc[:, 1].max()))

if max_row < 0 or max_col < 0:
    raise ValueError("No valid coordinates found.")

print(f"Coordinate grid size: ({max_row + 1}, {max_col + 1})")

# -------------------------
# Build heatmap
# -------------------------
heat_sum = np.zeros((max_row + 1, max_col + 1), dtype=np.float64)
heat_count = np.zeros((max_row + 1, max_col + 1), dtype=np.int32)

for mse, coords in zip(sample_mse, coord_sets):
    if USE_ALL_PATH_COORDS:
        coords_to_use = coords
    else:
        coords_to_use = coords[:1]

    for r, c in coords_to_use:
        if r < 0 or c < 0:
            continue
        heat_sum[r, c] += float(mse)
        heat_count[r, c] += 1

heatmap = np.full_like(heat_sum, np.nan, dtype=np.float64)
mask = heat_count > 0
heatmap[mask] = heat_sum[mask] / heat_count[mask]

visited = int(mask.sum())
print(f"Visited grid cells: {visited}")
print(f"Mean sample MSE: {sample_mse.mean():.6f}")
print(f"Min sample MSE:  {sample_mse.min():.6f}")
print(f"Max sample MSE:  {sample_mse.max():.6f}")

# -------------------------
# Save arrays for later use
# -------------------------
with h5py.File(OUTPUT_H5, "w") as out_f:
    out_f.create_dataset("sample_mse", data=sample_mse)
    out_f.create_dataset("rep_rows", data=rep_rows)
    out_f.create_dataset("rep_cols", data=rep_cols)
    out_f.create_dataset("heat_sum", data=heat_sum)
    out_f.create_dataset("heat_count", data=heat_count)
    out_f.create_dataset("heatmap", data=heatmap)
    out_f.attrs["paths_h5"] = PATHS_H5
    out_f.attrs["val_output_h5"] = VAL_OUTPUT_H5
    out_f.attrs["split"] = SPLIT
    out_f.attrs["use_all_path_coords"] = int(USE_ALL_PATH_COORDS)

print(f"Saved heatmap arrays to {OUTPUT_H5}")

# -------------------------
# Plot MSE heatmap
# -------------------------
plt.figure(figsize=(9, 7))
plt.imshow(heatmap, origin="upper")
plt.colorbar(label="Mean validation MSE")
plt.title("Validation MSE Heatmap")
plt.xlabel("col")
plt.ylabel("row")
plt.tight_layout()
plt.savefig(HEATMAP_PNG, dpi=200)
print(f"Saved heatmap image to {HEATMAP_PNG}")

# -------------------------
# Plot coverage / count heatmap
# -------------------------
plt.figure(figsize=(9, 7))
plt.imshow(heat_count, origin="upper")
plt.colorbar(label="Samples per coordinate")
plt.title("Validation Coordinate Coverage")
plt.xlabel("col")
plt.ylabel("row")
plt.tight_layout()
plt.savefig(COUNT_PNG, dpi=200)
print(f"Saved count image to {COUNT_PNG}")

plt.show()
