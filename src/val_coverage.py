from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from dataset import VAEPathDataset

PATHS_H5 = "vae_paths.h5"

dataset = VAEPathDataset(PATHS_H5, split="val")

coords_ds = dataset.group["coords"]

all_coords = []

print("Collecting coordinates...")

for i in tqdm(range(len(coords_ds))):
    coords = torch.tensor(coords_ds[i])   # (T+1, 2)
    all_coords.append(coords)

# -------------------------
# Concatenate
# -------------------------
all_coords = torch.cat(all_coords, dim=0)   # (N_total, 2)

rows = all_coords[:, 0]
cols = all_coords[:, 1]

print("Total points:", all_coords.shape[0])

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(6, 6))
plt.scatter(cols, rows, s=1, alpha=0.3)

plt.title("Validation Coverage (All Coordinates)")
plt.xlabel("Width (col)")
plt.ylabel("Height (row)")
plt.gca().invert_yaxis()   # match image coordinate system

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# X / column distribution
axes[0].hist(cols, bins=100)
axes[0].set_title("Column (X) Distribution")
axes[0].set_xlabel("X (col)")
axes[0].set_ylabel("Count")

# Y / row distribution
axes[1].hist(rows, bins=100)
axes[1].set_title("Row (Y) Distribution")
axes[1].set_xlabel("Y (row)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()