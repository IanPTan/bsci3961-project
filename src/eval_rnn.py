import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from rnn import RNN
from vae import VAE
from dataset import VAEPathDataset


# -------------------------
# Config
# -------------------------
PATHS_H5 = "vae_paths.h5"
RNN_WEIGHTS = "2048_3_rnn.pt"
VAE_WEIGHTS = "vae.pt"

# must match train_rnn.py
HIDDEN_DIM = 2048
HIDDEN_LOOPS = 7
K = 0

# indices of sequences to visualize
SEQ_IDCS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def decode_latents(vae, z):
    """
    z: (T, latent_dim) or (B, T, latent_dim)
    returns decoded patches
    """
    original_shape = z.shape[:-1]
    latent_dim = z.shape[-1]

    z_flat = z.reshape(-1, latent_dim)

    with torch.no_grad():
        x_recon = vae.decode(z_flat)

    x_recon = x_recon.reshape(*original_shape, *x_recon.shape[1:])
    return x_recon


def patch_to_imshow_format(patch):
    """
    Converts patch from torch shape to something matplotlib can show.
    Supports:
      (C, H, W) -> (H, W, C)
      (1, H, W) -> (H, W)
      (H, W) stays as is
    """
    patch = patch.detach().cpu()

    if patch.ndim == 3:
        if patch.shape[0] == 1:
            patch = patch.squeeze(0)          # (1, H, W) -> (H, W)
        else:
            patch = patch.permute(1, 2, 0)    # (C, H, W) -> (H, W, C)

    return patch


# -------------------------
# Load dataset
# -------------------------
dataset = VAEPathDataset(PATHS_H5)
subset = Subset(dataset, SEQ_IDCS)
loader = DataLoader(subset, batch_size=len(SEQ_IDCS), shuffle=False)

# Get all sequences in one batch for efficiency
moves, in_patches, out_patches = next(iter(loader))

print(f"Batch shapes for {len(SEQ_IDCS)} sequences:")
print("moves:", moves.shape)
print("in_patches:", in_patches.shape)
print("out_patches:", out_patches.shape)

patch_dim = in_patches.shape[-1]
move_dim = moves.shape[-1]
input_dim = patch_dim + move_dim
output_dim = out_patches.shape[-1]

moves = moves.to(device)             # (B, T, move_dim)
in_patches = in_patches.to(device)   # (B, T, patch_dim)
out_patches = out_patches.to(device) # (B, T, patch_dim)

x = torch.cat([in_patches, moves], dim=-1)        # (B, T, patch_dim + move_dim)

# -------------------------
# Load Models
# -------------------------
rnn = RNN(
    input_dim=input_dim,
    hidden_dim=HIDDEN_DIM,
    output_dim=output_dim,
    hidden_loops=HIDDEN_LOOPS,
    k=K,
).to(device)

rnn.load_state_dict(torch.load(RNN_WEIGHTS, map_location=device))
rnn.eval()

vae = VAE(
    conv_channels=[32, 64, 128, 256, 512, 1024],
    linear_features=[128, 64],
).to(device)

vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
vae.eval()

# -------------------------
# Run RNN
# -------------------------
with torch.no_grad():
    pred_out_patches, _ = rnn(x)

print("pred_out_patches batch shape:", pred_out_patches.shape)

# -------------------------
# Decode true and predicted latents (Batched)
# -------------------------
true_imgs_batch = decode_latents(vae, out_patches)       # (B, T, C, H, W)
pred_imgs_batch = decode_latents(vae, pred_out_patches)  # (B, T, C, H, W)

print("Decoded batch shapes:")
print("true_imgs_batch:", true_imgs_batch.shape)
print("pred_imgs_batch:", pred_imgs_batch.shape)

# -------------------------
# Plot and Save
# -------------------------
os.makedirs("figs/paths", exist_ok=True)

for i, seq_idx in enumerate(SEQ_IDCS):
    true_imgs = true_imgs_batch[i]
    pred_imgs = pred_imgs_batch[i]
    
    T = min(8, true_imgs.shape[0])
    fig, axes = plt.subplots(T, 2, figsize=(6, 2.5 * T))

    if T == 1:
        axes = axes[None, :]

    for t in range(T):
        true_patch = patch_to_imshow_format(true_imgs[t])
        pred_patch = patch_to_imshow_format(pred_imgs[t])

        axes[t, 0].imshow(true_patch, cmap="gray" if true_patch.ndim == 2 else None)
        axes[t, 0].set_title(f"True t={t}")
        axes[t, 0].axis("off")

        axes[t, 1].imshow(pred_patch, cmap="gray" if pred_patch.ndim == 2 else None)
        axes[t, 1].set_title(f"Pred t={t}")
        axes[t, 1].axis("off")

    plt.tight_layout()
    save_path = f"figs/paths/path_{seq_idx:03d}.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization for index {seq_idx} to {save_path}")

