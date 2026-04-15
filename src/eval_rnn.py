import torch
import matplotlib.pyplot as plt

from rnn import RNN
from vae import VAE
from dataset import VAEPathDataset


# -------------------------
# Config
# -------------------------
PATHS_H5 = "vae_paths.h5"
RNN_WEIGHTS = "2048_9_rnn.pt"
VAE_WEIGHTS = "vae.pt"

# must match train_rnn.py
HIDDEN_DIM = 2048
HIDDEN_LOOPS = 9
K = 0

# example index of sequence to visualize
SEQ_IDX = 4

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
        # change this if your VAE uses vae.decoder(...) instead
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

moves, in_patches, out_patches = dataset[SEQ_IDX]

print("Single sample shapes:")
print("moves:", moves.shape)
print("in_patches:", in_patches.shape)
print("out_patches:", out_patches.shape)

patch_dim = in_patches.shape[-1]
move_dim = moves.shape[-1]
input_dim = patch_dim + move_dim
output_dim = out_patches.shape[-1]

# add batch dimension
moves = moves.unsqueeze(0).to(device)             # (1, T, move_dim)
in_patches = in_patches.unsqueeze(0).to(device)   # (1, T, patch_dim)
out_patches = out_patches.unsqueeze(0).to(device) # (1, T, patch_dim)

x = torch.cat([in_patches, moves], dim=-1)        # (1, T, patch_dim + move_dim)

# -------------------------
# Load RNN
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

# -------------------------
# Load VAE
# -------------------------
# change constructor args to match your trained VAE
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

print("pred_out_patches:", pred_out_patches.shape)

# -------------------------
# Decode true and predicted latents
# -------------------------
true_imgs = decode_latents(vae, out_patches)[0]       # remove batch dim -> (T, ...)
pred_imgs = decode_latents(vae, pred_out_patches)[0]  # remove batch dim -> (T, ...)

print("Decoded shapes:")
print("true_imgs:", true_imgs.shape)
print("pred_imgs:", pred_imgs.shape)

# -------------------------
# Plot side by side
# -------------------------
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
plt.savefig(f"figs/paths/path_{SEQ_IDX}.png")
