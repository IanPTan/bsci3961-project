import os
import torch
import cv2
import numpy as np
import subprocess
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
HIDDEN_LOOPS = 5
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


def patch_to_numpy_bgr(patch, size=512):
    """
    Converts patch from torch tensor to BGR numpy array for OpenCV,
    and resizes it to 'size x size' using nearest neighbor interpolation.
    """
    patch = patch.detach().cpu().numpy()

    if patch.ndim == 3:
        if patch.shape[0] == 1:
            patch = patch[0]          # (1, H, W) -> (H, W)
        else:
            patch = np.transpose(patch, (1, 2, 0))    # (C, H, W) -> (H, W, C)

    # Scale to 0-255 uint8
    patch = (patch * 255).clip(0, 255).astype(np.uint8)

    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    else:
        # Assumes RGB -> BGR
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

    # Resize with no interpolation (nearest neighbor) to keep it pixelated
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)

    return patch


# -------------------------
# Load dataset
# -------------------------
dataset = VAEPathDataset(PATHS_H5, split="val")
subset = Subset(dataset, SEQ_IDCS)
loader = DataLoader(subset, batch_size=len(SEQ_IDCS), shuffle=False)

# Get all sequences in one batch for efficiency
moves, in_patches, out_patches, _ = next(iter(loader))

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
    pred_out_patches, _, _ = rnn(x)

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
# Save Videos
# -------------------------
os.makedirs("figs/paths", exist_ok=True)

fps = 2

for i, seq_idx in enumerate(SEQ_IDCS):
    true_latents = out_patches[i]
    pred_latents = pred_out_patches[i]

    true_imgs = true_imgs_batch[i]
    pred_imgs = pred_imgs_batch[i]

    T_full = true_imgs.shape[0]

    # Init video writer
    size = 512
    final_path = f"figs/paths/path_{seq_idx:03d}.mp4"
    temp_path = f"figs/paths/temp_{seq_idx:03d}.mp4"
    
    # Use mp4v for the temporary file as it's a reliable intermediate format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (size * 2, size))

    if not writer.isOpened():
        print(f"Error: Could not open VideoWriter for {temp_path}. skipping.")
        continue

    print(f"Generating temporary video: {temp_path} ({T_full} steps @ {fps}fps)")

    for t in range(T_full):
        tp = patch_to_numpy_bgr(true_imgs[t], size=512)
        pp = patch_to_numpy_bgr(pred_imgs[t], size=512)

        # Calculate MSE loss for this frame (on latents)
        mse = torch.nn.functional.mse_loss(true_latents[t], pred_latents[t]).item()

        # Add text labels (adjusted for 512x512)
        cv2.putText(tp, f"True t={t}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(pp, f"Pred t={t}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Add MSE loss on top right
        mse_text = f"MSE: {mse:.4f}"
        text_size = cv2.getTextSize(mse_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(pp, mse_text, (size - text_size[0] - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Concatenate side-by-side
        frame = np.hstack([tp, pp])
        writer.write(frame)

    writer.release()

    print(f"Converting to stable format: {final_path}")
    cmd = [
        "ffmpeg", "-y", "-i", temp_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        final_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_path) # Cleanup temp file
        size_bytes = os.path.getsize(final_path)
        print(f"Saved video to {final_path} (Size: {size_bytes / 1024:.1f} KB)")
    except subprocess.CalledProcessError as e:
        print(f"Error: ffmpeg conversion failed for {final_path}. Keeping {temp_path}.")


