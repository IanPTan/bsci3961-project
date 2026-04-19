import os
import torch
import cv2
import h5py
import numpy as np
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rnn import RNN
from vae import VAE


# -------------------------
# Config
# -------------------------
# must match the trained model/eval results
HIDDEN_DIM = 2048
HIDDEN_LOOPS = 3

VAL_H5 = f"{HIDDEN_DIM}_{HIDDEN_LOOPS}_val.h5"
VAE_WEIGHTS = "vae.pt"

# indices of sequences to visualize (indices into the validation set)
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
# Load Data from H5
# -------------------------
print(f"Loading predictions from {VAL_H5}...")
with h5py.File(VAL_H5, "r") as f:
    # Indexing into H5 datasets directly
    out_patches = torch.from_numpy(f["out_patches"][SEQ_IDCS]).to(device)
    pred_out_patches = torch.from_numpy(f["y"][SEQ_IDCS]).to(device)

print(f"Batch shapes for {len(SEQ_IDCS)} sequences:")
print("out_patches (targets):", out_patches.shape)
print("pred_out_patches (pred):", pred_out_patches.shape)

# -------------------------
# Load VAE for Decoding
# -------------------------
vae = VAE(
    conv_channels=[32, 64, 128, 256, 512, 1024],
    linear_features=[128, 64],
).to(device)

vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
vae.eval()

# -------------------------
# Decode true and predicted latents (Batched)
# -------------------------
print("Decoding latents to images...")
true_imgs_batch = decode_latents(vae, out_patches)       # (B, T, C, H, W)
pred_imgs_batch = decode_latents(vae, pred_out_patches)  # (B, T, C, H, W)

print("Decoded batch shapes:")
print("true_imgs_batch:", true_imgs_batch.shape)
print("pred_imgs_batch:", pred_imgs_batch.shape)

# -------------------------
# Calculate and Plot Losses
# -------------------------
# all_mses shape: (B, T)
# out_patches and pred_out_patches are (B, T, latent_dim)
all_mses_batch = torch.nn.functional.mse_loss(pred_out_patches, out_patches, reduction='none').mean(dim=-1).cpu().numpy()

os.makedirs("figs/paths", exist_ok=True)
print("Generating loss plots...")

for i, seq_idx in enumerate(SEQ_IDCS):
    plt.figure(figsize=(8, 4))
    plt.plot(all_mses_batch[i], label=f'Seq {seq_idx}')
    plt.xlabel('Timestep')
    plt.ylabel('MSE Loss')
    plt.title(f'MSE Loss over Time - Sequence {seq_idx}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"figs/paths/mse_{seq_idx:03d}.png")
    plt.close()

# Average MSE plot
avg_mse = all_mses_batch.mean(axis=0)
plt.figure(figsize=(8, 4))
plt.plot(avg_mse, color='red', linewidth=2, label='Average MSE')
plt.xlabel('Timestep')
plt.ylabel('Mean MSE Loss')
plt.title('Average MSE Loss over Time (All Samples)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(f"figs/paths/mse_average.png")
plt.close()
print("Loss plots saved to figs/paths/")

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


