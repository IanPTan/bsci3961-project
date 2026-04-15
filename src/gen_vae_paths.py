import torch as pt
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm
import h5py as hp

from dataset import Patcher, get_path
from vae import VAE


IMAGE_PATH = "frieren.png"
OUTPUT_PATH = "vae_paths.h5"
VAE_WEIGHTS_PATH = "vae.pt"

SPLIT_SIZES = {"train": 8000, "test": 2000}
NUM_MOVES = 128
PATCH_SIZE = 64
PATCH_STRIDE = 1
MOVE_MAX = 5

CONV_CHANNELS = [32, 64, 128, 256, 512, 1024]
LINEAR_FEATURES = [128, 64]


device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Using image path: {IMAGE_PATH}")
IMG = transforms.ToTensor()(transforms.Resize(1024)(Image.open(IMAGE_PATH).convert("RGB")))

print("Loading model...")
vae_model = VAE(conv_channels=CONV_CHANNELS, linear_features=LINEAR_FEATURES).to(device)
vae_model.load_state_dict(pt.load(VAE_WEIGHTS_PATH, map_location=device))
vae_model.eval()
print("Loaded model.")

patcher = Patcher(IMG, PATCH_SIZE, PATCH_STRIDE)
n_h, n_w = patcher.shape

enc_dim = LINEAR_FEATURES[-1]
seq_len_patches = NUM_MOVES + 1

with hp.File(OUTPUT_PATH, "w") as ds_file:

    start_i = 0

    for group, num_samples in SPLIT_SIZES.items():
        
        ds_file.create_group(group)

        ds_file[group].create_dataset(
            "patches",
            shape=(num_samples, seq_len_patches, enc_dim),
            dtype="f4",
        )
        ds_file[group].create_dataset(
            "moves",
            shape=(num_samples, NUM_MOVES, 2),
            dtype="i4",
        )
        ds_file[group].create_dataset(
            "coords",
            shape=(num_samples, seq_len_patches, 2),
            dtype="i4",
        )

        ds_patches = ds_file[group]["patches"]
        ds_moves = ds_file[group]["moves"]
        ds_coords = ds_file[group]["coords"]

        for i in tqdm(range(num_samples), desc=f"Encoding paths {group}", unit="path"):
            coords = get_path(
                start_i + i,
                patcher.shape,
                NUM_MOVES,
                x_min=-MOVE_MAX,
                x_max=MOVE_MAX,
            )

            coords[:, 0] = coords[:, 0].clamp(0, n_h - 1)
            coords[:, 1] = coords[:, 1].clamp(0, n_w - 1)

            moves = coords.diff(dim=0)
            patches = patcher(coords)

            x = patches.to(device)
            v, _ = vae_model.encode(x)

            ds_patches[i] = v.detach().cpu()
            ds_moves[i] = moves.cpu()
            ds_coords[i] = coords.cpu()
        start_i += num_samples

print("Done.")

