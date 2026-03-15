import torch as pt
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import h5py as hp

from dataset import PatchDataset
from vae import VAE

IMAGE_PATH = 'frieren128.png'
OUTPUT_PATH = "vae_patches.h5"
VAE_WEIGHTS_PATH = "vae.pt"

CONV_CHANNELS = [32, 64, 128, 256, 512, 1024]
LINEAR_FEATURES = [128, 64]

print(f"Using image path: {IMAGE_PATH}")
IMG = transforms.ToTensor()(transforms.Resize(128)(Image.open(IMAGE_PATH)))
dataset = PatchDataset(IMG, 64, 1)

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading model...")
vae_model = VAE(conv_channels=CONV_CHANNELS, linear_features=LINEAR_FEATURES).to(device)
vae_model.load_state_dict(pt.load(VAE_WEIGHTS_PATH, map_location=device))
vae_model.eval()
print("Loaded model.")

num_patches = len(dataset)
enc_dim = LINEAR_FEATURES[-1]

with hp.File(OUTPUT_PATH, "w") as ds_file:
    ds_file.create_dataset("patches", shape=(num_patches, enc_dim), dtype="f4")
    ds_file.create_dataset("coords", shape=(num_patches, 2), dtype="i8")

    ds_patches = ds_file["patches"]
    ds_coords = ds_file["coords"]

    for i, (img, coords) in tqdm(enumerate(dataset), desc="Encoding...", unit="sample", total=num_patches):
        x = img[None, ...].to(device)
        v, _ = vae_model.encode(x)

        ds_patches[i] = v[0].detach().cpu()
        ds_coords[i] = coords.cpu()

print("Done.")
