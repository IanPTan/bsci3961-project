import argparse
import tomllib
from pathlib import Path
import h5py as hp
import torch as pt
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm
import os
import numpy as np
import random

from dataset import PatchDataset
from vae import VAE

DEFAULT_CONFIG = {
    "vae_dir": "experiments/vae_1",
    "patch_size": 1024, # Special case for this script based on old code
    "patch_stride": 1,
    "rng_seed": 42
}

def get_latest_exp_dir(prefix="vae_"):
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

def get_exp_dir(args_exp_dir, prefix="patches_"):
    exp_root = Path("experiments")
    exp_root.mkdir(exist_ok=True)
    
    if args_exp_dir:
        exp_dir = Path(args_exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
        
    max_num = 0
    for d in exp_root.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            try:
                num = int(d.name.split("_")[1])
                max_num = max(max_num, num)
            except ValueError:
                pass
    new_dir = exp_root / f"{prefix}{max_num + 1}"
    new_dir.mkdir(exist_ok=True)
    return new_dir

def write_toml(config, path):
    with open(path, "w") as f:
        for k, v in config.items():
            if isinstance(v, str):
                f.write(f'{k} = "{v}"\n')
            else:
                f.write(f'{k} = {v}\n')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)
    pt.cuda.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Generate VAE Patches Dataset")
    parser.add_argument("--exp-dir", type=str, help="Path to patches experiment directory")
    parser.add_argument("--vae-dir", type=str, help="Path to VAE experiment directory")
    args = parser.parse_args()
    
    exp_dir = get_exp_dir(args.exp_dir, "patches_")
    print(f"Using experiment directory: {exp_dir}")
    
    config_path = exp_dir / "config.toml"
    config = DEFAULT_CONFIG.copy()
    if config_path.exists():
        with open(config_path, "rb") as f:
            config.update(tomllib.load(f))
    
    if args.vae_dir:
        config["vae_dir"] = args.vae_dir
    elif not config_path.exists():
        latest_vae = get_latest_exp_dir("vae_")
        if latest_vae:
            config["vae_dir"] = str(latest_vae)
            
    vae_dir = Path(config["vae_dir"])
    vae_config_path = vae_dir / "config.toml"
    if not vae_config_path.exists():
        print(f"Error: VAE config not found at {vae_config_path}")
        return
        
    with open(vae_config_path, "rb") as f:
        vae_config = tomllib.load(f)
        
    if not config_path.exists():
        write_toml(config, config_path)
        
    seed_everything(config["rng_seed"])
    
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_path = vae_config["image_path"]
    image_size = vae_config["image_size"]
    print(f"Using image path: {image_path} (size: {image_size})")
    
    IMG = transforms.ToTensor()(transforms.Resize(image_size)(Image.open(image_path).convert("RGB")))
    dataset = PatchDataset(IMG, config["patch_size"], config["patch_stride"])

    print(f"Loading VAE from {vae_dir}...")
    vae_model = VAE(
        conv_channels=vae_config["conv_channels"], 
        linear_features=vae_config["linear_features"]
    ).to(device)
    
    vae_weights_path = vae_dir / f"{vae_dir.name}.pt"
    if not vae_weights_path.exists():
        vae_weights_path = vae_dir / "vae.pt"
    
    vae_model.load_state_dict(pt.load(vae_weights_path, map_location=device))
    vae_model.eval()
    print("Loaded VAE model.")

    num_patches = len(dataset)
    enc_dim = vae_config["linear_features"][-1]
    output_h5 = exp_dir / "vae_patches.h5"

    with hp.File(output_h5, "w") as ds_file:
        ds_file.create_dataset("patches", shape=(num_patches, enc_dim), dtype="f4")
        ds_file.create_dataset("coords", shape=(num_patches, 2), dtype="i8")

        ds_patches = ds_file["patches"]
        ds_coords = ds_file["coords"]

        for i, (img, coords) in tqdm(enumerate(dataset), desc="Encoding...", unit="sample", total=num_patches):
            x = img[None, ...].to(device)
            with pt.no_grad():
                v, _ = vae_model.encode(x)

            ds_patches[i] = v[0].detach().cpu().numpy()
            ds_coords[i] = coords.cpu().numpy()

    print(f"Done. Dataset saved to {output_h5}")

if __name__ == "__main__":
    main()
