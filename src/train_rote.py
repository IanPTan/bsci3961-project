import torch as pt
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from dataset import PatchDataset
from vae import VAE

IMAGE_PATH = 'frieren128.png'
print(f"Using image path: {IMAGE_PATH}")
IMG = transforms.ToTensor()(transforms.Resize(128)(Image.open(IMAGE_PATH)))
dataset = PatchDataset(IMG, 64, 1)

conv_channels = [32, 64, 128, 256, 512, 1024]
linear_features = [128, 64]

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading model...")
vae_model = VAE(conv_channels=conv_channels, linear_features=linear_features).to(device)
vae_model.load_state_dict(pt.load("vae.pt", map_location=device))
vae_model.eval()
print("Loaded model.")


for (img, _) in dataset:
    x = img[None, ...]
    y, _, _ = vae_model(x)
    plt.imshow(y[0].detach().permute(1, 2, 0))
    plt.show()
    
