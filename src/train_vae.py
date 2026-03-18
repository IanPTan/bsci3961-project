import torch as pt
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from vae import VAE
from dataset import PatchDataset

IMAGE_PATH = 'frieren.png'
print(f"Using image path: {IMAGE_PATH}")
IMG = transforms.ToTensor()(transforms.Resize(1024)(Image.open(IMAGE_PATH)))
                            
# Hyperparameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
CONV_CHANNELS = [32, 64, 128, 256, 512, 1024]
LINEAR_FEATURES = [128, 64]
best_val = 8000
BETTER_BY = 1

# Device configuration
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate VAE model
vae_model = VAE(conv_channels=CONV_CHANNELS, linear_features=LINEAR_FEATURES).to(device)
optimizer = pt.optim.AdamW(vae_model.parameters(), lr=LEARNING_RATE)
scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

dataset = PatchDataset(IMG, 64, 1)

# Split Dataset (80% Train, 20% Validation)
val_ratio = 0.2
train_size = int((1 - val_ratio) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_losses = []
val_losses = []
lrs = []

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    vae_model.train()
    total_train_loss = 0

    # using train_loader now
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for batch_idx, (images, _) in enumerate(train_pbar):
        images = images.to(device)

        # Forward pass
        recon_images, mu, logvar = vae_model(images)
        loss = vae_model.vae_loss(recon_images, images, mu, logvar)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_pbar.set_postfix(L=f"{loss.item() / len(images):.4f}")

    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # --- Validation Phase ---
    vae_model.eval()
    total_val_loss = 0

    with pt.no_grad():
        for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            images = images.to(device)

            recon_images, mu, logvar = vae_model(images)
            loss = vae_model.vae_loss(recon_images, images, mu, logvar)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    # Update learning rate based on validation loss
    lr = optimizer.param_groups[0]['lr']
    lrs.append(lr)
    scheduler.step(avg_val_loss)

    if best_val - avg_val_loss > BETTER_BY:
        best_val = avg_val_loss
        pt.save(vae_model.state_dict(), "backup.pt")

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {lr}')

print("VAE training complete.")
pt.save(vae_model.state_dict(), "backup.pt")

# Create a figure with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Top Plot: Losses
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Bottom Plot: Learning Rate
ax2.plot(lrs, color='orange')
ax2.set_title('Learning Rate Schedule')
ax2.set_xlabel('Steps/Epochs')
ax2.set_ylabel('Learning Rate')

# Adjust layout to prevent titles from overlapping
plt.tight_layout()
plt.savefig("figs/vae_loss.png")

for i in range(len(images)):
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the original image
    axes[0].imshow(images[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title("Original")
    axes[0].axis('off') # Optional: hides the x/y axes for a cleaner look

    # Plot the reconstructed image
    axes[1].imshow(recon_images[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    plt.savefig(f"figs/vae_comp_{i}.png")
