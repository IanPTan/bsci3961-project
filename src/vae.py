import torch as pt
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, conv_channels, linear_features, input_channels=3):
        super().__init__()

        # --- Encoder ---
        self.enc_convs = nn.ModuleList()
        in_c = input_channels

        # Build Conv Layers
        for out_c in conv_channels:
            self.enc_convs.append(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
            )
            in_c = out_c

        # Build Linear Layers (FC)
        self.enc_linears = nn.ModuleList()
        # Input to first linear is the last conv channel count (due to Global Mean)
        in_f = conv_channels[-1]

        # All linears except the last one (which splits into mu/var)
        for out_f in linear_features[:-1]:
            self.enc_linears.append(nn.Linear(in_f, out_f))
            in_f = out_f

        # Final layers for Mu and LogVar
        latent_dim = linear_features[-1]
        self.fc_mu = nn.Linear(in_f, latent_dim)
        self.fc_logvar = nn.Linear(in_f, latent_dim)

        # --- Decoder ---
        # Mirroring Encoder Linears
        self.dec_linears = nn.ModuleList()
        # Start from latent dim
        in_f = latent_dim

        # Reverse features (excluding latent), ending at conv_channels[-1]
        rev_features = linear_features[:-1][::-1] + [conv_channels[-1]]

        for out_f in rev_features:
            self.dec_linears.append(nn.Linear(in_f, out_f))
            in_f = out_f

        # Mirroring Encoder Convs
        self.dec_convs = nn.ModuleList()
        # Start from last conv channel
        in_c = conv_channels[-1]

        # Reverse conv channels, ending at input_channels (3)
        rev_channels = conv_channels[:-1][::-1] + [input_channels]

        for i, out_c in enumerate(rev_channels):
            # Last layer usually doesn't need bias if followed by Sigmoid, but keeping standard
            is_last = (i == len(rev_channels) - 1)
            self.dec_convs.append(
                nn.ConvTranspose2d(
                    in_c, out_c,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1 # Ensures exactly 2x upsampling
                )
            )
            in_c = out_c

    def encode(self, x):
        # 1. Convolutions
        for conv in self.enc_convs:
            x = F.gelu(conv(x))

        # 2. Global Mean Pooling (No Flattening)
        # Shape: (Batch, Channels, H, W) -> (Batch, Channels)
        x = x.mean(dim=[2, 3])

        # 3. Linear Layers
        for linear in self.enc_linears:
            x = F.gelu(linear(x))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = pt.exp(0.5 * logvar)
        eps = pt.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 1. Linear Layers
        x = z
        for linear in self.dec_linears:
            x = F.gelu(linear(x))

        # 2. Reshape for Convs (Inverse of Global Mean)
        # We project to (Batch, C) then view as (Batch, C, 1, 1)
        x = x.view(x.size(0), x.size(1), 1, 1)

        # 3. Transposed Convolutions
        for i, deconv in enumerate(self.dec_convs):
            x = deconv(x)
            if i != len(self.dec_convs) - 1:
                x = F.gelu(x)
            else:
                x = pt.sigmoid(x) # Final activation
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def vae_loss(self, recon_x, x, mu, logvar):
        # Note: If dimensions mismatch due to GAP (1x1 -> upsampling),
        # you might need to interpolate recon_x or x.
        # Standard BCE requires matching shapes.
        if recon_x.shape != x.shape:
             recon_x = F.interpolate(recon_x, size=x.shape[2:], mode='bilinear', align_corners=False)

        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * pt.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

# Usage Example
# convs = [32, 64, 128, 256] -> 4 layers
# feats = [128, 64] -> Latent dim is 64

if __name__ == "__main__":
    model = VAE(conv_channels=[32, 64, 128, 256], linear_features=[128, 64])
    # test code here
    # Dummy input
    x = pt.randn(2, 3, 64, 64)
    recon, mu, logvar = model(x)
    print(f"Input: {x.shape}, Output: {recon.shape}")
