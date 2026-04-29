import torch as pt
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = pt.sigmoid(self.fc2(w))
        return x * w

class EncoderResidualBlock(nn.Module):
    def __init__(self, in_channels, inner_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(inner_channels),
            nn.SiLU(),
            nn.Conv2d(inner_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )
        self.se = SEBlock(in_channels)
        
    def forward(self, x):
        return x + self.se(self.conv2(self.conv1(x)))

class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels, inner_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(inner_channels),
            nn.SiLU(),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=5, padding=2, groups=inner_channels, bias=False)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(inner_channels),
            nn.SiLU(),
            nn.Conv2d(inner_channels, in_channels, kernel_size=1, bias=False)
        )
        self.se = SEBlock(in_channels)
        
    def forward(self, x):
        return x + self.se(self.conv3(self.conv2(self.conv1(x))))

class VAE(nn.Module):
    def __init__(self, 
                 enc_channels, enc_inner_channels, enc_scales, enc_linears,
                 dec_channels, dec_inner_channels, dec_scales, dec_linears,
                 latent_dim, image_size=1024, input_channels=3):
        super().__init__()
        
        self.image_size = image_size
        
        # --- Encoder ---
        self.enc_blocks = nn.ModuleList()
        in_c = input_channels
        
        spatial_dim = image_size
        
        for c, inner_c, scale in zip(enc_channels, enc_inner_channels, enc_scales):
            if in_c != c:
                self.enc_blocks.append(nn.Conv2d(in_c, c, kernel_size=1, bias=False))
                in_c = c
            self.enc_blocks.append(EncoderResidualBlock(c, inner_c))
            if scale > 1:
                self.enc_blocks.append(nn.AvgPool2d(scale))
                spatial_dim = spatial_dim // scale
                
        self.enc_spatial_dim = spatial_dim
        in_f = in_c * spatial_dim * spatial_dim
        
        self.enc_linears_seq = nn.ModuleList()
        for out_f in enc_linears:
            self.enc_linears_seq.append(nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.LayerNorm(out_f),
                nn.SiLU()
            ))
            in_f = out_f
            
        self.fc_mu = nn.Linear(in_f, latent_dim)
        self.fc_logvar = nn.Linear(in_f, latent_dim)

        # --- Decoder ---
        dec_scale_factor = 1
        for s in dec_scales:
            dec_scale_factor *= s
            
        assert image_size % dec_scale_factor == 0, f"Image size {image_size} not divisible by decoder scale {dec_scale_factor}"
        self.dec_start_spatial = image_size // dec_scale_factor
        self.dec_start_channels = dec_channels[0]
        
        dec_flat_size = self.dec_start_channels * self.dec_start_spatial * self.dec_start_spatial
        
        self.dec_linears_seq = nn.ModuleList()
        in_f = latent_dim
        for out_f in dec_linears:
            self.dec_linears_seq.append(nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.LayerNorm(out_f),
                nn.SiLU()
            ))
            in_f = out_f
            
        self.fc_dec_unflatten = nn.Linear(in_f, dec_flat_size)
        
        self.dec_blocks = nn.ModuleList()
        in_c = self.dec_start_channels
        
        for c, inner_c, scale in zip(dec_channels, dec_inner_channels, dec_scales):
            if scale > 1:
                self.dec_blocks.append(nn.Upsample(scale_factor=scale, mode='nearest'))
            if in_c != c:
                self.dec_blocks.append(nn.Conv2d(in_c, c, kernel_size=1, bias=False))
                in_c = c
            self.dec_blocks.append(DecoderResidualBlock(c, inner_c))
            
        self.dec_final = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, input_channels, kernel_size=3, padding=1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # SiLU approx
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        for block in self.enc_blocks:
            x = block(x)
            
        x = x.flatten(start_dim=1)
        
        for linear in self.enc_linears_seq:
            x = linear(x)
            
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = pt.clamp(logvar, -10, 10)
        std = pt.exp(0.5 * logvar)
        eps = pt.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = z
        for linear in self.dec_linears_seq:
            x = linear(x)
            
        x = self.fc_dec_unflatten(x)
        x = x.view(x.size(0), self.dec_start_channels, self.dec_start_spatial, self.dec_start_spatial)
        
        for block in self.dec_blocks:
            x = block(x)
            
        x = self.dec_final(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @pt.no_grad()
    def reconstruct(self, x):
        is_train = self.training
        self.eval()
        logits, _, _ = self.forward(x)
        pixels = pt.sigmoid(logits)
        if is_train:
            self.train()
        return pixels

    @pt.no_grad()
    def generate(self, z):
        is_train = self.training
        self.eval()
        logits = self.decode(z)
        pixels = pt.sigmoid(logits)
        if is_train:
            self.train()
        return pixels

    def vae_loss(self, recon_x, x, mu, logvar):
        if recon_x.shape != x.shape:
             recon_x = F.interpolate(recon_x, size=x.shape[2:], mode='bilinear', align_corners=False)

        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
        KLD = -0.5 * pt.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

if __name__ == "__main__":
    model = VAE(
        enc_channels=[32, 64], enc_inner_channels=[16, 32], enc_scales=[2, 2], enc_linears=[128],
        dec_channels=[64, 32], dec_inner_channels=[32, 16], dec_scales=[2, 2], dec_linears=[128],
        latent_dim=64, image_size=64
    )
    x = pt.randn(2, 3, 64, 64)
    recon, mu, logvar = model(x)
    print(f"Input: {x.shape}, Output: {recon.shape}")
