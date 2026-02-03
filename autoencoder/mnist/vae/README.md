# Variational Autoencoder (VAE) for MNIST

A PyTorch implementation of a Variational Autoencoder (VAE) for MNIST digit reconstruction and generation.

## Architecture

The VAE consists of:

### Encoder (Probabilistic)
- **Conv2d**: 1→16 channels, 3×3, stride=2 → 14×14×16
- **ReLU + BatchNorm2d**
- **Conv2d**: 16→32 channels, 3×3, stride=2 → 7×7×32
- **ReLU + BatchNorm2d**
- **Conv2d**: 32→64 channels, 3×3, stride=2 → 4×4×64
- **ReLU + BatchNorm2d**
- **Flatten**: 4×4×64 = 1024
- **Two Linear layers**: 1024 → latent_dim (one for μ, one for log σ²)

### Decoder (Deterministic)
- **Linear**: latent_dim → 1024
- **Reshape**: 1024 → 64×4×4
- **ConvTranspose2d**: 64→32, 3×3, stride=2 → 7×7×32
- **ReLU + BatchNorm2d**
- **ConvTranspose2d**: 32→16, 3×3, stride=2 → 14×14×16
- **ReLU + BatchNorm2d**
- **ConvTranspose2d**: 16→1, 3×3, stride=2 → 28×28×1

## Key Differences from Standard Autoencoder

1. **Probabilistic Latent Space**: The encoder outputs mean (μ) and log-variance (log σ²) instead of a deterministic latent vector
2. **Reparameterization Trick**: Samples z = μ + σ · ε where ε ~ N(0, I)
3. **VAE Loss**: Total loss = Reconstruction Loss (MSE) + β · KL Divergence
   - Reconstruction Loss: MSE between input and reconstructed image
   - KL Divergence: KL(q(z|x) || p(z)) regularizes the latent space to match N(0, I)
   - β: Weight for KL term (β=1.0 for standard VAE, β>1.0 for β-VAE)

## Loss Function

```python
# Reconstruction loss (MSE)
recon_loss = MSE(reconstructed, original)

# KL divergence: D_KL(N(μ, σ²) || N(0, I))
# = -0.5 * sum(1 + log(σ²) - μ² - σ²)
kl_div = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

# Total VAE loss
total_loss = recon_loss + β * kl_div
```

## Training

### Basic Usage

```bash
cd autoencoder/mnist/vae
python train.py
```

### Custom Hyperparameters

```bash
python train.py --latent-dim 16 --epochs 20 --batch-size 256 --lr 0.0005 --beta 1.0
```

### Arguments

- `--latent-dim`: Dimension of latent space (default: 12)
- `--beta`: Weight for KL divergence term (default: 1.0)
  - β=1.0: Standard VAE
  - β>1.0: β-VAE (stronger disentanglement, may sacrifice reconstruction quality)
  - β<1.0: Prioritizes reconstruction over regularization
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--save-plots`: Save visualization plots to disk
- `--plot-dir`: Directory to save plots (default: plots)

## Model Usage

### Training from Scratch

```python
from autoencoder.mnist.vae import VAE
from autoencoder.mnist.data import get_mnist_dataloaders
import torch

# Create model
model = VAE(latent_dim=12)
model = model.to('mps')  # or 'cuda', 'cpu'

# Load data
train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=128)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for images, targets in train_loader:
    images = images.to('mps')
    targets = targets.to('mps')

    # Forward pass
    reconstructed, mu, logvar = model(images)

    # Compute loss
    recon_loss = torch.nn.functional.mse_loss(reconstructed, targets, reduction='sum') / images.size(0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
    loss = recon_loss + kl_div

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Inference

```python
# Encode an image
mu, logvar = model.encode(images)

# Sample from latent distribution
z = model.reparameterize(mu, logvar)

# Decode back to image
reconstructed = model.decode(z)

# Generate new samples from prior
samples = model.sample(num_samples=16, device='mps')
```

## Outputs

After training, the script generates:

1. **Checkpoint file** (`vae_checkpoint.pth`): Contains model weights, optimizer state, and training history
2. **Training curves**: Loss over epochs (total, reconstruction, KL divergence)
3. **Reconstruction samples**: Original vs reconstructed images
4. **Generated samples**: Images sampled from the prior N(0, I)

## Comparison with Standard Autoencoder

| Feature | Standard Autoencoder | VAE |
|---------|---------------------|-----|
| Latent space | Deterministic (single vector) | Probabilistic (distribution) |
| Encoder output | z | μ, log σ² |
| Sampling | Not possible | Sample from N(μ, σ²) |
| Loss | Reconstruction only | Reconstruction + KL |
| Generation | ✗ (irregular latent space) | ✓ (smooth latent space) |
| Interpolation | Limited | Smooth and meaningful |

## Why VAE?

1. **Generative capability**: Can sample new images from the prior distribution
2. **Smooth latent space**: KL divergence regularizes the latent space to be continuous
3. **Interpolation**: Can smoothly interpolate between images in latent space
4. **Disentangled representations**: β-VAE can learn disentangled features

## References

- [Auto-Encoding Variational Bayes (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114)
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework (Higgins et al., 2017)](https://openreview.net/forum?id=Sy2fzU9gl)
