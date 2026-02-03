"""
Quick test to verify VAE model architecture and forward pass.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import VAE

def test_vae():
    """Test VAE model instantiation and forward pass."""
    print("Testing VAE Model")
    print("=" * 60)

    # Create model
    latent_dim = 12
    model = VAE(latent_dim=latent_dim)
    print(f"✓ Model created with latent_dim={latent_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    print(f"\n✓ Input shape: {x.shape}")

    # Forward pass
    reconstructed, mu, logvar = model(x)
    print(f"✓ Reconstructed shape: {reconstructed.shape}")
    print(f"✓ Mu shape: {mu.shape}")
    print(f"✓ Logvar shape: {logvar.shape}")

    # Test encoding
    mu_enc, logvar_enc = model.encode(x)
    print(f"\n✓ Encode output shapes: mu={mu_enc.shape}, logvar={logvar_enc.shape}")

    # Test reparameterization
    z = model.reparameterize(mu, logvar)
    print(f"✓ Sampled latent shape: {z.shape}")

    # Test decoding
    decoded = model.decode(z)
    print(f"✓ Decoded shape: {decoded.shape}")

    # Test sampling from prior
    samples = model.sample(num_samples=4, device='cpu')
    print(f"✓ Generated samples shape: {samples.shape}")

    # Test loss computation
    recon_loss = torch.nn.functional.mse_loss(reconstructed, x, reduction='sum') / batch_size
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = recon_loss + kl_div

    print(f"\n✓ Reconstruction loss: {recon_loss.item():.4f}")
    print(f"✓ KL divergence: {kl_div.item():.4f}")
    print(f"✓ Total loss: {total_loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")

if __name__ == '__main__':
    test_vae()
