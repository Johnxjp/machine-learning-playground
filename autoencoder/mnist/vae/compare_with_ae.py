"""
Compare VAE with standard autoencoder to illustrate key differences.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from model import VAE

# Import from parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(parent_dir, '..'))
from model import ConvAutoencoder

def compare_architectures():
    """Compare VAE and standard autoencoder architectures."""
    print("Architecture Comparison")
    print("=" * 80)

    latent_dim = 12
    vae = VAE(latent_dim=latent_dim)
    ae = ConvAutoencoder(latent_dim=latent_dim)

    vae_params = sum(p.numel() for p in vae.parameters())
    ae_params = sum(p.numel() for p in ae.parameters())

    print(f"\nStandard Autoencoder:")
    print(f"  Parameters: {ae_params:,}")
    print(f"  Latent space: Deterministic (single vector)")
    print(f"  Encoder output: z ∈ R^{latent_dim}")

    print(f"\nVariational Autoencoder (VAE):")
    print(f"  Parameters: {vae_params:,}")
    print(f"  Latent space: Probabilistic (distribution)")
    print(f"  Encoder output: μ ∈ R^{latent_dim}, log σ² ∈ R^{latent_dim}")
    print(f"  Additional params: {vae_params - ae_params:,} (for logvar head)")

    print("\n" + "=" * 80)


def compare_forward_pass():
    """Compare forward pass behavior."""
    print("\nForward Pass Comparison")
    print("=" * 80)

    latent_dim = 12
    batch_size = 4

    vae = VAE(latent_dim=latent_dim)
    ae = ConvAutoencoder(latent_dim=latent_dim)

    # Create dummy input
    x = torch.randn(batch_size, 1, 28, 28)

    print(f"\nInput: {x.shape}")

    # Standard AE forward pass
    print(f"\nStandard Autoencoder:")
    z_ae = ae.encode(x)
    print(f"  1. Encode: x → z")
    print(f"     z shape: {z_ae.shape}")
    recon_ae = ae.decode(z_ae)
    print(f"  2. Decode: z → x̂")
    print(f"     x̂ shape: {recon_ae.shape}")
    print(f"  → Deterministic: Same input always produces same latent code")

    # VAE forward pass
    print(f"\nVariational Autoencoder:")
    mu, logvar = vae.encode(x)
    print(f"  1. Encode: x → (μ, log σ²)")
    print(f"     μ shape: {mu.shape}")
    print(f"     log σ² shape: {logvar.shape}")
    z_vae = vae.reparameterize(mu, logvar)
    print(f"  2. Reparameterize: z = μ + σ · ε, where ε ~ N(0, I)")
    print(f"     z shape: {z_vae.shape}")
    recon_vae = vae.decode(z_vae)
    print(f"  3. Decode: z → x̂")
    print(f"     x̂ shape: {recon_vae.shape}")
    print(f"  → Stochastic: Same input can produce different latent codes")

    # Demonstrate stochasticity
    print(f"\nDemonstrating stochasticity:")
    z1 = vae.reparameterize(mu, logvar)
    z2 = vae.reparameterize(mu, logvar)
    diff = torch.abs(z1 - z2).mean().item()
    print(f"  Two samples from same (μ, σ²): mean absolute difference = {diff:.6f}")
    print(f"  (Non-zero difference confirms stochastic sampling)")

    print("\n" + "=" * 80)


def compare_loss_functions():
    """Compare loss functions."""
    print("\nLoss Function Comparison")
    print("=" * 80)

    latent_dim = 12
    batch_size = 4

    vae = VAE(latent_dim=latent_dim)
    ae = ConvAutoencoder(latent_dim=latent_dim)

    x = torch.randn(batch_size, 1, 28, 28)

    # Standard AE loss
    recon_ae = ae(x)
    loss_ae = torch.nn.functional.mse_loss(recon_ae, x, reduction='sum') / batch_size

    print(f"\nStandard Autoencoder Loss:")
    print(f"  L_AE = MSE(x̂, x)")
    print(f"  L_AE = {loss_ae.item():.4f}")
    print(f"  → Only reconstruction loss")

    # VAE loss
    recon_vae, mu, logvar = vae(x)
    recon_loss = torch.nn.functional.mse_loss(recon_vae, x, reduction='sum') / batch_size
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    loss_vae = recon_loss + kl_div

    print(f"\nVariational Autoencoder Loss:")
    print(f"  L_VAE = L_recon + β · L_KL")
    print(f"  L_recon = MSE(x̂, x) = {recon_loss.item():.4f}")
    print(f"  L_KL = D_KL(q(z|x) || p(z)) = {kl_div.item():.4f}")
    print(f"  L_VAE = {loss_vae.item():.4f} (with β=1.0)")
    print(f"  → Reconstruction + regularization")

    print(f"\nKL Divergence Interpretation:")
    print(f"  - Regularizes latent space to match prior N(0, I)")
    print(f"  - Prevents overfitting in latent space")
    print(f"  - Enables smooth interpolation and generation")

    print("\n" + "=" * 80)


def visualize_latent_space_sampling():
    """Visualize sampling difference."""
    print("\nLatent Space Sampling")
    print("=" * 80)

    latent_dim = 2  # Use 2D for easy visualization
    batch_size = 100

    vae = VAE(latent_dim=latent_dim)
    ae = ConvAutoencoder(latent_dim=latent_dim)

    x = torch.randn(batch_size, 1, 28, 28)

    # Get latent codes
    with torch.no_grad():
        z_ae = ae.encode(x)
        mu, logvar = vae.encode(x)

        # Sample multiple times from VAE
        z_vae_samples = []
        for _ in range(5):
            z_vae_samples.append(vae.reparameterize(mu, logvar))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AE latent space (deterministic)
    axes[0].scatter(z_ae[:, 0].numpy(), z_ae[:, 1].numpy(), alpha=0.6, s=20, c='blue')
    axes[0].set_title('Standard Autoencoder\n(Deterministic Latent Codes)', fontsize=12)
    axes[0].set_xlabel('Latent Dimension 1')
    axes[0].set_ylabel('Latent Dimension 2')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

    # VAE latent space (stochastic)
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, z_sample in enumerate(z_vae_samples):
        axes[1].scatter(
            z_sample[:, 0].numpy(),
            z_sample[:, 1].numpy(),
            alpha=0.3,
            s=20,
            c=colors[i],
            label=f'Sample {i+1}'
        )
    axes[1].scatter(mu[:, 0].numpy(), mu[:, 1].numpy(), alpha=0.8, s=40, c='black', marker='x', label='μ (mean)')
    axes[1].set_title('Variational Autoencoder\n(Probabilistic: Multiple Samples from Same Input)', fontsize=12)
    axes[1].set_xlabel('Latent Dimension 1')
    axes[1].set_ylabel('Latent Dimension 2')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('vae_vs_ae_latent_space.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: vae_vs_ae_latent_space.png")
    plt.show()

    print("\nKey Observations:")
    print("  - AE: Each input maps to a single point (deterministic)")
    print("  - VAE: Each input maps to a distribution (stochastic)")
    print("  - VAE samples cluster around μ with spread determined by σ")

    print("\n" + "=" * 80)


def main():
    """Run all comparisons."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "VAE vs Standard Autoencoder Comparison" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    compare_architectures()
    compare_forward_pass()
    compare_loss_functions()
    visualize_latent_space_sampling()

    print("\nSummary")
    print("=" * 80)
    print("\nWhen to use each:")
    print("\n  Standard Autoencoder:")
    print("    - Dimensionality reduction")
    print("    - Feature extraction")
    print("    - Denoising")
    print("    - Compression")
    print("\n  Variational Autoencoder:")
    print("    - Generative modeling (sample new data)")
    print("    - Smooth interpolation")
    print("    - Disentangled representations")
    print("    - Uncertainty estimation")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
