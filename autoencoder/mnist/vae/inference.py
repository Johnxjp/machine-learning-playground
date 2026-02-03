"""
Inference script for VAE - generate samples and reconstruct images.

Usage:
    # Generate random samples from prior
    python inference.py --checkpoint vae_high_kl.pth --num-samples 64

    # Reconstruct test images
    python inference.py --checkpoint vae_high_kl.pth --mode reconstruct --num-samples 10

    # Interpolate between two images
    python inference.py --checkpoint vae_high_kl.pth --mode interpolate --num-steps 10
"""

import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from model import VAE
from data import get_mnist_dataloaders
from utils import get_device


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='VAE Inference - Generate samples and reconstructions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='sample',
        choices=['sample', 'reconstruct', 'interpolate'],
        help='Inference mode: sample (generate from prior), reconstruct (encode-decode test images), or interpolate'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=64,
        help='Number of samples to generate or reconstruct'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=10,
        help='Number of interpolation steps (for interpolate mode)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/Users/johnlingi/programming/machine_learning/datasets/mnist/',
        help='Directory with MNIST data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path to save the figure'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu, cuda, mps). Auto-detect if not specified.'
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: str) -> VAE:
    """Load VAE model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config
    config = checkpoint['config']
    latent_dim = config['latent_dim']

    # Create model
    model = VAE(latent_dim=latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Latent dimension: {latent_dim}")
    print(f"  Trained epochs: {checkpoint['epoch']}")
    if 'beta' in config:
        print(f"  Beta (KL weight): {config['beta']}")
    print()

    return model


def generate_samples(model: VAE, num_samples: int, device: str, output: str = None):
    """Generate random samples from the prior N(0, I)."""
    print(f"Generating {num_samples} samples from prior N(0, I)...")

    samples = model.sample(num_samples=num_samples, device=device)
    samples = samples.cpu()

    # Determine grid size
    n_cols = min(8, num_samples)
    n_rows = (num_samples + n_cols - 1) // n_cols

    # Plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        row = idx // n_cols
        col = idx % n_cols

        axes[row, col].imshow(samples[idx, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')

    # Hide extra subplots
    for idx in range(num_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle(f'Generated Samples from Prior N(0, I)', fontsize=14)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output}")

    plt.show()


def reconstruct_images(model: VAE, num_samples: int, device: str, data_dir: str, output: str = None):
    """Reconstruct test images."""
    print(f"Reconstructing {num_samples} test images...")

    # Load test data
    _, _, test_loader = get_mnist_dataloaders(batch_size=num_samples, data_dir=data_dir)

    # Get a batch of test images
    test_images, _ = next(iter(test_loader))
    test_images = test_images[:num_samples].to(device)

    # Reconstruct
    with torch.no_grad():
        reconstructed, mu, logvar = model(test_images)

    test_images = test_images.cpu()
    reconstructed = reconstructed.cpu()
    mu = mu.cpu()
    logvar = logvar.cpu()

    # Plot original vs reconstructed
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 1.5, 4.5))

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(test_images[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # Reconstructed
        axes[1, i].imshow(reconstructed[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

        # Latent visualization (first 12 dims as bar chart)
        axes[2, i].bar(range(len(mu[i])), mu[i].numpy(), alpha=0.6)
        axes[2, i].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[2, i].set_ylim(-3, 3)
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([-3, 0, 3])
        axes[2, i].tick_params(labelsize=6)
        if i == 0:
            axes[2, i].set_title('Latent μ', fontsize=10)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output}")

    plt.show()

    # Print statistics
    print(f"\nLatent space statistics:")
    print(f"  μ mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")
    print(f"  σ mean: {torch.exp(0.5 * logvar).mean().item():.4f}, std: {torch.exp(0.5 * logvar).std().item():.4f}")


def interpolate_between_images(model: VAE, num_steps: int, device: str, data_dir: str, output: str = None):
    """Interpolate between two random test images in latent space."""
    print(f"Interpolating between two images with {num_steps} steps...")

    # Load test data
    _, _, test_loader = get_mnist_dataloaders(batch_size=2, data_dir=data_dir)

    # Get two test images
    test_images, _ = next(iter(test_loader))
    test_images = test_images[:2].to(device)

    # Encode to latent space
    with torch.no_grad():
        mu, logvar = model.encode(test_images)

        # Use mean for interpolation (deterministic)
        z1 = mu[0]
        z2 = mu[1]

        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, num_steps, device=device)
        interpolated_latents = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolated_latents.append(z_interp)

        interpolated_latents = torch.stack(interpolated_latents)

        # Decode interpolated latents
        interpolated_images = model.decode(interpolated_latents)

    # Move to CPU for plotting
    test_images = test_images.cpu()
    interpolated_images = interpolated_images.cpu()

    # Plot
    fig, axes = plt.subplots(2, num_steps, figsize=(num_steps * 1.5, 3))

    # First row: interpolated images
    for i in range(num_steps):
        axes[0, i].imshow(interpolated_images[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Start', fontsize=8, color='red')
        elif i == num_steps - 1:
            axes[0, i].set_title('End', fontsize=8, color='red')

    # Second row: show original images at start and end
    axes[1, 0].imshow(test_images[0, 0].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Original Start', fontsize=8)

    axes[1, -1].imshow(test_images[1, 0].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, -1].axis('off')
    axes[1, -1].set_title('Original End', fontsize=8)

    # Hide middle plots in second row
    for i in range(1, num_steps - 1):
        axes[1, i].axis('off')

    plt.suptitle('Latent Space Interpolation', fontsize=14)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output}")

    plt.show()


def main():
    """Main inference function."""
    args = parse_args()

    # Setup device
    device = args.device if args.device else get_device()
    print("=" * 60)
    print("VAE Inference")
    print("=" * 60)
    print(f"Device: {device}\n")

    # Load model
    model = load_checkpoint(args.checkpoint, device)

    # Run inference based on mode
    if args.mode == 'sample':
        generate_samples(model, args.num_samples, device, args.output)
    elif args.mode == 'reconstruct':
        reconstruct_images(model, args.num_samples, device, args.data_dir, args.output)
    elif args.mode == 'interpolate':
        interpolate_between_images(model, args.num_steps, device, args.data_dir, args.output)

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
