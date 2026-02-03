"""
Training script for MNIST Variational Autoencoder (VAE).

Run with default settings:
    python train.py

Run with custom hyperparameters:
    python train.py --latent-dim 16 --epochs 20 --batch-size 256 --lr 0.0005 --beta 1.0
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from vae.model import VAE
except ImportError:
    from model import VAE

from data import get_mnist_dataloaders
from utils import get_device, plot_reconstructions, plot_training_curves


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a Variational Autoencoder (VAE) on MNIST',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model hyperparameters
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=12,
        help='Dimension of the latent space'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.0,
        help='Weight for KL divergence term (beta-VAE). Default 1.0 for standard VAE.'
    )

    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )

    # Data
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/Users/johnlingi/programming/machine_learning/datasets/mnist/',
        help='Directory to store/load MNIST data'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.9,
        help='Fraction of training data to use for training (rest for validation)'
    )

    # Output
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default='vae_checkpoint.pth',
        help='Path to save model checkpoint'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save training plots to disk'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='plots',
        help='Directory to save plots'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu, cuda, mps). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = reconstruction_loss + beta * KL_divergence.

    Args:
        reconstructed: Reconstructed images (batch_size, 1, 28, 28)
        original: Original images (batch_size, 1, 28, 28)
        mu: Mean of latent distribution (batch_size, latent_dim)
        logvar: Log variance of latent distribution (batch_size, latent_dim)
        beta: Weight for KL divergence term (default: 1.0)

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (MSE)
    recon_loss = torch.nn.functional.mse_loss(reconstructed, original, reduction='sum')
    recon_loss = recon_loss / original.size(0)  # Average over batch

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # For standard normal prior N(0, I)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_div = kl_div / original.size(0)  # Average over batch

    # Total loss
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div


def train_epoch(
    model: VAE,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    beta: float = 1.0
) -> Tuple[float, float, float]:
    """
    Train the VAE for one epoch.

    Args:
        model: The VAE model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        beta: Weight for KL divergence term

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_kl_div)
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_div_sum = 0.0

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    optimizer.zero_grad()

    for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        reconstructed, mu, logvar = model(images)
        total_loss, recon_loss, kl_div = vae_loss(reconstructed, targets, mu, logvar, beta)

        # Backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_div_sum += kl_div.item()

        progress_bar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_div.item():.4f}'
        })

    n_batches = len(dataloader)
    return total_loss_sum / n_batches, recon_loss_sum / n_batches, kl_div_sum / n_batches


def validate(
    model: VAE,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    beta: float = 1.0
) -> Tuple[float, float, float]:
    """
    Validate the VAE.

    Args:
        model: The VAE model
        dataloader: Validation dataloader
        device: Device to run validation on
        beta: Weight for KL divergence term

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_kl_div), or (None, None, None) if no validation data
    """
    if len(dataloader) == 0:
        return None, None, None

    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_div_sum = 0.0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation', leave=False)
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            reconstructed, mu, logvar = model(images)
            total_loss, recon_loss, kl_div = vae_loss(reconstructed, targets, mu, logvar, beta)

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_div_sum += kl_div.item()

            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_div.item():.4f}'
            })

    n_batches = len(dataloader)
    return total_loss_sum / n_batches, recon_loss_sum / n_batches, kl_div_sum / n_batches


def save_checkpoint(
    model: VAE,
    optimizer: optim.Optimizer,
    epoch: int,
    train_losses: List[float],
    val_losses: List[float],
    config: dict,
    checkpoint_path: str
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = args.device if args.device else get_device()
    print("=" * 60)
    print("MNIST Variational Autoencoder (VAE) Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Beta (KL weight): {args.beta}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Random seed: {args.seed}")
    print()

    # Create plot directory if needed
    if args.save_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
        print(f"Plots will be saved to: {args.plot_dir}/")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        train_split=args.train_split
    )
    print(f"  Training samples: {len(train_loader.dataset):,}")
    print(f"  Validation samples: {len(val_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    print()

    # Create model
    print("Initializing VAE...")
    model = VAE(latent_dim=args.latent_dim)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Compression ratio: {28*28}/{args.latent_dim} = {28*28/args.latent_dim:.1f}x")
    print()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Track losses
    train_losses = []
    val_losses = []
    train_recon_losses = []
    train_kl_divs = []
    best_val_loss = float('inf')

    # Training loop
    print("Starting training...")
    print("-" * 60)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, args.beta
        )
        train_losses.append(train_loss)
        train_recon_losses.append(train_recon)
        train_kl_divs.append(train_kl)

        # Validate
        val_loss, val_recon, val_kl = validate(model, val_loader, device, args.beta)
        if val_loss is not None:
            val_losses.append(val_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
        if val_loss is not None:
            print(f"  Val Loss:   {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  ✓ New best validation loss!")
        else:
            print(f"  Val Loss:   N/A (no validation set)")

        print()

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete! Total time: {total_time/60:.1f} minutes")
    if val_losses:
        print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print(f"Final training loss: {train_losses[-1]:.4f}")
    print()

    # Save final checkpoint
    config = {
        'latent_dim': args.latent_dim,
        'beta': args.beta,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'seed': args.seed
    }
    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        train_losses,
        val_losses,
        config,
        args.checkpoint_path
    )
    print()

    # Generate visualizations
    print("Generating visualizations...")

    # 1. Training curves
    if val_losses:
        print("  1. Training curves...")
        plot_path = os.path.join(args.plot_dir, 'vae_training_curves.png') if args.save_plots else None
        plot_training_curves(train_losses, val_losses, save_path=plot_path)

    # 2. Reconstruction samples
    print("  2. Reconstruction samples...")
    model.eval()
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images.to(device)
        reconstructed, _, _ = model(test_images)

    plot_path = os.path.join(args.plot_dir, 'vae_reconstructions.png') if args.save_plots else None
    plot_reconstructions(test_images, reconstructed, n_samples=10, save_path=plot_path)

    # 3. Random samples from prior
    print("  3. Sampling from prior N(0, I)...")
    samples = model.sample(num_samples=16, device=device)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(16):
        row = i // 8
        col = i % 8
        axes[row, col].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')
    plt.suptitle('Random Samples from Prior N(0, I)', fontsize=12)
    plt.tight_layout()
    if args.save_plots:
        plot_path = os.path.join(args.plot_dir, 'vae_samples.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    print()
    print("=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Checkpoint: {args.checkpoint_path}")
    if args.save_plots:
        print(f"  Plots: {args.plot_dir}/")


if __name__ == '__main__':
    main()
