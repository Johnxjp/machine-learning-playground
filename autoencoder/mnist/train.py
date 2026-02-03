"""
Training script for MNIST Convolutional Autoencoder.

Run with default settings:
    python train.py

Run with custom hyperparameters:
    python train.py --latent-dim 16 --epochs 20 --batch-size 256 --lr 0.0005
"""

import argparse
import os
import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    from .model import ConvAutoencoder
    from .data import get_mnist_dataloaders
    from .utils import get_device, plot_reconstructions, plot_training_curves, visualize_latent_space
except ImportError:
    from model import ConvAutoencoder
    from data import get_mnist_dataloaders
    from utils import get_device, plot_reconstructions, plot_training_curves, visualize_latent_space


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a convolutional autoencoder on MNIST',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model hyperparameters
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=12,
        help='Dimension of the latent space'
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
        default='autoencoder_checkpoint.pth',
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


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The autoencoder model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    optimizer.zero_grad()
    for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        reconstructed = model(images)
        loss = criterion(reconstructed, targets)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Validate the model.

    Args:
        model: The autoencoder model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run validation on

    Returns:
        Average validation loss, or None if no validation data
    """
    if len(dataloader) == 0:
        return None

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation', leave=False)
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, targets)

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(dataloader)


def save_checkpoint(
    model: nn.Module,
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
    print("MNIST Convolutional Autoencoder Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Latent dimension: {args.latent_dim}")
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
    print("Initializing model...")
    model = ConvAutoencoder(latent_dim=args.latent_dim)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Compression ratio: {28*28}/{args.latent_dim} = {28*28/args.latent_dim:.1f}x")
    print()

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Track losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Training loop
    print("Starting training...")
    print("-" * 60)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        if val_loss is not None:
            val_losses.append(val_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f}")
        if val_loss is not None:
            print(f"  Val Loss:   {val_loss:.6f}")

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
        print(f"Best validation loss: {best_val_loss:.6f}")
    else:
        print(f"Final training loss: {train_losses[-1]:.6f}")
    print()

    # Save final checkpoint
    config = {
        'latent_dim': args.latent_dim,
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
        plot_path = os.path.join(args.plot_dir, 'training_curves.png') if args.save_plots else None
        plot_training_curves(train_losses, val_losses, save_path=plot_path)
    else:
        print("  1. Training curve (no validation)...")
        # Plot only training loss
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Training Loss', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        if args.save_plots:
            plot_path = os.path.join(args.plot_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()

    # 2. Reconstruction samples
    print("  2. Reconstruction samples...")
    model.eval()
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images.to(device)
        reconstructed = model(test_images)

    plot_path = os.path.join(args.plot_dir, 'reconstructions.png') if args.save_plots else None
    plot_reconstructions(test_images, reconstructed, n_samples=10, save_path=plot_path)

    # 3. Latent space visualization
    print("  3. Latent space (PCA)...")
    plot_path = os.path.join(args.plot_dir, 'latent_space_pca.png') if args.save_plots else None
    visualize_latent_space(
        model,
        test_loader,
        device,
        method='pca',
        n_samples=5000,
        save_path=plot_path
    )

    print()
    print("=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Checkpoint: {args.checkpoint_path}")
    if args.save_plots:
        print(f"  Plots: {args.plot_dir}/")
    print(f"\nTo run inference:")
    print(f"  python inference.py")


if __name__ == '__main__':
    main()
