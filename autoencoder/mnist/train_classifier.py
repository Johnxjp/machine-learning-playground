"""
Training script for MNIST Digit Classifier.

Supports both training from scratch and transfer learning from pre-trained autoencoder.

Examples:
    # Train from scratch
    python train_classifier.py --epochs 10

    # Transfer learning with frozen encoder
    python train_classifier.py --use-pretrained --encoder-checkpoint autoencoder_checkpoint.pth --freeze-encoder

    # Transfer learning with fine-tuning
    python train_classifier.py --use-pretrained --encoder-checkpoint autoencoder_checkpoint.pth
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
    from .classifier import MNISTClassifier, SimpleClassifier
    from .classifier_data import get_mnist_classification_dataloaders
    from .utils import get_device
except ImportError:
    from classifier import MNISTClassifier, SimpleClassifier
    from classifier_data import get_mnist_classification_dataloaders
    from utils import get_device


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MNIST digit classifier with optional transfer learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model hyperparameters
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=12,
        help='Dimension of the latent space (must match encoder if using pre-trained)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='Dimension of hidden layers in classifier head'
    )

    # Transfer learning
    parser.add_argument(
        '--use-pretrained',
        action='store_true',
        help='Use pre-trained encoder from autoencoder'
    )
    parser.add_argument(
        '--encoder-checkpoint',
        type=str,
        default='test_checkpoint.pth',
        help='Path to autoencoder checkpoint for pre-trained encoder'
    )
    parser.add_argument(
        '--freeze-encoder',
        action='store_true',
        help='Freeze encoder weights (only train classifier head)'
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
        default='classifier_checkpoint.pth',
        help='Path to save model checkpoint'
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
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The classifier model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()
        accuracy = 100.0 * correct / total

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, float]:
    """
    Validate the model.

    Args:
        model: The classifier model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run validation on

    Returns:
        Tuple of (average_loss, accuracy) or (None, None) if no validation data
    """
    if len(dataloader) == 0:
        return None, None

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation', leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Calculate accuracy
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()
            accuracy = 100.0 * correct / total

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.2f}%'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_losses: List[float],
    train_accuracies: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    config: dict,
    checkpoint_path: str
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
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
    print("MNIST Digit Classifier Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Random seed: {args.seed}")

    if args.use_pretrained:
        print(f"\nTransfer Learning:")
        print(f"  Using pre-trained encoder: {args.encoder_checkpoint}")
        print(f"  Freeze encoder: {args.freeze_encoder}")
    else:
        print(f"\nTraining from scratch (no pre-trained encoder)")
    print()

    # Load data
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_classification_dataloaders(
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
    if args.use_pretrained:
        model = MNISTClassifier(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            use_pretrained_encoder=True,
            encoder_checkpoint_path=args.encoder_checkpoint,
            freeze_encoder=args.freeze_encoder
        )
    else:
        model = SimpleClassifier(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim
        )

    model = model.to(device)

    # Print model info
    param_counts = model.count_parameters()
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    if param_counts['frozen'] > 0:
        print(f"  Frozen parameters: {param_counts['frozen']:,}")
    print()

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Track metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    # Training loop
    print("Starting training...")
    print("-" * 60)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if val_loss is not None:
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        if val_loss is not None:
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  ✓ New best validation accuracy!")
        else:
            print(f"  Val Loss:   N/A (no validation set)")

        print()

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete! Total time: {total_time/60:.1f} minutes")
    if val_accuracies:
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
    else:
        print(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
    print()

    # Save final checkpoint
    config = {
        'latent_dim': args.latent_dim,
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'seed': args.seed,
        'use_pretrained': args.use_pretrained,
        'freeze_encoder': args.freeze_encoder
    }
    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
        config,
        args.checkpoint_path
    )
    print()

    # Test set evaluation
    print("Evaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print()

    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
