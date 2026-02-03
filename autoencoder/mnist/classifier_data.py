"""
Data loading utilities for MNIST classification.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple


def get_mnist_classification_dataloaders(
    batch_size: int = 128,
    data_dir: str = "/Users/johnlingi/programming/machine_learning/datasets/mnist/",
    train_split: float = 0.9,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for MNIST classification.

    Downloads MNIST dataset if not already present. Splits the 60k training samples
    into train/val split. Uses the 10k test samples separately.

    Args:
        batch_size: Number of samples per batch (default: 128)
        data_dir: Directory to store/load MNIST data (default: datasets/mnist/)
        train_split: Fraction of training data to use for training vs validation (default: 0.9)
        num_workers: Number of worker processes for data loading (default: 0)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Define transform: convert to tensor (normalizes to [0, 1])
    transform = transforms.ToTensor()

    # Download and load MNIST training data
    mnist_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download and load MNIST test data
    mnist_test = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Split training data into train and validation sets
    train_size = int(train_split * len(mnist_train))
    val_size = len(mnist_train) - train_size

    mnist_train_split, mnist_val_split = random_split(
        mnist_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )

    # Create dataloaders
    train_loader = DataLoader(
        mnist_train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        mnist_val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
