"""
Convolutional Autoencoder for MNIST

A PyTorch implementation that compresses 28×28 MNIST images to a configurable
latent space (default: 12 dimensions) achieving ~65× compression.
"""

from .model import ConvAutoencoder, Encoder, Decoder
from .data import get_mnist_dataloaders, MNISTAutoencoderDataset
from .utils import (
    get_device,
    plot_reconstructions,
    plot_training_curves,
    visualize_latent_space
)
from .inference import AutoencoderInference

__all__ = [
    'ConvAutoencoder',
    'Encoder',
    'Decoder',
    'get_mnist_dataloaders',
    'MNISTAutoencoderDataset',
    'get_device',
    'plot_reconstructions',
    'plot_training_curves',
    'visualize_latent_space',
    'AutoencoderInference',
]
