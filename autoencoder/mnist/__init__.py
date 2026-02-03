"""
Convolutional Autoencoder for MNIST

A PyTorch implementation that compresses 28×28 MNIST images to a configurable
latent space (default: 12 dimensions) achieving ~65× compression.

Also includes MNIST digit classifier with transfer learning support.
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
from .classifier import MNISTClassifier, SimpleClassifier
from .classifier_data import get_mnist_classification_dataloaders

__all__ = [
    # Autoencoder
    'ConvAutoencoder',
    'Encoder',
    'Decoder',
    'get_mnist_dataloaders',
    'MNISTAutoencoderDataset',
    'AutoencoderInference',
    # Classifier
    'MNISTClassifier',
    'SimpleClassifier',
    'get_mnist_classification_dataloaders',
    # Utils
    'get_device',
    'plot_reconstructions',
    'plot_training_curves',
    'visualize_latent_space',
]
