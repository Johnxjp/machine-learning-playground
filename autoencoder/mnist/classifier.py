"""
MNIST Digit Classifier with optional pre-trained encoder.

Supports transfer learning by loading weights from a pre-trained autoencoder
and using the encoder as a feature extractor.
"""

import torch
import torch.nn as nn
from typing import Optional
from model import Encoder


class MNISTClassifier(nn.Module):
    """
    MNIST digit classifier that can optionally use a pre-trained encoder.

    Architecture:
        - Encoder (optional pre-trained from autoencoder)
        - Classifier head with 2 hidden layers
        - Final softmax layer with 10 outputs (one per digit)
    """

    def __init__(
        self,
        latent_dim: int = 12,
        hidden_dim: int = 128,
        num_classes: int = 10,
        use_pretrained_encoder: bool = False,
        encoder_checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = False
    ):
        """
        Initialize the classifier.

        Args:
            latent_dim: Dimension of the latent space (must match encoder)
            hidden_dim: Dimension of hidden layers in classifier head
            num_classes: Number of output classes (default: 10 for digits 0-9)
            use_pretrained_encoder: Whether to use pre-trained encoder weights
            encoder_checkpoint_path: Path to autoencoder checkpoint (required if use_pretrained_encoder=True)
            freeze_encoder: Whether to freeze encoder weights during training
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder

        # Create encoder
        self.encoder = Encoder(latent_dim=latent_dim)

        # Load pre-trained weights if requested
        if use_pretrained_encoder:
            if encoder_checkpoint_path is None:
                raise ValueError("encoder_checkpoint_path must be provided when use_pretrained_encoder=True")
            self._load_pretrained_encoder(encoder_checkpoint_path)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

        # Initialize classifier weights
        self._init_classifier_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and classifier.

        Args:
            x: Input images of shape (batch_size, 1, 28, 28)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Encode
        latent = self.encoder(x)

        # Classify
        logits = self.classifier(latent)

        return logits

    def _load_pretrained_encoder(self, checkpoint_path: str) -> None:
        """
        Load pre-trained encoder weights from autoencoder checkpoint.

        Args:
            checkpoint_path: Path to autoencoder checkpoint file
        """
        print(f"Loading pre-trained encoder from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Verify latent dimension matches
        checkpoint_latent_dim = checkpoint['config']['latent_dim']
        if checkpoint_latent_dim != self.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: model has latent_dim={self.latent_dim}, "
                f"but checkpoint has latent_dim={checkpoint_latent_dim}. "
                f"Please set --latent-dim {checkpoint_latent_dim} to match the checkpoint."
            )

        # Extract encoder weights from checkpoint
        encoder_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('encoder.'):
                # Remove 'encoder.' prefix
                new_key = key[8:]
                encoder_state_dict[new_key] = value

        # Load weights
        self.encoder.load_state_dict(encoder_state_dict)
        print(f"✓ Loaded encoder weights (latent_dim={self.encoder.latent_dim})")

    def _freeze_encoder(self) -> None:
        """Freeze encoder weights so they are not updated during training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder weights frozen")

    def _init_classifier_weights(self) -> None:
        """Initialize classifier weights using Kaiming normal initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent feature vector from encoder.

        Args:
            x: Input images of shape (batch_size, 1, 28, 28)

        Returns:
            Latent vectors of shape (batch_size, latent_dim)
        """
        return self.encoder(x)

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights to allow fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
        print("✓ Encoder weights unfrozen")

    def count_parameters(self) -> dict:
        """
        Count total, trainable, and frozen parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'encoder': encoder_params,
            'classifier': classifier_params
        }


class SimpleClassifier(nn.Module):
    """
    Simple baseline classifier without pre-trained encoder.

    Uses the same architecture but trains from scratch.
    """

    def __init__(
        self,
        latent_dim: int = 12,
        hidden_dim: int = 128,
        num_classes: int = 10
    ):
        """
        Initialize the simple classifier.

        Args:
            latent_dim: Dimension of the encoder output
            hidden_dim: Dimension of hidden layers in classifier head
            num_classes: Number of output classes (default: 10)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Same architecture as MNISTClassifier
        self.encoder = Encoder(latent_dim=latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

        # Initialize all weights
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images of shape (batch_size, 1, 28, 28)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits

    def _init_weights(self) -> None:
        """Initialize all weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self) -> dict:
        """Count parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': 0
        }
