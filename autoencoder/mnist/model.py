import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional encoder that compresses 28×28 MNIST images to latent vectors.

    Architecture:
        - Conv2d: 1→16 channels, 3×3, stride=2, padding=1 → 14×14×16
        - ReLU + BatchNorm2d
        - Conv2d: 16→32 channels, 3×3, stride=2, padding=1 → 7×7×32
        - ReLU + BatchNorm2d
        - Conv2d: 32→64 channels, 3×3, stride=2, padding=1 → 4×4×64
        - ReLU + BatchNorm2d
        - Flatten: 4×4×64 = 1024
        - Linear: 1024 → latent_dim
    """

    def __init__(self, latent_dim: int = 12):
        """
        Initialize the encoder.

        Args:
            latent_dim: Dimension of the latent space (default: 12)
        """
        super().__init__()
        self.latent_dim = latent_dim

        # First convolutional block: 1 → 16 channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        # Second convolutional block: 16 → 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # Third convolutional block: 32 → 64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        # Fully connected layer: 1024 → latent_dim
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input images to latent vectors.

        Args:
            x: Input images of shape (batch_size, 1, 28, 28)

        Returns:
            Latent vectors of shape (batch_size, latent_dim)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Flatten and project to latent space
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 1024)
        x = self.fc(x)

        return x


class Decoder(nn.Module):
    """
    Convolutional decoder that reconstructs 28×28 MNIST images from latent vectors.

    Architecture:
        - Linear: latent_dim → 1024
        - Reshape: 1024 → 64×4×4
        - ConvTranspose2d: 64→32, 3×3, stride=2, padding=1, output_padding=0 → 7×7×32
        - ReLU + BatchNorm2d
        - ConvTranspose2d: 32→16, 3×3, stride=2, padding=1, output_padding=1 → 14×14×16
        - ReLU + BatchNorm2d
        - ConvTranspose2d: 16→1, 3×3, stride=2, padding=1, output_padding=1 → 28×28×1
    """

    def __init__(self, latent_dim: int = 12):
        """
        Initialize the decoder.

        Args:
            latent_dim: Dimension of the latent space (default: 12)
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Fully connected layer: latent_dim → 1024
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)

        # First transposed convolutional block: 64 → 32 channels
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        # Second transposed convolutional block: 32 → 16 channels
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        # Third transposed convolutional block: 16 → 1 channel
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to reconstructed images.

        Args:
            z: Latent vectors of shape (batch_size, latent_dim)

        Returns:
            Reconstructed images of shape (batch_size, 1, 28, 28)
        """
        # Project latent to feature maps
        x = self.fc(z)
        x = x.view(x.size(0), 64, 4, 4)  # Reshape to (batch_size, 64, 4, 4)

        # First deconv block
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second deconv block
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Third deconv block (no activation)
        x = self.deconv3(x)

        return x


class ConvAutoencoder(nn.Module):
    """
    Complete convolutional autoencoder for MNIST images.

    Compresses 28×28 grayscale images to a configurable latent space and reconstructs them.
    Uses MSE loss for training.
    """

    def __init__(self, latent_dim: int = 12):
        """
        Initialize the autoencoder.

        Args:
            latent_dim: Dimension of the latent space (default: 12)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete autoencoder.

        Args:
            x: Input images of shape (batch_size, 1, 28, 28)

        Returns:
            Reconstructed images of shape (batch_size, 1, 28, 28)
        """
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors.

        Args:
            x: Input images of shape (batch_size, 1, 28, 28)

        Returns:
            Latent vectors of shape (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Args:
            z: Latent vectors of shape (batch_size, latent_dim)

        Returns:
            Reconstructed images of shape (batch_size, 1, 28, 28)
        """
        return self.decoder(z)

    def _init_weights(self) -> None:
        """
        Initialize weights using Kaiming normal initialization for convolutional layers.
        Follows ResNet initialization pattern.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
