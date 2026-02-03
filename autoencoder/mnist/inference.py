import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Handle both package import and standalone script execution
try:
    from .model import ConvAutoencoder
    from .data import get_mnist_dataloaders
    from .utils import get_device, plot_reconstructions
except ImportError:
    from model import ConvAutoencoder
    from data import get_mnist_dataloaders
    from utils import get_device, plot_reconstructions


class AutoencoderInference:
    """
    Inference wrapper for trained autoencoder.

    Provides methods for encoding, decoding, reconstruction, and latent space interpolation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        latent_dim: int = 12,
        device: Optional[str] = None
    ):
        """
        Initialize the inference wrapper.

        Args:
            checkpoint_path: Path to the saved model checkpoint
            latent_dim: Dimension of the latent space (default: 12)
            device: Device to run inference on (default: auto-detect)
        """
        self.latent_dim = latent_dim
        self.device = device if device else get_device()

        # Load model
        self.model = ConvAutoencoder(latent_dim=latent_dim)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")
        print(f"Final validation loss: {checkpoint['val_losses'][-1]:.6f}")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors.

        Args:
            images: Input images of shape (batch_size, 1, 28, 28)

        Returns:
            Latent vectors of shape (batch_size, latent_dim)
        """
        with torch.no_grad():
            images = images.to(self.device)
            latent = self.model.encode(images)
        return latent

    def decode_latent(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Args:
            latent_vectors: Latent vectors of shape (batch_size, latent_dim)

        Returns:
            Reconstructed images of shape (batch_size, 1, 28, 28)
        """
        with torch.no_grad():
            latent_vectors = latent_vectors.to(self.device)
            images = self.model.decode(latent_vectors)
        return images

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images through the full autoencoder.

        Args:
            images: Input images of shape (batch_size, 1, 28, 28)

        Returns:
            Reconstructed images of shape (batch_size, 1, 28, 28)
        """
        with torch.no_grad():
            images = images.to(self.device)
            reconstructed = self.model(images)
        return reconstructed

    def interpolate_latent(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        n_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two images in latent space.

        Args:
            img1: First image of shape (1, 1, 28, 28)
            img2: Second image of shape (1, 1, 28, 28)
            n_steps: Number of interpolation steps (default: 10)

        Returns:
            Interpolated images of shape (n_steps, 1, 28, 28)
        """
        with torch.no_grad():
            # Encode both images
            z1 = self.encode_images(img1)
            z2 = self.encode_images(img2)

            # Create interpolation weights
            alphas = torch.linspace(0, 1, n_steps).to(self.device)

            # Interpolate in latent space
            interpolated_latents = []
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                interpolated_latents.append(z_interp)

            interpolated_latents = torch.cat(interpolated_latents, dim=0)

            # Decode interpolated latents
            interpolated_images = self.decode_latent(interpolated_latents)

        return interpolated_images


def main():
    """
    Main function demonstrating inference usage.
    """
    print("Loading trained autoencoder for inference...\n")

    # Initialize inference wrapper
    inference = AutoencoderInference(
        checkpoint_path='autoencoder_checkpoint.pth',
        latent_dim=12
    )

    # Load test data
    _, _, test_loader = get_mnist_dataloaders(batch_size=128)

    # Get a batch of test images
    test_images, _ = next(iter(test_loader))

    print("\n1. Generating reconstructions...")
    reconstructed = inference.reconstruct(test_images)
    plot_reconstructions(
        test_images,
        reconstructed,
        n_samples=10,
        save_path='inference_results.png'
    )
    print("Saved reconstruction visualization to inference_results.png")

    print("\n2. Encoding images to latent space...")
    latent_vectors = inference.encode_images(test_images[:5])
    print(f"Latent vectors shape: {latent_vectors.shape}")
    print(f"Sample latent vector:\n{latent_vectors[0].cpu().numpy()}")

    print("\n3. Generating latent space interpolation...")
    # Pick two different digits
    img1 = test_images[0:1]
    img2 = test_images[5:6]

    interpolated = inference.interpolate_latent(img1, img2, n_steps=10)

    # Visualize interpolation
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(interpolated[i, 0].cpu().numpy(), cmap='gray')
        axes[i].axis('off')
    plt.suptitle('Latent Space Interpolation', fontsize=14)
    plt.tight_layout()
    plt.savefig('latent_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved interpolation visualization to latent_interpolation.png")

    print("\n4. Compression statistics:")
    original_size = 28 * 28  # pixels
    compressed_size = inference.latent_dim  # latent dimensions
    compression_ratio = original_size / compressed_size
    print(f"Original image size: {original_size} pixels")
    print(f"Compressed size: {compressed_size} dimensions")
    print(f"Compression ratio: {compression_ratio:.1f}x")

    print("\nInference complete!")


if __name__ == '__main__':
    main()
