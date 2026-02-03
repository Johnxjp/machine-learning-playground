import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Optional


def get_device() -> str:
    """
    Get the best available device for PyTorch.

    Returns:
        'mps' for Mac GPU, 'cuda' for NVIDIA GPU, or 'cpu'
    """
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def plot_reconstructions(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    n_samples: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot original and reconstructed images side-by-side.

    Args:
        original_images: Original images of shape (batch_size, 1, 28, 28)
        reconstructed_images: Reconstructed images of shape (batch_size, 1, 28, 28)
        n_samples: Number of samples to display (default: 10)
        save_path: Optional path to save the figure
    """
    n_samples = min(n_samples, original_images.size(0))

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3))

    for i in range(n_samples):
        # Original images
        axes[0, i].imshow(original_images[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # Reconstructed images
        axes[1, i].imshow(reconstructed_images[i, 0].cpu().detach().numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the figure
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_latent_space(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    method: str = 'pca',
    n_samples: int = 5000,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the latent space using PCA or t-SNE.

    Args:
        model: Trained autoencoder model
        dataloader: DataLoader containing MNIST images
        device: Device to run inference on
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_samples: Number of samples to visualize (default: 5000)
        save_path: Optional path to save the figure
    """
    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for images, _ in dataloader:
            if len(latent_vectors) * images.size(0) >= n_samples:
                break

            images = images.to(device)
            z = model.encode(images)
            latent_vectors.append(z.cpu().numpy())

            # Get original labels from the wrapped dataset
            batch_labels = []
            for i in range(images.size(0)):
                idx = len(labels) + i
                if idx < len(dataloader.dataset.mnist_dataset):
                    _, label = dataloader.dataset.mnist_dataset[idx]
                    batch_labels.append(label)
            labels.extend(batch_labels)

    # Concatenate all latent vectors
    latent_vectors = np.vstack(latent_vectors)[:n_samples]
    labels = np.array(labels[:n_samples])

    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(latent_vectors)
        title = f'PCA Visualization of Latent Space'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced = reducer.fit_transform(latent_vectors)
        title = f't-SNE Visualization of Latent Space'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
