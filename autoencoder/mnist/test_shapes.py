"""
Quick test to verify model shapes are correct.
"""
import torch
from model import ConvAutoencoder

def test_model_shapes():
    """Test that the autoencoder produces correct output shapes."""
    print("Testing ConvAutoencoder shape transformations...\n")

    # Create model
    latent_dim = 12
    model = ConvAutoencoder(latent_dim=latent_dim)
    model.eval()

    # Create dummy input (batch of 4 MNIST images)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 28, 28)

    print(f"Input shape: {dummy_input.shape}")

    # Test encoding
    with torch.no_grad():
        latent = model.encode(dummy_input)
        print(f"Latent shape: {latent.shape}")
        assert latent.shape == (batch_size, latent_dim), f"Expected ({batch_size}, {latent_dim}), got {latent.shape}"

        # Test decoding
        reconstructed = model.decode(latent)
        print(f"Reconstructed shape: {reconstructed.shape}")
        assert reconstructed.shape == dummy_input.shape, f"Expected {dummy_input.shape}, got {reconstructed.shape}"

        # Test full forward pass
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == dummy_input.shape, f"Expected {dummy_input.shape}, got {output.shape}"

    print("\n✓ All shape tests passed!")
    print(f"✓ Compression ratio: {28*28}/{latent_dim} = {28*28/latent_dim:.1f}x")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

if __name__ == '__main__':
    test_model_shapes()
