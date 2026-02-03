# Convolutional Autoencoder for MNIST

A PyTorch implementation of a convolutional autoencoder that compresses 28×28 MNIST images to a 12-dimensional latent space and reconstructs them.

## Architecture

**Compression ratio:** 784 pixels → 12 dimensions ≈ 65× compression

### Encoder (28×28×1 → latent_dim)
- Conv2d: 1→16 channels, 3×3, stride=2, padding=1 → 14×14×16
- Conv2d: 16→32 channels, 3×3, stride=2, padding=1 → 7×7×32
- Conv2d: 32→64 channels, 3×3, stride=2, padding=1 → 4×4×64
- Flatten + Linear: 1024 → latent_dim

### Decoder (latent_dim → 28×28×1)
- Linear: latent_dim → 1024
- ConvTranspose2d: 64→32, 7×7
- ConvTranspose2d: 32→16, 14×14
- ConvTranspose2d: 16→1, 28×28

## Files

- `model.py` - Encoder, Decoder, and ConvAutoencoder classes
- `data.py` - MNIST data loading utilities
- `utils.py` - Visualization and helper functions
- `train.py` - Training script (CLI version)
- `training.ipynb` - Training notebook with experiments
- `inference.py` - Inference API for trained models
- `test_shapes.py` - Shape verification tests
- `examples.sh` - Example training commands

## Quick Start

You can train the model using either the Jupyter notebook or the Python script.

### Option 1: Training Script (Recommended)

Run with default settings:
```bash
uv run python train.py
```

Or customize the training:
```bash
uv run python train.py --latent-dim 16 --epochs 20 --batch-size 256 --lr 0.0005 --save-plots
```

Available options:
- `--latent-dim`: Latent space dimension (default: 12)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--save-plots`: Save all plots to disk
- `--plot-dir`: Directory for plots (default: plots)
- `--checkpoint-path`: Path to save checkpoint (default: autoencoder_checkpoint.pth)
- `--device`: Force specific device (cpu/cuda/mps, default: auto-detect)
- `--seed`: Random seed for reproducibility (default: 42)

See `examples.sh` for more usage examples.

### Option 2: Training Notebook

Open `training.ipynb` in Jupyter and run all cells:

```python
# Configuration
LATENT_DIM = 12
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10

# Training completes in ~5-10 minutes on MPS
```

The notebook will:
1. Download MNIST dataset
2. Train the autoencoder for 10 epochs
3. Plot training/validation curves
4. Show reconstruction samples
5. Visualize latent space with PCA
6. Save checkpoint to `autoencoder_checkpoint.pth`

### Inference

Run the inference script after training:

```bash
python inference.py
```

This will:
1. Load the trained model
2. Generate reconstruction visualizations
3. Show latent space interpolation between digits
4. Save results to PNG files

## Configuration

The latent dimension is configurable throughout:

```python
# In training.ipynb
LATENT_DIM = 12  # Try 8, 16, 32, etc.

# In inference.py
inference = AutoencoderInference(
    checkpoint_path='autoencoder_checkpoint.pth',
    latent_dim=12
)
```

## Dependencies

- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn (for PCA/t-SNE visualization)
- tqdm

## Training Details

- **Optimizer:** Adam (lr=0.001)
- **Loss:** MSE (mean squared error)
- **Device:** MPS for Mac GPU acceleration
- **Data split:** 54k train / 6k validation / 10k test
- **Expected time:** 5-10 minutes for 10 epochs

## Expected Outputs

After training, you should have:
- `autoencoder_checkpoint.pth` - Trained model weights
- `inference_results.png` - Reconstruction visualization
- `latent_interpolation.png` - Latent space interpolation

## Success Criteria

- Final MSE loss < 0.01 (lower is better)
- Reconstructed digits are clearly recognizable
- Smooth interpolation in latent space
- Similar train/validation losses (no overfitting)
