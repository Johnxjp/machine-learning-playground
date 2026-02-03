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
- `train.py` - Training script for autoencoder (CLI version)
- `training.ipynb` - Training notebook for autoencoder
- `latent_exploration.ipynb` - Interactive notebook for exploring latent space
- `inference.py` - Inference API for trained autoencoder
- `classifier.py` - MNIST digit classifier with transfer learning support
- `classifier_data.py` - Data loading for classification
- `train_classifier.py` - Training script for classifier
- `test_shapes.py` - Shape verification tests
- `examples.sh` - Example training commands
- `CLASSIFIER_RESULTS.md` - Transfer learning results and comparisons

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

### Latent Space Exploration

Open `latent_exploration.ipynb` to interactively explore the latent space:

**Features:**
- Load trained models and analyze latent space statistics
- Generate images from random latent vectors (normal, uniform, or learned distributions)
- Input custom latent vectors and see decoder outputs
- Explore what individual latent dimensions control
- Perform latent space arithmetic (averaging, addition, subtraction)
- Compare real MNIST digits with generated images

This notebook is perfect for understanding what the autoencoder has learned and how the latent space is structured.

### MNIST Digit Classification (Transfer Learning)

Train a digit classifier using the pre-trained autoencoder encoder:

**From scratch:**
```bash
python train_classifier.py --epochs 10 --latent-dim 16
```

**Transfer learning with frozen encoder (fast):**
```bash
python train_classifier.py \
  --epochs 10 \
  --latent-dim 16 \
  --use-pretrained \
  --encoder-checkpoint test_checkpoint.pth \
  --freeze-encoder
```

**Transfer learning with fine-tuning (best accuracy):**
```bash
python train_classifier.py \
  --epochs 10 \
  --latent-dim 16 \
  --use-pretrained \
  --encoder-checkpoint test_checkpoint.pth
```

**Results (1 epoch):**
- From scratch: 94.14% test accuracy
- Transfer (frozen): 95.48% test accuracy (+1.34%, 65% faster)
- Transfer (fine-tuned): 97.34% test accuracy (+3.20%)

See `CLASSIFIER_RESULTS.md` for detailed comparison and analysis.

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
