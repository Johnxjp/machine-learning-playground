# Latent Space Exploration Guide

An interactive notebook for exploring the autoencoder's latent space by generating and manipulating latent vectors.

## Quick Start

```bash
jupyter notebook latent_exploration.ipynb
```

## Features

### 1. Load Trained Model
- Automatically loads checkpoint and configuration
- Displays training statistics and model info
- Works with any trained autoencoder checkpoint

### 2. Analyze Latent Space Statistics
- Encodes 1000+ real MNIST images
- Computes mean, std, min, max for each latent dimension
- Visualizes distribution of latent values
- Shows box plots per dimension

### 3. Generate from Random Distributions

**Three sampling methods:**

1. **Standard Normal**: `N(0, 1)` - Simple Gaussian sampling
2. **Learned Distribution**: Uses statistics from real data
3. **Uniform**: Uniform random values in a range

Example:
```python
z, images = generate_and_visualize_random(
    n_samples=16,
    distribution='learned',  # or 'normal', 'uniform'
    scale=1.0
)
```

### 4. Custom Latent Vectors

Input your own latent vector:
```python
custom_latent = [0.5, -0.3, 1.2, -0.8, 0.0, 0.7, -1.1, 0.4, ...]
decoded = decode_custom_vector(custom_latent)
```

Visualizes both the latent vector (bar chart) and decoded image.

### 5. Explore Individual Dimensions

Vary one dimension at a time to see what it controls:
```python
explore_dimension(
    dim_index=0,        # Which dimension to vary
    value_range=(-3, 3), # Range to sweep
    n_steps=10,         # Number of steps
    base_vector=None    # Starting point (default: zeros)
)
```

Shows a sequence of images as the dimension changes.

### 6. Latent Space Arithmetic

Perform operations on latent vectors:
- **Average**: `(z1 + z2) / 2` - Blend two images
- **Difference**: `z1 - z2` - Extract features
- **Sum**: `z1 + z2` - Combine features

```python
latent_arithmetic_demo()  # Shows examples
```

### 7. Compare Real vs Generated

Side-by-side comparison of:
- Real MNIST digits
- Generated images from random latent vectors

Helps evaluate the quality and diversity of the latent space.

## Use Cases

### Understanding the Model
- **What has it learned?** Generate random samples to see the range of digits it can produce
- **What do dimensions control?** Explore individual dimensions to understand their semantic meaning
- **Is the latent space smooth?** Check if similar latent vectors produce similar images

### Creative Exploration
- **Generate new digits**: Sample from the learned distribution
- **Digit morphing**: Interpolate between latent vectors (combine with inference.py)
- **Feature isolation**: Use arithmetic to isolate specific features

### Debugging
- **Check for mode collapse**: Are random samples diverse?
- **Understand reconstruction failures**: Why do some digits reconstruct poorly?
- **Analyze latent statistics**: Are all dimensions being used?

## Tips

### Getting Good Results

1. **Use the learned distribution**: `distribution='learned'` typically produces the most digit-like images
2. **Start with smaller scales**: If images look strange, try `scale=0.5`
3. **Explore systematically**: Use the dimension exploration to understand each dimension
4. **Compare with real data**: Always check against real MNIST to calibrate your intuition

### Common Observations

- **Some dimensions are more important**: Not all dimensions contribute equally
- **Smooth transitions**: Nearby latent vectors usually produce similar images
- **Out-of-distribution samples**: Random samples far from the learned distribution may produce artifacts
- **Semantic structure**: Some dimensions may control specific features (thickness, slant, style)

## Example Session

```python
# 1. Load model
# Run the first few cells to load checkpoint

# 2. Understand the data distribution
# See mean/std of latent dimensions from real images

# 3. Generate samples from learned distribution
z, imgs = generate_and_visualize_random(16, 'learned', scale=1.0)

# 4. Explore what dimension 0 controls
explore_dimension(0, value_range=(-2, 2), n_steps=10)

# 5. Try custom vectors
my_latent = [0.0] * latent_dim  # All zeros
my_latent[0] = 2.0  # Set first dimension
decoded = decode_custom_vector(my_latent)

# 6. Compare real vs generated
# Run the comparison cell to see quality
```

## Output Examples

Each function produces matplotlib visualizations:
- **Grid of images**: Generated digits
- **Bar charts**: Latent vector values
- **Box plots**: Distribution statistics
- **Image sequences**: Dimension sweeps
- **Side-by-side comparisons**: Real vs generated

## Requirements

All dependencies are already installed:
- PyTorch
- matplotlib
- numpy
- model.py, data.py, utils.py (from this package)

## Next Steps

After exploring the latent space:
1. Train with different `latent_dim` to see how it affects the space
2. Compare latent spaces from different training runs
3. Implement latent space interpolation (see inference.py)
4. Try clustering latent vectors by digit class
