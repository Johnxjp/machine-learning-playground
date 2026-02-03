# VAE Latent Space Explorer - Interactive App

An interactive Streamlit application for exploring the VAE latent space by sampling from multidimensional Gaussians.

## Features

- **Sample from N(μ, σ²)**: Control mean and standard deviation for each latent dimension independently
- **Sample from N(0, I)**: Sample from the prior distribution (standard normal)
- **Deterministic Mode**: Set exact latent values without randomness
- **Real-time Visualization**: See generated images update as you adjust sliders
- **Latent Vector Inspection**: View the sampled latent vector as a bar chart
- **Multiple Dimensions**: Works with any latent dimension size

## Installation

```bash
pip install streamlit torch torchvision matplotlib numpy
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
streamlit run app.py
```

The app will look for `vae_high_kl.pth` in the current directory by default.

### Specify Custom Checkpoint

You can also specify the checkpoint path in the app's sidebar text input field.

## How to Use

### 1. Sample from N(μ, σ²) Mode

This mode lets you control each latent dimension's distribution independently:

- **μ sliders**: Set the mean for each dimension (-3.0 to 3.0)
- **σ sliders**: Set the standard deviation for each dimension (0.0 to 2.0)
- The app samples `z ~ N(μ, σ²)` and decodes it to an image

**Example Use Cases:**
- Set μ₁ = 2.0, μ₂ = -1.5 (others at 0) to explore a specific region
- Set σ = 0 for all dimensions to get deterministic outputs
- Increase σ for certain dimensions to add variability

### 2. Sample from N(0, I) Mode

Sample from the prior distribution (standard normal):
- All dimensions use μ = 0, σ = 1
- Click "🎲 Resample" to generate new random samples
- This is what the VAE was trained to match (via KL divergence)

### 3. Custom Deterministic z Mode

Set exact latent vector values:
- Direct control over each z dimension
- No randomness (σ = 0)
- Useful for systematic exploration

## Understanding Multidimensional Distributions

Each latent dimension has its own Gaussian distribution:

```
z₁ ~ N(μ₁, σ₁²)
z₂ ~ N(μ₂, σ₂²)
...
zₙ ~ N(μₙ, σₙ²)
```

The dimensions are **independent**, so the full distribution is:

```
z ~ N(μ, Σ)
```

where Σ is a diagonal covariance matrix (since dimensions are independent).

## UI Components

### Main Display
- **Generated Image**: Shows the decoded image from the sampled latent vector
- **Latent Vector Chart**: Bar chart visualization of the sampled z values
- **Numeric Values**: Exact z values for each dimension

### Sidebar
- **Configuration**: Checkpoint path and device selection
- **Model Info**: Latent dimension, beta value, training epochs
- **Tips**: Helpful information about different modes

### Control Buttons
- **Reset to N(0,1)**: Reset all μ to 0 and σ to 1
- **Randomize μ**: Set random μ values (keeping σ = 1)
- **Resample** (N(0,I) mode): Generate new random sample

## Exploring the Effects of High KL Weight

When using a model trained with high β (KL weight):

1. **Tight Latent Space**: The latent space is more regular and closer to N(0, I)
2. **Smooth Interpolation**: Small changes in μ produce smooth transitions
3. **Lower Quality**: Reconstructions may be blurrier (high β sacrifices reconstruction for regularization)
4. **Better Generation**: Samples from N(0, I) tend to look more realistic

**Experiment:**
- Train with β = 0.1: Sharp reconstructions, irregular latent space
- Train with β = 1.0: Balanced
- Train with β = 10.0: Regular latent space, blurrier outputs

## Tips for Exploration

1. **Start with N(0, I)**: See what the model naturally generates
2. **Adjust Single Dimensions**: Change μ for one dimension at a time to see its effect
3. **Increase σ**: Add uncertainty to specific dimensions
4. **Compare Models**: Load checkpoints with different β values to see differences
5. **Systematic Exploration**: Use deterministic mode to map out the latent space

## Example Workflow

```bash
# Train models with different β values
python train.py --beta 0.1 --epochs 10 --checkpoint-path vae_low_kl.pth
python train.py --beta 1.0 --epochs 10 --checkpoint-path vae_normal.pth
python train.py --beta 10.0 --epochs 10 --checkpoint-path vae_high_kl.pth

# Explore each model
streamlit run app.py
# Then change checkpoint path in the sidebar to compare
```

## Troubleshooting

**App won't start:**
- Make sure streamlit is installed: `pip install streamlit`
- Check Python version (3.8+)

**Model not found:**
- Train a model first: `python train.py`
- Check the checkpoint path in the sidebar

**Slow generation:**
- Use CPU instead of MPS if on Mac (MPS can be slower for small models)
- The first generation is slower (model loading), subsequent ones are fast

## Technical Details

The app uses:
- **Streamlit**: For the interactive web interface
- **PyTorch**: For model inference
- **Matplotlib**: For visualization
- **Session State**: To maintain slider values between reruns

The sampling follows:
```python
z = μ + σ · ε, where ε ~ N(0, I)
```

This is the reparameterization trick used during VAE training.
