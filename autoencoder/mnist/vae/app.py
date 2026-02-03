"""
Interactive Streamlit app for exploring VAE latent space.

Run with:
    streamlit run app.py -- --checkpoint vae_high_kl.pth
"""

import argparse
import os
import sys

import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add current and parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Try different import strategies
try:
    from vae.model import VAE
except ImportError:
    try:
        from model import VAE
    except ImportError:
        import model as model_module
        VAE = model_module.VAE

try:
    from utils import get_device
except ImportError:
    import utils as utils_module
    get_device = utils_module.get_device


@st.cache_resource
def load_model(checkpoint_path: str, device: str):
    """Load VAE model from checkpoint (cached)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    latent_dim = config['latent_dim']

    model = VAE(latent_dim=latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def sample_and_decode(model: VAE, mu_values: np.ndarray, sigma_values: np.ndarray, device: str):
    """
    Sample from Gaussian with specified mu and sigma for each dimension.

    Args:
        model: VAE model
        mu_values: Array of means for each latent dimension
        sigma_values: Array of standard deviations for each latent dimension
        device: Device to run on

    Returns:
        Generated image tensor
    """
    # Convert to torch tensors
    mu = torch.tensor(mu_values, dtype=torch.float32, device=device).unsqueeze(0)
    sigma = torch.tensor(sigma_values, dtype=torch.float32, device=device).unsqueeze(0)

    # Sample: z = mu + sigma * eps, where eps ~ N(0, 1)
    eps = torch.randn_like(mu)
    z = mu + sigma * eps

    # Decode
    with torch.no_grad():
        image = model.decode(z)

    return image.cpu().squeeze().numpy(), z.cpu().squeeze().numpy()


def main():
    st.set_page_config(
        page_title="VAE Latent Space Explorer",
        page_icon="🎨",
        layout="wide"
    )

    st.title("🎨 VAE Latent Space Explorer")
    st.markdown("Explore the VAE latent space by sampling from multidimensional Gaussians")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Checkpoint path
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint path",
        value="vae_high_kl.pth",
        help="Path to the VAE checkpoint file"
    )

    # Device selection
    device_option = st.sidebar.selectbox(
        "Device",
        ["Auto-detect", "cpu", "mps", "cuda"]
    )

    if device_option == "Auto-detect":
        device = get_device()
    else:
        device = device_option

    st.sidebar.info(f"Using device: **{device}**")

    # Load model
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.info("Please train a model first or provide a valid checkpoint path.")
        return

    try:
        model, config = load_model(checkpoint_path, device)
        latent_dim = config['latent_dim']

        st.sidebar.success(f"✓ Model loaded")
        st.sidebar.markdown(f"**Latent dimension:** {latent_dim}")
        st.sidebar.markdown(f"**Beta (KL weight):** {config.get('beta', 'N/A')}")
        st.sidebar.markdown(f"**Trained epochs:** {config.get('epochs', 'N/A')}")

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Main content
    st.markdown("---")

    # Control mode
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Sampling Mode")
        mode = st.radio(
            "Choose sampling mode:",
            ["Sample from N(μ, σ²)", "Sample from N(0, I)", "Custom deterministic z"],
            help="Control how to sample from the latent space"
        )

        if mode == "Sample from N(0, I)":
            st.info("Sampling from the prior distribution N(0, I)")
            resample_button = st.button("🎲 Resample", key="resample")

        st.markdown("---")
        st.subheader("Latent Space Controls")

    # Initialize session state for sliders if not exists
    if 'mu_values' not in st.session_state:
        st.session_state.mu_values = np.zeros(latent_dim)
        st.session_state.sigma_values = np.ones(latent_dim)

    # Sliders for each latent dimension
    mu_values = np.zeros(latent_dim)
    sigma_values = np.ones(latent_dim)

    with col2:
        if mode == "Sample from N(μ, σ²)":
            st.markdown("**Adjust μ and σ for each dimension:**")

            # Preset buttons
            preset_col1, preset_col2 = st.columns(2)
            with preset_col1:
                if st.button("Reset to N(0,1)"):
                    st.session_state.mu_values = np.zeros(latent_dim)
                    st.session_state.sigma_values = np.ones(latent_dim)
                    st.rerun()

            with preset_col2:
                if st.button("Randomize μ"):
                    st.session_state.mu_values = np.random.randn(latent_dim)
                    st.rerun()

            # Create expander for each dimension
            for i in range(latent_dim):
                with st.expander(f"Dimension {i+1}", expanded=(i < 3)):
                    mu_values[i] = st.slider(
                        f"μ_{i+1}",
                        min_value=-3.0,
                        max_value=3.0,
                        value=float(st.session_state.mu_values[i]),
                        step=0.1,
                        key=f"mu_{i}",
                        help=f"Mean for latent dimension {i+1}"
                    )

                    sigma_values[i] = st.slider(
                        f"σ_{i+1}",
                        min_value=0.0,
                        max_value=2.0,
                        value=float(st.session_state.sigma_values[i]),
                        step=0.05,
                        key=f"sigma_{i}",
                        help=f"Standard deviation for latent dimension {i+1}"
                    )

            # Update session state
            st.session_state.mu_values = mu_values
            st.session_state.sigma_values = sigma_values

        elif mode == "Sample from N(0, I)":
            mu_values = np.zeros(latent_dim)
            sigma_values = np.ones(latent_dim)

        elif mode == "Custom deterministic z":
            st.markdown("**Set z values directly (no sampling):**")

            for i in range(latent_dim):
                with st.expander(f"Dimension {i+1}", expanded=(i < 3)):
                    mu_values[i] = st.slider(
                        f"z_{i+1}",
                        min_value=-3.0,
                        max_value=3.0,
                        value=0.0,
                        step=0.1,
                        key=f"z_{i}",
                        help=f"Value for latent dimension {i+1}"
                    )

            sigma_values = np.zeros(latent_dim)  # No sampling

    # Generate image
    with col1:
        st.subheader("Generated Image")

        if mode == "Custom deterministic z":
            # Directly use mu_values as z
            z = torch.tensor(mu_values, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                image = model.decode(z)
            image = image.cpu().squeeze().numpy()
            sampled_z = mu_values
        else:
            # Sample from Gaussian
            image, sampled_z = sample_and_decode(model, mu_values, sigma_values, device)

        # Display image
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

        # Show latent vector
        with st.expander("📊 View Latent Vector (z)"):
            st.markdown("**Sampled latent vector:**")

            # Create bar chart
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            x = np.arange(latent_dim)
            ax2.bar(x, sampled_z, alpha=0.7, color='steelblue')
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax2.set_xlabel('Latent Dimension')
            ax2.set_ylabel('Value')
            ax2.set_title('Sampled Latent Vector')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{i+1}' for i in range(latent_dim)])
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()

            # Show numeric values
            cols = st.columns(min(4, latent_dim))
            for i in range(latent_dim):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    st.metric(f"z_{i+1}", f"{sampled_z[i]:.3f}")

        # Show distribution info
        if mode == "Sample from N(μ, σ²)":
            with st.expander("📈 Distribution Statistics"):
                st.markdown(f"**Input Distribution:** N(μ, σ²)")
                st.markdown(f"- Mean (μ): {mu_values.mean():.3f} ± {mu_values.std():.3f}")
                st.markdown(f"- Std Dev (σ): {sigma_values.mean():.3f} ± {sigma_values.std():.3f}")
                st.markdown(f"**Sampled z:**")
                st.markdown(f"- Mean: {sampled_z.mean():.3f}")
                st.markdown(f"- Std: {sampled_z.std():.3f}")
                st.markdown(f"- Min: {sampled_z.min():.3f}, Max: {sampled_z.max():.3f}")

    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    - Each latent dimension has a mean (μ) and standard deviation (σ)
    - We sample from N(μ, σ²) for each dimension independently
    - The sampled vector z is decoded to generate an image
    - Adjust sliders to explore different regions of the latent space
    """)

    # Additional info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tips")
    st.sidebar.markdown("""
    - **N(0, I)**: Sample from the prior (what the model was trained to match)
    - **N(μ, σ²)**: Control each dimension independently
    - **Deterministic**: No randomness, directly set z values
    - Try different β values during training to see effects on generation
    """)


if __name__ == '__main__':
    main()
