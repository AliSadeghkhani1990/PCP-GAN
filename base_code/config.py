"""
Configuration settings for PCP-GAN
"""
import os

# Get the project root directory automatically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths (users should place their data here)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# Default U-Net model path (used for porosity calculation)
UNET_MODEL_PATH = os.path.join(MODELS_DIR, "unet_model.h5")
UNET_MODEL_URL = "https://zenodo.org/records/17271919/files/unet_model.h5"

# Trained GAN generator + metadata (used by base_code/generate.py for inference).
# Zenodo v1.0.0: https://doi.org/10.5281/zenodo.21279479
GAN_MODEL_PATH = os.path.join(MODELS_DIR, "G.h5")
GAN_MODEL_URL = "https://zenodo.org/records/21279479/files/G.h5"
GAN_METADATA_PATH = os.path.join(MODELS_DIR, "metadata.zip")
GAN_METADATA_URL = "https://zenodo.org/records/21279479/files/metadata.zip"

# Training parameters - ALL parameters from main.py
TRAINING_CONFIG = {
    # Core training parameters
    'patch_size': 480,
    'epochs': 200,  # value used in the paper; lower it for a quick smoke test
    'n_batch': 12,
    'latent_dim': 100,
    'learning_rate': 0.0002,
    'beta_1': 0.5,

    # Data processing parameters
    'threshold_value': 85,
    'num_patches_per_category_class': 1200,
    'n_classes_porosity': 10,
    'desired_images_per_class': 160,
    'min_images_per_class': 20,
    'bat_per_epo': 300,

    # Saving parameters
    'saving_step': 5,
    'save_interval': True,
    'last_epochs_to_save': 20,

    # Truncation parameters
    'truncation_percentage_left': 0.0,
    'truncation_percentage_right': 0.05,
}