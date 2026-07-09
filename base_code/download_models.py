import os
import zipfile
import requests
from tqdm import tqdm


def download_file(url, destination, desc="Downloading"):
    """Download file with progress bar"""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Download the file in chunks and show progress
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=desc
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def ensure_file(url, path, desc):
    """Download `url` to `path` unless it already exists (skips if url is None)."""
    if not url:
        return
    if os.path.exists(path):
        print(f"{os.path.basename(path)} already exists - skipping.")
        return
    print(f"Downloading {os.path.basename(path)} from Zenodo...")
    download_file(url, path, desc=desc)
    print(f"Saved to {path}")


if __name__ == "__main__":
    from base_code.config import (
        UNET_MODEL_PATH, UNET_MODEL_URL,
        GAN_MODEL_PATH, GAN_MODEL_URL,
        GAN_METADATA_PATH, GAN_METADATA_URL,
        MODELS_DIR,
    )

    # 1) U-Net porosity-segmentation model (required for training/evaluation)
    ensure_file(UNET_MODEL_URL, UNET_MODEL_PATH, desc="Downloading U-Net model")

    # 2) Trained GAN generator (required for inference with generate.py)
    ensure_file(GAN_MODEL_URL, GAN_MODEL_PATH, desc="Downloading GAN generator")

    # 3) Metadata archive -> unzip into saved_models/metadata/
    ensure_file(GAN_METADATA_URL, GAN_METADATA_PATH, desc="Downloading metadata")
    if GAN_METADATA_URL and os.path.exists(GAN_METADATA_PATH):
        metadata_dir = os.path.join(MODELS_DIR, "metadata")
        if not os.path.exists(os.path.join(metadata_dir, "config.json")):
            print("Extracting metadata.zip ...")
            with zipfile.ZipFile(GAN_METADATA_PATH, "r") as z:
                z.extractall(MODELS_DIR)  # archive already contains a 'metadata/' folder
            print(f"Metadata extracted to {metadata_dir}")

    print("Done.")
