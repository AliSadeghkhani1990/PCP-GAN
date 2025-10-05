import os
import requests
from tqdm import tqdm


def download_file(url, destination):
    """Download file with progress bar"""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Download the file in chunks and show progress
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc="Downloading U-Net model"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


if __name__ == "__main__":
    from base_code.config import UNET_MODEL_PATH, UNET_MODEL_URL

    # Only download if the file doesn't exist
    if not os.path.exists(UNET_MODEL_PATH):
        print("Downloading U-Net model from Zenodo...")
        download_file(UNET_MODEL_URL, UNET_MODEL_PATH)
        print(f"Model saved to {UNET_MODEL_PATH}")
    else:
        print("Model already exists")