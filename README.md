# PCP-GAN

**P**roperty-**C**onstrained **P**ore-scale imaging via conditional **G**enerative **A**dversarial **N**etworks

## 🚧 Status
Repository under active development - Core functionality complete, documentation in progress!

## 📖 Overview
PCP-GAN is a conditional GAN framework for generating synthetic pore-scale images with controllable properties:
- **Porosity control**: Generate images with specific pore fractions
- **Multi-depth support**: Handle samples from different imaging depths  
- **High fidelity**: U-Net based porosity calculation ensures accurate property preservation

## 🚀 Quick Start

### Prerequisites
- Python 3.9
- CUDA 11.3+ (for GPU support)
- 8GB+ RAM (16GB recommended for training)

### Installation
```bash
# Clone repository
git clone https://github.com/AliSadeghkhani1990/PCP-GAN.git
cd PCP-GAN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the pre-trained U-Net model
# The U-Net model (377 MB) is required for porosity calculation

# Option A: Automatic download (Recommended)
python download_models.py

# Option B: Manual download
# Download from: https://doi.org/10.5281/zenodo.17271919
# Place unet_model.h5 in the saved_models/ directory
```
### Sample Images
Representative example images demonstrating the expected data format are available in the `examples/sample_images/` folder, organized by depth (1879.50m - 1943.50m). These show the required RGB thin-section format with blue-dyed epoxy in pore spaces.
### Prepare Your Data
Place your training images in the `data/` directory following this structure:
```
data/
├── 1879.50/    # Depth category 1
├── 1881.90/    # Depth category 2
├── 1918.50/    # Depth category 3
└── 1943.50/    # Depth category 4
```