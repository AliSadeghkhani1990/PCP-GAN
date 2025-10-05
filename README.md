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