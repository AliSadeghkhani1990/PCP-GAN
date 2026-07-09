# PCP-GAN

**P**roperty-**C**onstrained **P**ore-scale image reconstruction via conditional **G**enerative **A**dversarial **N**etworks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1007%2Fs10596--026--10465--y-blue.svg)](https://doi.org/10.1007/s10596-026-10465-y)
[![Trained GAN](https://img.shields.io/badge/Zenodo-Trained%20GAN-1682D4.svg)](https://doi.org/10.5281/zenodo.21279478)
[![U-Net model](https://img.shields.io/badge/Zenodo-U--Net%20model-1682D4.svg)](https://doi.org/10.5281/zenodo.17271919)

Official code for the paper *"PCP-GAN: Property-Constrained Pore-scale image reconstruction via conditional Generative Adversarial Networks"* (**Computational Geosciences**, 2026).

## 📖 Overview
PCP-GAN is a multi-conditional GAN framework that generates synthetic pore-scale (thin-section) images with **precisely controlled properties**, addressing both property-matching and data-scarcity challenges in subsurface characterization:
- **Porosity control**: generate images at a specified target porosity (overall **R² = 0.95**, mean absolute error 0.0099–0.0197).
- **Multi-depth support**: a single model conditions on **four depths** of a carbonate formation (1879.50 m – 1943.50 m) simultaneously.
- **RGB fidelity**: works on RGB thin sections, preserving mineralogical cues (anhydrite–dolomite differentiation, grain boundaries, pore-type distinctions) that are lost in grayscale/binarized representations.
- **Automated porosity quantification**: a U-Net segmentation model measures porosity of every patch during training and evaluation.

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
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** all commands below are run from the **repository root** using module syntax
> (`python -m base_code.<script>`), so the `base_code` package resolves correctly.

### Download the pre-trained models
Both models are fetched automatically into `saved_models/` (the U-Net, 377 MB, is required
for porosity calculation during training/evaluation; the trained GAN generator is required
for [inference](#-example-generate-an-image-at-a-target-depth-and-porosity)):

```bash
# Option A: automatic download (recommended) — gets the U-Net and, if configured,
# the trained GAN generator + metadata, and unzips the metadata into saved_models/.
python -m base_code.download_models

# Option B: manual download
# U-Net: https://doi.org/10.5281/zenodo.17271919  -> saved_models/unet_model.h5
# GAN:   https://doi.org/10.5281/zenodo.21279479  -> saved_models/G.h5
#        and unzip metadata.zip so that saved_models/metadata/ contains config.json & conditions.json
```

## 🧪 Example: generate an image at a target depth and porosity

![PCP-GAN generated thin sections: depth (rows) by target porosity (columns)](examples/generated_example.png)

*Synthetic thin sections produced by the trained generator. Each **row** is a depth and
each **column** a higher target porosity (φ). Blue = pore space (dyed-epoxy); note how the
pore fraction grows left→right while each depth keeps its characteristic grain fabric — the
deepest sample (1943.50 m) shows fine, dispersed intercrystalline porosity typical of its
crystalline texture. Figure generated with the command below (5 porosity steps per depth).*

With a **trained generator** (`G.h5`) and its `metadata/` folder in `saved_models/`
(see [Trained models](#-trained-models)), generate synthetic thin-section images
without retraining:

```bash
python -m base_code.generate \
    --model saved_models/G.h5 \
    --metadata saved_models/metadata \
    --category-class 0 \
    --porosity 0.15 \
    --num 5 \
    --output results/generated
```

Depth categories are ordered by increasing depth:

| `--category-class` | Depth      | Rock texture              |
|:------------------:|:-----------|:--------------------------|
| 0                  | 1879.50 m  | Grainstone (dolomite)     |
| 1                  | 1881.90 m  | Grainstone (dolomite)     |
| 2                  | 1918.50 m  | Crystalline (dolomite–anhydrite) |
| 3                  | 1943.50 m  | Crystalline (dolomite)    |

The script prints the porosity range the chosen depth was trained on (and warns if
your target is outside it), then writes PNGs to `results/generated/`. Each generated
image is an RGB thin section whose blue (dyed-epoxy) fraction matches the requested
porosity. Sweeping `--porosity` across a depth's range reproduces the *depth × porosity*
grids shown in Figs. 4 and 10–13 of the paper.

## 🏋️ Training from scratch

### Prepare your data
Place your training images in the `data/` directory, organised by depth (folder names
must be numeric depth values):
```
data/
├── 1879.50/    # Depth category 0
├── 1881.90/    # Depth category 1
├── 1918.50/    # Depth category 2
└── 1943.50/    # Depth category 3
```
Representative example images demonstrating the required format (RGB thin sections with
blue-dyed epoxy in pore spaces) are in [`examples/sample_images/`](examples/sample_images).
See [`examples/README.md`](examples/README.md) for image specifications.

### Run training
```bash
python -m base_code.main
```
Training parameters live in [`base_code/config.py`](base_code/config.py) (patch size 480,
200 epochs, 10 porosity classes, etc. — the values used in the paper). Lower `epochs`
for a quick smoke test. Outputs (plots, saved models, metadata, training sub-images) are
written to a timestamped folder under `results/`.

## 📦 Trained models

| Model | Files | Purpose | Location |
|:------|:------|:--------|:---------|
| **U-Net** | `unet_model.h5` (377 MB) | Porosity segmentation | [Zenodo · 10.5281/zenodo.17271919](https://doi.org/10.5281/zenodo.17271919) |
| **PCP-GAN generator + discriminator** | `G.h5` (185 MB), `D.h5` (21 MB), `metadata.zip` | Image generation / inference (and resumed training) | [Zenodo · 10.5281/zenodo.21279479](https://doi.org/10.5281/zenodo.21279479) |

To use the generator for inference, place `G.h5` in `saved_models/` and unzip `metadata.zip`
so that `saved_models/metadata/` contains `config.json` and `conditions.json`
(the layout `base_code/generate.py` expects). The DOI
[10.5281/zenodo.21279478](https://doi.org/10.5281/zenodo.21279478) always resolves to the
latest version.

## 📁 Repository structure
```
PCP-GAN/
├── base_code/          # Source code
│   ├── main.py             # Full training pipeline (entry point)
│   ├── generate.py         # Standalone inference / image generation
│   ├── download_models.py  # Fetch the pre-trained models from Zenodo
│   ├── config.py           # Paths and training hyper-parameters
│   ├── models.py           # Generator & discriminator architectures
│   ├── training.py         # GAN training loop
│   ├── data_processing.py  # Loading, patching, porosity, balancing
│   ├── data_augmentation.py
│   ├── conditions.py       # Modular condition manager (depth, porosity)
│   └── utils.py            # Visualisation, evaluation, I/O
├── examples/           # 20 representative sample images (5 per depth)
├── data/               # <- place your training data here (git-ignored)
├── results/            # <- generated outputs & runs (git-ignored)
├── saved_models/       # <- downloaded / trained models (git-ignored)
└── requirements.txt
```
`data/`, `results/`, and `saved_models/` ship empty (only a `.gitignore` placeholder):
their large contents are intentionally excluded from version control and created at run time.

## 📄 Citation
If you use this code or the associated models, please cite:

```bibtex
@article{sadeghkhani2026pcpgan,
  title   = {PCP-GAN: Property-Constrained Pore-scale image reconstruction via conditional Generative Adversarial Networks},
  author  = {Sadeghkhani, Ali and Bennett, Brandon and Babaei, Masoud and Rabbani, Arash},
  journal = {Computational Geosciences},
  volume  = {30},
  number  = {62},
  year    = {2026},
  doi     = {10.1007/s10596-026-10465-y}
}
```

> Sadeghkhani, A., Bennett, B., Babaei, M., & Rabbani, A. (2026). PCP-GAN: Property-Constrained
> Pore-scale image reconstruction via conditional Generative Adversarial Networks.
> *Computational Geosciences*, 30, 62. https://doi.org/10.1007/s10596-026-10465-y

## 🗂️ Data availability
The complete thin-section dataset is available from the authors on reasonable request.
Twenty representative images (five per depth) are provided in `examples/sample_images/`
for reproducibility. The trained models are archived on Zenodo: the PCP-GAN generator +
discriminator at [10.5281/zenodo.21279479](https://doi.org/10.5281/zenodo.21279479) and the
U-Net segmentation model at [10.5281/zenodo.17271919](https://doi.org/10.5281/zenodo.17271919).

## 📜 License
This project is released under the [MIT License](LICENSE).

## ✉️ Contact
Ali Sadeghkhani — University of Leeds. Questions and issues are welcome via the
[GitHub issue tracker](https://github.com/AliSadeghkhani1990/PCP-GAN/issues).
