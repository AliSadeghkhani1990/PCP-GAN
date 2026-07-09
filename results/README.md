# `results/` — generated outputs (not tracked)

This folder is where the code writes its outputs. It ships empty on purpose: everything
inside is machine-generated at run time and is excluded from version control via `.gitignore`.

## What gets written here
- **Training** (`python -m base_code.main`) creates a timestamped run folder containing
  the training sub-images, distribution/heatmap plots, evaluation plots (including the
  R² porosity accuracy figure), saved model checkpoints, and inference metadata.
- **Inference** (`python -m base_code.generate ... --output results/generated`) writes the
  generated images here.

Nothing needs to be placed here manually — the folder simply defines the output location
(`RESULTS_DIR` in `base_code/config.py`).
