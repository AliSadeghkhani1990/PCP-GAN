# `saved_models/` — pre-trained models (not tracked)

This folder holds the pre-trained models used by the code. It ships empty on purpose:
the model files are large (hundreds of MB) and are hosted on Zenodo rather than in Git.
Its contents are excluded from version control via `.gitignore`.

## How to populate it
From the repository root, run:

```bash
python -m base_code.download_models
```

This downloads:
- `unet_model.h5` — U-Net porosity-segmentation model (377 MB), required for training/evaluation
  — [Zenodo 10.5281/zenodo.17271919](https://doi.org/10.5281/zenodo.17271919)
- `G.h5` — trained PCP-GAN generator (185 MB), required for inference with `base_code/generate.py`
  — [Zenodo 10.5281/zenodo.21279479](https://doi.org/10.5281/zenodo.21279479)
- `metadata/` — `config.json` and `conditions.json` (unzipped automatically), required by `generate.py`

After running it, this folder should contain `unet_model.h5`, `G.h5`, and a `metadata/` sub-folder.
