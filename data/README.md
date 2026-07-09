# `data/` — training images (not tracked)

Place your training images here before running `python -m base_code.main`.
This folder ships empty on purpose: it is a placeholder that defines where the
model reads its input from (`DATA_DIR` in `base_code/config.py`). Its contents are
excluded from version control via `.gitignore`.

## Expected layout
Organise images into one sub-folder per depth; the **folder name must be the numeric
depth value**, which the model uses as the categorical (depth) condition:

```
data/
├── 1879.50/    # depth category 0
├── 1881.90/    # depth category 1
├── 1918.50/    # depth category 2
└── 1943.50/    # depth category 3
```

## Image requirements
- RGB thin-section images with blue-dyed epoxy filling the pore space
- Larger than 480×480 pixels (the model extracts 480×480 patches automatically)
- Each depth folder should contain several images from the same formation

See [`examples/sample_images/`](../examples/sample_images) for representative examples
and [`examples/README.md`](../examples/README.md) for the full format specification.
