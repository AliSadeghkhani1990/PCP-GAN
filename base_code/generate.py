"""
Standalone inference script for PCP-GAN.

Loads a trained conditional GAN generator and produces synthetic pore-scale
images conditioned on depth (category) and porosity, WITHOUT retraining.

Requires a trained generator (e.g. ``G.h5``) and the ``metadata`` directory
produced during training (``config.json`` + ``conditions.json``). Both are
provided with the released model (see the README "Trained models" section).

Example (from the repository root):
    python -m base_code.generate \
        --model saved_models/G.h5 \
        --metadata saved_models/metadata \
        --category-class 0 --porosity 0.15 --num 5

Depth categories are ordered by increasing depth. For the dataset used in the
paper the mapping is:
    class 0 -> 1879.50 m,  class 1 -> 1881.90 m,
    class 2 -> 1918.50 m,  class 3 -> 1943.50 m
"""

import os
import sys
import json
import argparse

# Make the project root importable when run as `python base_code/generate.py`
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


def load_metadata(metadata_dir):
    """Load the config and condition metadata saved during training."""
    with open(os.path.join(metadata_dir, 'config.json')) as f:
        config = json.load(f)
    with open(os.path.join(metadata_dir, 'conditions.json')) as f:
        conditions = json.load(f)
    return config, conditions


def build_generator_inputs(latent_dim, category_dimension, category_class,
                           porosity, n_samples, active_conditions):
    """Assemble [noise, one-hot category, porosity] inputs for the generator."""
    noise = np.random.randn(n_samples, latent_dim).astype(np.float32)
    inputs = [noise]
    for cond in active_conditions:
        if cond == 'category':
            one_hot = np.zeros((n_samples, category_dimension), dtype=np.float32)
            one_hot[:, category_class] = 1.0
            inputs.append(one_hot)
        elif cond == 'porosity':
            inputs.append(np.full((n_samples, 1), porosity, dtype=np.float32))
    return inputs


def save_image(img, path, n_channels):
    """Save a single image given in the [0, 1] range as an 8-bit PNG."""
    arr = (img * 255.0).clip(0, 255).astype(np.uint8)
    if n_channels == 1:
        arr = arr.squeeze()
    Image.fromarray(arr).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate pore-scale images with a trained PCP-GAN generator")
    parser.add_argument('--model', required=True,
                        help='Path to the trained generator (e.g. G.h5)')
    parser.add_argument('--metadata', required=True,
                        help='Path to the metadata directory (config.json, conditions.json)')
    parser.add_argument('--category-class', type=int, default=0,
                        help='Depth category class index (0-based; see module docstring)')
    parser.add_argument('--porosity', type=float, required=True,
                        help='Target porosity as a fraction, e.g. 0.15')
    parser.add_argument('--num', type=int, default=1,
                        help='Number of images to generate')
    parser.add_argument('--output', default='results/generated',
                        help='Output directory for the generated images')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional random seed for reproducible noise')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    config, conditions = load_metadata(args.metadata)
    latent_dim = config['latent_dim']
    n_channels = config['n_channels']
    active_conditions = conditions['active_conditions']
    category_dimension = conditions.get('category', {}).get('n_classes', 1)

    # Report (and sanity-check against) the porosity range this class was trained on
    cat_key = str(args.category_class)
    por_meta = conditions.get('porosity', {}).get(
        'category_specific_details', {}).get(cat_key)
    if por_meta is not None:
        print(f"Category class {args.category_class}: trained porosity range "
              f"[{por_meta['min']:.4f}, {por_meta['max']:.4f}] "
              f"(mean {por_meta['mean']:.4f})")
        if not (por_meta['min'] <= args.porosity <= por_meta['max']):
            print(f"  WARNING: requested porosity {args.porosity} is outside the "
                  f"trained range for this depth; the result may be unreliable.")

    print(f"Loading generator from {args.model} ...")
    generator = load_model(args.model, compile=False)

    inputs = build_generator_inputs(
        latent_dim, category_dimension, args.category_class,
        args.porosity, args.num, active_conditions)

    print(f"Generating {args.num} image(s) ...")
    images = generator.predict(inputs)
    images = (images + 1) / 2.0  # tanh output [-1, 1] -> [0, 1]

    os.makedirs(args.output, exist_ok=True)
    for k in range(args.num):
        fname = f"gen_cat{args.category_class}_por{args.porosity:.3f}_{k + 1}.png"
        out_path = os.path.join(args.output, fname)
        save_image(images[k], out_path, n_channels)
        print(f"  saved {out_path}")

    print(f"Done. {args.num} image(s) written to {args.output}/")


if __name__ == "__main__":
    main()
