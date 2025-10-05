# Sample Training Images

This folder contains representative examples of the training data format used in PCP-GAN.

## Folder Structure
```sample_images/
├── 1879.50/  # Sample from depth 1879.50m
├── 1881.90/  # Sample from depth 1881.90m
├── 1918.50/  # Sample from depth 1918.50m
└── 1943.50/  # Sample from depth 1943.50m
```

## Image Specifications
- **Format**: RGB thin-section images
- **Resolution**: 768×516 pixels (automatically cropped to 768×515 during processing)
- **Processing**: Model automatically extracts 480×480 pixel sub-images during training
- **Blue regions**: Pore spaces filled with blue-dyed epoxy resin
- **Samples**: Carbonate formations from depths 1879.50m - 1943.50m

## Usage
These 20 example images (5 per depth) demonstrate the expected image format. Users should:

1. Prepare their own thin-section images following this format
2. Organize images by depth in separate folders (folder names represent depth values)
3. Place organized folders in the `data/` directory
4. Run the training script - the model will automatically:
   - Extract 480×480 pixel sub-images
   - Calculate porosity using the U-Net model
   - Balance the dataset across porosity classes

## Image Requirements
- RGB color images showing thin sections
- Blue-dyed epoxy clearly visible in pore spaces
- Minimum size: larger than 480×480 pixels
- Each depth folder should contain multiple images from the same formation
- Folder names should be numeric depth values (e.g., 1879.50, 2000.0)

## Notes
For the complete dataset used in the paper or questions about data preparation, please contact the authors or refer to the main README.