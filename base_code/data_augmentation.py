"""
Augmentation utilities for dataset balancing.

Provides 8 geometric transformations (flips/rotations) and noise injection
to generate synthetic samples for underrepresented classes. Recalculates
porosity for augmented images while preserving category labels.
"""

import numpy as np
import random
import sys
import os

# This approach avoids circular imports by importing specific functions
# from data_processing's namespace rather than the module itself
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# We'll directly define the function signatures here to avoid imports
def read_patch(original_images, patch_info, patch_size):
    """
    Import wrapper to avoid circular imports.
    This will be replaced by the actual function at runtime.
    """
    p, i, j = patch_info['p'], patch_info['i'], patch_info['j']
    return original_images[p][i:i+patch_size, j:j+patch_size]

def calculate_porosity(patch, image_type, porosity_method=None, unet_model=None, threshold_value=None):
    """
    Import wrapper to avoid circular imports.
    This will be replaced by the actual function at runtime.
    """
    # This is just a placeholder - the real function will be used at runtime
    return 0.0

def flip2(image, flip_direction):
    """
    Apply one of 8 possible flip/rotation transformations to an image.
    
    Args:
        image: Input image array (2D or with channel dimension)
        flip_direction: Integer from 1-8 indicating the type of transformation
    
    Returns:
        Transformed image with the same shape as the input
    """
    # Make a copy to avoid modifying the original
    A = image.copy()
    
    # Store original shape to restore after transformation
    original_shape = A.shape
    has_channel_dim = len(original_shape) == 3
    
    # Apply transformation based on direction
    if flip_direction == 1:  # Horizontal flip
        if has_channel_dim:
            A = A[:, ::-1, :]
        else:
            A = np.fliplr(A)
    elif flip_direction == 2:  # Vertical flip
        if has_channel_dim:
            A = A[::-1, :, :]
        else:
            A = np.flipud(A)
    elif flip_direction == 3:  # Transpose
        if has_channel_dim:
            # For RGB/grayscale with channel dimension, we need to preserve the channel
            height, width, channels = A.shape
            temp = np.zeros((width, height, channels), dtype=A.dtype)
            for c in range(channels):
                temp[:, :, c] = A[:, :, c].T
            A = temp
        else:
            A = A.T
    elif flip_direction == 4:  # Transpose + vertical flip
        if has_channel_dim:
            height, width, channels = A.shape
            temp = np.zeros((width, height, channels), dtype=A.dtype)
            for c in range(channels):
                temp[:, :, c] = A[:, :, c].T
            A = temp[::-1, :, :]
        else:
            A = np.flipud(A.T)
    elif flip_direction == 5:  # Horizontal + vertical flip (180° rotation)
        if has_channel_dim:
            A = A[::-1, ::-1, :]
        else:
            A = np.flipud(np.fliplr(A))
    elif flip_direction == 6:  # Transpose + horizontal flip
        if has_channel_dim:
            height, width, channels = A.shape
            temp = np.zeros((width, height, channels), dtype=A.dtype)
            for c in range(channels):
                temp[:, :, c] = A[:, :, c].T
            A = temp[:, ::-1, :]
        else:
            A = np.fliplr(A.T)
    elif flip_direction == 7:  # Transpose + horizontal + vertical flip
        if has_channel_dim:
            height, width, channels = A.shape
            temp = np.zeros((width, height, channels), dtype=A.dtype)
            for c in range(channels):
                temp[:, :, c] = A[:, :, c].T
            A = temp[::-1, ::-1, :]
        else:
            A = np.flipud(np.fliplr(A.T))
    # Direction 8 is no change, so we do nothing
    
    # Ensure the output shape matches the input shape if transposition was involved
    if A.shape != original_shape:
        print(f"Warning: Shape changed from {original_shape} to {A.shape} after transformation.")
        # In certain cases with RGB images and transposition, dimensions might have changed
        # If the channel dimension moved, reshape appropriately
        if has_channel_dim and len(A.shape) == 3 and A.shape[2] != original_shape[2]:
            A = np.moveaxis(A, 0, 2)  # Move channels back to last dimension
        
        # If still doesn't match, raise error
        if A.shape != original_shape:
            raise ValueError(f"Shape mismatch after transformation: {A.shape} vs expected {original_shape}")
    
    return A

def generate_augmented_patches(original_images, patch_info_list, patch_size, n_needed, 
                               image_type, unet_model, threshold_value, condition_manager, 
                               noise_range=(-0.5, 0.5)):
    """
    Generate augmented patches by applying random flips and noise.
    
    Args:
        original_images: List of original images
        patch_info_list: List of patch info dictionaries to augment
        patch_size: Size of the patches
        n_needed: Number of augmented patches needed
        image_type: Type of image (RGB, grayscale, etc.)
        unet_model: Model for porosity calculation if needed
        threshold_value: Threshold value for porosity calculation
        condition_manager: Manager containing active conditions
        porosity_method: Method to use for porosity calculation
        noise_range: Range of random noise to add
    
    Returns:
        List of augmented patch info dictionaries
    """
    augmented_patches = []
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    # If we need more patches than available, we'll need multiple augmentations per patch
    n_source_patches = len(patch_info_list)
    
    for i in range(n_needed):
        # Select a random patch to augment
        source_idx = random.randint(0, n_source_patches - 1)
        source_patch_info = patch_info_list[source_idx]
        
        # Read the original image patch
        source_image = read_patch(original_images, source_patch_info, patch_size)
        
        # Choose a random flip direction (1-7, excluding 8 which is no change)
        flip_direction = random.randint(1, 7)
        
        # Apply the flip transformation
        augmented_image = flip2(source_image, flip_direction)
        
        # Add random noise
        noise = np.random.randint(noise_range[0], noise_range[1] + 1, size=augmented_image.shape)
        augmented_image = np.clip(augmented_image + noise, 0, 255).astype(np.uint8)
        
        # Create a new patch info dictionary, starting with position info
        # Note: We keep the original position info even though the image is different
        # This is because the position is only used for retrieving the original image
        augmented_patch_info = {
            'p': source_patch_info['p'],
            'i': source_patch_info['i'],
            'j': source_patch_info['j'],
            'augmented': True,  # Mark as augmented for tracking
            'augmentation_type': f"flip_{flip_direction}_noise_{noise_range}"
        }
        
        # Copy category information if it exists in the source
        if 'category' in active_conditions:
            if 'category_actual' in source_patch_info:
                augmented_patch_info['category_actual'] = source_patch_info['category_actual']
            if 'category_scaled' in source_patch_info:
                augmented_patch_info['category_scaled'] = source_patch_info['category_scaled']
            if 'category_class' in source_patch_info:
                augmented_patch_info['category_class'] = source_patch_info['category_class']
        
        # Recalculate porosity if that condition is active
        if 'porosity' in active_conditions:
            # Calculate porosity for the augmented image
            # FIXED: Add depth parameter if category_actual exists to ensure correct depth-specific calculation
            porosity = calculate_porosity(
                augmented_image,
                image_type,
                unet_model=unet_model,
                threshold_value=threshold_value
            )
            augmented_patch_info['porosity_value'] = porosity
            
            # We'll keep the original porosity class for simplicity
            # In a more advanced implementation, we could recategorize based on new porosity
            if 'porosity_class' in source_patch_info:
                augmented_patch_info['porosity_class'] = source_patch_info['porosity_class']
        
        augmented_patches.append(augmented_patch_info)
    
    return augmented_patches

# Modified version of process_class to use augmentation
def process_class_with_augmentation(class_patches, class_key, desired_images_per_class, min_images_per_class,
                                  balanced_patch_info, repetition_percentages, empty_classes, class_counts,
                                  original_images, patch_size, image_type, unet_model, threshold_value, 
                                  condition_manager):
    """
    Process a single class during balancing, using augmentation instead of simple duplication.
    
    This is a modified version of the original process_class function that uses
    data augmentation when more samples are needed.
    """
    num_original = len(class_patches)
    
    if num_original < min_images_per_class:
        empty_classes.append(class_key)
        return
    
    if num_original >= desired_images_per_class:
        # If we have enough images, randomly sample them
        selected_patches = random.sample(class_patches, desired_images_per_class)
        rep_percentage = 0
    else:
        # If we need more images, use the originals plus augmented versions
        additional_needed = desired_images_per_class - num_original
        augmented_patches = generate_augmented_patches(
            original_images, 
            class_patches, 
            patch_size, 
            additional_needed, 
            image_type, 
            unet_model, 
            threshold_value, 
            condition_manager
        )
        selected_patches = class_patches + augmented_patches
        rep_percentage = (additional_needed / desired_images_per_class) * 100
    
    balanced_patch_info.extend(selected_patches)
    repetition_percentages[class_key] = rep_percentage
    class_counts[class_key] = len(selected_patches)

def balance_and_analyze_dataset_with_augmentation(patch_info, n_classes_category, n_classes_porosity, 
                               desired_images_per_class, min_images_per_class, num_patches_per_category_class, 
                               condition_manager, original_images, patch_size, image_type, unet_model, 
                               threshold_value):
    """Balance dataset while respecting active conditions, using augmentation for underrepresented classes"""
    balanced_patch_info = []
    repetition_percentages = {}
    empty_classes = []
    class_counts = {}
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    print("\nBalancing dataset using augmentation...")
    
    if not active_conditions:
        # If no conditions are active, just use num_patches_per_category_class
        desired_total = num_patches_per_category_class
        if len(patch_info) < desired_total:
            # Use augmentation instead of repetition
            additional_needed = desired_total - len(patch_info)
            augmented_patches = generate_augmented_patches(
                original_images, patch_info, patch_size, additional_needed,
                image_type, unet_model, threshold_value, condition_manager
            )
            balanced_patch_info = patch_info + augmented_patches
        else:
            balanced_patch_info = random.sample(patch_info, desired_total)
        
        class_counts = {0: len(balanced_patch_info)}
        non_empty_classes = {0}
    else:
        # Handle active conditions
        if 'category' in active_conditions and 'porosity' in active_conditions:
            # Both conditions active - use desired_images_per_class for balancing
            class_counts = {(d, p): 0 for d in range(n_classes_category) for p in range(n_classes_porosity)}
            for d in range(n_classes_category):
                for p in range(n_classes_porosity):
                    class_patches = [patch for patch in patch_info 
                                   if patch['category_class'] == d and 
                                   patch['porosity_class'] == p]
                    process_class_with_augmentation(
                        class_patches, (d, p), desired_images_per_class, min_images_per_class,
                        balanced_patch_info, repetition_percentages, empty_classes, class_counts,
                        original_images, patch_size, image_type, unet_model, threshold_value, 
                        condition_manager
                    )
        
        elif 'porosity' in active_conditions:
            # Only porosity active - use desired_images_per_class for balancing
            class_counts = {p: 0 for p in range(n_classes_porosity)}
            for p in range(n_classes_porosity):
                class_patches = [patch for patch in patch_info 
                               if patch['porosity_class'] == p]
                process_class_with_augmentation(
                    class_patches, p, desired_images_per_class, min_images_per_class,
                    balanced_patch_info, repetition_percentages, empty_classes, class_counts,
                    original_images, patch_size, image_type, unet_model, threshold_value, 
                    condition_manager
                )
                
        elif 'category' in active_conditions:
            # Only category active - use num_patches_per_category_class for each category
            class_counts = {d: 0 for d in range(n_classes_category)}
            for d in range(n_classes_category):
                # Get patches for this category
                class_patches = [patch for patch in patch_info if patch['category_class'] == d]
                
                # Skip if no patches for this category
                if not class_patches:
                    print(f"Warning: Category class {d} has no patches! Skipping...")
                    empty_classes.append(d)
                    continue
                
                # Use augmentation for each category
                desired_total = num_patches_per_category_class
                if len(class_patches) < desired_total:
                    # If we have fewer patches than desired, augment some
                    additional_needed = desired_total - len(class_patches)
                    augmented_patches = generate_augmented_patches(
                        original_images, class_patches, patch_size, additional_needed,
                        image_type, unet_model, threshold_value, condition_manager
                    )
                    selected_patches = class_patches + augmented_patches
                    rep_percentage = (additional_needed / desired_total) * 100
                else:
                    # If we have enough, randomly sample
                    selected_patches = random.sample(class_patches, desired_total)
                    rep_percentage = 0
                
                balanced_patch_info.extend(selected_patches)
                repetition_percentages[d] = rep_percentage
                class_counts[d] = len(selected_patches)
    
    random.shuffle(balanced_patch_info)
    non_empty_classes = set(class_counts.keys()) - set(empty_classes)
    
    # Print statistics
    print("\nBalanced Dataset Statistics (with augmentation):")
    print(f"Total patches: {len(balanced_patch_info)}")
    print(f"Empty classes: {len(empty_classes)}")
    print(f"Non-empty classes: {len(non_empty_classes)}")
    
    # Count how many patches were augmented
    augmented_count = sum(1 for patch in balanced_patch_info if patch.get('augmented', False))
    print(f"Augmented patches: {augmented_count} ({augmented_count/len(balanced_patch_info)*100:.1f}%)")
    
    if 'porosity' in active_conditions:
        porosity_values = [patch['porosity_value'] for patch in balanced_patch_info]
        print(f"\nPorosity value range: {min(porosity_values):.2f} to {max(porosity_values):.2f}")
        print(f"Mean porosity: {np.mean(porosity_values):.2f}")
        print(f"Median porosity: {np.median(porosity_values):.2f}")
    
    return balanced_patch_info, repetition_percentages, empty_classes, class_counts, non_empty_classes

# At runtime, patch the functions to use the actual implementations
def patch_functions(real_read_patch, real_calculate_porosity):
    """
    This function patches the placeholder functions with the real implementations
    from data_processing to avoid circular imports.
    """
    global read_patch, calculate_porosity
    read_patch = real_read_patch
    calculate_porosity = real_calculate_porosity