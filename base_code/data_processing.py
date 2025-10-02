import cv2
import numpy as np
import os
import random
from tensorflow.keras.models import load_model
from enum import Enum
from base_code.data_augmentation import generate_augmented_patches, process_class_with_augmentation, flip2


class ImageType(Enum):
    RGB = "rgb"
    GRAYSCALE = "grayscale"
    BINARY = "binary"
    ONE_HOT = "one_hot"

def extract_category_from_dirname(dirname):
    """Extract category value from directory name"""
    try:
        return float(dirname)
    except ValueError:
        raise ValueError(f"Directory name '{dirname}' cannot be converted to a number")

def scale_category(x, min_category=None, max_category=None):
    """
    Scale category values to range (0,1) - strictly between 0 and 1
    
    Args:
        x: Input category value or array
        min_category: Optional minimum category. If None, will be calculated from x if x is array
        max_category: Optional maximum category. If None, will be calculated from x if x is array
    
    Returns:
        scaled: Scaled values in range (0,1), never exactly 0 or 1
    """
    # First determine min and max if not provided
    if min_category is None:
        if isinstance(x, (list, np.ndarray)):
            min_category = np.min(x)
        else:
            raise ValueError("min_category must be provided if x is a single value")
    
    if max_category is None:
        if isinstance(x, (list, np.ndarray)):
            max_category = np.max(x)
        else:
            raise ValueError("max_category must be provided if x is a single value")
    
    # Small delta to prevent exactly 0 or 1
    delta = 1e-6
    
    # First scale to [0,1]
    scaled = (x - min_category) / (max_category - min_category)
    
    # Then squeeze to (0,1)
    scaled = scaled * (1 - 2*delta) + delta
    
    return scaled

def process_categories(directory_path, n_channels):
    """Process categories from directory structure and return category information and images.
    
    Args:
        directory_path (str): Path to the base directory containing category subdirectories
        n_channels (int): Number of channels for image processing
        
    Returns:
        tuple: (original_images, category_values, category_classes)
            - original_images: List of processed images (uncropped)
            - category_values: List of dicts containing category information
            - category_classes: Dict mapping categories to class indices
    """
    subdirs = [d for d in os.listdir(directory_path) 
              if os.path.isdir(os.path.join(directory_path, d))]
    
    if not subdirs:
        raise ValueError("Category condition is enabled but no subdirectories found")
        
    all_categories = []
    original_images = []
    
    print(f"Found {len(subdirs)} category folders: {subdirs}")
    
    for subdir in subdirs:
        try:
            category = extract_category_from_dirname(subdir)
            subdir_path = os.path.join(directory_path, subdir)
            image_filenames = [f for f in os.listdir(subdir_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            
            print(f"Processing category {category} with {len(image_filenames)} images")
            
            for filename in image_filenames:
                file_path = os.path.join(subdir_path, filename)
                image = cv2.imread(file_path, cv2.IMREAD_COLOR if n_channels==3 else cv2.IMREAD_GRAYSCALE)
                if n_channels == 1:
                    image = image[..., np.newaxis]
                elif n_channels == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                original_images.append(image)
                all_categories.append(category)
                
        except ValueError as e:
            print(f"Skipping directory '{subdir}': {str(e)}")
            continue
    
    # Process category information
    min_category = min(all_categories)
    max_category = max(all_categories)
    unique_categories = sorted(set(all_categories))
    category_classes = {category: idx for idx, category in enumerate(unique_categories)}
    
    print(f"Category mapping: {category_classes}")
    
    category_values = []
    for category in all_categories:
        scaled_category = scale_category(category, min_category=min_category, max_category=max_category)
        category_values.append({
            'actual': category,
            'scaled': scaled_category,
            'class': category_classes[category]
        })
    
    return original_images, category_values, category_classes


def load_images(directory_path, image_type, n_channels, condition_manager):
    """Load images with category and property processing based on active conditions"""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")
    
    print(f"Using base directory: {directory_path}")
    original_images = []
    category_values = None
    category_classes = None
    
    # Check if category condition is enabled
    category_enabled = 'category' in [cond.name for cond in condition_manager.active_conditions]
    
    if category_enabled:
        # Process categories using the new function
        uncropped_images, category_values, category_classes = process_categories(
            directory_path, n_channels
        )
        # Apply cropping to all images
        original_images = [img[:-1, :] for img in uncropped_images]
    else:
        # Load images directly from the main directory
        image_filenames = [f for f in os.listdir(directory_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        for filename in image_filenames:
            file_path = os.path.join(directory_path, filename)
            image = cv2.imread(file_path, cv2.IMREAD_COLOR if n_channels==3 else cv2.IMREAD_GRAYSCALE)
            if n_channels == 1:
                image = image[..., np.newaxis]
            elif n_channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply cropping
            cropped_image = image[:-1, :]
            original_images.append(cropped_image)
    
    return original_images, category_values, category_classes


def load_unet_model(model_path):
    return load_model(model_path, compile=False)


def extract_and_save_patch_info(original_images, category_values, patch_size, num_patches_per_category_class, 
                              unet_model, image_type, threshold_value,
                              truncation_percentage_left, truncation_percentage_right, condition_manager):
    """Modified version to handle disabled category condition and support different porosity methods"""
    all_patches = []
    
    # Check if category is enabled (category_values is not None)
    if category_values is not None:
        unique_category_classes = set(d['class'] for d in category_values)
        
        # Process with categories
        for category_class in unique_category_classes:
            class_indices = [i for i, d in enumerate(category_values) if d['class'] == category_class]
            
            # Get the actual depth value for this class
            category_actual = category_values[class_indices[0]]['actual'] if class_indices else None
            print(f"Processing category class {category_class} (depth {category_actual})")
            
            for _ in range(num_patches_per_category_class * 2):
                p = random.choice(class_indices)
                image = original_images[p]
                height, width = image.shape[:2]
                i = random.randint(0, height - patch_size)
                j = random.randint(0, width - patch_size)
                patch = image[i:i+patch_size, j:j+patch_size]
                
                patch_info = {
                    'p': p,
                    'i': i,
                    'j': j,
                }
                
                # Only add category information if category is enabled
                if 'category' in [cond.name for cond in condition_manager.active_conditions]:
                    patch_info.update({
                        'category_actual': category_values[p]['actual'],
                        'category_scaled': category_values[p]['scaled'],
                        'category_class': category_values[p]['class'],
                    })
                
                # Only calculate porosity if condition is active
                if 'porosity' in [cond.name for cond in condition_manager.active_conditions]:
                    # Calculate porosity using the appropriate method for this category/depth
                    porosity = calculate_porosity(
                        patch,
                        image_type,
                        unet_model=unet_model,
                        threshold_value=threshold_value
                    )
                    patch_info['porosity_value'] = porosity
                
                all_patches.append(patch_info)
    else:
        # Process without categories - treat all images as one group
        total_patches = num_patches_per_category_class * 2
        for _ in range(total_patches):
            p = random.randint(0, len(original_images) - 1)
            image = original_images[p]
            height, width = image.shape[:2]
            i = random.randint(0, height - patch_size)
            j = random.randint(0, width - patch_size)
            patch = image[i:i+patch_size, j:j+patch_size]
            
            patch_info = {
                'p': p,
                'i': i,
                'j': j,
            }
            
            # Only calculate porosity if condition is active
            if 'porosity' in [cond.name for cond in condition_manager.active_conditions]:
                porosity = calculate_porosity(
                    patch,
                    image_type,
                    unet_model=unet_model,
                    threshold_value=threshold_value
                )
                patch_info['porosity_value'] = porosity
            
            all_patches.append(patch_info)
    
    # Apply porosity truncation only if porosity condition is active
    if 'porosity' in [cond.name for cond in condition_manager.active_conditions]:
        # When category is enabled, process each category separately
        if 'category' in [cond.name for cond in condition_manager.active_conditions]:
            filtered_patches = {}
            for category_class in set(patch.get('category_class', 0) for patch in all_patches):
                # Get patches for this category
                category_patches = [p for p in all_patches if p.get('category_class', 0) == category_class]
                
                # Calculate bounds for this category only
                category_porosities = [p['porosity_value'] for p in category_patches]
                if category_porosities:  # Check if we have any patches for this category
                    lower_bound = np.percentile(category_porosities, truncation_percentage_left * 100)
                    upper_bound = np.percentile(category_porosities, (1 - truncation_percentage_right) * 100)
                    
                    # Filter patches for this category using category-specific bounds
                    filtered_category_patches = [p for p in category_patches 
                                                if lower_bound <= p['porosity_value'] <= upper_bound]
                    
                    # Take up to num_patches_per_category_class patches
                    filtered_patches[category_class] = filtered_category_patches[:num_patches_per_category_class]
                    
                    print(f"Category {category_class}: Porosity range [{lower_bound:.4f}, {upper_bound:.4f}], "
                          f"Selected {len(filtered_patches[category_class])} patches")
            
            # Combine all filtered patches
            patch_info = []
            for category_class, patches in filtered_patches.items():
                patch_info.extend(patches)
        else:
            # Original behavior when category is disabled (use single class 0)
            porosities = [patch['porosity_value'] for patch in all_patches]
            lower_bound = np.percentile(porosities, truncation_percentage_left * 100)
            upper_bound = np.percentile(porosities, (1 - truncation_percentage_right) * 100)
            
            filtered_patches = {}
            for patch in all_patches:
                if lower_bound <= patch['porosity_value'] <= upper_bound:
                    category_class = patch.get('category_class', 0)  # Default to 0 if no category
                    if category_class not in filtered_patches:
                        filtered_patches[category_class] = []
                    filtered_patches[category_class].append(patch)
            
            patch_info = []
            unique_classes = filtered_patches.keys()
            for category_class in unique_classes:
                if category_class in filtered_patches:
                    class_patches = filtered_patches[category_class][:num_patches_per_category_class]
                    patch_info.extend(class_patches)
    else:
        # If porosity is not active, use all patches without truncation
        if category_values is not None and 'category' in [cond.name for cond in condition_manager.active_conditions]:
            # When category is enabled and porosity is disabled, take samples per category
            patch_info = []
            unique_category_classes = set(patch.get('category_class', 0) for patch in all_patches)
            
            for category_class in unique_category_classes:
                # Get all patches for this category
                class_patches = [p for p in all_patches if p.get('category_class', 0) == category_class]
                
                # Take up to num_patches_per_category_class patches from this category
                if len(class_patches) > num_patches_per_category_class:
                    class_patches = random.sample(class_patches, num_patches_per_category_class)
                
                patch_info.extend(class_patches)
        else:
            # If no category is enabled, use first set of patches
            patch_info = all_patches[:num_patches_per_category_class]
    
    random.shuffle(patch_info)
    return patch_info

def categorize_patches(patch_info, n_classes_porosity, condition_manager):
    """
    Categorize patches based on active conditions.
    Each condition will be categorized according to its own logic.
    When both category and porosity are active, porosity classes are created per category.
    
    Args:
        patch_info: List of dictionaries containing patch information
        n_classes_porosity: Number of classes for porosity categorization
        condition_manager: Manager containing information about active conditions
    
    Returns:
        patch_info: Updated patch information with categorization for active conditions
    """
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    if 'porosity' in active_conditions:
        if 'category' in active_conditions:
            # Get all unique category classes
            category_classes = sorted(set(patch['category_class'] for patch in patch_info if 'category_class' in patch))
            
            # Process each category separately
            for category_class in category_classes:
                # Get patches for this category
                category_patches = [p for p in patch_info if p.get('category_class') == category_class]
                
                # Skip if no patches for this category
                if not category_patches:
                    continue
                
                # Get porosity values for this category
                category_porosities = [p['porosity_value'] for p in category_patches]
                
                # Calculate quantile values for this category's porosity
                quantile_values_porosity = np.linspace(np.min(category_porosities), 
                                                     np.max(category_porosities), 
                                                     n_classes_porosity + 1)
                
                # Categorize porosity values for this category
                for patch in category_patches:
                    porosity = patch['porosity_value']
                    patch['porosity_class'] = get_porosity_class(porosity, quantile_values_porosity)
                
                print(f"Category {category_class}: Porosity range [{min(category_porosities):.4f}, "
                      f"{max(category_porosities):.4f}] divided into {n_classes_porosity} classes")
        else:
            # Original behavior when only porosity is active
            # Collect all porosity values
            all_porosities = [patch['porosity_value'] for patch in patch_info]
            
            # Calculate quantile values for porosity
            quantile_values_porosity = np.linspace(np.min(all_porosities), 
                                                 np.max(all_porosities), 
                                                 n_classes_porosity + 1)
            
            # Categorize porosity values
            for patch in patch_info:
                porosity = patch['porosity_value']
                patch['porosity_class'] = get_porosity_class(porosity, quantile_values_porosity)
    
    # Example structure for future conditions:
    if 'permeability' in active_conditions:
        # Add permeability categorization logic here
        pass
    
    if 'specific_surface_area' in active_conditions:
        # Add SSA categorization logic here
        pass
    
    return patch_info

def get_porosity_class(value, bins):
    """Helper function to determine class based on value and bins"""
    for i in range(len(bins) - 1):
        if bins[i] <= value <= bins[i + 1]:
            return i
    return len(bins) - 2  # For values equal to the maximum


def read_patch(original_images, patch_info, patch_size):
    """
    Read a patch from original images, applying augmentations if specified.
    
    Args:
        original_images: List of original images
        patch_info: Dictionary with patch information
        patch_size: Size of the patch to extract
    
    Returns:
        The image patch, with augmentations applied if needed
    """
    p, i, j = patch_info['p'], patch_info['i'], patch_info['j']
    original_patch = original_images[p][i:i+patch_size, j:j+patch_size]
    
    # Check if this is an augmented patch
    if patch_info.get('augmented', False):
        # Extract augmentation information
        aug_type = patch_info.get('augmentation_type', '')
        if 'flip_' in aug_type:
            # Extract flip direction from augmentation_type string
            parts = aug_type.split('_')
            flip_direction = int(parts[1])
            
            # Import flip2 function if needed

            
            # Apply the flip transformation
            augmented_patch = flip2(original_patch, flip_direction)
            
            # Apply noise if specified in the augmentation type
            if 'noise_' in aug_type:
                try:
                    # Parse noise range from string like "noise_(-2, 2)"
                    noise_str = aug_type.split('noise_')[1]
                    noise_range = eval(noise_str)  # Convert string to tuple
                    
                    # Apply consistent noise using a seed based on patch position
                    seed = hash(f"{p}_{i}_{j}") % 10000
                    rng = np.random.RandomState(seed)
                    noise = rng.randint(noise_range[0], noise_range[1] + 1, 
                                         size=augmented_patch.shape)
                    augmented_patch = np.clip(augmented_patch + noise, 0, 255).astype(np.uint8)
                except:
                    # Use default noise range if parsing fails
                    noise_range = (-2, 2)
                    seed = hash(f"{p}_{i}_{j}") % 10000
                    rng = np.random.RandomState(seed)
                    noise = rng.randint(noise_range[0], noise_range[1] + 1, 
                                         size=augmented_patch.shape)
                    augmented_patch = np.clip(augmented_patch + noise, 0, 255).astype(np.uint8)
            
            return augmented_patch
    
    return original_patch

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

def process_class(class_patches, class_key, desired_images_per_class, min_images_per_class,
                 balanced_patch_info, repetition_percentages, empty_classes, class_counts):
    """Helper function to process a single class during balancing"""
    num_original = len(class_patches)
    
    if num_original < min_images_per_class:
        empty_classes.append(class_key)
        return
    
    if num_original >= desired_images_per_class:
        selected_patches = random.sample(class_patches, desired_images_per_class)
        rep_percentage = 0
    else:
        repetitions = desired_images_per_class - num_original
        selected_patches = class_patches + random.choices(class_patches, k=repetitions)
        rep_percentage = (repetitions / desired_images_per_class) * 100
    
    balanced_patch_info.extend(selected_patches)
    repetition_percentages[class_key] = rep_percentage
    class_counts[class_key] = len(selected_patches)


def calculate_porosity(patch, image_type, unet_model=None, threshold_value=None):
    """
    Calculate porosity using U-Net model for RGB images.

    Args:
        patch: Input image patch
        image_type: Type of image (RGB, GRAYSCALE, etc.)
        unet_model: Pre-trained U-Net model for segmentation
        threshold_value: Not used (kept for backwards compatibility)

    Returns:
        float: Calculated porosity value
    """
    if image_type == ImageType.RGB:
        if unet_model is None:
            raise ValueError("U-Net model is required for RGB images")
        patch_float = patch.astype('float32') / 255.0
        patch_input = np.expand_dims(patch_float, axis=0)
        binary_prediction = unet_model.predict(patch_input)[0]
        binary_patch = (binary_prediction > 0.5).astype(np.uint8)
        return np.sum(binary_patch) / binary_patch.size

    elif image_type == ImageType.GRAYSCALE:
        # For grayscale images, we need the U-Net model too
        if unet_model is None:
            raise ValueError("U-Net model is required for grayscale images")
        # Convert grayscale to 3-channel for U-Net input
        if len(patch.shape) == 2:
            patch = np.stack([patch, patch, patch], axis=-1)
        patch_float = patch.astype('float32') / 255.0
        patch_input = np.expand_dims(patch_float, axis=0)
        binary_prediction = unet_model.predict(patch_input)[0]
        binary_patch = (binary_prediction > 0.5).astype(np.uint8)
        return np.sum(binary_patch) / binary_patch.size

    elif image_type == ImageType.BINARY:
        # For binary images, directly calculate porosity
        return np.sum(patch) / patch.size

    elif image_type == ImageType.ONE_HOT:
        # For one-hot encoded images, sum the channel representing pores
        pore_channel = patch[..., 0]
        return np.sum(pore_channel) / pore_channel.size

    else:
        raise ValueError(f"Unsupported image type: {image_type}")
