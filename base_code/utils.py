"""
Utility functions for visualization and evaluation.

Includes training/inference visualization, porosity accuracy evaluation (R²),
dataset distribution analysis, results directory management, and metadata saving.
Handles all plotting, file organization, and experiment tracking.
"""

import os
import sys
# Add the parent directory of 'multi_condition' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import random
from base_code.data_processing import read_patch, calculate_porosity

def get_plots_directory(base_dir):
    """Get or create plots directory"""
    plots_dir = os.path.join(base_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def generate_real_samples(original_images, balanced_patch_info, n_samples, patch_size, condition_manager):
    indices = np.random.randint(0, len(balanced_patch_info), n_samples)

    selected_patches = [read_patch(original_images, balanced_patch_info[i], patch_size) for i in indices]
    selected_X = np.array(selected_patches, dtype=np.float32)
    selected_X = (selected_X / 255.0 - 0.5) * 2  # Scale to [-1, 1]

    # Get condition values
    condition_inputs = []
    for condition in condition_manager.active_conditions:
        condition_values = [balanced_patch_info[i][f'{condition.name}_scaled' if condition.name == 'category' else f'{condition.name}_value']
                          for i in indices]
        condition_inputs.append(np.array(condition_values, dtype=np.float32))

    return [selected_X] + condition_inputs, np.ones((n_samples, 1))


def visualize_generated_samples(original_images, balanced_patch_info, n_samples, patch_size, n_channels, condition_manager):
    # Get the outputs using the training.py version of generate_real_samples
    [selected_X, *condition_values], _ = generate_real_samples(
        original_images, balanced_patch_info, n_samples, patch_size, condition_manager
    )
    
    # Sort by porosity value if porosity condition is active
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    if 'porosity' in active_conditions:
        # Find the index of porosity in the active conditions
        porosity_idx = active_conditions.index('porosity')
        
        # Get porosity values
        porosity_values = condition_values[porosity_idx]
        
        # Create indices sorted by porosity
        sorted_indices = np.argsort(porosity_values)
        
        # Reorder everything based on sorted indices
        selected_X = selected_X[sorted_indices]
        condition_values = [cv[sorted_indices] for cv in condition_values]
    
    # Convert images back to 0-255 range for visualization
    display_images = ((selected_X + 1) * 0.5 * 255).astype(np.uint8)
    
    # Create subplot grid
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4.5, n_rows*4.5))  # Increased figure size
    plt.subplots_adjust(hspace=0.3, wspace=0.0)  # Further reduced spacing
    
    # Handle single row case
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    
    # Plot each image with its conditions
    for i in range(n_samples):
        row = i // n_cols
        col = i % n_cols
        
        if n_channels == 1:
            axes[row][col].imshow(display_images[i], cmap='gray')
        else:
            axes[row][col].imshow(display_images[i])
        
        # Remove axis ticks for cleaner look
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
        
        # Create title text based on active conditions - only show Porosity with capital P
        title_parts = []
        for j, condition in enumerate(condition_manager.active_conditions):
            value = condition_values[j][i]
            if condition.name.lower() == "porosity":
                title_parts.append(f"Porosity: {value:.4f}")  # Explicitly use "Porosity" with capital P
            else:
                # For other conditions, still capitalize the first letter
                condition_name = condition.name.capitalize()
                title_parts.append(f"{condition_name}: {value:.4f}")
        
        # Set title with larger font size and some padding
        axes[row][col].set_title('\n'.join(title_parts), fontsize=16, pad=10)
    
    # Apply tight layout but with reduced padding
    plt.tight_layout(pad=0.4)
    from __main__ import results_directory
    plots_dir = get_plots_directory(results_directory)
    plt.savefig(os.path.join(plots_dir, 'generated_samples_visualization.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: generated_samples_visualization.png")
    plt.close()

def analyze_sample_distribution(samples_data, n_bins=10):
    """
    Analyzes the distribution of category and porosity values in the generated samples.
    """
    # Create figure for distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot category distribution
    ax1.hist(samples_data['category_values']['actual'], bins=n_bins, edgecolor='black')
    ax1.set_title('Distribution of Category Values')
    ax1.set_xlabel('Actual Category')
    ax1.set_ylabel('Frequency')
    
    # Plot porosity distribution
    ax2.hist(samples_data['porosity_values'], bins=n_bins, edgecolor='black')
    ax2.set_title('Distribution of Porosity Values')
    ax2.set_xlabel('Porosity')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    from __main__ import results_directory
    plots_dir = get_plots_directory(results_directory)
    plt.savefig(os.path.join(plots_dir, 'sample_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: sample_distribution.png")
    plt.close()
    
    # Print additional statistics
    print("\nDetailed Statistics:")
    print("\nCategory Values:")
    print(f"Mean (actual): {np.mean(samples_data['category_values']['actual']):.2f}")
    print(f"Std (actual): {np.std(samples_data['category_values']['actual']):.2f}")
    print(f"Mean (scaled): {np.mean(samples_data['category_values']['scaled']):.4f}")
    print(f"Std (scaled): {np.std(samples_data['category_values']['scaled']):.4f}")
    
    print("\nPorosity Values:")
    print(f"Mean: {np.mean(samples_data['porosity_values']):.4f}")
    print(f"Std: {np.std(samples_data['porosity_values']):.4f}")

# Replace this function in your utils.py file

def plot_generated_images_enhanced(generator, epoch, latent_dim, n_channels, condition_manager, balanced_patch_info, fixed_noise=None, fixed_classes=None):
    """
    Enhanced visualization function to better debug category values.
    Modified to handle one-hot encoding for categories.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Import the global category_dimension variable
    # (You'll need to add "from __main__ import category_dimension" at the top of utils.py)
    from __main__ import category_dimension
    if category_dimension is None:
        raise ValueError("category_dimension must be set before plotting")
    
    print("\n==== Enhanced Image Generation Debug ====")
    
    # Check if category condition is active
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    # Determine the number of rows (categories) and columns (porosity values)
    n_cols = 5  # Number of porosity values to show per category
    
    # Get unique category values
    if 'category' in active_conditions:
        # Get all unique category values and their information
        category_info = []
        for patch in balanced_patch_info:
            if 'category_actual' in patch and 'category_scaled' in patch and 'category_class' in patch:
                info = (patch['category_actual'], patch['category_scaled'], patch['category_class'])
                if info not in category_info:
                    category_info.append(info)
        
        # Sort by actual value
        category_info.sort(key=lambda x: x[0])
        n_rows = len(category_info)
        
        print(f"Found {n_rows} unique category values:")
        for actual, scaled, class_idx in category_info:
            print(f"  Actual: {actual:.2f}, Scaled: {scaled:.6f}, Class: {class_idx}")
    else:
        n_rows = 1
        category_info = [(0, 0, 0)]  # Dummy value
        print("Category condition is not active")
    
    # Get porosity values if active
    porosity_values = []
    if 'porosity' in active_conditions:
        # Get min/max porosity for each category to show range
        for actual_cat, scaled_cat, class_cat in category_info:
            category_porosities = [p['porosity_value'] for p in balanced_patch_info 
                                   if p.get('category_class', 0) == class_cat]
            if category_porosities:
                min_p = min(category_porosities)
                max_p = max(category_porosities)
                # Create evenly spaced values between min and max
                for i in range(n_cols):
                    p_val = min_p + (max_p - min_p) * i / (n_cols - 1) if n_cols > 1 else min_p
                    porosity_values.append((class_cat, p_val))
                print(f"  Category {class_cat} porosity range: {min_p:.4f} to {max_p:.4f}")
            else:
                print(f"  Category {class_cat} has no porosity values")
    else:
        # If no porosity condition, just use dummy values
        for class_cat in [info[2] for info in category_info]:
            for i in range(n_cols):
                porosity_values.append((class_cat, 0.1 * i))
        print("Porosity condition is not active")

    # Create inputs for the generator
    total_images = len(porosity_values)
    
    if fixed_noise is None:
        fixed_noise = np.random.randn(total_images, latent_dim)
    
    # Prepare all generator inputs and labels for tracking
    all_inputs = []
    all_labels = []
    
    for i, (cat_class, porosity) in enumerate(porosity_values):
        # Get the actual category value for this class
        cat_info = [info for info in category_info if info[2] == cat_class][0]
        actual_cat, scaled_cat, _ = cat_info
        
        noise = fixed_noise[i].reshape(1, -1)
        generator_inputs = [noise]
        
        input_labels = {
            'noise_idx': i,
            'actual_category': actual_cat,
            'scaled_category': scaled_cat,
            'category_class': cat_class,
            'porosity': porosity
        }
        
        # Add condition inputs based on active conditions
        for condition in condition_manager.active_conditions:
            if condition.name == 'category':
                # Create one-hot vector for category
                one_hot = np.zeros((1, category_dimension), dtype=np.float32)
                one_hot[0, cat_class] = 1.0
                generator_inputs.append(one_hot)
            elif condition.name == 'porosity':
                generator_inputs.append(np.array([[porosity]]))
        
        all_inputs.append(generator_inputs)
        all_labels.append(input_labels)
    
    # Generate images
    all_images = []
    for inputs in all_inputs:
        img = generator.predict(inputs)
        all_images.append((img[0] + 1) / 2)  # Convert from [-1,1] to [0,1]
    
    # Display generated images
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    # Handle single row case
    if n_rows == 1:
        axes = [axes]
    
    # Plot each image with its conditions
    for i, (img, labels) in enumerate(zip(all_images, all_labels)):
        row = i // n_cols
        col = i % n_cols
        
        if n_channels == 1:
            axes[row][col].imshow(img, cmap='gray')
        else:
            axes[row][col].imshow(img)
        
        axes[row][col].axis('off')
        
        # Create detailed title with exact values
        title_parts = []
        title_parts.append(f"Cat: {labels['actual_category']:.2f}")
        title_parts.append(f"Class: {labels['category_class']}")
        
        if 'porosity' in active_conditions:
            title_parts.append(f"Por: {labels['porosity']:.4f}")
        
        axes[row][col].set_title('\n'.join(title_parts), fontsize=8)
    
    # Add row labels for category values
    for i, (actual, scaled, _) in enumerate(category_info):
        if i < len(axes):
            ylabel = f"Category: {actual:.2f}\n(Class: {i})"
            axes[i][0].set_ylabel(ylabel, fontsize=10, rotation=90, labelpad=15)
    
    plt.tight_layout()
    fig.suptitle(f'Generated Images at Epoch {epoch}', fontsize=16, y=1.02)
    from __main__ import model_path_saving
    epoch_dir = os.path.join(model_path_saving, "epoch_images")
    os.makedirs(epoch_dir, exist_ok=True)
    plt.savefig(os.path.join(epoch_dir, f'generated_epoch_{epoch:04d}.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: generated_epoch_{epoch:04d}.png")
    plt.close()
    
    # Print a summary of inputs used
    print("\nGenerator inputs used:")
    for condition in condition_manager.active_conditions:
        print(f"  {condition.name} values:")
        if condition.name == 'category':
            values = [labels['category_class'] for labels in all_labels]
            print(f"    Classes used: {sorted(set(values))}")
        elif condition.name == 'porosity':
            values = [labels['porosity'] for labels in all_labels]
            print(f"    Range: {min(values):.6f} to {max(values):.6f}")
    
    return fixed_noise, fixed_classes

def analyze_patch_info(patch_info, condition_manager):
    """Analyze patch information based on active conditions"""
    analysis_results = {}
    
    # Check which conditions are active
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    if 'porosity' in active_conditions:
        porosities = [patch['porosity_value'] for patch in patch_info]
        analysis_results['porosity'] = {
            'values': porosities,
            'classes': [patch['porosity_class'] for patch in patch_info],
            'n_classes': max([patch['porosity_class'] for patch in patch_info]) + 1
        }
        
        plt.figure(figsize=(10, 6))
        plt.hist(porosities, bins=50, edgecolor='black')
        plt.title('Porosity Distribution')
        plt.xlabel('Porosity')
        plt.ylabel('Frequency')
        from __main__ import results_directory
        plots_dir = get_plots_directory(results_directory)
        plt.savefig(os.path.join(plots_dir, 'porosity_distribution.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: porosity_distribution.png")
        plt.close()

    if 'category' in active_conditions:
        category_classes = [patch['category_class'] for patch in patch_info]
        analysis_results['category'] = {
            'classes': category_classes,
            'n_classes': max(category_classes) + 1
        }

    print("\nDistribution of Images:")
    if 'category' in active_conditions and 'porosity' in active_conditions:
        # Just print summary statistics
        total_images = len(patch_info)
        n_categories = analysis_results['category']['n_classes']
        n_porosity_classes = analysis_results['porosity']['n_classes']
        print(f"Total images: {total_images}")
        print(f"Categories: {n_categories}, Porosity classes: {n_porosity_classes}")
    elif 'porosity' in active_conditions:
        # Process only porosity
        porosity_class_counts = np.bincount(
            analysis_results['porosity']['classes'],
            minlength=analysis_results['porosity']['n_classes']
        )
        for i, count in enumerate(porosity_class_counts):
            print(f"Porosity Category {i}: {count} images")

    # Process ranges for active conditions
    print("\nValue ranges for active conditions:")
    ranges = {}

    if 'category' in active_conditions and 'porosity' in active_conditions:
        for d in range(analysis_results['category']['n_classes']):
            ranges[d] = []
            for p in range(analysis_results['porosity']['n_classes']):
                class_values = [
                    patch['porosity_value'] for patch in patch_info
                    if patch['category_class'] == d and patch['porosity_class'] == p
                ]
                if class_values:
                    min_val = min(class_values)
                    max_val = max(class_values)
                    print(f"Category {d}, Porosity Class {p}: [{min_val:.4f}, {max_val:.4f}]")
                    ranges[d].append((min_val, max_val))
                else:
                    print(f"Category {d}, Porosity Class {p}: No images")
                    ranges[d].append((0, 0))
    elif 'porosity' in active_conditions:
        ranges[0] = []
        for p in range(analysis_results['porosity']['n_classes']):
            class_values = [
                patch['porosity_value'] for patch in patch_info 
                if patch['porosity_class'] == p
            ]
            if class_values:
                min_val = min(class_values)
                max_val = max(class_values)
                print(f"Porosity Class {p}: [{min_val:.4f}, {max_val:.4f}]")
                ranges[0].append((min_val, max_val))
            else:
                print(f"Porosity Class {p}: No images")
                ranges[0].append((0, 0))
    
    return ranges, analysis_results

def draw_heatmaps(original_patch_info, balanced_patch_info, condition_manager):
    """Draw heatmaps based on active conditions"""
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    # Determine dimensions based on active conditions
    n_classes_category = 1
    n_classes_porosity = 1
    
    if 'category' in active_conditions:
        n_classes_category = max(patch['category_class'] for patch in original_patch_info) + 1
    if 'porosity' in active_conditions:
        n_classes_porosity = max(patch['porosity_class'] for patch in original_patch_info) + 1
    
    original_counts = np.zeros((n_classes_category, n_classes_porosity))
    balanced_counts = np.zeros((n_classes_category, n_classes_porosity))
    
    # Count occurrences based on active conditions
    for patch in original_patch_info:
        category_class = patch.get('category_class', 0) if 'category' in active_conditions else 0
        porosity_class = patch.get('porosity_class', 0) if 'porosity' in active_conditions else 0
        original_counts[category_class, porosity_class] += 1
    
    for patch in balanced_patch_info:
        category_class = patch.get('category_class', 0) if 'category' in active_conditions else 0
        porosity_class = patch.get('porosity_class', 0) if 'porosity' in active_conditions else 0
        balanced_counts[category_class, porosity_class] += 1
    
    # Only create visualization if at least one condition is active
    if active_conditions:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        sns.heatmap(original_counts, cmap='Blues', annot=True, fmt='g', 
                   linewidths=0.5, linecolor='black', ax=ax1)
        ax1.set_title('Distribution of Original Images')
        
        if 'porosity' in active_conditions:
            ax1.set_xlabel('Porosity Classes')
        if 'category' in active_conditions:
            ax1.set_ylabel('Category Classes')
        else:
            ax1.set_ylabel('Single Category')
        
        sns.heatmap(balanced_counts, cmap='Blues', annot=True, fmt='g',
                   linewidths=0.5, linecolor='black', ax=ax2)
        ax2.set_title('Distribution of Balanced Images')
        
        if 'porosity' in active_conditions:
            ax2.set_xlabel('Porosity Classes')
        if 'category' in active_conditions:
            ax2.set_ylabel('Category Classes')
        else:
            ax2.set_ylabel('Single Category')
        
        plt.tight_layout()
        from __main__ import results_directory
        plots_dir = get_plots_directory(results_directory)
        plt.savefig(os.path.join(plots_dir, 'distribution_heatmaps.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: distribution_heatmaps.png")
        plt.close()
    else:
        print("No conditions active - skipping heatmap visualization")

# Replace this function in your utils.py file

def plot_generated_images_inference_enhanced(generator, latent_dim, n_channels, condition_manager, model_path_saving, patch_size, balanced_patch_info, non_empty_classes):
    """
    Enhanced inference visualization function with depth-specific porosity ranges and noise vectors.
    Modified to use a different noise vector for each depth category.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Import the global category_dimension variable
    from __main__ import category_dimension
    if category_dimension is None:
        raise ValueError("category_dimension must be set before inference")
    
    print("\n==== Enhanced Inference Image Generation ====")
    
    # Check if category and porosity conditions are active
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    # Get unique category values if category is active
    if 'category' in active_conditions:
        # Extract all unique category information
        category_info = []
        for patch in balanced_patch_info:
            if 'category_actual' in patch and 'category_scaled' in patch and 'category_class' in patch:
                info = (patch['category_actual'], patch['category_scaled'], patch['category_class'])
                if info not in category_info:
                    category_info.append(info)
        
        # Sort by actual value
        category_info.sort(key=lambda x: x[0])
        n_rows = len(category_info)
        
        print(f"Found {n_rows} unique category values:")
        for actual, scaled, class_idx in category_info:
            print(f"  Actual: {actual:.2f}, Scaled: {scaled:.6f}, Class: {class_idx}")
    else:
        n_rows = 1
        category_info = [(0, 0, 0)]  # Dummy value
        print("Category condition is not active")
    
    # Set number of porosity steps (columns)
    n_cols = 10  # Fixed to 10 for the 10 porosity classes
    
    # Get porosity range per category if porosity is active
    category_porosity_ranges = {}
    empty_classes = set()
    
    if 'porosity' in active_conditions:
        for actual_cat, scaled_cat, class_cat in category_info:
            # Get all patches for this category
            category_patches = [p for p in balanced_patch_info if p.get('category_class', 0) == class_cat]
            
            if category_patches:
                # Get porosity values for this category
                category_porosities = [p['porosity_value'] for p in category_patches]
                min_porosity = min(category_porosities)
                max_porosity = max(category_porosities)
                
                # Get porosity classes for this category
                porosity_classes = set(p.get('porosity_class', 0) for p in category_patches)
                
                # Find missing classes
                missing_classes = set(range(n_cols)) - porosity_classes
                
                # Create porosity steps
                porosity_steps = []
                for i in range(n_cols):
                    if i in missing_classes:
                        # Use None to indicate missing class
                        porosity_steps.append(None)
                        empty_classes.add((class_cat, i))
                    else:
                        # Calculate evenly spaced value in the range
                        por_val = min_porosity + (max_porosity - min_porosity) * i / (n_cols - 1)
                        porosity_steps.append(por_val)
                
                category_porosity_ranges[class_cat] = {
                    'min': min_porosity,
                    'max': max_porosity,
                    'steps': porosity_steps
                }
                
                print(f"  Category {class_cat} (depth {actual_cat:.2f}m) porosity range: {min_porosity:.4f} to {max_porosity:.4f}")
                if missing_classes:
                    print(f"    Missing porosity classes: {sorted(missing_classes)}")
            else:
                # If no patches for this category, use dummy values
                category_porosity_ranges[class_cat] = {
                    'min': 0,
                    'max': 0.1,
                    'steps': [None] * n_cols  # All None to indicate no valid classes
                }
                for i in range(n_cols):
                    empty_classes.add((class_cat, i))
                print(f"  Category {class_cat} has no patches")
    else:
        # If no porosity condition, use dummy values
        for actual_cat, scaled_cat, class_cat in category_info:
            category_porosity_ranges[class_cat] = {
                'min': 0,
                'max': 0.1,
                'steps': [0.01 * i for i in range(n_cols)]
            }
        print("Porosity condition is not active")
    
    # Create different noise vectors for each depth category
    depth_noise_vectors = {}
    for actual_cat, scaled_cat, cat_class in category_info:
        # Generate a unique noise vector for this depth
        depth_noise_vectors[cat_class] = np.random.randn(1, latent_dim)
        print(f"  Generated unique noise vector for depth {actual_cat:.2f}m (category {cat_class})")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = [axes]  # Handle single row case
    
    # Generate images for each category and porosity combination
    for row, (actual_cat, scaled_cat, cat_class) in enumerate(category_info):
        # Get the noise vector for this depth
        depth_noise = depth_noise_vectors[cat_class]
        
        for col in range(n_cols):
            # Check if this is an empty class
            if (cat_class, col) in empty_classes or category_porosity_ranges[cat_class]['steps'][col] is None:
                # Create a black image for empty classes
                axes[row][col].imshow(np.zeros((patch_size, patch_size, 3)), cmap='gray')
                axes[row][col].set_title("Empty", fontsize=8)
                axes[row][col].axis('off')
                continue
            
            # Get porosity value for this column
            porosity = category_porosity_ranges[cat_class]['steps'][col]
            
            # Prepare generator inputs
            generator_inputs = [depth_noise.copy()]  # Use the depth-specific noise
            
            for condition in condition_manager.active_conditions:
                if condition.name == 'category':
                    # Create one-hot vector for category
                    one_hot = np.zeros((1, category_dimension), dtype=np.float32)
                    one_hot[0, cat_class] = 1.0
                    generator_inputs.append(one_hot)
                elif condition.name == 'porosity':
                    generator_inputs.append(np.array([[porosity]]))
            
            # Generate and display image
            img = generator.predict(generator_inputs)
            img = (img[0] + 1) / 2  # Convert from [-1,1] to [0,1]
            
            if n_channels == 1:
                axes[row][col].imshow(img, cmap='gray')
            else:
                axes[row][col].imshow(img)
            
            axes[row][col].axis('off')
            
            # Add detailed title
            if 'porosity' in active_conditions:
                title = f"Por: {porosity:.3f}"
                axes[row][col].set_title(title, fontsize=8)
    
    # Add category labels to left side
    for row, (actual_cat, scaled_cat, cat_class) in enumerate(category_info):
        ylabel = f"Cat: {actual_cat:.2f}\n(Class: {cat_class})"
        axes[row][0].set_ylabel(ylabel, fontsize=9, rotation=90, labelpad=15)
    
    # Add title
    conditions_str = " & ".join(active_conditions)
    title = f"Generated Images with Varying {conditions_str}\n(Different noise vector per depth)"
    fig.suptitle(title, fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.1, right=0.98, bottom=0.05, wspace=0.02, hspace=0.15)
    
    # Save the figure
    save_filename = os.path.join(model_path_saving, 'generated_inference_grid_enhanced.png')
    plt.savefig(save_filename, dpi=300, bbox_inches='tight', format='png')
    print(f"Image saved to: {save_filename}")

    plt.close(fig)
    
    # Print a summary of inputs used
    print("\nInference generator inputs:")
    print(f"  Number of unique noise vectors: {len(depth_noise_vectors)}")
    for condition in condition_manager.active_conditions:
        print(f"  {condition.name} values:")
        if condition.name == 'category':
            values = [info[2] for info in category_info]  # class values
            print(f"    Classes used: {sorted(values)}")
        elif condition.name == 'porosity':
            all_min = min(r['min'] for r in category_porosity_ranges.values())
            all_max = max(r['max'] for r in category_porosity_ranges.values())
            print(f"    Overall range: {all_min:.6f} to {all_max:.6f}")
            for cat_class in sorted(category_porosity_ranges.keys()):
                range_info = category_porosity_ranges[cat_class]
                print(f"    Category {cat_class} range: {range_info['min']:.6f} to {range_info['max']:.6f}")

def generate_latent_points(latent_dim, n_samples=1):
    return np.random.randn(n_samples, latent_dim)


def evaluate_generator_accuracy(generator, latent_dim, balanced_patch_info, n_classes_category, unet_model, image_type,
                                threshold_value, condition_manager, num_samples=100):
    # Import global category_dimension
    from __main__ import category_dimension

    # Safety check
    if category_dimension is None:
        raise ValueError("category_dimension must be set before evaluation")

    
    # Adjust num_samples if it's larger than available patches
    num_samples = min(num_samples, len(balanced_patch_info))
    
    generated_porosities = []
    target_porosities = []
    category_values = []
    
    selected_samples = random.sample(balanced_patch_info, num_samples)
    
    # Check if category is enabled
    has_category = 'category_actual' in balanced_patch_info[0] if balanced_patch_info else False
    
    for sample in selected_samples:
        # Create generator inputs starting with noise
        generator_inputs = [np.random.randn(1, latent_dim)]
        
        # Add condition inputs based on active conditions
        for condition in condition_manager.active_conditions:
            if condition.name == 'category':
                # Create one-hot vector for the category
                one_hot = np.zeros((1, category_dimension), dtype=np.float32)
                one_hot[0, sample['category_class']] = 1.0
                generator_inputs.append(one_hot)
            elif condition.name == 'porosity':
                generator_inputs.append(np.array([sample['porosity_value']]).reshape(1, 1))
        
        # Generate image
        generated_image = generator.predict(generator_inputs)
        generated_image = ((generated_image[0] + 1) / 2.0 * 255).astype(np.uint8)

        # Calculate porosity using U-Net
        calculated_porosity = calculate_porosity(
            generated_image,
            image_type,
            unet_model=unet_model,
            threshold_value=threshold_value
        )
        
        generated_porosities.append(calculated_porosity)
        target_porosities.append(sample['porosity_value'])
        if has_category:
            category_values.append(sample['category_actual'])
        else:
            category_values.append(0)  # Default category when disabled
    
    generated_porosities = np.array(generated_porosities)
    target_porosities = np.array(target_porosities)
    category_values = np.array(category_values)
    
    correlation_matrix = np.corrcoef(target_porosities, generated_porosities)
    r_squared = correlation_matrix[0, 1] ** 2
    
    plt.figure(figsize=(10, 8))
    
    # Get unique categories for coloring
    unique_categories = np.unique(category_values)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    
    for category, color in zip(unique_categories, colors):
        mask = np.isclose(category_values, category, rtol=1e-5)
        plt.scatter(target_porosities[mask], generated_porosities[mask], 
                   c=[color], label=f'Category {category:.2f}', alpha=0.6)
    
    min_val = min(min(target_porosities), min(generated_porosities))
    max_val = max(max(target_porosities), max(generated_porosities))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Correlation')
    
    z = np.polyfit(target_porosities, generated_porosities, 1)
    p = np.poly1d(z)
    plt.plot(target_porosities, p(target_porosities), 'r-', alpha=0.8, 
             label=f'Best Fit (R² = {r_squared:.4f})')
    
    plt.xlabel('Expected Porosity')
    plt.ylabel('Generated Porosity')
    
    # Add method to title
    method_str = "Porosity Method: Enhanced U-Net"
    plt.title(f'Generated vs Expected Porosity Values ({method_str})')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    from __main__ import model_path_saving
    plt.savefig(os.path.join(model_path_saving, 'generator_accuracy_evaluation.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: generator_accuracy_evaluation.png")
    plt.close()
    
    return {
        'r_squared': r_squared,
        'target_porosities': target_porosities,
        'generated_porosities': generated_porosities,
        'category_values': category_values
    }


def print_data_ranges(balanced_patch_info, condition_manager):
    """Print data ranges for active conditions"""
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    print("\nData Ranges:")
    if 'category' in active_conditions:
        categories = [patch['category_scaled'] for patch in balanced_patch_info]
        print(f"Category scaled range: [{min(categories):.6f}, {max(categories):.6f}]")
        
        # Print unique category value for each class
        category_by_class = {}
        for patch in balanced_patch_info:
            category_by_class[patch['category_class']] = patch['category_scaled']
        
        print("\nCategory value for each class:")
        for category_class in sorted(category_by_class.keys()):
            print(f"Class {category_class}: {category_by_class[category_class]:.6f}")
    else:
        print("Category condition is disabled - no category ranges to display")
    
    if 'porosity' in active_conditions:
        porosities = [patch['porosity_value'] for patch in balanced_patch_info]
        print(f"Porosity range: [{min(porosities):.6f}, {max(porosities):.6f}]")
        
        # Print non-empty pairs
        if 'category' in active_conditions:
            non_empty_pairs = set()
            for patch in balanced_patch_info:
                non_empty_pairs.add((patch['category_class'], patch['porosity_class']))
            print("\nNon-empty class combinations (category_class, porosity_class):")
            print(sorted(non_empty_pairs))
        else:
            non_empty_classes = set()
            for patch in balanced_patch_info:
                non_empty_classes.add(patch.get('porosity_class', 0))
            print("\nNon-empty porosity classes:")
            print(sorted(non_empty_classes))
    else:
        print("Porosity condition is disabled - no porosity ranges to display")

def save_inference_metadata(model_path, n_channels, patch_size, 
                           latent_dim, balanced_patch_info, condition_manager,
                           image_type, porosity_threshold_value=None):
    import os
    import json
    import pickle
    import numpy as np

    
    # Create metadata directory if it doesn't exist
    metadata_dir = os.path.join(model_path, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Collect metadata
    metadata = {
        'n_channels': n_channels,
        'patch_size': patch_size,
        'latent_dim': latent_dim,
        'image_type': image_type.value if hasattr(image_type, 'value') else str(image_type),
        'porosity_threshold_value': porosity_threshold_value
    }
    
    # Save basic metadata as JSON
    with open(os.path.join(metadata_dir, 'config.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Extract and save condition-related information
    condition_data = {
        'active_conditions': [cond.name for cond in condition_manager.active_conditions]
    }
    
    # Collect statistics for each active condition
    if balanced_patch_info:
        for condition in condition_manager.active_conditions:
            if condition.name == 'category':
                # Save category information
                category_values = [patch.get('category_scaled', 0) for patch in balanced_patch_info 
                                   if 'category_scaled' in patch]
                category_classes = [patch.get('category_class', 0) for patch in balanced_patch_info 
                                    if 'category_class' in patch]
                
                if category_values:
                    condition_data['category'] = {
                        'min': float(min(category_values)),
                        'max': float(max(category_values)),
                        'unique_values': [float(v) for v in sorted(set(category_values))],
                        'n_classes': max(category_classes) + 1 if category_classes else 0,
                        'class_to_value': {}
                    }
                    
                    # Map class indices to representative values
                    for cls in range(condition_data['category']['n_classes']):
                        class_values = [patch['category_scaled'] for patch in balanced_patch_info 
                                        if patch.get('category_class') == cls]
                        if class_values:
                            condition_data['category']['class_to_value'][str(cls)] = float(np.mean(class_values))
            
            elif condition.name == 'porosity':
                # Save porosity information with category-specific details
                condition_data['porosity'] = {
                    'min': float(min(patch['porosity_value'] for patch in balanced_patch_info)),
                    'max': float(max(patch['porosity_value'] for patch in balanced_patch_info)),
                    'mean': float(np.mean([patch['porosity_value'] for patch in balanced_patch_info])),
                    'threshold_value': float(
                        porosity_threshold_value) if porosity_threshold_value is not None else None,
                    'category_specific_details': {}
                }
                
                # Collect porosity details for each category
                unique_categories = set(patch.get('category_class', 0) for patch in balanced_patch_info)
                
                for category_class in unique_categories:
                    # Get patches for this specific category
                    category_patches = [
                        patch for patch in balanced_patch_info 
                        if patch.get('category_class', 0) == category_class
                    ]
                    
                    # Analyze porosity for this category
                    category_porosity_data = {
                        'min': float(min(p['porosity_value'] for p in category_patches)),
                        'max': float(max(p['porosity_value'] for p in category_patches)),
                        'mean': float(np.mean([p['porosity_value'] for p in category_patches])),
                        'n_classes': max(p.get('porosity_class', 0) for p in category_patches) + 1,
                        'class_to_value': {},
                        'class_ranges': {}
                    }
                    
                    # Get porosity class details
                    for porosity_class in range(category_porosity_data['n_classes']):
                        class_patches = [
                            p for p in category_patches 
                            if p.get('porosity_class', 0) == porosity_class
                        ]
                        
                        if class_patches:
                            # Representative value (mean)
                            category_porosity_data['class_to_value'][str(porosity_class)] = float(
                                np.mean([p['porosity_value'] for p in class_patches])
                            )
                            
                            # Class range
                            category_porosity_data['class_ranges'][str(porosity_class)] = [
                                float(min(p['porosity_value'] for p in class_patches)),
                                float(max(p['porosity_value'] for p in class_patches))
                            ]
                    
                    # Store category-specific porosity details
                    condition_data['porosity']['category_specific_details'][str(category_class)] = category_porosity_data
    
    # Save condition data as JSON
    with open(os.path.join(metadata_dir, 'conditions.json'), 'w') as f:
        json.dump(condition_data, f, indent=2)
    
    # Also save sample patch info to help with debugging (optional)
    sample_patches = balanced_patch_info[:10] if balanced_patch_info else []
    with open(os.path.join(metadata_dir, 'sample_patches.pkl'), 'wb') as f:
        pickle.dump(sample_patches, f)
    
    print(f"Inference metadata saved to {metadata_dir}")


def create_results_directory(patch_size, image_type, condition_manager, code_version, base_run_number=1):
    """
    Automatically create and return the results directory path based on current date, 
    active conditions, image type, and patch size. Automatically finds the next available
    run number to prevent overwriting existing data.
    
    Args:
        patch_size (int): Size of image patches (e.g., 128, 256)
        image_type (ImageType): Type of image (RGB, GRAYSCALE, etc.)
        condition_manager (ConditionManager): Manager containing active conditions
        code_version (str): Code version string (e.g., "version35_article")
        base_run_number (int): Starting run number to check (default: 1)
        
    Returns:
        str: Full path to the results directory
    """
    # Base directory (constant part)
    base_dir = r"D:\OneDrive - University of Leeds\6. Running Result of GANs code\3. Universal_GAN_Project"
    
    # Get current date in format YYMMDD
    current_date = datetime.datetime.now().strftime("%y%m%d")
    
    # Image type (lowercase)
    img_type_str = image_type.value.lower()
    
    # Determine condition type based on active conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    if 'category' in active_conditions and 'porosity' in active_conditions:
        condition_str = "conditional_category_porosity"
    elif 'category' in active_conditions:
        condition_str = "conditional_category"
    elif 'porosity' in active_conditions:
        condition_str = "conditional_porosity"
    else:
        condition_str = "unconditional"
    
    # Auto-increment run number until finding the first available one
    run_number = base_run_number
    while True:
        # Format run number and patch size
        run_size_str = f"R{run_number}_{patch_size}"
        
        # Generate full path
        full_path = os.path.join(base_dir, current_date, code_version, img_type_str, 
                                condition_str, run_size_str)
        
        # Check if this directory already exists
        if not os.path.exists(full_path):
            break
            
        # If it exists, try the next run number
        run_number += 1
        print(f"Run {run_number-1} already exists, trying run {run_number}...")
    
    # Create directory
    os.makedirs(full_path, exist_ok=True)
    print(f"Created results directory for run {run_number}: {full_path}")
    
    return full_path

def save_training_images_by_class(original_images, balanced_patch_info, patch_size, save_dir, condition_manager):
    """
    Save training images organized by their class to the specified directory.
    
    Args:
        original_images: List of original images
        balanced_patch_info: List of balanced patch info dictionaries
        patch_size: Size of the patches
        save_dir: Directory to save the images
        condition_manager: Manager containing active conditions
    """
    import os
    import cv2
    
    # Determine active conditions
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    
    # Create the "Training sub-images" directory
    training_dir = os.path.join(save_dir, "Training sub-images")
    os.makedirs(training_dir, exist_ok=True)
    
    print(f"\nSaving training images by class to: {training_dir}")
    
    # Counter for tracking success/failures
    save_counts = {}
    failed_count = 0
    
    # Create dictionaries to track unique filenames
    class_file_counters = {}
    
    # Create class directories and save images based on active conditions
    if 'category' in active_conditions and 'porosity' in active_conditions:
        # Both conditions active - organize by combination of classes
        for patch_idx, patch in enumerate(balanced_patch_info):
            try:
                cat_class = patch['category_class']
                por_class = patch['porosity_class']
                
                # Create directory for this class combination
                class_dir = os.path.join(training_dir, f"Cat{cat_class}_Por{por_class}")
                os.makedirs(class_dir, exist_ok=True)
                
                # Initialize counter for this class if not exists
                class_key = f"Cat{cat_class}_Por{por_class}"
                if class_key not in save_counts:
                    save_counts[class_key] = 0
                    class_file_counters[class_key] = 0
                
                # Read and save the patch
                patch_img = read_patch(original_images, patch, patch_size)
                
                # Generate a unique filename using a counter for this class
                counter = class_file_counters[class_key]
                augmented_tag = "_aug" if patch.get('augmented', False) else ""
                base_name = f"img_p{patch['p']}_i{patch['i']}_j{patch['j']}"
                img_filename = f"{base_name}_{counter}{augmented_tag}.png"
                img_path = os.path.join(class_dir, img_filename)
                
                # Increment the counter for this class
                class_file_counters[class_key] += 1
                
                # Save as RGB or grayscale based on channels
                if len(patch_img.shape) >= 3 and patch_img.shape[-1] == 3:  # RGB
                    cv2.imwrite(img_path, cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
                else:  # Grayscale
                    cv2.imwrite(img_path, patch_img.squeeze())
                
                # Increment counter
                save_counts[class_key] += 1
            except Exception as e:
                failed_count += 1
                print(f"Error saving patch {patch_idx}: {str(e)}")
            
    elif 'category' in active_conditions:
        # Only category active - organize by category class
        for patch_idx, patch in enumerate(balanced_patch_info):
            try:
                cat_class = patch['category_class']
                
                # Create directory for this category class
                class_dir = os.path.join(training_dir, f"Cat{cat_class}")
                os.makedirs(class_dir, exist_ok=True)
                
                # Initialize counter for this class if not exists
                class_key = f"Cat{cat_class}"
                if class_key not in save_counts:
                    save_counts[class_key] = 0
                    class_file_counters[class_key] = 0
                
                # Read and save the patch
                patch_img = read_patch(original_images, patch, patch_size)
                
                # Generate a unique filename using a counter for this class
                counter = class_file_counters[class_key]
                augmented_tag = "_aug" if patch.get('augmented', False) else ""
                base_name = f"img_p{patch['p']}_i{patch['i']}_j{patch['j']}"
                img_filename = f"{base_name}_{counter}{augmented_tag}.png"
                img_path = os.path.join(class_dir, img_filename)
                
                # Increment the counter for this class
                class_file_counters[class_key] += 1
                
                # Save as RGB or grayscale based on channels
                if len(patch_img.shape) >= 3 and patch_img.shape[-1] == 3:  # RGB
                    cv2.imwrite(img_path, cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
                else:  # Grayscale
                    cv2.imwrite(img_path, patch_img.squeeze())
                
                # Increment counter
                save_counts[class_key] += 1
            except Exception as e:
                failed_count += 1
                print(f"Error saving patch {patch_idx}: {str(e)}")
    
    elif 'porosity' in active_conditions:
        # Only porosity active - organize by porosity class
        for patch_idx, patch in enumerate(balanced_patch_info):
            try:
                por_class = patch['porosity_class']
                
                # Create directory for this porosity class
                class_dir = os.path.join(training_dir, f"Por{por_class}")
                os.makedirs(class_dir, exist_ok=True)
                
                # Initialize counter for this class if not exists
                class_key = f"Por{por_class}"
                if class_key not in save_counts:
                    save_counts[class_key] = 0
                    class_file_counters[class_key] = 0
                
                # Read and save the patch
                patch_img = read_patch(original_images, patch, patch_size)
                
                # Generate a unique filename using a counter for this class
                counter = class_file_counters[class_key]
                augmented_tag = "_aug" if patch.get('augmented', False) else ""
                base_name = f"img_p{patch['p']}_i{patch['i']}_j{patch['j']}"
                img_filename = f"{base_name}_{counter}{augmented_tag}.png"
                img_path = os.path.join(class_dir, img_filename)
                
                # Increment the counter for this class
                class_file_counters[class_key] += 1
                
                # Save as RGB or grayscale based on channels
                if len(patch_img.shape) >= 3 and patch_img.shape[-1] == 3:  # RGB
                    cv2.imwrite(img_path, cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
                else:  # Grayscale
                    cv2.imwrite(img_path, patch_img.squeeze())
                
                # Increment counter
                save_counts[class_key] += 1
            except Exception as e:
                failed_count += 1
                print(f"Error saving patch {patch_idx}: {str(e)}")
    
    else:
        # No conditions active - just save all images in one directory
        for patch_idx, patch in enumerate(balanced_patch_info):
            try:
                # Initialize counter
                if "all" not in save_counts:
                    save_counts["all"] = 0
                    class_file_counters["all"] = 0
                
                # Read and save the patch
                patch_img = read_patch(original_images, patch, patch_size)
                
                # Generate a unique filename
                counter = class_file_counters["all"]
                augmented_tag = "_aug" if patch.get('augmented', False) else ""
                img_filename = f"img_{counter}{augmented_tag}.png"
                img_path = os.path.join(training_dir, img_filename)
                
                # Increment the counter
                class_file_counters["all"] += 1
                
                # Save as RGB or grayscale based on channels
                if len(patch_img.shape) >= 3 and patch_img.shape[-1] == 3:  # RGB
                    cv2.imwrite(img_path, cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
                else:  # Grayscale
                    cv2.imwrite(img_path, patch_img.squeeze())
                
                # Increment counter
                save_counts["all"] += 1
            except Exception as e:
                failed_count += 1
                print(f"Error saving patch {patch_idx}: {str(e)}")

    # Count and print summary statistics only
    num_classes_saved = len(save_counts)
    print(f"\nTraining images saved: {num_classes_saved} classes")
    
    # Generate verification statistics 
    expected_total = len(balanced_patch_info)
    actual_total = sum(save_counts.values())
    
    print(f"Total saved training images: {actual_total}")
    print(f"Total patches in balanced_patch_info: {expected_total}")
    if failed_count > 0:
        print(f"Failed to save {failed_count} images")
    
    # Count augmented patches saved
    augmented_count = sum(1 for patch in balanced_patch_info if patch.get('augmented', False))
    augmented_saved = 0
    for class_name, count in save_counts.items():
        class_patches = [p for p in balanced_patch_info if 
                        (('category_class' in p and f"Cat{p['category_class']}" in class_name) or
                         ('porosity_class' in p and f"Por{p['porosity_class']}" in class_name) or
                         class_name == "all")]
        augmented_saved += sum(1 for p in class_patches if p.get('augmented', False))
    
    print(f"Augmented patches in balanced_patch_info: {augmented_count}")
    print(f"Augmented patches saved: {augmented_saved}")
    

def create_models_subdirectory(main_directory):
    """
    Creates a 'Saved Models' subdirectory within the specified main directory.
    
    Args:
        main_directory (str): Path to the main directory
        
    Returns:
        str: Path to the 'Saved Models' subdirectory
    """
    models_dir = os.path.join(main_directory, "Saved Models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"Created models directory: {models_dir}")
    return models_dir