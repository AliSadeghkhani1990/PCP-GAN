import os
import sys
# Add the parent directory of 'multi_condition' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from base_code.data_processing import load_images, extract_and_save_patch_info, categorize_patches, load_unet_model, ImageType, read_patch, calculate_porosity
from base_code.data_augmentation import balance_and_analyze_dataset_with_augmentation, patch_functions
from base_code.models import generator_model, discriminator_model
from base_code.training import train
from base_code.conditions import create_default_condition_manager
from base_code.utils import (analyze_patch_info, draw_heatmaps,
                                   plot_generated_images_inference_enhanced, evaluate_generator_accuracy,
                                   visualize_generated_samples, print_data_ranges, save_inference_metadata, 
                                   create_results_directory, save_training_images_by_class, create_models_subdirectory
                                   )
from tensorflow.keras.models import load_model

# Configuration
code_version = "Version48_Article_Multidepth_Publication"
patch_size = 480
threshold_value = 85
num_patches_per_category_class = 120
n_batch = 8
epochs = 1
latent_dim = 100
learning_rate = 0.0002
beta_1 = 0.5
saving_step = 5
save_interval = True
last_epochs_to_save = 20
n_classes_porosity = 10
desired_images_per_class = 16
min_images_per_class = 2
bat_per_epo = 300
truncation_percentage_left = 0.0
truncation_percentage_right = 0.05

# Image type configuration
image_type = ImageType.RGB
n_channels = {
    ImageType.RGB: 3,
    ImageType.GRAYSCALE: 1,
    ImageType.BINARY: 1,
    ImageType.ONE_HOT: 4
}[image_type]

# Directory with all depths
directory_path = r'D:\OneDrive - University of Leeds\6. Running Result of GANs code\6. Training Images\1. Colorful\2. CGAN Article\Set 5\Set5.1\Scale Bar_500\cropped\Single run\data2\non_combined'
reload = False  # Start with new models since we're using multiple depths now
model_names = ['G.h5', 'D.h5']
local_model_path_loading = r'D:\OneDrive - University of Leeds\6. Running Result of GANs code\3. Universal_GAN_Project\250710\Version48_Article_Multidepth\rgb\conditional_category_porosity\R1_512\Load Model'
unet_model_path = r"D:\OneDrive - University of Leeds\29.unet\models\best_model.h5"

category_dimension = 4  # Expected to have 4 depths: 1879.50, 1881.90, 1883.75, and 1884.25

def load_or_create_models(n_channels, patch_size, condition_manager):
    """Load existing models or create new ones"""
    if reload:
        model_path = local_model_path_loading
        # Check if all model files exist locally
        for model_name in model_names:
            model_file_path = os.path.join(local_model_path_loading, model_name)
            if not os.path.exists(model_file_path):
                print(f"Model file {model_name} not found in {local_model_path_loading}")
                return create_new_models(n_channels, patch_size, condition_manager)
        
        print(f"Models will be loaded from: {model_path}")
        g_model = load_model(os.path.join(model_path, 'G.h5'), compile=False)
        d_model = load_model(os.path.join(model_path, 'D.h5'), compile=False)
        print("Loaded saved models.")
    else:
        return create_new_models(n_channels, patch_size, condition_manager)
    
    return g_model, d_model

def create_new_models(n_channels, patch_size, condition_manager):
    """Create new models for training"""
    print("Creating new models...")
    g_model = generator_model(latent_dim, n_channels, patch_size, condition_manager)
    d_model = discriminator_model((patch_size, patch_size, n_channels), condition_manager)
    return g_model, d_model

def main():
    global category_dimension, model_path_saving, results_directory
    print("Starting multi-depth conditional GAN training process...")
    # Create and configure condition manager
    print("Setting up condition manager...")
    condition_manager = create_default_condition_manager()
    # IMPORTANT: Enable BOTH category and porosity conditions
    condition_manager.enable_condition('category')
    condition_manager.enable_condition('porosity')
    active_conditions = [cond.name for cond in condition_manager.active_conditions]
    print("Active conditions:", active_conditions)
    # Print porosity calculation information
    print("Using enhanced U-Net model for porosity calculation")
    
    # Create results directory
    results_directory = create_results_directory(patch_size, image_type, condition_manager, code_version)
    print(f"Results will be saved to: {results_directory}")
    
    # Create models subdirectory
    model_path_saving = create_models_subdirectory(results_directory)
    print(f"Models will be saved to: {model_path_saving}")
    
    # Patch the data_augmentation module with the real functions
    patch_functions(read_patch, calculate_porosity)
    
    # Load images from all depth directories
    print("Loading images from multiple depths...")
    original_images, category_values, category_classes = load_images(directory_path, image_type, n_channels, condition_manager)
    
    # Update category dimension based on actual data
    n_classes_category = len(category_classes) if category_classes is not None else 1
    print(f"Found {n_classes_category} depth categories")
    
    if 'category' in active_conditions and category_classes is not None:
        category_dimension = n_classes_category
        print(f"Set category dimension to {category_dimension} based on dataset")
        # Print the actual depth values for each category class
        for depth, class_idx in category_classes.items():
            print(f"  Category class {class_idx}: Depth {depth}m")

    # Load U-Net model if needed
    unet_model = None
    if 'porosity' in active_conditions:
        print("Loading U-Net model...")
        unet_model = load_unet_model(unet_model_path)
    
    # Extract and categorize patches
    print("Extracting and categorizing patches for all depths...")
    patch_info = extract_and_save_patch_info(
        original_images, category_values, patch_size,
        num_patches_per_category_class, unet_model, image_type,
        threshold_value,
        truncation_percentage_left, truncation_percentage_right, condition_manager
    )
    
    # Categorize patches (importantly, this will categorize porosity per-depth)
    patch_info = categorize_patches(patch_info, n_classes_porosity, condition_manager)
    
    # Count patches per depth/category
    if 'category' in active_conditions:
        depth_counts = {}
        for patch in patch_info:
            depth = patch['category_actual']
            if depth not in depth_counts:
                depth_counts[depth] = 0
            depth_counts[depth] += 1
        
        print("\nPatch distribution by depth:")
        for depth, count in sorted(depth_counts.items()):
            print(f"  Depth {depth}m: {count} patches")
    
    # Balance dataset using augmentation
    print("Balancing dataset with augmentation across all depths...")
    balanced_patch_info, repetition_percentages, empty_classes, class_counts, non_empty_classes = balance_and_analyze_dataset_with_augmentation(
        patch_info,
        n_classes_category,
        n_classes_porosity,
        desired_images_per_class,
        min_images_per_class,
        num_patches_per_category_class,
        condition_manager,
        original_images,
        patch_size,
        image_type,
        unet_model,
        threshold_value
    )

    # Save training images organized by depth and porosity class
    save_training_images_by_class(original_images, balanced_patch_info, patch_size, results_directory, condition_manager)
    
    # Debugging info
    print("\n=== DEBUGGING INFO ===")
    print(f"Total number of images used for training: {len(balanced_patch_info)}")
    print(f"Original patch count before balancing: {len(patch_info)}")
    print(f"Class distribution after balancing: {class_counts}")
    
    # Check distribution of balanced dataset by depth
    if 'category' in active_conditions:
        depth_counts = {}
        for patch in balanced_patch_info:
            depth = patch['category_actual']
            if depth not in depth_counts:
                depth_counts[depth] = 0
            depth_counts[depth] += 1
        
        print("\nBalanced patch distribution by depth:")
        for depth, count in sorted(depth_counts.items()):
            print(f"  Depth {depth}m: {count} patches")
    print("======================\n")

    # Analyze data ranges for generation
    print("\nAnalyzing data ranges for generation...")
    print_data_ranges(balanced_patch_info, condition_manager)
    # Analyze patch info and get porosity ranges
    print("Analyzing patch info...")
    analyze_patch_info(patch_info, condition_manager)
    # Draw heatmaps to visualize dataset balance
    print("Drawing heatmaps...")
    draw_heatmaps(patch_info, balanced_patch_info, condition_manager)
    
    # Load or create models with condition manager
    g_model, d_model = load_or_create_models(n_channels, patch_size, condition_manager)
    
    # Print model summaries
    print("Generator Summary:")
    g_model.summary()
    print("\nDiscriminator Summary:")
    d_model.summary()
    
    # Prepare dataset for training
    dataset = [original_images]
    
    # Test sample generation before training
    print("\nTesting sample generation before training...")
    visualize_generated_samples(original_images, balanced_patch_info, 10, patch_size, n_channels, condition_manager)
    
    # Train the model
    print("Starting model training...")
    train(g_model, d_model, dataset, balanced_patch_info, latent_dim, epochs, n_batch, 
          saving_step, non_empty_classes, patch_size, model_path_saving, n_classes_category, n_classes_porosity,
          save_interval, last_epochs_to_save, bat_per_epo, n_channels, condition_manager)
    # After training, evaluate generator accuracy
    print("Evaluating generator...")
    if 'porosity' in active_conditions:
        evaluation_results = evaluate_generator_accuracy(g_model, latent_dim, balanced_patch_info,
                                                         n_classes_category, unet_model, image_type,
                                                         threshold_value,
                                                         condition_manager)
        print(f"\nOverall R² value: {evaluation_results['r_squared']:.4f}")
    else:
        print("Porosity condition is disabled - skipping porosity-based evaluation")

    # Generate inference images
    plot_generated_images_inference_enhanced(g_model, latent_dim, n_channels, condition_manager, 
                                           model_path_saving, patch_size, balanced_patch_info,
                                           non_empty_classes)
    # Save metadata for inference
    print("Saving metadata for inference...")
    save_inference_metadata(
        model_path=model_path_saving,
        n_channels=n_channels,
        patch_size=patch_size,
        latent_dim=latent_dim,
        balanced_patch_info=balanced_patch_info,
        condition_manager=condition_manager,
        image_type=image_type,
        porosity_threshold_value=threshold_value
    )
    print("Multi-depth conditional GAN training process completed.")

if __name__ == "__main__":
    main()