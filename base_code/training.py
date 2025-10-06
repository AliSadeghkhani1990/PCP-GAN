"""
Training pipeline for Conditional GAN.

Handles batch preparation, alternating G/D training with TensorFlow GradientTape,
learning rate scheduling, model checkpointing, and loss visualization.
Uses separate Adam optimizers with polynomial decay for stable training.
"""

import sys
import os
# Add the parent directory of 'multi_condition' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from base_code.data_processing import read_patch
from base_code.utils import plot_generated_images_enhanced

import numpy as np
import random

# In training.py
def generate_real_samples(original_images, balanced_patch_info, n_samples, patch_size, condition_manager):
    # Import the global category_dimension variable
    from __main__ import category_dimension

    # Safety check
    if category_dimension is None:
        raise ValueError("category_dimension must be set before training")
    
    indices = np.random.randint(0, len(balanced_patch_info), n_samples)
    
    selected_patches = [read_patch(original_images, balanced_patch_info[i], patch_size) for i in indices]
    selected_X = np.array(selected_patches, dtype=np.float32)
    selected_X = (selected_X / 255.0 - 0.5) * 2  # Scale to [-1, 1]
    
    # Get condition values
    condition_inputs = []
    for condition in condition_manager.active_conditions:
        if condition.name == 'category':
            # One-hot encode the category values
            one_hot_categories = np.zeros((n_samples, category_dimension), dtype=np.float32)
            
            for i, idx in enumerate(indices):
                # Get the class index (0-based)
                class_idx = balanced_patch_info[idx]['category_class']
                # Set the corresponding one-hot bit
                one_hot_categories[i, class_idx] = 1.0
                
            condition_inputs.append(one_hot_categories)
        else:
            # For other conditions like porosity, keep as before
            condition_values = [balanced_patch_info[i][f'{condition.name}_value'] 
                              for i in indices]
            condition_inputs.append(np.array(condition_values, dtype=np.float32))
    
    return [selected_X] + condition_inputs, np.ones((n_samples, 1))

def generate_latent_points(latent_dim, n_samples, condition_manager, balanced_patch_info, non_empty_classes):
    """Generate latent points for GAN input with one-hot encoding for categories"""
    # Import the global category_dimension variable
    from __main__ import category_dimension
    # Safety check
    if category_dimension is None:
        raise ValueError("category_dimension must be set before generating latent points")
    
    z_input = np.random.randn(latent_dim * n_samples).reshape(n_samples, latent_dim)
    
    if not condition_manager.active_conditions:
        return [z_input]
    
    # Create mappings for continuous values and prepare one-hot encoding
    unique_values = {}
    condition_values = []
    
    # First pass: collect all unique values for each condition
    for condition in condition_manager.active_conditions:
        if condition.name == 'category':
            # Get unique category classes
            if balanced_patch_info:
                unique_classes = set(patch['category_class'] for patch in balanced_patch_info)
                
                # Generate random category indices
                selected_classes = np.random.choice(list(unique_classes), size=n_samples)
                
                # Create one-hot vectors
                one_hot_categories = np.zeros((n_samples, category_dimension), dtype=np.float32)
                for i, cls_idx in enumerate(selected_classes):
                    one_hot_categories[i, cls_idx] = 1.0
                    
                condition_values.append(one_hot_categories)
            else:
                # Fallback if no balanced_patch_info
                random_classes = np.random.randint(0, category_dimension, size=n_samples)
                one_hot = np.zeros((n_samples, category_dimension), dtype=np.float32)
                for i, cls_idx in enumerate(random_classes):
                    one_hot[i, cls_idx] = 1.0
                condition_values.append(one_hot)
        else:
            # For other conditions like porosity, keep as before
            if condition.name == 'porosity':
                values = [patch['porosity_value'] for patch in balanced_patch_info]
                unique_values[condition.name] = sorted(set(values))
                available_values = unique_values[condition.name]
                selected_values = np.random.choice(available_values, size=n_samples)
                condition_values.append(selected_values.astype(np.float32))
    
    return [z_input] + condition_values

def generate_fake_samples(generator, latent_dim, n_samples, condition_manager, balanced_patch_info, non_empty_classes):
    # Get latent points and condition inputs
    inputs = generate_latent_points(latent_dim, n_samples, condition_manager, balanced_patch_info, non_empty_classes)
    # Generate images
    images = generator.predict(inputs)
    # Create labels for discriminator (all fake = 0)
    y = np.zeros((n_samples, 1))
    return [images] + inputs[1:] if len(inputs) > 1 else [images], y

def train(g_model, d_model, dataset, balanced_patch_info, latent_dim, epochs, n_batch, 
          saving_step, non_empty_classes, patch_size, model_path_saving, n_classes_category, n_classes_porosity,
          save_interval, last_epochs_to_save, bat_per_epo, n_channels, condition_manager):
    
    original_images = dataset[0]
    
    gen_losses = []
    disc_losses = []
    # Calculate total steps for the learning rate schedule
    total_steps = epochs * bat_per_epo
    # Initialize learning rate schedules
    lr_scheduleG = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=2e-4,
        decay_steps=total_steps,
        end_learning_rate=2e-6,
        power=1.0
    )
    lr_scheduleD = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=2e-4,
        decay_steps=total_steps,
        end_learning_rate=2e-6,
        power=1.0
    )
    # Initialize optimizers
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduleG, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduleD, beta_1=0.5)
    d_model.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
    # Define loss function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    @tf.function
    def train_step(real_samples_and_conditions):
        # Unpack the inputs dynamically
        real_images = real_samples_and_conditions[0]
        condition_inputs = real_samples_and_conditions[1:]
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, latent_dim])
    
        # Train Discriminator
        with tf.GradientTape() as disc_tape:
            generator_inputs = [noise] + list(condition_inputs)
            generated_images = g_model(generator_inputs, training=True)
            discriminator_real_inputs = [real_images] + list(condition_inputs)
            discriminator_fake_inputs = [generated_images] + list(condition_inputs)
        
            real_output = d_model(discriminator_real_inputs, training=True)
            fake_output = d_model(discriminator_fake_inputs, training=True)

            d_loss_real = loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake

        d_gradients = disc_tape.gradient(d_loss, d_model.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, d_model.trainable_variables))

        # Train Generator
        with tf.GradientTape() as gen_tape:
            generator_inputs = [noise] + list(condition_inputs)
            generated_images = g_model(generator_inputs, training=True)
            discriminator_inputs = [generated_images] + list(condition_inputs)
            fake_output = d_model(discriminator_inputs, training=False)
            g_loss = loss_fn(tf.ones_like(fake_output), fake_output)

        g_gradients = gen_tape.gradient(g_loss, g_model.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, g_model.trainable_variables))

        return d_loss, g_loss

    # Training loop
    fixed_noise = None
    fixed_classes = None

    progress_bar = tqdm(range(epochs), desc='Epochs')
    for i in progress_bar:
        total_gen_loss = 0
        total_disc_loss = 0

        for j in range(bat_per_epo):
            # Get real samples with conditions
            real_samples_and_labels = generate_real_samples(original_images, balanced_patch_info, n_batch, patch_size, condition_manager)
            real_samples, y_real = real_samples_and_labels
            
            # Convert all inputs to tensors
            real_samples = [tf.convert_to_tensor(x, dtype=tf.float32) for x in real_samples]

            d_loss, g_loss = train_step(real_samples)
            
            total_disc_loss += d_loss
            total_gen_loss += g_loss

        mean_gen_loss = total_gen_loss / bat_per_epo
        mean_disc_loss = total_disc_loss / bat_per_epo

        gen_losses.append(mean_gen_loss)
        disc_losses.append(mean_disc_loss)
        
        # Get current learning rates
        current_step = tf.constant(i * bat_per_epo, dtype=tf.float32)
        current_g_lr = g_optimizer.learning_rate(current_step).numpy()
        current_d_lr = d_optimizer.learning_rate(current_step).numpy()

        progress_bar.set_postfix({
            'gen_loss': mean_gen_loss.numpy(),
            'disc_loss': mean_disc_loss.numpy()
        })
        print(f"\nCurrent learning rates - Generator: {current_g_lr:.2e}, Discriminator: {current_d_lr:.2e}")
        # Save models logic
        save_current_epoch = (i >= epochs - last_epochs_to_save) or (save_interval and (i + 1) % saving_step == 0)
        if save_current_epoch:
            g_model.save(os.path.join(model_path_saving, f'G_epoch_{i + 1}.h5'), save_format='tf', include_optimizer=True)
            d_model.save(os.path.join(model_path_saving, f'D_epoch_{i + 1}.h5'), save_format='tf', include_optimizer=True)
            print(f"Models saved at epoch {i + 1}")

          # Enhanced visualization call
        fixed_noise, fixed_classes = plot_generated_images_enhanced(
            g_model, i+1, latent_dim, n_channels, condition_manager, balanced_patch_info, 
            fixed_noise, fixed_classes
        )
    plot_training_results(gen_losses, disc_losses, epochs)

def plot_training_results(gen_losses, disc_losses, epochs):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot for generator and discriminator losses
    ax1.set_xlabel('Epoch')
    ax1.set_title('Generator and Discriminator Loss')

    color = 'tab:red'
    ax1.set_ylabel('Generator Loss', color=color)
    ax1.plot(range(1, epochs + 1), gen_losses, color=color, label='Generator Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax1_twin = ax1.twinx()
    color = 'tab:blue'
    ax1_twin.set_ylabel('Discriminator Loss', color=color)
    ax1_twin.plot(range(1, epochs + 1), disc_losses, color=color, label='Discriminator Loss')
    ax1_twin.tick_params(axis='y', labelcolor=color)

    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    plt.tight_layout()
    from __main__ import results_directory
    from base_code.utils import get_plot_path
    save_path = get_plot_path(results_directory, 'evaluation_plots', 'training_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: training_results.png to evaluation_plots/")
    plt.close(fig)