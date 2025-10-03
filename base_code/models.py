"""
Generator and Discriminator architectures for Conditional GAN.

Generator: Transforms noise + conditions into images using transposed convolutions.
Discriminator: Classifies real vs fake images considering condition values.
Both models support multiple conditions (category via one-hot, porosity as scalar).
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Concatenate, Dropout


def generator_model(latent_dim, n_channels, patch_size, condition_manager):
    # Import the global category_dimension from main module
    from __main__ import category_dimension

    # Safety check
    if category_dimension is None:
        raise ValueError("category_dimension must be set before creating generator model")
        
    noise_input = Input(shape=(latent_dim,))
    # Create condition inputs
    condition_inputs = []
    condition_embeddings = []
    initial_size = patch_size // 16    
    
    for condition in condition_manager.active_conditions:
        if condition.name == 'category':
            # Create one-hot input for categories (n_categories channels)
            cond_input = Input(shape=(category_dimension,), dtype='float32')
            condition_inputs.append(cond_input)
            
            # Reshape for spatial concatenation - create individual channel for each category
            li = Reshape((1, 1, category_dimension))(cond_input)
            li = tf.tile(li, [1, initial_size, initial_size, 1])
            condition_embeddings.append(li)
        else:
            # For other conditions like porosity, keep as before
            cond_input = Input(shape=(1,), dtype='float32')
            condition_inputs.append(cond_input)
            
            li = Reshape((1, 1, 1))(cond_input)
            li = tf.tile(li, [1, initial_size, initial_size, 1])
            condition_embeddings.append(li)        
 
    # Generate from noise
    gen = Dense(initial_size * initial_size * 512)(noise_input)
    gen = LeakyReLU(0.2)(gen)
    gen = Reshape((initial_size, initial_size, 512))(gen)
     
    # Combine with conditions if any are active
    if condition_embeddings:
        gen = Concatenate()([gen] + condition_embeddings)
    
    # Transposed convolutions
    x = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same')(gen)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    generated_output = Conv2DTranspose(n_channels, (5, 5), strides=(2, 2), padding='same', activation='tanh')(x)
    
    # Create model with appropriate inputs
    model_inputs = [noise_input] + condition_inputs
    model = Model(inputs=model_inputs, outputs=generated_output)    
    return model


def discriminator_model(in_shape, condition_manager):
    # Import the global category_dimension from main module
    from __main__ import category_dimension

    # Safety check
    if category_dimension is None:
        raise ValueError("category_dimension must be set before creating discriminator model")
        
    image_input = Input(shape=in_shape)
    
    # Create condition inputs
    condition_inputs = []
    condition_embeddings = []
    
    for condition in condition_manager.active_conditions:
        if condition.name == 'category':
            # Create one-hot input for categories
            cond_input = Input(shape=(category_dimension,), dtype='float32')
            condition_inputs.append(cond_input)
            
            # Reshape for spatial concatenation
            li = Reshape((1, 1, category_dimension))(cond_input)
            li = tf.tile(li, [1, in_shape[0], in_shape[1], 1])
            condition_embeddings.append(li)
        else:
            # For other conditions like porosity, keep as before
            cond_input = Input(shape=(1,), dtype='float32')
            condition_inputs.append(cond_input)
            
            li = Reshape((1, 1, 1))(cond_input)
            li = tf.tile(li, [1, in_shape[0], in_shape[1], 1])
            condition_embeddings.append(li)
    
    # Combine image with conditions if any are active
    if condition_embeddings:
        x = Concatenate()([image_input] + condition_embeddings)
    else:
        x = image_input

    # Convolutional layers
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    validity_output = Dense(1, activation='sigmoid')(x)
    model_inputs = [image_input] + condition_inputs
    model = Model(inputs=model_inputs, outputs=validity_output)    
    return model