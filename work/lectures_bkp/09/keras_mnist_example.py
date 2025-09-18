#!/usr/bin/env python3
"""
Keras MNIST Classification Example

This script demonstrates a complete machine learning workflow using Keras/TensorFlow:
1. Data Loading and Preprocessing
2. Model Architecture Design
3. Training and Validation
4. Performance Evaluation

Dataset: MNIST Handwritten Digit Recognition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. DATA PREPARATION
def prepare_data():
    """
    Prepare MNIST dataset for training and validation
    
    Steps:
    - Load train and test datasets
    - Normalize pixel values
    - Convert labels to categorical
    """
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Reshape and normalize images
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    
    # Convert labels to categorical
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    return train_images, train_labels, test_images, test_labels

# 2. MODEL ARCHITECTURE
def create_model():
    """
    Create a simple neural network for MNIST classification
    
    Architecture:
    - Convolutional layers
    - Fully connected layers
    - Dropout for regularization
    """
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 3. TRAINING PROCESS
def train_model(model, train_images, train_labels, test_images, test_labels):
    """
    Train and validate the model
    
    Steps:
    - Fit the model
    - Evaluate on test data
    """
    # Train the model
    history = model.fit(
        train_images, train_labels, 
        epochs=5, 
        batch_size=64, 
        validation_data=(test_images, test_labels),
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f'\nTest accuracy: {test_accuracy * 100:.2f}%')
    
    return history

def main():
    """
    Main function to orchestrate the machine learning workflow
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Prepare data
    train_images, train_labels, test_images, test_labels = prepare_data()
    
    # Create and train model
    model = create_model()
    train_model(model, train_images, train_labels, test_images, test_labels)

if __name__ == '__main__':
    main()
