"""Brief examples of Keras usage"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 28, 28, 1)  # Simulated MNIST-like data
y = np.random.randint(0, 10, 1000)  # 10 classes

# Sequential API example
def create_sequential_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Functional API example
def create_functional_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs, outputs)

# Custom model example
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
        self.pool = keras.layers.MaxPooling2D(2)
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        return self.dense(x)

# Data preprocessing
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

if __name__ == '__main__':
    # Train sequential model
    model = create_sequential_model()
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Add callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5'),
        keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    
    # Train with data augmentation
    model.fit(datagen.flow(X, y, batch_size=32),
              epochs=5,
              callbacks=callbacks)
    
    # Save and load model
    model.save('sequential_model.h5')
    loaded_model = keras.models.load_model('sequential_model.h5')
