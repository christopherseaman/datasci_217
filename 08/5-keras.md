# Keras Implementation (1/8)

## Layers & Activation Functions
```python
from tensorflow import keras

# Dense layers
model.add(keras.layers.Dense(64))

# Convolutional layers
model.add(keras.layers.Conv2D(32, (3, 3)))

# RNN layers
model.add(keras.layers.LSTM(64))

# Activation options
keras.activations.relu
keras.activations.sigmoid
keras.activations.tanh
```

<!--
Talking points:
- Keras provides high-level neural network APIs
- Common layer types available out of box
- Built-in activation functions
- Easy layer stacking with add()
-->

---

# Keras Implementation (2/8)

## Building Models - Part 1
```python
# Sequential API
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Model lifecycle
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
```

<!--
Talking points:
- Sequential API for linear layer stacks
- Easy model definition and compilation
- Common optimizers and losses built-in
- Multiple metrics can be tracked
-->

---

# Keras Implementation (3/8)

## Building Models - Part 2
```python
# Functional API
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(64, activation='relu')(inputs)
outputs = keras.layers.Dense(10)(x)
model = keras.Model(inputs, outputs)

# Model subclassing
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(64)
        self.dense2 = keras.layers.Dense(10)
```

<!--
Talking points:
- Functional API for complex architectures
- Multiple inputs/outputs possible
- Subclassing for custom behavior
- Different APIs for different needs
-->

---

# Keras Implementation (4/8)

## Data Preprocessing
```python
# Image data generators
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Sequence generators
seq = keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=100
)

# Text preprocessing
vectorizer = keras.layers.TextVectorization(max_tokens=20000)
```

<!--
Talking points:
- Built-in data augmentation tools
- Sequence handling utilities
- Text preprocessing capabilities
- Real-time data generation
-->

---

# Keras Implementation (5/8)

## Loss Functions & Optimizers
```python
# Common loss functions
keras.losses.BinaryCrossentropy()
keras.losses.CategoricalCrossentropy()
keras.losses.MeanSquaredError()

# Optimizer selection
keras.optimizers.Adam(learning_rate=0.001)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Learning rate scheduling
keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.9
)
```

<!--
Talking points:
- Comprehensive loss function library
- Multiple optimizer options
- Learning rate scheduling
- Easy parameter tuning
-->

---

# Keras Implementation (6/8)

## Model Training
```python
# Fit method
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=3),
    keras.callbacks.ModelCheckpoint('best_model.h5'),
    keras.callbacks.TensorBoard(log_dir='./logs')
]
```

<!--
Talking points:
- Simple fit API for training
- Built-in validation support
- Rich callback system
- Training monitoring tools
-->

---

# Keras Implementation (7/8)

## Transfer Learning
```python
# Pre-trained models
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False
)

# Feature extraction
base_model.trainable = False

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False
```

<!--
Talking points:
- Pre-trained models available
- Easy feature extraction
- Fine-tuning capabilities
- Layer freezing control
-->

---

# Keras Implementation (8/8)

## Model Evaluation
```python
# Metrics
model.evaluate(x_test, y_test)

# Custom metrics
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()

# Visualization
keras.utils.plot_model(model, show_shapes=True)
```

<!--
Talking points:
- Built-in evaluation methods
- Custom metric definition
- Model visualization tools
- Comprehensive evaluation options
-->

---
