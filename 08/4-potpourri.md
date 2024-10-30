# Patsy (1/3)

## Formula Specification
```python
import patsy

# Basic syntax
y, X = patsy.dmatrices('y ~ x1 + x2', data)

# Transformations
'y ~ standardize(x1) + center(x2)'

# Interactions
'y ~ x1 + x2 + x1:x2'
```

<!--
Talking points:
- Patsy provides R-like formula syntax
- Easy specification of model terms
- Built-in transformations available
- Interaction terms easily specified
-->

---

# Patsy (2/3)

## Design Matrices
```python
# Creating matrices
y, X = patsy.dmatrices('y ~ x1 + C(category)', data)

# Categorical data
'y ~ C(category, Treatment)'
'y ~ C(category, Sum)'

# Missing data
y, X = patsy.dmatrices('y ~ x1', data, NA_action='drop')
```

<!--
Talking points:
- Automatic creation of design matrices
- Special handling for categorical data
- Multiple coding schemes available
- Missing data handling options
-->

---

# Patsy (3/3)

## Advanced Features
```python
# Custom transformations
def log_plus_1(x): return np.log(x + 1)
'y ~ log_plus_1(x1)'

# Splines
'y ~ bs(x, df=3)'

# Integration with statsmodels
model = sm.OLS.from_formula('y ~ x1 + x2', data)
```

<!--
Talking points:
- Custom transformations possible
- Spline basis functions supported
- Seamless statsmodels integration
- Flexible formula specification
-->

---

# Deep Learning Frameworks Introduction (1/2)

## Framework Overview
```python
# Keras/TensorFlow
import tensorflow as tf
from tensorflow import keras

# PyTorch
import torch
import torch.nn as nn

# Core concepts comparison
keras_model = keras.Sequential()  # Layer-based
torch_model = nn.Module()        # Class-based
```

<!--
Talking points:
- Multiple frameworks available
- Different programming paradigms
- Keras focuses on simplicity
- PyTorch emphasizes flexibility
-->

---

# Deep Learning Frameworks Introduction (2/2)

Hand-wavy explanation (many caveats apply)
- When to use Keras:
  - Quick prototyping
  - High-level APIs needed
  - Production with TensorFlow Serving

- When to use PyTorch:
  - Research/experimentation
  - Custom architectures
  - Dynamic computation graphs

Note: _most_ neural nets perform better with normalized data regardless of platform

<!--
Talking points:
- Framework choice depends on use case
- Keras excels in production deployment
- PyTorch popular in research
- Data preprocessing important for all
-->

---
