"""Brief examples of Patsy usage"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'y': np.random.randn(100),
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Basic formula
print("\n=== Basic Formula ===")
y, X = patsy.dmatrices('y ~ x1 + x2', data)
print("Design matrix shape:", X.shape)

# Transformations
print("\n=== Transformations ===")
y, X = patsy.dmatrices('y ~ standardize(x1) + center(x2)', data)
print("Transformed x1 mean:", X[:, 1].mean())
print("Transformed x2 mean:", X[:, 2].mean())

# Categorical variables
print("\n=== Categorical Variables ===")
y, X = patsy.dmatrices('y ~ C(category)', data)
print("Category design matrix:\n", X)

# Interactions
print("\n=== Interactions ===")
y, X = patsy.dmatrices('y ~ x1 + x2 + x1:x2', data)
print("Interaction design matrix shape:", X.shape)

# Custom transformations
print("\n=== Custom Transformations ===")
def log_plus_1(x):
    return np.log(x - x.min() + 1)

y, X = patsy.dmatrices('y ~ log_plus_1(x1)', data)
print("Custom transformation result:", X[:, 1][:5])

# Integration with statsmodels
print("\n=== Statsmodels Integration ===")
model = sm.OLS.from_formula('y ~ x1 + x2', data)
results = model.fit()
print(results.params)
