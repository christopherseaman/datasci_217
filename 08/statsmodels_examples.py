"""Brief examples of statsmodels usage"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import PanelOLS

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)  # 3 features
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100)

# Linear regression
print("\n=== Linear Regression ===")
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const)
results = model.fit()
print(results.summary().tables[1])  # Coefficient table

# GLM example
print("\n=== GLM ===")
# Binary outcome for logistic regression
y_binary = (y > y.mean()).astype(int)
glm_model = sm.GLM(y_binary, X_with_const, family=sm.families.Binomial())
glm_results = glm_model.fit()
print("GLM coefficients:", glm_results.params)

# Panel data example
print("\n=== Panel Data ===")
# Generate panel data
entities = 10
times = 5
panel_idx = pd.MultiIndex.from_product([range(entities), range(times)],
                                     names=['entity', 'time'])
panel_data = pd.DataFrame({
    'y': np.random.randn(entities * times),
    'x1': np.random.randn(entities * times)
}, index=panel_idx)

model_panel = PanelOLS(panel_data.y, panel_data.x1, entity_effects=True)
panel_results = model_panel.fit()
print("Panel regression coefficients:", panel_results.params)

# Robust regression
print("\n=== Robust Regression ===")
rlm_model = sm.RLM(y, X_with_const)
rlm_results = rlm_model.fit()
print("Robust regression coefficients:", rlm_results.params)

if __name__ == '__main__':
    # Optional: Diagnostic plots
    import matplotlib.pyplot as plt
    fig = sm.graphics.plot_regress_exog(results, 'x1')
    plt.show()
