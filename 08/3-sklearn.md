# scikit-learn (1/9)

## Data Preprocessing
```python
# Input: X (array-like of shape (n_samples, n_features))
# Output: standardized features with zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Input: X_cat (array-like of categorical variables)
# Output: one-hot encoded matrix
enc = OneHotEncoder(sparse=False)
X_encoded = enc.fit_transform(X_cat)
```

<!--
Talking points:
- Preprocessing is crucial for model performance
- Scaling ensures features are comparable
- One-hot encoding handles categorical variables
- Always fit preprocessors on training data only
-->

---

# scikit-learn (2/9)

## Feature Selection
```python
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier

# SelectKBest
selector = SelectKBest(k=5)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
rfe = RFE(RandomForestClassifier(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```

<!--
Talking points:
- Feature selection reduces dimensionality
- SelectKBest uses univariate statistical tests
- RFE iteratively removes least important features
- Different methods suit different problems
-->

---

# scikit-learn (3/9)

## Supervised Learning: Classification
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Models
lr = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC(kernel='rbf')

# Common interface
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

<!--
Talking points:
- Classification predicts categorical outcomes
- Different algorithms suit different problems
- Consistent API across all models
- Model selection depends on data characteristics
-->

---

# scikit-learn (4/9)

## Supervised Learning: Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Models
lr = LinearRegression()
rf = RandomForestRegressor()
svr = SVR(kernel='rbf')

# Common interface
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

<!--
Talking points:
- Regression predicts continuous outcomes
- Similar interface to classification
- Different algorithms handle nonlinearity differently
- Consider interpretability vs performance
-->

---

# scikit-learn (5/9)

## Unsupervised Learning
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Clustering
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

<!--
Talking points:
- Unsupervised learning finds patterns without labels
- Clustering groups similar data points
- PCA reduces dimensions while preserving variance
- Different methods reveal different patterns
-->

---

# scikit-learn (6/9)

## Model Selection
```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Grid search
params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(), params, cv=5)
grid.fit(X, y)
```

<!--
Talking points:
- Cross-validation assesses model generalization
- Grid search optimizes hyperparameters
- Multiple metrics available for evaluation
- Balance between computation and thoroughness
-->

---

# scikit-learn (7/9)

## Model Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Classification metrics
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# Regression metrics
print(f"MSE: {mean_squared_error(y_true, y_pred)}")
print(f"R2: {r2_score(y_true, y_pred)}")
```

<!--
Talking points:
- Different metrics suit different problems
- Classification report provides comprehensive view
- Confusion matrix shows error types
- R-squared and MSE common for regression
-->

---

# scikit-learn (8/9)

## Pipeline Construction
```python
from sklearn.pipeline import Pipeline, FeatureUnion

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=5)),
    ('classifier', LogisticRegression())
])

# Feature union
union = FeatureUnion([
    ('numeric', numeric_pipeline),
    ('text', text_pipeline)
])
```

<!--
Talking points:
- Pipelines chain operations together
- Ensures consistent preprocessing
- Feature union combines different features
- Helps prevent data leakage
-->

---

# scikit-learn (9/9)

## Model Persistence
```python
import joblib

# Save model
joblib.dump(model, 'model.joblib')

# Load model
model = joblib.load('model.joblib')

# Update model
model.partial_fit(X_new, y_new)
```

<!--
Talking points:
- Model persistence saves trained models
- Joblib efficient for large NumPy arrays
- Partial fitting allows incremental updates
- Consider version compatibility
-->

---
