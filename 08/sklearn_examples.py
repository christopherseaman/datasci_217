"""Brief examples of scikit-learn usage"""
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 20)  # 20 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary classification
X_cat = np.random.choice(['A', 'B', 'C'], size=(1000, 2))  # Categorical features

# Preprocessing
print("\n=== Preprocessing ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled data mean:", X_scaled.mean(axis=0)[:3])

enc = OneHotEncoder(sparse=False)
X_encoded = enc.fit_transform(X_cat)
print("Encoded categories:", enc.categories_)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Classification example
print("\n=== Classification ===")
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation
print("\n=== Cross-validation ===")
scores = cross_val_score(clf, X_scaled, y, cv=5)
print("CV scores:", scores)

# Pipeline example
print("\n=== Pipeline ===")
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])
pipe.fit(X, y)
print("Pipeline score:", pipe.score(X, y))

# Dimensionality reduction
print("\n=== PCA ===")
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print("Explained variance ratio:", pca.explained_variance_ratio_)

if __name__ == '__main__':
    # Optional: Plot PCA results
    import matplotlib.pyplot as plt
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
    plt.title('PCA visualization')
    plt.show()
