# Demo 2: seaborn Statistical Visualization

## Learning Objectives
- Create statistical plots with seaborn
- Visualize relationships between variables
- Analyze distributions and patterns
- Apply seaborn styling and themes

## Setup

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set inline plotting for Jupyter
%matplotlib inline

# Set seaborn style
sns.set_style('whitegrid')
```

## Part 1: Statistical Plot Types

### Box Plots and Violin Plots

```python
# Load sample data
tips = sns.load_dataset('tips')

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Box plot
sns.boxplot(data=tips, x='day', y='tip', ax=axes[0])
axes[0].set_title('Tip Distribution by Day')

# Violin plot
sns.violinplot(data=tips, x='day', y='tip', ax=axes[1])
axes[1].set_title('Tip Distribution Shape by Day')

plt.tight_layout()
plt.show()
```

### Distribution Plots

```python
# Create sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histogram
sns.histplot(data=data, ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Histogram')

# Density plot
sns.kdeplot(data=data, ax=axes[0, 1])
axes[0, 1].set_title('Density Plot')

# Combined histogram and density
sns.histplot(data=data, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Histogram + Density')

# Multiple distributions
data1 = np.random.normal(0, 1, 500)
data2 = np.random.normal(2, 1, 500)
sns.kdeplot(data=data1, label='Distribution 1', ax=axes[1, 1])
sns.kdeplot(data=data2, label='Distribution 2', ax=axes[1, 1])
axes[1, 1].set_title('Multiple Distributions')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

## Part 2: Relationship Visualization

### Scatter Plots and Regression

```python
# Create scatter plot with regression line
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Basic scatter plot
sns.scatterplot(data=tips, x='total_bill', y='tip', ax=axes[0])
axes[0].set_title('Total Bill vs Tip')

# Scatter plot with regression line
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[1])
axes[1].set_title('Total Bill vs Tip with Regression')

plt.tight_layout()
plt.show()
```

### Categorical Relationships

```python
# Create categorical plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Strip plot
sns.stripplot(data=tips, x='day', y='tip', ax=axes[0, 0])
axes[0, 0].set_title('Strip Plot')

# Swarm plot
sns.swarmplot(data=tips, x='day', y='tip', ax=axes[0, 1])
axes[0, 1].set_title('Swarm Plot')

# Bar plot
sns.barplot(data=tips, x='day', y='tip', ax=axes[1, 0])
axes[1, 0].set_title('Bar Plot')

# Point plot
sns.pointplot(data=tips, x='day', y='tip', ax=axes[1, 1])
axes[1, 1].set_title('Point Plot')

plt.tight_layout()
plt.show()
```

## Part 3: Advanced Statistical Plots

### Pair Plots

```python
# Create pair plot
iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species')
plt.suptitle('Iris Dataset Pair Plot', y=1.02)
plt.show()
```

### Joint Plots

```python
# Create joint plot
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')
plt.suptitle('Total Bill vs Tip Joint Plot', y=1.02)
plt.show()
```

### Heatmaps

```python
# Create correlation heatmap
# Select only numeric columns for correlation
numeric_tips = tips.select_dtypes(include=[np.number])
correlation_matrix = numeric_tips.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

## Part 4: Multi-dimensional Analysis

### Faceted Plots

```python
# Create faceted plots
g = sns.FacetGrid(tips, col='day', hue='time', col_wrap=2)
g.map(plt.scatter, 'total_bill', 'tip', alpha=0.6)
g.add_legend()
plt.suptitle('Total Bill vs Tip by Day and Time', y=1.02)
plt.show()
```

### Complex Relationships

```python
# Create complex relationship plot
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot with multiple dimensions
scatter = sns.scatterplot(data=tips, x='total_bill', y='tip', 
                         hue='day', size='size', sizes=(50, 200), ax=ax)
ax.set_title('Multi-dimensional Relationship Plot')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
```

## Part 5: Styling and Themes

### Different Styles

```python
# Try different styles
styles = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, style in enumerate(styles):
    sns.set_style(style)
    sns.scatterplot(data=tips, x='total_bill', y='tip', ax=axes[i])
    axes[i].set_title(f'Style: {style}')

# Remove extra subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()
```

### Color Palettes

```python
# Try different color palettes
palettes = ['viridis', 'plasma', 'coolwarm', 'Set1', 'husl']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, palette in enumerate(palettes):
    sns.scatterplot(data=tips, x='total_bill', y='tip', 
                   hue='day', palette=palette, ax=axes[i])
    axes[i].set_title(f'Palette: {palette}')

# Remove extra subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()
```

## Key Takeaways

1. **Statistical Plots**: Use seaborn for box plots, violin plots, and distribution analysis
2. **Relationships**: Visualize correlations and patterns between variables
3. **Multi-dimensional**: Show multiple variables simultaneously with color, size, and facets
4. **Styling**: Apply consistent themes and color palettes for professional appearance
5. **Best Practices**: Choose appropriate plot types for your data and analysis goals

## Next Steps

- Practice with your own datasets
- Experiment with different statistical plot types
- Learn about pandas plotting for quick exploration
- Explore modern visualization tools like altair
