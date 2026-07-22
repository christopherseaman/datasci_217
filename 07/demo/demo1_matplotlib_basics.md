# Demo 1: matplotlib Fundamentals

## Learning Objectives
- Create figures and subplots
- Customize plot appearance
- Use different plot types effectively
- Apply colors, markers, and line styles

## Setup

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set inline plotting for Jupyter
%matplotlib inline
```

## Part 1: Figures and Subplots

Every matplotlib plot requires a Figure object. The Figure contains one or more subplots (Axes objects) where the actual plotting occurs.

### Create a Basic Figure

The basic matplotlib workflow: create data, create figure, plot data, customize, display.

```python
# Create a simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True)
plt.show()
```

### Create Multiple Subplots

Use `plt.subplots()` to create a grid of subplots. Access individual subplots using array indexing: `axes[row, col]`.

```python
# Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot on each subplot
x = np.linspace(0, 10, 100)

# Top-left: Line plot
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine Wave')
axes[0, 0].grid(True)

# Top-right: Cosine
axes[0, 1].plot(x, np.cos(x), color='red')
axes[0, 1].set_title('Cosine Wave')
axes[0, 1].grid(True)

# Bottom-left: Scatter plot
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100)
axes[1, 0].scatter(x_scatter, y_scatter, alpha=0.6)
axes[1, 0].set_title('Scatter Plot')
axes[1, 0].grid(True)

# Bottom-right: Histogram
data = np.random.normal(0, 1, 1000)
axes[1, 1].hist(data, bins=30, alpha=0.7)
axes[1, 1].set_title('Histogram')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

## Part 2: Plot Customization

Control plot appearance through color, marker, and line style parameters.

### Colors, Markers, and Line Styles

Format string syntax: `'markerstyle-linestyle-color'` (e.g., `'o-', 's--', '^-.'`).

```python
# Create sample data
x = np.linspace(0, 10, 20)
y1 = x
y2 = x**0.5
y3 = np.log(x + 1)
y4 = np.sin(x)

# Create figure with multiple lines
fig, ax = plt.subplots(figsize=(10, 6))

# Different line styles and markers
ax.plot(x, y1, 'o-', label='Linear', color='blue', markersize=8)
ax.plot(x, y2, 's--', label='Square Root', color='red', markersize=6)
ax.plot(x, y3, '^-.', label='Logarithm', color='green', markersize=8)
ax.plot(x, y4, '*:', label='Sine', color='purple', markersize=10)

# Customize appearance
ax.set_title('Different Line Styles and Markers')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Advanced Customization

Set font sizes, colors, and annotations for publication-quality output.

```python
# Create a more sophisticated plot
fig, ax = plt.subplots(figsize=(12, 8))

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y1 = np.sin(x) + 0.1 * np.random.randn(100)
y2 = np.cos(x) + 0.1 * np.random.randn(100)

# Plot with customization
ax.plot(x, y1, label='Sine + Noise', color='#1f77b4', linewidth=2, alpha=0.8)
ax.plot(x, y2, label='Cosine + Noise', color='#ff7f0e', linewidth=2, alpha=0.8)

# Add trend lines
z1 = np.polyfit(x, y1, 1)
p1 = np.poly1d(z1)
ax.plot(x, p1(x), '--', color='#1f77b4', alpha=0.5, label='Sine Trend')

z2 = np.polyfit(x, y2, 1)
p2 = np.poly1d(z2)
ax.plot(x, p2(x), '--', color='#ff7f0e', alpha=0.5, label='Cosine Trend')

# Customize appearance
ax.set_title('Trigonometric Functions with Trend Lines', fontsize=16, fontweight='bold')
ax.set_xlabel('X values', fontsize=12)
ax.set_ylabel('Y values', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')

# Add annotations
ax.annotate('Peak', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

plt.tight_layout()
plt.show()
```

## Part 3: Different Plot Types

Common plot types: line plots, bar charts, histograms, scatter plots, box plots.

### Bar Charts and Histograms

Bar charts: categorical data. Histograms: numerical data distribution.

```python
# Create sample data
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 12, 78]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart
axes[0].bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[0].set_title('Bar Chart')
axes[0].set_xlabel('Categories')
axes[0].set_ylabel('Values')
axes[0].grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(values):
    axes[0].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

# Histogram
data = np.random.normal(0, 1, 1000)
axes[1].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[1].set_title('Histogram')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Scatter Plots and Box Plots

Scatter plots: relationships between two variables. Box plots: distribution and outliers.

```python
# Create sample data
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = 2 * x + np.random.randn(n)
categories = np.random.choice(['Group A', 'Group B', 'Group C'], n)
data_by_group = [np.random.normal(i, 1, 50) for i in range(3)]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scatter plot
scatter = axes[0].scatter(x, y, c=range(n), cmap='viridis', alpha=0.6)
axes[0].set_title('Scatter Plot with Color Mapping')
axes[0].set_xlabel('X values')
axes[0].set_ylabel('Y values')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0])

# Box plot
box_plot = axes[1].boxplot(data_by_group, labels=['Group 1', 'Group 2', 'Group 3'])
axes[1].set_title('Box Plot')
axes[1].set_xlabel('Groups')
axes[1].set_ylabel('Values')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Part 4: Saving and Exporting

### Save Figures

Use `plt.savefig()` with appropriate format and DPI settings.

```python
# Create a publication-quality figure
fig, ax = plt.subplots(figsize=(10, 6))

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/5)

# Create plot
ax.plot(x, y, linewidth=2, color='#2E8B57')
ax.set_title('Damped Sine Wave', fontsize=16, fontweight='bold')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.grid(True, alpha=0.3)

# Save in different formats
plt.savefig('damped_sine.png', dpi=300, bbox_inches='tight')
plt.savefig('damped_sine.pdf', bbox_inches='tight')
plt.savefig('damped_sine.svg', bbox_inches='tight')

plt.show()
print("Figures saved as PNG, PDF, and SVG")
```

## Key Takeaways

1. **Figures and Subplots**: Use `plt.subplots()` to create multiple plots in one figure
2. **Customization**: Control colors, markers, line styles, and text elements
3. **Plot Types**: Choose the right plot type for your data (line, bar, scatter, histogram, box)
4. **Export**: Save figures in high quality for presentations and publications
5. **Best Practices**: Use clear labels, appropriate colors, and good contrast

