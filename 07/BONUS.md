# Advanced Data Visualization Topics

## Annotations and Drawing on Plots

### Adding Text and Annotations

**Reference:**

- `ax.text(x, y, 'text')` - Add text at coordinates
- `ax.annotate('text', xy=(x, y), xytext=(x2, y2))` - Add annotation with arrow
- `ax.arrow(x, y, dx, dy)` - Add arrow
- `ax.axhline(y=value)` - Add horizontal line
- `ax.axvline(x=value)` - Add vertical line

**Example:**

```python
# Annotate important points
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data)

# Add text annotation
ax.text(50, data[50], 'Peak Value', fontsize=12, ha='center')

# Add arrow annotation
ax.annotate('Important Event', 
           xy=(100, data[100]), 
           xytext=(150, data[100] + 10),
           arrowprops=dict(arrowstyle='->', color='red'))

# Add reference lines
ax.axhline(y=data.mean(), color='gray', linestyle='--', alpha=0.7)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
```

### Drawing Shapes and Patches

**Reference:**

```python
from matplotlib.patches import Rectangle, Circle, Polygon

# Add shapes to plots
rect = Rectangle((x, y), width, height, color='blue', alpha=0.3)
circle = Circle((x, y), radius, color='red', alpha=0.3)
polygon = Polygon([(x1, y1), (x2, y2), (x3, y3)], color='green', alpha=0.3)

ax.add_patch(rect)
ax.add_patch(circle)
ax.add_patch(polygon)
```

## matplotlib Configuration

### Global Configuration

**Reference:**

- `plt.rcParams` - Access all configuration parameters
- `plt.rc('font', size=12)` - Set font size
- `plt.rc('figure', figsize=(8, 6))` - Set default figure size
- `plt.rcdefaults()` - Reset to defaults

**Example:**

```python
# Custom matplotlib configuration
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Create plot with custom settings
fig, ax = plt.subplots()
ax.plot(data)
```

### Style Sheets

**Reference:**

```python
# Available styles
plt.style.available  # List all available styles

# Use a style
plt.style.use('seaborn-v0_8')
plt.style.use('ggplot')
plt.style.use('bmh')

# Create custom style
plt.style.use({
    'figure.facecolor': 'white',
    'axes.facecolor': 'lightgray',
    'axes.grid': True,
    'grid.color': 'white'
})
```

## Advanced pandas Plotting

### Subplot Layouts

**Reference:**

```python
# Advanced subplot options
df.plot(subplots=True, layout=(2, 2), sharex=True, sharey=True)
df.plot(subplots=True, figsize=(12, 8), title='Custom Title')
```

### Stacked and Grouped Plots

**Reference:**

```python
# Stacked bar plots
df.plot.bar(stacked=True, alpha=0.7)

# Grouped bar plots
df.plot.bar(x='category', y='value', color=['red', 'blue', 'green'])

# Area plots
df.plot.area(alpha=0.7, stacked=True)
```

## Advanced seaborn Features

### Statistical Visualization

**Reference:**

- `sns.pairplot()` - Pairwise relationships
- `sns.jointplot()` - Joint distributions
- `sns.violinplot()` - Distribution shapes
- `sns.heatmap()` - Correlation matrices
- `sns.clustermap()` - Hierarchical clustering heatmap

**Example:**

```python
# Advanced seaborn statistical plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Pair plot for correlation analysis
sns.pairplot(df, hue='category')

# Joint plot with regression
sns.jointplot(data=df, x='x', y='y', kind='reg')

# Violin plot for distribution comparison
sns.violinplot(data=df, x='category', y='value')

# Heatmap for correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

### Facet Grids and Categorical Plots

**Reference:**

```python
# Advanced categorical plots
sns.catplot(data=df, x='category', y='value', hue='group', kind='box')
sns.catplot(data=df, x='category', y='value', col='time', row='group')

# Facet grid
g = sns.FacetGrid(df, col='category', row='group')
g.map(sns.scatterplot, 'x', 'y')
```

## Advanced matplotlib Customization

### Publication-Quality Plots

**Reference:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality defaults
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False
})

# Create publication-quality plot
fig, ax = plt.subplots(figsize=(8, 6))

# Your plotting code here
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, linewidth=2, label='sin(x)')
ax.set_xlabel('X values', fontsize=14)
ax.set_ylabel('Y values', fontsize=14)
ax.set_title('Publication-Quality Plot', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Save with high DPI
plt.savefig('publication_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Custom Color Palettes

**Reference:**

```python
# Define custom color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

# Or use colormap
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors = ['#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Use in plot
plt.imshow(data, cmap=cmap)
plt.colorbar()
```

## Interactive Visualizations

### Bokeh for Interactive Plots

**Reference:**

```python
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool
import numpy as np

# Enable notebook output
output_notebook()

# Create interactive plot
p = figure(title="Interactive Scatter Plot", 
           x_axis_label='X', y_axis_label='Y',
           width=600, height=400)

# Add hover tool
hover = HoverTool(tooltips=[("index", "$index"),
                           ("(x,y)", "($x, $y)")])
p.add_tools(hover)

# Generate data
x = np.random.randn(100)
y = np.random.randn(100)

# Add scatter plot
p.circle(x, y, size=10, alpha=0.6, color='blue')

# Show plot
show(p)
```

### Plotly for Interactive Dashboards

**Reference:**

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create interactive scatter plot
fig = px.scatter(df, x='total_bill', y='tip', 
                 color='time', size='size',
                 hover_data=['day', 'smoker'],
                 title='Interactive Tips Analysis')

# Add trend line
fig.add_trace(go.Scatter(x=df['total_bill'], 
                        y=df['tip'],
                        mode='lines',
                        name='Trend',
                        line=dict(dash='dash')))

# Show plot
fig.show()
```

## Advanced seaborn Features

### Statistical Plotting

**Reference:**

```python
import seaborn as sns

# Statistical plots
sns.regplot(data=df, x='x', y='y')  # Regression plot
sns.residplot(data=df, x='x', y='y')  # Residual plot
sns.distplot(data=df, x='column')  # Distribution plot
sns.kdeplot(data=df, x='x', y='y')  # 2D density plot

# Advanced statistical plots
sns.pairplot(df, hue='category', diag_kind='kde')
sns.jointplot(data=df, x='x', y='y', kind='hex')
sns.clustermap(df.corr(), annot=True, cmap='coolwarm')
```

### Custom Themes and Styles

**Reference:**

```python
# Set custom theme
sns.set_theme(style="whitegrid", 
              palette="husl",
              font_scale=1.2,
              rc={"figure.figsize": (10, 8)})

# Or create custom style
custom_style = {
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
}

sns.set_style("white", rc=custom_style)
```

## Animation and Dynamic Plots

### matplotlib Animation

**Reference:**

```python
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# Create animated plot
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', linewidth=2)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

def animate(frame):
    x = np.linspace(0, 10, 100)
    y = np.sin(x + frame * 0.1)
    line.set_data(x, y)
    return line,

# Create animation
anim = FuncAnimation(fig, animate, frames=100, 
                    interval=50, blit=True)

# Save as GIF
anim.save('sine_wave.gif', writer='pillow', fps=20)
```

### Real-time Data Visualization

**Reference:**

```python
import time
import random

# Real-time plotting
fig, ax = plt.subplots()
x_data, y_data = [], []

def update_plot():
    # Add new data point
    x_data.append(time.time())
    y_data.append(random.random())
    
    # Keep only last 100 points
    if len(x_data) > 100:
        x_data.pop(0)
        y_data.pop(0)
    
    # Update plot
    ax.clear()
    ax.plot(x_data, y_data)
    ax.set_title('Real-time Data')
    plt.pause(0.1)

# Run for 10 seconds
start_time = time.time()
while time.time() - start_time < 10:
    update_plot()
```

## Advanced Color Theory

### Colorblind-Friendly Palettes

**Reference:**

```python
# Colorblind-friendly palettes
colorblind_palettes = {
    'colorblind': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    'viridis': ['#440154', '#31688e', '#35b779', '#fde725'],
    'plasma': ['#0d0887', '#7e03a8', '#cc4778', '#f0f921']
}

# Use in plots
sns.set_palette(colorblind_palettes['viridis'])
```

### Color Psychology in Data Visualization

**Reference:**

```python
# Emotional color associations
emotional_colors = {
    'trust': '#1f77b4',      # Blue
    'energy': '#ff7f0e',     # Orange
    'growth': '#2ca02c',     # Green
    'danger': '#d62728',     # Red
    'luxury': '#9467bd',     # Purple
    'warmth': '#bcbd22'      # Yellow
}

# Use contextually
def choose_color_for_data(data_type, value):
    if data_type == 'sales' and value > 1000:
        return emotional_colors['growth']
    elif data_type == 'errors' and value > 10:
        return emotional_colors['danger']
    else:
        return emotional_colors['trust']
```

## Performance Optimization

### Large Dataset Visualization

**Reference:**

```python
# For large datasets, use sampling
def plot_large_dataset(df, sample_size=10000):
    if len(df) > sample_size:
        df_sample = df.sample(sample_size)
        print(f"Sampled {sample_size} points from {len(df)} total")
    else:
        df_sample = df
    
    # Use efficient plot types
    plt.scatter(df_sample['x'], df_sample['y'], alpha=0.1, s=1)
    plt.show()

# Or use hexbin for density
plt.hexbin(df['x'], df['y'], gridsize=50, cmap='Blues')
plt.colorbar()
```

### Memory-Efficient Plotting

**Reference:**

```python
# Clear memory between plots
import gc

def memory_efficient_plotting():
    # Create plot
    fig, ax = plt.subplots()
    ax.plot(data)
    plt.show()
    
    # Clean up
    plt.close(fig)
    gc.collect()
```

## Export and Sharing

### Multiple Format Export

**Reference:**

```python
# Export to multiple formats
def export_plot(fig, filename_base):
    # High-res PNG
    fig.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
    
    # Vector formats
    fig.savefig(f'{filename_base}.svg', bbox_inches='tight')
    fig.savefig(f'{filename_base}.pdf', bbox_inches='tight')
    
    # Web formats
    fig.savefig(f'{filename_base}.jpg', dpi=150, bbox_inches='tight')
```

### Interactive HTML Export

**Reference:**

```python
# Export interactive plots to HTML
import plotly.offline as pyo

# Create plotly figure
fig = px.scatter(df, x='x', y='y')

# Export to HTML
pyo.plot(fig, filename='interactive_plot.html', auto_open=False)
```

## Advanced Statistical Visualization

### Confidence Intervals

**Reference:**

```python
# Add confidence intervals to plots
def plot_with_confidence(x, y, ax):
    # Calculate confidence interval
    mean_y = np.mean(y)
    std_y = np.std(y)
    n = len(y)
    se = std_y / np.sqrt(n)
    ci = 1.96 * se  # 95% confidence interval
    
    # Plot mean line
    ax.axhline(mean_y, color='red', linestyle='-', linewidth=2)
    
    # Plot confidence interval
    ax.axhspan(mean_y - ci, mean_y + ci, alpha=0.3, color='red')
    
    # Add labels
    ax.text(0.02, 0.98, f'Mean: {mean_y:.2f} ± {ci:.2f}', 
            transform=ax.transAxes, verticalalignment='top')
```

### Statistical Annotations

**Reference:**

```python
# Add statistical annotations
from scipy import stats

def add_statistical_annotations(ax, x, y):
    # Calculate correlation
    r, p_value = stats.pearsonr(x, y)
    
    # Add text annotation
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_value:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
```

## Custom Plot Types

### Waterfall Charts

**Reference:**

```python
def create_waterfall_chart(data, labels):
    """Create waterfall chart for showing cumulative changes"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate cumulative values
    cumulative = np.cumsum([0] + data)
    
    # Create bars
    for i, (label, value) in enumerate(zip(labels, data)):
        color = 'green' if value >= 0 else 'red'
        ax.bar(i, value, bottom=cumulative[i], color=color, alpha=0.7)
        ax.text(i, cumulative[i] + value/2, f'{value:.1f}', 
                ha='center', va='center')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title('Waterfall Chart')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### Sankey Diagrams

**Reference:**

```python
# Sankey diagram for flow visualization
def create_sankey_diagram():
    import plotly.graph_objects as go
    
    # Define flows
    source = [0, 1, 0, 2, 3, 3]
    target = [2, 3, 3, 4, 4, 5]
    value = [8, 4, 2, 8, 4, 2]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["A", "B", "C", "D", "E", "F"]
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(title_text="Sankey Diagram", font_size=10)
    fig.show()
```

## Visualization Testing and Validation

### Automated Plot Testing

**Reference:**

```python
# Test plot properties
def test_plot_properties(fig, expected_properties):
    """Test that plot has expected properties"""
    ax = fig.axes[0]
    
    # Test title
    if 'title' in expected_properties:
        assert ax.get_title() == expected_properties['title']
    
    # Test axis labels
    if 'xlabel' in expected_properties:
        assert ax.get_xlabel() == expected_properties['xlabel']
    
    # Test data range
    if 'xlim' in expected_properties:
        xlim = ax.get_xlim()
        assert xlim[0] == expected_properties['xlim'][0]
        assert xlim[1] == expected_properties['xlim'][1]
```

### Plot Quality Metrics

**Reference:**

```python
# Calculate plot quality metrics
def calculate_plot_quality(fig):
    """Calculate various quality metrics for a plot"""
    ax = fig.axes[0]
    
    metrics = {
        'has_title': bool(ax.get_title()),
        'has_xlabel': bool(ax.get_xlabel()),
        'has_ylabel': bool(ax.get_ylabel()),
        'has_legend': bool(ax.get_legend()),
        'has_grid': ax.grid,
        'aspect_ratio': fig.get_figwidth() / fig.get_figheight()
    }
    
    return metrics
```

These advanced topics will help you create professional, publication-ready visualizations and handle complex visualization challenges in your data science work.
