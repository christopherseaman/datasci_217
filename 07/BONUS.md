---
notion:
  title_line: "# DLC: Advanced Data Visualization Topics"
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-8172-a97a-dc1e9e5b577a"
  url: "https://app.notion.com/p/3d2d9fdd1a1a8172a97adc1e9e5b577a"
---

# DLC: Advanced Data Visualization Topics

# Annotations and Drawing on Plots

## Reference Lines and Shaded Spans

The lecture's annotation card covers `ax.text()` and `ax.annotate()` for labeling one point. Reference lines and shaded spans mark context that runs across the whole Axes, such as a season average, the week a program started, or an outbreak period.

### Reference Card: Reference Lines and Spans

- `ax.axhline(y, color='gray', linestyle='--')`: Draw a horizontal line across the whole Axes, such as a target or a mean.
- `ax.axvline(x, linestyle=':')`: Draw a vertical line across the whole Axes, such as the week an intervention started.
- `ax.axhspan(ymin, ymax, alpha=0.2)` / `ax.axvspan(xmin, xmax, alpha=0.2)`: Shade a horizontal or vertical band, such as a normal range or an outbreak period.
- `ax.arrow(x, y, dx, dy)`: Draw a bare arrow from `(x, y)` that moves `dx` across and `dy` up; `ax.annotate()` is usually easier because it pairs the arrow with text.

### Code Snippet: Mark a Mean, an Event, and a Period

```python
import matplotlib.pyplot as plt
import numpy as np

weeks = np.arange(1, 13)  # weeks 1 through 12
flu_visits = np.array([40, 42, 45, 51, 60, 72, 80, 76, 64, 55, 48, 44])
print(round(flu_visits.mean(), 1))  # 56.4

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(weeks, flu_visits, marker='o')
ax.axhline(flu_visits.mean(), color='gray', linestyle='--')
ax.text(1, flu_visits.mean() + 2, 'Season mean')
ax.axvline(3, color='gray', linestyle=':')
ax.text(3.1, 85, 'Vaccine clinic opens')
ax.axvspan(6, 8, color='orange', alpha=0.2)
ax.text(7, 30, 'Outbreak', ha='center')
ax.set(xlabel='Week', ylabel='Flu clinic visits', ylim=(0, 90))
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

Expected output: `56.4` prints, and the plot shows twelve weekly points with a dashed gray line labeled "Season mean" at 56.4, a dotted line at week 3 labeled "Vaccine clinic opens", and weeks 6 through 8 shaded and labeled "Outbreak".

## Drawing Shapes and Patches

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

# matplotlib Configuration

## Global Configuration

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

## Style Sheets

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

# Advanced pandas Plotting

## Subplot Layouts

**Reference:**

```python
# Advanced subplot options
df.plot(subplots=True, layout=(2, 2), sharex=True, sharey=True)
df.plot(subplots=True, figsize=(12, 8), title='Custom Title')
```

## Stacked and Grouped Plots

**Reference:**

```python
# Stacked bar plots
df.plot.bar(stacked=True, alpha=0.7)

# Grouped bar plots
df.plot.bar(x='category', y='value', color=['red', 'blue', 'green'])

# Area plots
df.plot.area(alpha=0.7, stacked=True)
```

# Advanced seaborn Features

## Statistical Visualization

**Figure-level** functions such as `pairplot()`, `jointplot()`, `catplot()`, and `clustermap()` build a whole Figure of their own, so they do not take `ax=` and cannot share a `plt.subplots()` grid. The other functions below draw into one Axes, like the plots in the lecture.

### Reference Card: More seaborn Plot Types

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `sns.pairplot(df, hue=...)` | Compare every numeric pair | Figure-level grid |
| `sns.jointplot(data=df, x=..., y=..., kind=...)` | Combine a relationship with marginal distributions | Figure-level plot |
| `sns.catplot(kind='box', data=df, x=..., y=...)` | Build a faceted categorical plot | Figure-level grid |
| `sns.clustermap(matrix)` | Heatmap with rows and columns reordered by hierarchical clustering (needs SciPy) | Figure-level grid |
| `sns.violinplot(data=df, x=..., y=...)` | Show distribution shape by category | `Axes` |
| `sns.stripplot(data=df, x=..., y=..., hue=...)` | Show individual observations by category | `Axes` |
| `sns.regplot(data=df, x=..., y=...)` | Scatter plot with a fitted regression line and its confidence band | `Axes` |
| `sns.residplot(data=df, x=..., y=...)` | Residuals from that fitted line, to check for leftover pattern | `Axes` |

## Facet Grids and Categorical Plots

**Reference:**

```python
# Advanced categorical plots
sns.catplot(data=df, x='category', y='value', hue='group', kind='box')
sns.catplot(data=df, x='category', y='value', col='time', row='group')

# Facet grid
g = sns.FacetGrid(df, col='category', row='group')
g.map(sns.scatterplot, 'x', 'y')
```

## Custom Themes and Styles

**Reference:**

```python
sns.set_theme(style="whitegrid",
              palette="husl",
              font_scale=1.2,
              rc={"figure.figsize": (10, 8)})

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

# Advanced matplotlib Customization

## Publication-Quality Plots

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

## Custom Color Palettes

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

# Interactive Visualizations

*The Python visualization ecosystem is constantly evolving. While matplotlib and seaborn are the workhorses, modern libraries offer exciting new approaches.*

This survey names alternatives to the lecture's tools; the same visible-context rules still apply. Extended Altair, Bokeh, and Plotly examples follow it.

## Ecosystem at a Glance

- **plotnine** brings a layered grammar-of-graphics interface familiar to ggplot2 users.
- **Bokeh** targets browser-based visualizations, custom interactions, and server applications.
- **Plotly** offers a high-level Express API plus lower-level graph objects for interactive charts and dashboards.

## Tool Selection Guide

| Tool | Best For | Learning Curve | Interactivity | Output Formats | Grammar |
|------|----------|----------------|---------------|----------------|---------|
| matplotlib | Custom plots, publication quality | High | Pan/zoom in desktop or widget backends | PNG/SVG/PDF | Imperative |
| seaborn | Statistical plots, beautiful defaults | Low | Pan/zoom in desktop or widget backends | PNG/SVG/PDF | Imperative |
| pandas | Quick exploration, basic charts | Very Low | Pan/zoom in desktop or widget backends | PNG/SVG/PDF | Imperative |
| altair | Interactive plots, grammar of graphics | Medium | Built-in | PNG/SVG/HTML/JSON | Declarative |
| plotnine | R users, layered approach | Medium | Pan/zoom in desktop or widget backends | PNG/SVG/PDF | Declarative |
| bokeh | Interactive web visualizations | High | High | HTML/JS | Imperative |
| plotly | Dashboards, web applications | Medium | High | HTML/JS | Declarative |

## Altair for Declarative Interactive Charts

**Reference:**

```python
import altair as alt

base = alt.Chart(df).encode(
    x='x:Q',
    y='y:Q',
    color='category:N',
    tooltip=['x', 'y', 'category']
)

# Layer points with a fitted line, then enable pan and zoom.
chart = (
    base.mark_circle()
    + base.transform_regression('x', 'y').mark_line()
).interactive()

chart.save('interactive_chart.html')
```

## Bokeh for Interactive Plots

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

## Plotly for Interactive Dashboards

**Reference:**

```python
import plotly.express as px

# Add a fitted ordinary-least-squares line. A raw line connecting rows in
# dataframe order would not represent a statistical trend. Plotly delegates
# OLS fitting to its optional statsmodels dependency, and color='time' fits
# one line per time group rather than one pooled line.
fig = px.scatter(df, x='total_bill', y='tip',
                 color='time', size='size',
                 hover_data=['day', 'smoker'],
                 trendline='ols',
                 title='Interactive Tips Analysis with fitted trend')

# Show plot
fig.show()
```

# Animation and Dynamic Plots

## matplotlib Animation

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

## Real-time Data Visualization

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

# Advanced Color Theory

## Colorblind-Friendly Palettes

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

## Color Psychology in Data Visualization

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

# Performance Optimization

## Large Dataset Visualization

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

## Memory-Efficient Plotting

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

# Export and Sharing

## Multiple Format Export

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

## Interactive HTML Export

**Reference:**

```python
# Export interactive plots to HTML
import plotly.offline as pyo

# Create plotly figure
fig = px.scatter(df, x='x', y='y')

# Export to HTML
pyo.plot(fig, filename='interactive_plot.html', auto_open=False)
```

# Advanced Statistical Visualization

## Confidence Intervals

**Reference:**

```python
# Add a one-sample t confidence interval for a mean. This is appropriate when
# y is an independent sample from one population and its distribution is
# reasonably symmetric (or n is large enough for the t approximation).
import numpy as np
from scipy import stats

def plot_with_confidence(y, ax):
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        raise ValueError('at least two observations are required')
    mean_y = y.mean()
    se = stats.sem(y)
    half_width = stats.t.ppf(0.975, df=n - 1) * se

    ax.axhline(mean_y, color='red', linewidth=2, label='sample mean')
    ax.axhspan(mean_y - half_width, mean_y + half_width,
               alpha=0.3, color='red', label='95% t interval')
    ax.text(0.02, 0.98,
            f'Mean: {mean_y:.2f} (95% t interval: '
            f'{mean_y - half_width:.2f}–{mean_y + half_width:.2f})',
            transform=ax.transAxes, va='top')
```

## Statistical Annotations

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

# Custom Plot Types

## Waterfall Charts

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

## Sankey Diagrams

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

# Visualization Testing and Validation

## Automated Plot Testing

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

## Plot Quality Metrics

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
        'has_grid': any(
            line.get_visible()
            for line in ax.get_xgridlines() + ax.get_ygridlines()
        ),
        'aspect_ratio': fig.get_figwidth() / fig.get_figheight()
    }
    
    return metrics
```

# Further Reading

## Tufte's Books and Essays

- [The Visual Display of Quantitative Information](https://www.edwardtufte.com/tufte/books_vdqi) - Tufte's seminal work
- [Envisioning Information](https://www.edwardtufte.com/book/envisioning-information/) - Color, layering, and detail
- [Tufte's website](https://www.edwardtufte.com/) - Essays and resources

## Color Tools

- [ColorBrewer 2.0](https://colorbrewer2.org/) - Interactive color advice for maps and visualizations
- [Colorblind-Safe Palettes](https://sronpersonalpages.nl/~pault/) - Paul Tol's color schemes
- [Adobe Color](https://color.adobe.com/) - Create and explore color schemes
