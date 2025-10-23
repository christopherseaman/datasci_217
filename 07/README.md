Data Visualization: From Exploration to Communication

See [BONUS.md](BONUS.md) for advanced topics:

- Advanced matplotlib customization and publication-quality plots
- Interactive visualization with Bokeh and Plotly
- Statistical visualization with seaborn advanced features
- Custom color palettes and themes
- Animation and dynamic plots

*Fun fact: The word "visualization" comes from the Latin "visus" meaning "sight." In data science, we're literally making data visible - turning numbers into stories that our eyes can understand and our brains can process.*

![xkcd 1945: Scientific Paper Graph Quality](media/xkcd_1945.png)

*"The data clearly shows that our hypothesis is correct, assuming we ignore all the data that doesn't support our hypothesis."*

Data visualization is the art and science of turning data into insights. This lecture covers the essential tools for creating effective visualizations: **matplotlib for customization**, **seaborn for statistical plots**, and **modern alternatives** for interactive and grammar-of-graphics approaches.

**Learning Objectives:**

- Master matplotlib fundamentals (figures, subplots, customization)
- Create statistical visualizations with seaborn
- Use pandas plotting for quick data exploration
- Understand visualization principles and best practices
- Explore modern visualization libraries (altair, plotnine)
- Manage persistent computing sessions with tmux

# The Visualization Ecosystem

*Reality check: There are more Python visualization libraries than there are ways to mess up a bar chart. But don't worry - we'll focus on the essential tools that actually matter for daily data science work.*

Python's visualization landscape has evolved dramatically. While matplotlib remains the foundation, modern tools like seaborn, altair, and plotnine offer more intuitive interfaces for common tasks.

**Visual Guide - Python Visualization Stack:**

```
FOUNDATION LAYER
┌─────────────────────────────────────┐
│           matplotlib                │  ← Low-level, highly customizable
│     (The foundation of everything)   │
└─────────────────────────────────────┘
                    ↑
                    │
            PANDAS LAYER
┌─────────────────────────────────────┐
│         pandas.plot()              │  ← Quick exploration, built on matplotlib
│     (DataFrame/Series plotting)     │
└─────────────────────────────────────┘
                    ↑
                    │
            STATISTICAL LAYER
┌─────────────────────────────────────┐
│           seaborn                   │  ← Statistical plots, beautiful defaults
│     (Built on matplotlib)           │
└─────────────────────────────────────┘
                    ↑
                    │
            MODERN LAYER
┌─────────────────────────────────────┐
│    altair (vega-lite)               │  ← Grammar of graphics, interactive
│    plotnine (ggplot2)               │  ← R's ggplot2 in Python
└─────────────────────────────────────┘
```

## Choosing the Right Tool

**When to use what:**

- **pandas.plot()** - Quick exploration, basic charts
- **matplotlib** - Custom plots, publication quality, fine control
- **seaborn** - Statistical plots, beautiful defaults, relationship analysis
- **altair** - Interactive plots, grammar of graphics, web-ready
- **plotnine** - If you know ggplot2, consistent API

**Pro tip:** Start with pandas for exploration, seaborn for analysis, matplotlib for customization, and modern tools for interactive/sharing needs.

# LIVE DEMO!

# matplotlib Fundamentals

*Think of matplotlib as the foundation of your visualization house - you can build anything on it, but you need to understand the plumbing before you can install the fancy fixtures.*

matplotlib is the bedrock of Python visualization. While it can be verbose, understanding its core concepts gives you the power to create any visualization you can imagine.

## Figures and Subplots

Every matplotlib plot lives within a `Figure` object, which can contain multiple `subplots` (individual plot areas).

**Reference:**

- `plt.figure(figsize=(width, height))` - Create a new figure
- `fig.add_subplot(rows, cols, position)` - Add subplot to figure
- `plt.subplots(rows, cols)` - Create figure with multiple subplots
- `fig.savefig('filename.png', dpi=300)` - Save figure to file
- `plt.show()` - Display the plot

**Example:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot on each subplot
axes[0, 0].plot([1, 2, 3, 4], [1, 4, 2, 3])
axes[0, 0].set_title('Line Plot')

axes[0, 1].hist(np.random.normal(0, 1, 1000), bins=30)
axes[0, 1].set_title('Histogram')

axes[1, 0].scatter(np.random.randn(100), np.random.randn(100))
axes[1, 0].set_title('Scatter Plot')

axes[1, 1].bar(['A', 'B', 'C'], [3, 7, 2])
axes[1, 1].set_title('Bar Chart')

plt.tight_layout()
plt.show()
```

## Customizing Plots

matplotlib's power comes from its extensive customization options.

**Reference:**

- `ax.set_title('Title')` - Set plot title
- `ax.set_xlabel('X Label')` - Set x-axis label
- `ax.set_ylabel('Y Label')` - Set y-axis label
- `ax.set_xlim(min, max)` - Set x-axis limits
- `ax.set_ylim(min, max)` - Set y-axis limits
- `ax.grid(True)` - Add grid lines
- `ax.legend()` - Add legend
- `ax.set_style('seaborn')` - Change plot style

**Example:**

```python
# Create a customized plot
fig, ax = plt.subplots(figsize=(8, 6))

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot with customization
ax.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
ax.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')

# Customize appearance
ax.set_title('Trigonometric Functions')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
```

## Colors, Markers, and Line Styles

matplotlib offers extensive control over visual elements.

**Reference:**

**Colors:**
- Named colors: `'red'`, `'blue'`, `'green'`
- Hex colors: `'#FF5733'`, `'#2E8B57'`
- RGB tuples: `(0.1, 0.2, 0.5)`

**Line Styles:**
- `'-'` solid, `'--'` dashed, `'-.'` dash-dot, `':'` dotted

**Markers:**
- `'o'` circle, `'s'` square, `'^'` triangle, `'*'` star

**Example:**

```python
# Demonstrate different styles
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 20)

# Different line styles and markers
ax.plot(x, x, 'o-', label='circles', color='blue', markersize=8)
ax.plot(x, x**0.5, 's--', label='squares', color='red', markersize=6)
ax.plot(x, np.log(x+1), '^-.', label='triangles', color='green', markersize=8)
ax.plot(x, np.sin(x), '*:', label='stars', color='purple', markersize=10)

ax.set_title('Different Line Styles and Markers')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

# pandas Plotting: Quick Exploration

*Think of pandas plotting as your data exploration Swiss Army knife - not the most specialized tool, but incredibly useful for getting a quick sense of your data.*

pandas provides convenient plotting methods that build on matplotlib, perfect for quick data exploration.

**Reference:**

- `df.plot()` - Line plot (default)
- `df.plot(kind='bar')` - Bar chart
- `df.plot(kind='hist')` - Histogram
- `df.plot(kind='scatter', x='col1', y='col2')` - Scatter plot
- `df.plot(kind='box')` - Box plot
- `df.plot(kind='pie')` - Pie chart

**Example:**

```python
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100)
})

# Quick exploration with pandas
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Line plot
df.plot(ax=axes[0, 0], title='Line Plot')

# Histogram
df.plot(kind='hist', ax=axes[0, 1], alpha=0.7, title='Histogram')

# Scatter plot
df.plot(kind='scatter', x='A', y='B', ax=axes[1, 0], title='Scatter Plot')

# Box plot
df.plot(kind='box', ax=axes[1, 1], title='Box Plot')

plt.tight_layout()
plt.show()
```

## DataFrame Plotting Options

**Reference:**

- `subplots=True` - Create separate subplots for each column
- `figsize=(width, height)` - Set figure size
- `title='Title'` - Set plot title
- `xlabel='X Label'` - Set x-axis label
- `ylabel='Y Label'` - Set y-axis label
- `legend=True` - Show legend
- `grid=True` - Add grid lines

**Example:**

```python
# Sales data example
sales_data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Product_A': [100, 120, 110, 130, 140, 135],
    'Product_B': [80, 90, 95, 105, 110, 115],
    'Product_C': [60, 70, 75, 80, 85, 90]
})

# Set Month as index for better plotting
sales_data.set_index('Month', inplace=True)

# Create subplots for each product
sales_data.plot(subplots=True, figsize=(10, 8), 
                title='Sales by Product Over Time',
                grid=True, legend=True)
plt.tight_layout()
plt.show()
```

# LIVE DEMO!

# seaborn: Statistical Visualization

*seaborn is like having a data visualization expert sitting next to you, automatically choosing the right colors, styles, and statistical methods to make your plots look professional and informative.*

seaborn builds on matplotlib to provide beautiful statistical visualizations with minimal code. It's the go-to choice for most data analysis tasks.

**Reference:**

- `sns.set_style('whitegrid')` - Set plot style
- `sns.set_palette('husl')` - Set color palette
- `sns.scatterplot(x='col1', y='col2', data=df)` - Scatter plot
- `sns.lineplot(x='col1', y='col2', data=df)` - Line plot
- `sns.histplot(data=df, x='col')` - Histogram
- `sns.boxplot(data=df, x='col1', y='col2')` - Box plot
- `sns.heatmap(data=df)` - Heatmap

**Example:**

```python
import seaborn as sns

# Set seaborn style
sns.set_style('whitegrid')
tips = sns.load_dataset('tips')

# Create multiple plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Scatter plot
sns.scatterplot(data=tips, x='total_bill', y='tip', 
                hue='time', ax=axes[0, 0])
axes[0, 0].set_title('Total Bill vs Tip')

# Box plot
sns.boxplot(data=tips, x='day', y='tip', ax=axes[0, 1])
axes[0, 1].set_title('Tip by Day')

# Histogram
sns.histplot(data=tips, x='total_bill', hue='time', 
             alpha=0.7, ax=axes[1, 0])
axes[1, 0].set_title('Bill Distribution')

plt.tight_layout()
plt.show()
```

## Advanced seaborn Features

**Reference:**

- `sns.pairplot(df)` - Pairwise relationships
- `sns.jointplot(x='col1', y='col2', data=df)` - Joint distribution
- `sns.violinplot(data=df, x='col1', y='col2')` - Violin plot
- `sns.stripplot(data=df, x='col1', y='col2')` - Strip plot
- `sns.catplot(kind='box', data=df, x='col1', y='col2')` - Categorical plot

**Example:**

```python
# Advanced seaborn visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Pair plot (shows all pairwise relationships)
# Note: This creates its own figure, so we'll use a subset
sample_data = tips.sample(50)
sns.pairplot(sample_data, hue='time', height=3)

# Joint plot (scatter + histograms)
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')

# Violin plot (shows distribution shape)
sns.violinplot(data=tips, x='day', y='tip', ax=axes[0, 0])
axes[0, 0].set_title('Tip Distribution by Day (Violin Plot)')

# Strip plot (shows individual points)
sns.stripplot(data=tips, x='day', y='tip', hue='time', ax=axes[0, 1])
axes[0, 1].set_title('Individual Tips by Day and Time')

plt.tight_layout()
plt.show()
```

# Density Plots and Distribution Visualization

*Density plots show the shape of your data distribution - they're like histograms but smoother, revealing patterns that might be hidden in discrete bins.*

Density plots (also called KDE - Kernel Density Estimation) provide a smooth representation of data distribution.

**Reference:**

- `df.plot.density()` - Create density plot
- `sns.histplot(data=df, x='col', kde=True)` - Histogram with density overlay
- `sns.kdeplot(data=df, x='col')` - Pure density plot
- `sns.distplot(data=df, x='col')` - Combined histogram and density

**Example:**

```python
# Create sample data with different distributions
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
bimodal_data = np.concatenate([
    np.random.normal(-2, 0.5, 500),
    np.random.normal(2, 0.5, 500)
])

# Density plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# pandas density plot
pd.Series(normal_data).plot.density(ax=axes[0, 0], title='Normal Distribution')
axes[0, 0].grid(True, alpha=0.3)

# seaborn density plot
sns.kdeplot(data=normal_data, ax=axes[0, 1], title='Normal Distribution (seaborn)')
axes[0, 1].grid(True, alpha=0.3)

# Bimodal distribution
sns.kdeplot(data=bimodal_data, ax=axes[1, 0], title='Bimodal Distribution')
axes[1, 0].grid(True, alpha=0.3)

# Combined histogram and density
sns.histplot(data=normal_data, kde=True, ax=axes[1, 1], title='Histogram + Density')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

# Modern Visualization Libraries

*The Python visualization ecosystem is constantly evolving. While matplotlib and seaborn are the workhorses, modern libraries offer exciting new approaches.*

## vega-altair: Grammar of Graphics

altair implements the grammar of graphics (like ggplot2 in R), making it intuitive to build complex visualizations.

**Reference:**

```python
import altair as alt

# Basic altair syntax
alt.Chart(df).mark_point().encode(
    x='column1',
    y='column2',
    color='category'
)
```

**Example:**

```python
# altair example (if installed)
try:
    import altair as alt
    
    # Create interactive scatter plot
    chart = alt.Chart(tips).mark_circle().encode(
        x='total_bill:Q',
        y='tip:Q',
        color='time:N',
        size='size:O'
    ).interactive()
    
    # Display (requires Jupyter or altair viewer)
    chart.show()
    
except ImportError:
    print("altair not installed. Install with: pip install altair")
```

## Other Modern Tools: plotnine, Bokeh, and Plotly

**plotnine** brings R's ggplot2 syntax to Python, perfect for those familiar with R.

**Bokeh** and **Plotly** create interactive web-based visualizations.

**Example:**

```python
# plotnine example (if installed)
try:
    from plotnine import ggplot, aes, geom_point, geom_smooth, theme_minimal
    
    # Create ggplot2-style plot
    p = (ggplot(tips, aes(x='total_bill', y='tip', color='time')) 
         + geom_point(alpha=0.6)
         + geom_smooth(method='lm')
         + theme_minimal())
    
    print(p)
    
except ImportError:
    print("plotnine not installed. Install with: pip install plotnine")

# Bokeh and Plotly for interactive plots
# These require separate installation and are great for web dashboards
```

# Command Line: Persistent Sessions with tmux

*When you're working on long-running data analysis or remote servers, you need sessions that survive network hiccups and accidental terminal closures. tmux is your friend.*

tmux (terminal multiplexer) allows you to create persistent terminal sessions that survive disconnections and can be shared between multiple terminal windows.

**Reference:**

- `tmux` - Start new session
- `tmux new-session -s session_name` - Start named session
- `tmux list-sessions` - List all sessions
- `tmux attach-session -t session_name` - Attach to session
- `tmux kill-session -t session_name` - Kill session
- `Ctrl+b d` - Detach from session (keeps it running)
- `Ctrl+b c` - Create new window
- `Ctrl+b n` - Next window
- `Ctrl+b p` - Previous window

**Example:**

```bash
# Start a new tmux session for data analysis
tmux new-session -s data_analysis

# Inside tmux, start your Python environment
conda activate datasci_217
jupyter notebook

# Detach from session (Ctrl+b, then d)
# Session keeps running in background

# Later, reattach to the same session
tmux attach-session -t data_analysis

# Your Jupyter notebook is still running!
```

## tmux Configuration

**Reference:**

Create `~/.tmux.conf` for custom settings:

```bash
# Enable mouse support
set -g mouse on

# Set default terminal
set -g default-terminal "screen-256color"

# Start windows and panes at 1
set -g base-index 1
setw -g pane-base-index 1

# Reload config file
bind r source-file ~/.tmux.conf \; display "Config reloaded!"
```

# Visualization Best Practices

*Good visualization is like good writing - it should be clear, honest, and serve the reader (or viewer) first.*

# FIXME: Add before/after visualization examples showing good vs bad design

# FIXME: Add color palette examples for different data types

## The Right Chart for the Job

**Chart Selection Guide:**

- **Line charts**: Time series, trends over time
- **Bar charts**: Categories, comparisons
- **Scatter plots**: Relationships between two variables
- **Histograms**: Distribution of single variable
- **Box plots**: Distribution with outliers
- **Heatmaps**: Patterns in 2D data
- **Pie charts**: Parts of a whole (use sparingly!)

**Example:**

```python
# Demonstrate chart selection
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Line chart - time series
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100))
axes[0, 0].plot(dates, values)
axes[0, 0].set_title('Line Chart: Time Series')
axes[0, 0].tick_params(axis='x', rotation=45)

# Bar chart - categories
categories = ['A', 'B', 'C', 'D']
counts = [23, 45, 56, 12]
axes[0, 1].bar(categories, counts)
axes[0, 1].set_title('Bar Chart: Categories')

# Scatter plot - relationships
x = np.random.randn(100)
y = 2 * x + np.random.randn(100)
axes[0, 2].scatter(x, y, alpha=0.6)
axes[0, 2].set_title('Scatter Plot: Relationships')

# Histogram - distribution
data = np.random.normal(0, 1, 1000)
axes[1, 0].hist(data, bins=30, alpha=0.7)
axes[1, 0].set_title('Histogram: Distribution')

# Box plot - distribution with outliers
data_by_group = [np.random.normal(i, 1, 50) for i in range(3)]
axes[1, 1].boxplot(data_by_group)
axes[1, 1].set_title('Box Plot: Distribution + Outliers')

# Heatmap - 2D patterns
heatmap_data = np.random.randn(10, 10)
im = axes[1, 2].imshow(heatmap_data, cmap='coolwarm')
axes[1, 2].set_title('Heatmap: 2D Patterns')
plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.show()
```

## Design Principles

**Key Principles:**

1. **Clarity**: Make your message obvious
2. **Honesty**: Don't mislead with scale or design
3. **Simplicity**: Remove unnecessary elements
4. **Consistency**: Use consistent colors, fonts, styles
5. **Accessibility**: Consider colorblind users, use patterns/textures

**Example:**

```python
# Good vs Bad visualization example
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# BAD: Misleading scale, poor colors, no labels
data = [10, 20, 30, 40]
ax1.bar(['A', 'B', 'C', 'D'], data, color=['red', 'blue', 'green', 'yellow'])
ax1.set_title('Sales by Region')
ax1.set_ylim(0, 50)  # Misleading scale

# GOOD: Clear scale, good colors, proper labels
ax2.bar(['A', 'B', 'C', 'D'], data, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_title('Sales by Region (in thousands)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Sales (thousands)', fontsize=12)
ax2.set_xlabel('Region', fontsize=12)
ax2.set_ylim(0, 45)  # Appropriate scale
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(data):
    ax2.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

# LIVE DEMO!

# Key Takeaways

1. **Start with pandas** for quick exploration
2. **Use seaborn** for statistical visualizations
3. **Customize with matplotlib** when needed
4. **Choose the right chart** for your data and message
5. **Follow design principles** for effective communication
6. **Use tmux** for persistent computing sessions
7. **Explore modern tools** like altair and plotnine for specific needs

You now have the skills to create effective visualizations that tell compelling data stories. These are essential skills for any data scientist.

Next week: We'll dive into data aggregation and group operations!

Practice Challenge

Before next class:
1. **Create visualizations:**
   - Use pandas.plot() for quick exploration
   - Use seaborn for statistical plots
   - Customize with matplotlib
   
2. **Practice tmux:**
   - Start a persistent session
   - Run Jupyter notebook in tmux
   - Detach and reattach
   
3. **Follow best practices:**
   - Choose appropriate chart types
   - Use clear labels and titles
   - Consider your audience

Remember: Good visualization is about communication - make your data tell a story!