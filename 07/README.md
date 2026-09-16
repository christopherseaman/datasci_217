---
notion:
  title_line: "Data Visualization: From Exploration to Communication"
  role: lecture
  status: mapped
  page_id: "29ad9fdd-1a1a-803c-a031-f791f9043193"
  url: "https://app.notion.com/p/29ad9fdd1a1a803ca031f791f9043193"
---

Data Visualization: From Exploration to Communication

See [BONUS.md](BONUS.md) for advanced topics:

- Advanced matplotlib customization and publication-quality plots
- Advanced interactive visualization with Bokeh and Plotly
- Statistical visualization with seaborn advanced features
- Custom color palettes and themes
- Animation and dynamic plots

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/07/demo/demo1_matplotlib_basics.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/07/demo/demo2_seaborn_statistical.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/07/demo/demo3_pandas_altair.ipynb)

*Fun fact: The word "visualization" comes from the Latin "visus" meaning "sight." In data science, we're literally making data visible - turning numbers into stories that our eyes can understand and our brains can process.*

This lecture uses prepared plotting tables so you can focus on choosing honest encodings and communicating what those tables show. Lecture 08 then teaches how to produce grouped summaries that can become plotting tables.

# Outline

- Visualization contract: question, audience, unit, grain, and encodings
- Tufte's principles for honest, readable displays
- Visualization ecosystem and tool selection
- matplotlib fundamentals (figures, subplots, customization)
- pandas plotting for quick data exploration
- seaborn statistical graphics and distribution views
- Altair: declarative charts, typed encodings, tooltips, and basic interaction
- Optional survey: plotnine, Bokeh, and Plotly


![xkcd 1945: Scientific Paper Graph Quality](media/xkcd_1945.png)

*"The data clearly shows that our hypothesis is correct, assuming we ignore all the data that doesn't support our hypothesis."*

# Start with a visualization contract

A **visualization** maps data values to visible properties so that a reader can make a comparison. Before choosing an API or chart type, write these four plain-language statements:

1. **Question:** What comparison or pattern should the chart help the reader understand?
2. **Audience and claim:** Who will use the chart, what context do they bring, and what descriptive conclusion should the finished chart support? A visual pattern alone does not prove why that pattern occurred.
3. **Unit and grain:** What does one mark or summarized position represent, and what does one row in the plotting table represent?
4. **Variables:** What is each variable's data type, what analytical role does it play, and which visible property will encode it?

## State the unit and grain shown

The **unit displayed** is what one mark or summarized position in the chart represents. Its **grain** is the corresponding row meaning in the plotting table. State both because a chart made from participant rows answers a different question from a chart made from program summaries.

## Separate data type from role

A variable's **data type** describes the meaning and valid operations of its values:

- **categorical** values place observations into named groups;
- **quantitative** values record numeric magnitudes for which arithmetic is meaningful;
- **ordinal** values are categories with a meaningful order; and
- **temporal** values represent dates or times, whose order and spacing may matter.

A variable's **role** describes how it participates in this particular analysis: for example, a quantitative column can be the measure being compared, a categorical column can define groups, and a temporal column can establish observation order. An identifier labels or links records; even when stored as a number, it is not automatically a quantitative measure. The same data type can play different roles in different charts, so record both type and role before choosing x, y, color, or another encoding.

## Separate exploratory and explanatory work

An **exploratory visualization** helps the analyst inspect patterns, distributions, or unexpected values while the question is still being refined. It may be quick, but it still needs truthful scales and labels.

An **explanatory visualization** communicates one selected finding to a named audience. It removes irrelevant alternatives, adds context and annotation, and uses a title or caption that states what the reader should notice without overstating the evidence.

![xkcd 1845, “State Word Map”: a satirical U.S. map labeled with supposedly distinctive search words, followed by notes about arbitrary methods and random noise.](media/xkcd_1845.png)

*xkcd 1845, “State Word Map”* — If flexible method choices can produce any headline, the chart is not evidence.

## Think in marks and encodings

A **mark** is a visible object such as a point, line, or rectangle. An **encoding** maps a data value to a visible property such as horizontal position, vertical position, length, color, marker shape, or line style.

Position along a common scale usually supports more precise comparison than area or decorative volume. Color can distinguish categories, but color alone is fragile: some readers cannot distinguish the selected hues, and grayscale reproduction may remove the distinction. When category identity matters, pair color with a redundant encoding such as marker shape, line style, direct labeling, or position.

### Reference Card: Visualization Contract

| Question | What to record | Useful output |
| :--- | :--- | :--- |
| **Question** | The comparison or pattern the reader should inspect | One-sentence chart claim |
| **Audience** | Who will use the chart and what context they have | Appropriate labels and annotation |
| **Unit and grain** | What one mark and one plotting-table row represent | A defensible aggregation level |
| **Variable role** | Type plus role: measure, group, time, or identifier | Candidate x, y, color, or shape encoding |
| **Accessibility** | A redundant cue and text alternative for the main comparison | A chart usable without color or hover |

### Code Snippet: State the Contract Before Plotting

```python
plot_spec = {
    "question": "Do guided sessions finish with higher prepared scores?",
    "unit": "one program-by-round summary",
    "measure": "mean prepared score",
    "group": "pathway",
    "encoding": "x=round, y=score, color+marker=pathway",
}
```

# Edward Tufte's Principles of Data Visualization

*Good visualization is like good writing - it should be clear, honest, and serve the reader (or viewer) first.*

**"Above all else, show the data."** - Edward Tufte

Edward Tufte, the pioneer of information design, established fundamental principles that remain essential for effective data visualization.

**Essential Reading:**

- [The Visual Display of Quantitative Information](https://www.edwardtufte.com/tufte/books_vdqi) - Tufte's seminal work
- [Envisioning Information](https://www.edwardtufte.com/book/envisioning-information/) - Color, layering, and detail
- [Tufte's website](https://www.edwardtufte.com/) - Essays and resources

**1. Data-Ink Ratio: Maximize the Data-Ink**

The **data-ink ratio** is the proportion of ink (or pixels) used to present actual data compared to the total ink used in the entire display.

```
Data-Ink Ratio = Data-Ink / Total Ink Used
```

**Tufte's Goal:** Maximize this ratio by eliminating non-data ink (chartjunk).

**Key Practices:**
- Remove unnecessary gridlines (or make them subtle)
- Eliminate decorative elements that don't convey information
- Use direct labeling instead of legends when possible
- Avoid 3D effects and shadows that distort perception
- Remove redundant labels and tick marks

![Data-Ink Ratio Comparison](media/tufte_data_ink_ratio.png)

*Left: Low data-ink ratio with excessive decoration. Right: High data-ink ratio focusing on the data.*

**2. Chartjunk: Eliminate Visual Noise**

**Chartjunk** includes any visual elements that do not convey information:
- Unnecessary 3D effects
- Heavy grid lines
- Decorative fills and patterns
- Excessive colors
- Redundant labels

**3. Lie Factor: Maintain Visual Integrity**

The **lie factor** measures how much a visualization distorts the data:

```
Lie Factor = (Size of effect shown in graphic) / (Size of effect in data)
```

**Ideal Lie Factor:** Close to 1.0 (no distortion)

**Common distortions to avoid:**
- Axis limits that hide relevant context or exaggerate differences. Bars normally need a zero baseline because length encodes magnitude; line charts do not always need to start at zero, but their range and any axis break must be clear and appropriate to the question.
- 3D perspective that distorts area/volume comparisons
- Inconsistent scales
- Cherry-picked time ranges

**4. Small Multiples: Show Comparisons**

Use small, repeated charts with the same scale to enable easy comparison across categories or time.

![Small Multiples Example](media/tufte_small_multiples.png)

*Small multiples enable quick visual comparison across multiple dimensions while maintaining consistent scales.*

**5. High-Resolution Data Graphics**

Show as much detail as the data allows - don't oversimplify or aggregate unnecessarily.

### Reference Card: Tufte's Checks

| Check | Ask | Typical remedy |
| :--- | :--- | :--- |
| **Data-ink** | Does every prominent mark carry information? | Remove decoration, heavy grids, and redundant labels |
| **Lie factor** | Does visual effect size match the data effect? | Use honest limits, units, and an appropriate baseline |
| **Small multiples** | Are repeated groups comparable? | Keep scale and encoding consistent across panels |
| **Resolution** | Did aggregation hide meaningful variation? | Show raw points or label the summary clearly |

## Before/After Examples: Applying Tufte's Principles

### Example 1: Bar Chart Redesign

![Bar Chart Comparison](media/tufte_bar_comparison.png)

*Before (left): Excessive colors, patterns, and heavy gridlines distract from the data. After (right): Clean design with direct labeling maximizes data-ink ratio.*

### Example 2: Line Chart with Truncated Axis (Lie Factor)

![Lie Factor Example](media/tufte_lie_factor.png)

*Before (left): The narrow y-range exaggerates modest growth. After (right): In this example, starting at zero restores useful magnitude context. A zero baseline is not a universal requirement for line charts; use a clearly labeled range that supports the intended comparison without distortion.*

## Color Palette Best Practices

Different data types require different color strategies:

![Color Palette Guide](media/color_palettes.png)

**Color Selection Guidelines:**
- **Sequential:** Use for ordered data (temperature, age, income) - single hue gradient
- **Diverging:** Use for data with meaningful zero/midpoint (profit/loss, correlation) - two contrasting hues
- **Qualitative:** Use for categories with no inherent order - distinct, unrelated colors
- **Accessibility:** Always test for colorblind accessibility using tools like [ColorBrewer](https://colorbrewer2.org/)

**Additional Resources:**
- [ColorBrewer 2.0](https://colorbrewer2.org/) - Interactive color advice for maps and visualizations
- [Colorblind-Safe Palettes](https://sronpersonalpages.nl/~pault/) - Paul Tol's color schemes
- [Adobe Color](https://color.adobe.com/) - Create and explore color schemes

## The Right Chart for the Job

**Chart Selection Guide:**

- **Line charts**: Time series, trends over time
- **Bar charts**: Categories, comparisons
- **Scatter plots**: Relationships between two variables
- **Histograms**: Distribution of single variable
- **Box plots**: Distribution with outliers
- **Heatmaps**: Patterns in 2D data
- **Pie charts**: Parts of a whole (use sparingly!)

![Chart Selection Guide](media/chart_selection.png)

*Different chart types are optimized for different data relationships and questions. Choose the right chart for your message.*

## Make the chart accessible

An accessible chart is designed so more readers can recover its comparison.

- Use readable type, complete labels, and adequate contrast against the background.
- Use a colorblind-safe palette, but do not treat palette choice as the whole accessibility task.
- Add redundant encoding when color distinguishes important categories.
- Prefer direct labels or a clearly associated legend over a distant decoding task.
- Do not rely on hover interaction to reveal essential values.
- Provide a concise **text alternative** or caption that states the chart type, axes, main pattern, and a relevant limitation.

Example text alternative:

> Line chart of mean prepared score by study round for standard and guided programs. Both rise across five rounds; the guided series rises from 61 to 79 and finishes seven points above the standard series. These are descriptive prepared summaries and do not establish a causal program effect.

That text alternative describes this actual example. Color is reinforced with marker shape, line style, and direct labels, so the comparison does not depend on color or a hover interaction alone:

```python
import matplotlib.pyplot as plt

rounds = [1, 2, 3, 4, 5]
standard = [60, 62, 65, 68, 72]
guided = [61, 65, 70, 74, 79]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rounds, standard, color='#E69F00', marker='s', linestyle='--')
ax.plot(rounds, guided, color='#0072B2', marker='o', linestyle='-')

ax.text(5.08, standard[-1], 'Standard', va='center')
ax.text(5.08, guided[-1], 'Guided', va='center')
ax.set(xlabel='Study round', ylabel='Mean prepared score',
       title='Guided program finishes 7 points higher by round 5')
ax.set_xticks(rounds)
ax.set_xlim(1, 5.7)
ax.grid(axis='y', alpha=0.25)
fig.tight_layout()
plt.show()
```


# The Visualization Ecosystem

*Reality check: There are more Python visualization libraries than there are ways to mess up a bar chart. But don't worry - we'll focus on the essential tools that actually matter for daily data science work.*

**Visual Guide - Plotting Backends:**

```
matplotlib ← pandas .plot() (default backend)
           ← seaborn
           ← plotnine

Vega-Lite  ← Altair

Arrows mean “renders through,” not a required learning order.
```

## Choosing the Right Tool

**When to use what:**

- **pandas.plot()** - Quick exploration, basic charts
- **matplotlib** - Custom plots, publication quality, fine control
- **seaborn** - Statistical plots, beautiful defaults, relationship analysis
- **altair** - Interactive plots, grammar of graphics, web-ready
- **plotnine** - If you know ggplot2, consistent API

**Pro tip:** Learn the foundation in sequence: matplotlib for figures and axes, pandas for quick exploration, then seaborn for statistical graphics. Use modern tools when interactive or web delivery is part of the contract.

# matplotlib: Foundation Layer

*Think of matplotlib as the foundation of your visualization house - you can build anything on it, but you need to understand the plumbing before you can install the fancy fixtures.*

matplotlib is the bedrock of Python visualization. While it can be verbose, understanding its core concepts gives you the power to create any visualization you can imagine.

## Figures and Subplots

Every matplotlib plot lives within a `Figure` object, which can contain multiple `subplots` (individual plot areas).

### Reference Card: Figures and Subplots

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `plt.figure(figsize=(w, h))` | Create a figure, with dimensions in inches | `Figure` |
| `plt.subplots(rows, cols, figsize=(w, h))` | Create a figure and grid of axes | Figure plus one Axes or an array of Axes |
| `fig.add_subplot(rows, cols, position)` | Add one axes to an existing figure | `Axes` |
| `fig.savefig(path, dpi=300)` | Save the current figure at a chosen resolution | Image file |
| `plt.show()` | Render the figure in an interactive session | Displayed figure |

### Code Snippet: Figure and Subplot Layout

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

![Matplotlib Subplots Example](media/matplotlib_subplots.png)

*Creating multiple subplots in a single figure allows for easy comparison across different visualization types.*

## Customizing Plots

matplotlib's power comes from its extensive customization options.

### Reference Card: Axes Customization

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `ax.set(title=..., xlabel=..., ylabel=...)` | Set visible context for the reader | Updated `Axes` |
| `ax.set_xlim(left, right)` / `ax.set_ylim(bottom, top)` | Control displayed ranges; use deliberately | Updated limits |
| `ax.grid(axis='y', alpha=0.3)` | Add restrained reference lines | Updated `Axes` |
| `ax.legend()` | Decode labeled series when direct labels are not enough | Legend artist |
| `plt.style.use(name)` | Apply a named style before creating figures | Global style setting |

### Code Snippet: Axes Customization

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

![Matplotlib Customization Example](media/matplotlib_customization.png)

*Customization allows you to create publication-quality plots with precise control over every visual element.*

## Colors, Markers, and Line Styles

matplotlib offers extensive control over visual elements.

### Reference Card: Colors, Markers, and Line Styles

| Property | Examples | Use |
| :--- | :--- | :--- |
| **Color** | `'steelblue'`, `'#0072B2'`, `(0.1, 0.2, 0.5)` | Distinguish categories or encode magnitude |
| **Line style** | `'-'`, `'--'`, `'-.'`, `':'` | Reinforce series identity or meaning |
| **Marker** | `'o'`, `'s'`, `'^'`, `'*'` | Reinforce category identity and show observations |

### Code Snippet: Visual Styles

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

![Matplotlib Colors and Styles](media/matplotlib_styles.png)

*matplotlib provides extensive options for colors, markers, and line styles to create visually distinct data series.*

![xkcd 833: Convincing](media/xkcd_833.png)

*"And if you don't label your axes, I'm leaving you." - The importance of proper chart labeling, illustrated.*

# LIVE DEMO!

# pandas: Quick Data Exploration

*Think of pandas plotting as your data exploration Swiss Army knife - not the most specialized tool, but incredibly useful for getting a quick sense of your data.*

### Reference Card: pandas Plotting

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `df.plot()` | Quick line plot of numeric columns | `Axes` |
| `df.plot(kind='bar')` | Compare values across categories | `Axes` |
| `df.plot(kind='hist', bins=...)` | Inspect numeric distributions | `Axes` |
| `df.plot(kind='scatter', x='col1', y='col2')` | Inspect a two-variable relationship | `Axes` |
| `df.plot(kind='box')` | Compare distributions and outliers | `Axes` |
| `df.plot(kind='pie', y='col')` | Show nonnegative values as parts of their total | `Axes` |

### Code Snippet: Quick pandas Plots

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

![Pandas Plotting Examples](media/pandas_plotting.png)

*pandas plotting methods provide quick, convenient visualization for data exploration with minimal code.*

## DataFrame Plotting Options

### Reference Card: DataFrame Plot Options

| Option | Purpose and arguments | Output |
| :--- | :--- | :--- |
| `subplots=True` | Give each selected column its own axes | Array of `Axes` |
| `figsize=(width, height)` | Set figure dimensions in inches | Larger or smaller figure |
| `title=...`, `xlabel=...`, `ylabel=...` | Add reader-facing context | Labeled plot |
| `legend=True`, `grid=True` | Decode series or add restrained guides | Updated `Axes` |

### Code Snippet: DataFrame Plot Options

```python
# Sales data example
sales_data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Product_A': [100, 120, 110, 130, 140, 135],
    'Product_B': [80, 90, 95, 105, 110, 115],
    'Product_C': [60, 70, 75, 80, 85, 90]
})

# Set Month as index for better plotting
sales_data = sales_data.set_index('Month')

# Create subplots for each product
sales_data.plot(subplots=True, figsize=(10, 8), 
                title='Sales by Product Over Time',
                grid=True, legend=True)
plt.tight_layout()
plt.show()
```

# seaborn: Statistical Graphics

*seaborn is like having a data visualization expert sitting next to you, automatically choosing the right colors, styles, and statistical methods to make your plots look professional and informative.*

seaborn builds on matplotlib to provide beautiful statistical visualizations with minimal code. It's the go-to choice for most data analysis tasks.

### Reference Card: seaborn Statistical Graphics

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `sns.set_style(name)` / `sns.set_palette(name)` | Set defaults for readable plots | Updated seaborn defaults |
| `sns.scatterplot(data=df, x=..., y=..., hue=...)` | Show relationships and optional groups | `Axes` |
| `sns.lineplot(data=df, x=..., y=..., hue=...)` | Show ordered trends and summaries | `Axes` |
| `sns.histplot(data=df, x=..., kde=True)` | Show distribution, optionally with density | `Axes` |
| `sns.boxplot(data=df, x=..., y=...)` | Compare distributions and outliers | `Axes` |
| `sns.heatmap(data=df)` | Encode a matrix as color | `Axes` |

### Code Snippet: Statistical Plots

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

![Seaborn Statistical Plots](media/seaborn_statistical.png)

*seaborn excels at creating beautiful statistical visualizations with automatic styling and color choices.*

## Optional: Advanced Seaborn Features

The following figure-level and distribution tools are an optional extension; the core lecture uses the axes-level plots above.

### Reference Card: Optional seaborn Extensions

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `sns.pairplot(df, hue=...)` | Compare every numeric pair | Figure-level grid |
| `sns.jointplot(data=df, x=..., y=..., kind=...)` | Combine a relationship with marginal distributions | Figure-level plot |
| `sns.violinplot(data=df, x=..., y=...)` | Show distribution shape by category | `Axes` |
| `sns.stripplot(data=df, x=..., y=..., hue=...)` | Show individual observations by category | `Axes` |
| `sns.catplot(kind='box', data=df, x=..., y=...)` | Build a faceted categorical plot | Figure-level grid |

### Code Snippet: Optional seaborn Extensions

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

![KDE curves for a normal sample centered near zero and a bimodal sample with peaks near minus two and two.](media/distribution_reference.png)

### Reference Card: Distribution Plots

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `df.plot.density()` | Quick KDE for numeric columns | `Axes` |
| `sns.histplot(data=df, x='col', kde=True)` | Compare bins with a smooth density estimate | `Axes` |
| `sns.kdeplot(data=df, x='col')` | Show a smoothed distribution alone | `Axes` |

### Code Snippet: Density Comparisons

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
sns.kdeplot(data=normal_data, ax=axes[0, 1])
axes[0, 1].set_title('Normal Distribution (seaborn)')
axes[0, 1].grid(True, alpha=0.3)

# Bimodal distribution
sns.kdeplot(data=bimodal_data, ax=axes[1, 0])
axes[1, 0].set_title('Bimodal Distribution')
axes[1, 0].grid(True, alpha=0.3)

# Combined histogram and density
sns.histplot(data=normal_data, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Histogram + Density')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

# LIVE DEMO!

# Altair: Declarative Charts and Interaction

Altair expresses a chart as **data → mark → typed encodings**. This makes the mapping from a table to visible properties explicit and produces a portable Vega-Lite specification. Use type shorthands deliberately: `:Q` for a quantitative measure, `:N` for a nominal category, `:O` for an ordered category, and `:T` for a temporal value.

![Six sessions show reflection scores increasing with activities completed; color and shape distinguish independent and guided pathways. This tiny example demonstrates encodings, not a causal effect.](media/altair_study_reference.png)

### Reference Card: Altair chart construction

| Call | Purpose / arguments | Result |
| :--- | :--- | :--- |
| `alt.Chart(study)` | Supply the source DataFrame | Chart to configure |
| `.mark_point(filled=True, size=90)` | Choose filled points and their area | Chart with point marks |
| `.encode(x='field:Q', color='group:N')` | Map quantitative and categorical fields to visible properties | Encoded chart |
| `.encode(tooltip=['field:Q'])` | Choose values shown on hover | Chart with tooltips |
| `.interactive()` | Add scale-bound pan/zoom interaction | Interactive chart |
| `alt.hconcat(left, right)` | Place two charts side by side | Compound chart |

### Code Snippet: Encode the study table

```python
import altair as alt
import pandas as pd

study = pd.DataFrame({
    'activities_completed': [1, 2, 3, 4, 5, 6],
    'reflection_score': [54, 58, 63, 66, 71, 75],
    'pathway': ['Independent', 'Independent', 'Independent',
                'Guided', 'Guided', 'Guided'],
})

scatter = alt.Chart(study).mark_point(filled=True, size=90).encode(
    x=alt.X('activities_completed:Q', title='Activities completed (count)'),
    y=alt.Y('reflection_score:Q', title='Reflection score (points)'),
    color=alt.Color('pathway:N', title='Pathway'),
    shape=alt.Shape('pathway:N', title='Pathway'),
    tooltip=['activities_completed:Q', 'reflection_score:Q', 'pathway:N'],
).properties(title='Prepared sessions: reflection score and activity count')

scatter.interactive()
```

`Chart(study)` supplies the table, `mark_point(filled=True)` chooses points, and the nominal color-plus-shape encodings redundantly identify pathways. `encode()` states the mapping. Tooltips and `.interactive()` can help a reader inspect a value or zoom, but the title, axes, legend, and main comparison must remain visible without hover. For a compact comparison, compose already honest charts with `alt.hconcat(left, right)` or `alt.vconcat(top, bottom)`; the end-of-lecture demo practices that pattern after the basic path above.

Altair does not replace the visualization contract: state the row grain and variable roles first, choose truthful scales and marks, use redundant cues when category identity matters, and supply a text alternative for the rendered or shared view.

# Optional Survey: Other Modern Visualization Libraries

*The Python visualization ecosystem is constantly evolving. While matplotlib and seaborn are the workhorses, modern libraries offer exciting new approaches.*

This optional, unassessed survey names alternatives; the same visible-context rules still apply. Extended Bokeh and Plotly examples live in [BONUS.md](BONUS.md).

## Ecosystem at a glance

- **plotnine** brings a layered grammar-of-graphics interface familiar to ggplot2 users.
- **Bokeh** targets browser-based visualizations, custom interactions, and server applications.
- **Plotly** offers a high-level Express API plus lower-level graph objects for interactive charts and dashboards.

## Tool Selection Guide

**When to use each tool:**

- **matplotlib**: Custom plots, publication quality, fine control
- **pandas**: Quick exploration, basic charts
- **seaborn**: Statistical plots, beautiful defaults, relationship analysis
- **altair**: Interactive plots, grammar of graphics, web-ready
- **plotnine**: R users, layered approach, statistical plots
- **Bokeh**: High-performance web visualizations, custom interactions
- **Plotly**: Dashboards, web applications, easy interactivity

| Tool | Best For | Learning Curve | Interactivity | Output Formats | Grammar |
|------|----------|----------------|---------------|----------------|---------|
| matplotlib | Custom plots, publication quality | High | None | PNG/SVG/PDF | Imperative |
| seaborn | Statistical plots, beautiful defaults | Low | None | PNG/SVG/PDF | Imperative |
| pandas | Quick exploration, basic charts | Very Low | None | PNG/SVG/PDF | Imperative |
| altair | Interactive plots, grammar of graphics | Medium | Built-in | PNG/SVG/HTML/JSON | Declarative |
| plotnine | R users, layered approach | Medium | None | PNG/SVG/PDF | Declarative |
| bokeh | Interactive web visualizations | High | High | HTML/JS | Imperative |
| plotly | Dashboards, web applications | Medium | High | HTML/JS | Declarative |


![xkcd 1138: Heatmap](media/xkcd_1138.png)

*"Every single map of the United States looks the same because it's just a population density map." - A reminder that your visualization should show meaningful patterns, not just expected distributions.*


# LIVE DEMO!
