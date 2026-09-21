---
notion:
  title_line: "# Data Visualization: From Exploration to Communication"
  role: lecture
  status: mapped
  page_id: "29ad9fdd-1a1a-803c-a031-f791f9043193"
  url: "https://app.notion.com/p/29ad9fdd1a1a803ca031f791f9043193"
---

# Data Visualization: From Exploration to Communication

See [BONUS.md](BONUS.md) for the optional extensions.

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/07/demo/demo1_matplotlib_basics.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/07/demo/demo2_seaborn_statistical.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/07/demo/demo3_pandas_altair.ipynb)

*Fun fact: The word "visualization" comes from the Latin "visus" meaning "sight." In data science, we're literally making data visible - turning numbers into stories that our eyes can understand and our brains can process.*

This lecture uses prepared plotting tables so you can focus on choosing honest encodings and communicating what those tables show. Lecture 08 then teaches how to produce grouped summaries that can become plotting tables.

![xkcd 1945: Scientific Paper Graph Quality](media/xkcd_1945.png)

*Chart quality in scientific papers dipped during the PowerPoint/MSPaint era; the tools in this lecture keep you on the rising end of the curve.*

# Start with a visualization contract

In the 1850s, Florence Nightingale had monthly counts of British Army deaths in the Crimean War, split by cause. Her diagram made one comparison impossible to miss: the blue wedges (deaths from preventable disease) dwarf the red wedges (deaths from wounds). She used it to argue for sanitary reform in army hospitals. The lesson for us: decide what the reader should compare before drawing anything.

![Florence Nightingale's Diagram of the Causes of Mortality in the Army in the East: blue wedges for preventable disease are far larger than red wedges for wounds.](media/Nightingale-mortality-1600.jpg)

A **visualization** maps data values to visible properties so that a reader can make a comparison. Think of it as a translation: each column of your table becomes something the eye can compare, such as a position, a length, or a color. Before choosing an API or chart type, write these four plain-language statements:

1. **Question:** What comparison or pattern should the chart help the reader understand?
2. **Audience and claim:** Who will use the chart, what context do they bring, and what descriptive conclusion should the finished chart support? A visual pattern alone does not prove why that pattern occurred.
3. **Unit and grain:** What does one mark or summarized position represent, and what does one row in the plotting table represent?
4. **Variables:** What is each variable's data type, what analytical role does it play, and which visible property will encode it?

## State the unit and grain shown

The **unit displayed** is what one mark or summarized position in the chart represents. Its grain, Lecture 06's term for what one row represents, is the plotting table's row meaning. You met this idea as row meaning in Lecture 05 (one row is one clinic visit, not one patient) and as long format in Lecture 06 (one row per entity-variable observation): a plotting table is a table whose rows match the marks you want to draw.

| Plotting table | One row is | One mark is | Question it can answer |
| --- | --- | --- | --- |
| Patient visits | One blood-pressure reading at one visit | One point | How much do individual readings vary? |
| Clinic summary | One clinic in one month | One bar or line point | Which clinic's average changed more? |

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

## Write the contract down

Record the four answers before you write any plotting code. The card adds a fifth row for accessibility, which the accessibility section later in this lecture develops.

### Reference Card: Visualization Contract

| Question | What to record | Useful output |
| :--- | :--- | :--- |
| **Question** | The comparison or pattern the reader should inspect | One-sentence chart claim |
| **Audience and claim** | Who reads it and the one descriptive conclusion it supports | Appropriate labels, title, and annotation |
| **Unit and grain** | What one mark and one plotting-table row represent | A defensible aggregation level |
| **Variable role** | Type plus role: measure, group, time, or identifier | Candidate x, y, color, or shape encoding |
| **Accessibility** | A redundant cue and text alternative for the main comparison | A chart usable without color or hover |

### Code Snippet: Storage dtype Is Not Visualization Type

A pandas dtype (Lecture 04) records how values are stored; the visualization data type records what they mean.

```python
import pandas as pd

visits = pd.DataFrame({
    'patient_id': [101, 101, 102, 102],
    'clinic': ['North', 'North', 'South', 'South'],
    'visit_month': [1, 2, 1, 2],
    'systolic_bp': [142, 136, 128, 131],
})
print(visits.dtypes)
```

```text
patient_id     int64
clinic           str
visit_month    int64
systolic_bp    int64
dtype: object
```

Three columns are `int64`, but only `systolic_bp` (mmHg) is a quantitative measure. `patient_id` is an identifier (its average means nothing), `visit_month` is temporal (order and spacing matter), and `clinic` is categorical. One row is one visit, so one scatter point would be one visit.

## The Right Chart for the Job

The question and the variable types narrow the choice of chart:

- **Line charts**: Time series, trends over time
- **Bar charts**: Categories, comparisons
- **Scatter plots**: Relationships between two variables
- **Histograms**: Distribution of single variable
- **Box plots**: Distribution with outliers
- **Heatmaps**: Patterns in 2D data
- **Pie charts**: Parts of a whole (use sparingly!)

![Chart Selection Guide](media/chart_selection.png)

*Different chart types are optimized for different data relationships and questions. Choose the right chart for your message.*

# The Visualization Ecosystem

*Reality check: There are more Python visualization libraries than there are ways to mess up a bar chart. But don't worry - we'll focus on the essential tools that actually matter for daily data science work.*

A **plotting backend** is the engine that actually draws a library's charts. matplotlib draws images from Python; **Vega-Lite** draws charts in a web browser.

```
matplotlib ← pandas .plot() (default backend)
           ← seaborn
           ← plotnine

Vega-Lite  ← Altair

Arrows mean “renders through,” not a required learning order.
```

## Choosing the Right Tool

Each tool below solves a different job; most of them draw through matplotlib. In the Typical output column, a `Figure` is matplotlib's whole image and an `Axes` is one plotting panel inside it; the next section explains both.

### Reference Card: Choosing a Plotting Tool

| Tool | Reach for it when | Draws through | Typical output |
| --- | --- | --- | --- |
| pandas `.plot()` | You want a first look at a DataFrame | matplotlib | `Axes` |
| matplotlib | You need full control or a publication figure | itself | `Figure`/`Axes`; PNG, SVG, PDF |
| seaborn | You want statistical plots from long data, with groups shown by color (`hue=`) | matplotlib | `Axes` |
| Altair | You want encodings that state each column's data type, hover tooltips, or a chart for a web page or JSON file | Vega-Lite | Altair `Chart` object; HTML, JSON |

plotnine, Bokeh, and Plotly are surveyed in [BONUS.md](BONUS.md).

# matplotlib: Foundation Layer

*Think of matplotlib as the foundation of your visualization house - you can build anything on it, but you need to understand the plumbing before you can install the fancy fixtures.*

pandas and seaborn draw through matplotlib, so the charts they make are matplotlib objects you can adjust with the same methods (Altair, at the end of this lecture, draws through Vega-Lite instead). Two objects are enough to fix almost any of these charts:

- A **Figure** is the whole canvas: the image you display or save.
- An **Axes** is one plotting area inside the Figure, with its own x-axis, y-axis, title, and marks. "Axes" names one plotting area; it is not the plural of "axis".

If the Figure is a canvas, each Axes is one painting on it: `fig.savefig()` saves the whole canvas, and `ax.set_title()` titles one painting.

The main path is `fig, ax = plt.subplots()` followed by Axes methods such as `ax.plot()` and `ax.set_xlabel()`. An alternative you will see in older examples is the pyplot shortcut style, such as `plt.plot()` and `plt.title()`, which draws on whichever Axes is current. It is fine for a quick sketch but confusing with several panels, so this course uses Axes methods.

## Figures and Subplots

With several panels, `plt.subplots(rows, cols)` returns the Axes in a NumPy array, so Lecture 03 indexing applies: one row of panels gives a 1-D array (`axes[0]`, `axes[1]`), and a grid gives a 2-D array (`axes[0, 1]` is row 0, column 1).

```text
Figure (fig)
├── Axes axes[0]  ← its own title, labels, lines, bars
└── Axes axes[1]  ← its own scales
```

### Reference Card: Figures and Subplots

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `plt.subplots(rows, cols, figsize=(w, h))` | Create a figure and a grid of Axes, with dimensions in inches | `Figure` plus one Axes or an array of Axes |
| `plt.figure(figsize=(w, h))` | Create an empty figure (pyplot shortcut style) | `Figure` |
| `plt.plot()`, `plt.title()`, `plt.xlabel()` | Pyplot shortcuts for `ax.plot()`, `ax.set_title()`, `ax.set_xlabel()` on the current Axes | Updated current Axes |
| `fig.add_subplot(rows, cols, position)` | Add one Axes to an existing figure | `Axes` |
| `fig.tight_layout()` / `plt.tight_layout()` | Adjust spacing so titles and labels do not overlap | Rearranged figure |
| `plt.show()` | Render the figure in an interactive session | Displayed figure |

### Code Snippet: What `plt.subplots()` Returns

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
print(type(fig))
print(axes.shape)
print(type(axes[0]))
```

```text
<class 'matplotlib.figure.Figure'>
(2,)
<class 'matplotlib.axes._axes.Axes'>
```

Read each class name from its last part: one `Figure`, and an array of two `Axes`, one per panel.

## Draw Marks on an Axes

Most chart types from the contract topic have a matching Axes method. Call the method on the Axes where that chart should appear.

### Reference Card: Axes Plotting Methods

- `ax.plot(x, y)`: Draw a line through the points in order; use for trends over an ordered x such as time.
- `ax.scatter(x, y, alpha=0.6)`: Draw one point per observation; `alpha` (0 to 1) makes overlapping points see-through.
- `ax.bar(categories, heights)`: Draw one bar per category, starting at zero.
- `ax.hist(values, bins=30)`: Count values in 30 equal-width bins and draw one bar per bin.
- `ax.boxplot([group_a, group_b], tick_labels=['A', 'B'])`: Draw the median, quartiles, and outliers of each group.

### Code Snippet: One Mark Type per Panel

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot([1, 2, 3, 4], [1, 4, 2, 3])
axes[0, 0].set_title('Line Plot')

axes[0, 1].hist(rng.normal(0, 1, 1000), bins=30)  # 1,000 draws: mean 0, standard deviation 1
axes[0, 1].set_title('Histogram')

axes[1, 0].scatter(rng.standard_normal(100), rng.standard_normal(100))
axes[1, 0].set_title('Scatter Plot')

axes[1, 1].bar(['A', 'B', 'C'], [3, 7, 2])
axes[1, 1].set_title('Bar Chart')

plt.tight_layout()
plt.show()
```

![Matplotlib Subplots Example](media/matplotlib_subplots.png)

*One Figure, four Axes: each panel has its own title and its own scales.*

## Customizing Plots

Titles, axis labels with units, deliberate limits, and a restrained grid give the reader the context the contract asks for.

### Reference Card: Axes Customization

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `ax.set(title=..., xlabel=..., ylabel=...)` | Set visible context for the reader | Updated `Axes` |
| `ax.set_title(text)`, `ax.set_xlabel(text)`, `ax.set_ylabel(text)` | Set one label at a time | Updated `Axes` |
| `ax.set_xlim(left, right)` / `ax.set_ylim(bottom, top)` | Control displayed ranges; use deliberately | Updated limits |
| `ax.set_xticks(positions, labels)` | Choose where ticks sit and, optionally, what they say | Updated ticks |
| `ax.grid(axis='y', alpha=0.3)` | Add restrained reference lines | Updated `Axes` |
| `ax.legend()` | Decode labeled series when direct labels are not enough | Legend artist |
| `plt.style.use(name)` | Apply a named style before creating figures | Global style setting |

### Code Snippet: Axes Customization

```python
fig, ax = plt.subplots(figsize=(8, 6))

x = np.linspace(0, 10, 100)  # 100 evenly spaced values from 0 to 10
y1 = np.sin(x)               # sine of every value, element-wise like Lecture 03's np.sqrt()
y2 = np.cos(x)

ax.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
ax.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')

ax.set_title('Trigonometric Functions')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
```

![Matplotlib Customization Example](media/matplotlib_customization.png)

*Two labeled series with a title, axis labels, a legend, and a light grid.*

## Colors, Markers, and Line Styles

Each series can combine a color, a marker, and a line style. Combining them keeps lines distinguishable even in grayscale, a point the accessibility section returns to.

### Reference Card: Colors, Markers, and Line Styles

| Property | Examples | Use |
| :--- | :--- | :--- |
| **Color** | `'steelblue'`, `'#0072B2'`, `(0.1, 0.2, 0.5)` | Distinguish categories or encode magnitude |
| **Line style** | `'-'`, `'--'`, `'-.'`, `':'` | Reinforce series identity or meaning; set thickness with `linewidth=` |
| **Marker** | `'o'`, `'s'`, `'^'`, `'*'` | Reinforce category identity and show observations; set size with `markersize=` |
| **Format string** | `'o-'`, `'s--'` | Shorthand for marker + line style: `'o--'` equals `marker='o', linestyle='--'` |

### Code Snippet: Visual Styles

```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 20)

ax.plot(x, x, 'o-', label='circles', color='blue', markersize=8)
ax.plot(x, x**0.5, 's--', label='squares', color='red', markersize=6)
ax.plot(x, np.log(x+1), '^-.', label='triangles', color='green', markersize=8)
ax.plot(x, np.sin(x), '*:', label='stars', color='purple', markersize=10)

ax.set_title('Different Line Styles and Markers')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

![Four series titled Different Line Styles and Markers: blue circles on a solid line, red squares on a dashed line, green triangles on a dash-dot line, and purple stars on a dotted line.](media/matplotlib_styles.png)

*Each series combines its own color, marker, and line style.*

## Annotate, Declutter, and Save

An explanatory chart usually points at one thing. An **annotation** is text attached to a specific data point, often with an arrow, so the reader does not have to hunt for it. The top and right frame lines (**spines**) carry no data; hiding them leaves more attention for the marks. `fig.savefig()` writes the whole Figure to a file, and the file type comes from the extension.

### Reference Card: Annotate, Declutter, and Save

- `ax.annotate(text, xy=(x, y), xytext=(x2, y2), arrowprops=dict(arrowstyle='->'))`: Put `text` at `xytext` with an arrow to the data point `xy`. Add `color=...` for the text and `color=...` inside `arrowprops` for the arrow, such as the color of the line being labeled.
- `ax.text(x, y, text, va='center')`: Put text at a point with no arrow, such as a direct line label; `va=` (vertical alignment) and `ha=` (horizontal alignment) set which part of the text sits on that point.
- `ax.spines[['top', 'right']].set_visible(False)`: Hide the two frame lines that carry no data.
- `ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)`: Place the legend just outside the right edge, without a box.
- `fig.set_size_inches(w, h)`: Resize an existing Figure before saving.
- `fig.savefig('chart.png', dpi=150, bbox_inches='tight')`: Save a PNG at 150 dots per inch; `bbox_inches='tight'` trims extra margin so labels are not cut off. Use `.svg` or `.pdf` for vector output.

### Code Snippet: Point to the Peak and Save

```python
import matplotlib.pyplot as plt

weeks = [1, 2, 3, 4]
flu_visits = [42, 45, 51, 48]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(weeks, flu_visits, marker='o')
ax.annotate('Peak: 51 visits', xy=(3, 51), xytext=(1.5, 55),
            arrowprops=dict(arrowstyle='->'))
ax.set(xlabel='Week', ylabel='Flu clinic visits', ylim=(0, 60))
ax.set_xticks(weeks)
ax.spines[['top', 'right']].set_visible(False)
fig.savefig('flu_visits.png', dpi=150, bbox_inches='tight')
```

Expected result: `flu_visits.png` appears next to your notebook, showing one line with an arrow from 'Peak: 51 visits' to the week-3 point and no top or right frame line.

![xkcd 833: Convincing](media/xkcd_833.png)

*"I just think I can do better than someone who doesn't label her axes." Label your axes.*

# LIVE DEMO!

# pandas: Quick Data Exploration

*Think of pandas plotting as your data exploration Swiss Army knife - not the most specialized tool, but incredibly useful for getting a quick sense of your data.*

When you have just loaded a table and want to see it, `df.plot()` is the fastest path: one method call on the DataFrame you already have (Lecture 04). `df.plot()` returns a matplotlib `Axes`, so matplotlib methods still work afterward, and `ax=` draws into one panel of a `plt.subplots()` grid.

## Index to x, Columns to Series

pandas hands the data to matplotlib using two rules:

- The **index** becomes the x-axis.
- Each numeric **column** becomes one series (a line, a set of bars, ...), and the column names fill the legend.

So check the index before plotting. If it is the default row number 0, 1, 2, ..., the x-axis shows row numbers; `set_index()` from Lecture 06 makes it something meaningful, such as the week.

### Code Snippet: The Index Becomes the x-Axis

```python
import pandas as pd

weekly = pd.DataFrame({
    'week': [1, 2, 3, 4],
    'North': [42, 45, 51, 48],
    'South': [30, 33, 31, 36],
}).set_index('week')

ax = weekly.plot(marker='o', ylabel='Flu clinic visits', title='Weekly visits by clinic')
print(ax.get_xlabel())  # week
```

Expected output: two lines, one per clinic, with `week` on the x-axis and a legend reading North and South.

## Plot Kinds

### Reference Card: pandas Plotting

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `df.plot()` | Line plot: index on x, one line per numeric column | `Axes` |
| `df.plot(ax=axes[0, 1])` | Draw into an existing Axes instead of a new figure | That Axes |
| `df.plot(kind='bar')` | Compare values across categories | `Axes` |
| `df.plot(kind='hist', bins=...)` | Inspect numeric distributions | `Axes` |
| `df.plot(kind='scatter', x='col1', y='col2')` | Inspect a two-variable relationship | `Axes` |
| `df.plot(kind='box')` | Compare distributions and outliers | `Axes` |
| `df.plot(kind='pie', y='col')` | Show nonnegative values as parts of their total | `Axes` |
| `df.plot.bar()`, `df.plot.hist()` | Same as `kind='bar'` / `kind='hist'`; `df.plot.density()` below uses this form | `Axes` |

### Code Snippet: Several Plot Kinds in One Grid

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
weekly.plot(ax=axes[0, 0], marker='o', title='Line: visits each week')
weekly.plot(kind='bar', ax=axes[0, 1], title='Bar: visits each week')
weekly.plot(kind='scatter', x='North', y='South', ax=axes[1, 0], title='Scatter: one point per week')
weekly.plot(kind='box', ax=axes[1, 1], title='Box: spread of weekly visits')
plt.tight_layout()
plt.show()
```

![Four pandas plots of the same four weeks of clinic visits: a line per clinic, grouped bars per week, a scatter of North against South with one point per week, and one box per clinic.](media/pandas_plotting.png)

*One table, four views: each `kind=` answers a different question about the same rows.*

## DataFrame Plotting Options

### Reference Card: DataFrame Plot Options

| Option | Purpose and arguments | Output |
| :--- | :--- | :--- |
| `subplots=True` | Give each selected column its own Axes | Array of `Axes` |
| `sharey=True` | Give every subplot the same y-scale so panels are comparable | Shared limits |
| `figsize=(width, height)` | Set figure dimensions in inches | Larger or smaller figure |
| `title=...`, `xlabel=...`, `ylabel=...` | Add reader-facing context | Labeled plot |
| `legend=True`, `grid=True` | Decode series or add restrained guides | Updated `Axes` |

### Code Snippet: DataFrame Plot Options

```python
monthly = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'North': [100, 120, 110, 130, 140, 135],
    'South': [80, 90, 95, 105, 110, 115],
    'East': [60, 70, 75, 80, 85, 90],
}).set_index('month')

monthly.plot(subplots=True, sharey=True, figsize=(10, 8),
             title='Clinic visits by month', ylabel='Visits', grid=True)
plt.tight_layout()
plt.show()
```

Expected output: three stacked panels (North, South, East), each labeled Visits, that share one y-axis running from about 56 to 144, so East's lower counts sit visibly lower than North's.

# seaborn: Statistical Graphics

*seaborn is like having a data visualization expert sitting next to you, automatically choosing the right colors, styles, and statistical methods to make your plots look professional and informative.*

seaborn builds on matplotlib to draw statistical graphics from a DataFrame in one call. It expects long data from Lecture 06: one row per observation, one column per variable. You pass column names, and seaborn maps each one to an encoding:

- `data=`: the DataFrame the column names come from
- `x=` and `y=`: horizontal and vertical position
- `hue=`: color, one color per category

That is the visualization contract written as code: `sns.scatterplot(data=visits, x='visit_month', y='systolic_bp', hue='clinic')` means one point per visit, month across, blood pressure up, clinic by color. Each plotting function in the card below returns a matplotlib `Axes` and accepts `ax=` to draw into one panel of a `plt.subplots()` grid.

## Statistical Plots from Long Data

### Reference Card: seaborn Statistical Graphics

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `sns.load_dataset(name)` | Download a small example table, such as `'healthexp'` (needs internet) | `DataFrame` |
| `sns.set_style(name)` / `sns.set_palette(name)` | Set defaults for readable plots | Updated seaborn defaults |
| `sns.scatterplot(data=df, x=..., y=..., hue=...)` | Show relationships and optional groups | `Axes` |
| `sns.lineplot(data=df, x=..., y=..., hue=...)` | Show ordered trends; averages rows that share an x value | `Axes` |
| `sns.barplot(data=df, x=..., y=..., hue=...)` | One bar per category showing the mean of y, with an error bar | `Axes` |
| `sns.histplot(data=df, x=..., kde=True)` | Show distribution, optionally with density | `Axes` |
| `sns.boxplot(data=df, x=..., y=...)` | Compare distributions and outliers | `Axes` |
| `sns.heatmap(data=df, annot=True)` | Encode a wide table of numbers (index as rows, columns as columns) as color; `annot=True` writes each value in its cell | `Axes` |
| `errorbar=None` | Hide the error band or bar on lineplot/barplot | Updated `Axes` |

### Code Snippet: Statistical Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
health = sns.load_dataset('healthexp')  # one row per country-year, 1970-2020

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.scatterplot(data=health, x='Spending_USD', y='Life_Expectancy', hue='Country', ax=axes[0])
axes[0].set(xlabel='Health spending per person (USD)', ylabel='Life expectancy (years)')
sns.lineplot(data=health, x='Year', y='Life_Expectancy', hue='Country', legend=False, ax=axes[1])
axes[1].set(ylabel='Life expectancy (years)')
sns.boxplot(data=health, x='Life_Expectancy', y='Country', ax=axes[2])
axes[2].set(xlabel='Life expectancy (years)', ylabel='')
fig.tight_layout()
plt.show()
```

![Three seaborn panels from the healthexp table: health spending against life expectancy with one colored point per country-year, life expectancy over time with one line per country, and one box of yearly life expectancy per country.](media/seaborn_statistical.png)

*Left: one point per country-year. Middle: one line per country. Right: each box summarizes one country's yearly values.*

## Watch the Grain

When several rows share an x value, `sns.lineplot()` and `sns.barplot()` draw the **mean** of those rows plus an error band or bar showing its uncertainty, not the rows themselves. The unit displayed silently changes from one reading to an average reading, so say so in the axis label or title. Lecture 08 shows how to build such summaries yourself.

### Code Snippet: seaborn Averages Repeated x Values

```python
import pandas as pd
import seaborn as sns

readings = pd.DataFrame({
    'week': [1, 1, 1, 2, 2, 2],
    'patient_id': ['A', 'B', 'C', 'A', 'B', 'C'],
    'systolic_bp': [150, 138, 142, 144, 136, 139],
})
ax = sns.lineplot(data=readings, x='week', y='systolic_bp', errorbar=None)
print(ax.get_lines()[0].get_ydata())  # [143.33333333 139.66666667]
```

`ax.get_lines()[0].get_ydata()` reads back the y-values of the line seaborn drew. Six readings became two points: the week 1 mean (143.3) and the week 2 mean (139.7).

# Density Plots and Distribution Visualization

*Density plots show the shape of your data distribution - they're like histograms but smoother, revealing patterns that might be hidden in discrete bins.*

A histogram counts values in bins, so its shape depends on where the bins start and how wide they are. A **density plot**, or **KDE** (kernel density estimate), instead puts a small smooth bump on every observation and adds the bumps up. The curve's total area is 1, so the y-axis is **density**, not a count. The bump width is the **bandwidth**: wider bumps smooth more, narrower bumps show more detail and more noise.

Density plots are good at revealing shape. Fasting glucose readings from a clinic that serves people with and without diabetes may show two peaks (**bimodal**), which a single mean or box plot would hide; the bimodal panel below shows that pattern with simulated data.

![KDE curves for a normal sample centered near zero and a bimodal sample with peaks near minus two and two.](media/distribution_reference.png)

## Density Plots in pandas and seaborn

`df.plot.density()` computes the KDE with SciPy, so it needs `scipy` installed (Colab has it); seaborn's `kdeplot()` and `histplot(kde=True)` work without it.

### Reference Card: Distribution Plots

| Call | Purpose and key arguments | Output |
| :--- | :--- | :--- |
| `df.plot.density()` | Quick KDE for numeric columns (needs SciPy) | `Axes` |
| `sns.histplot(data=df, x='col', kde=True)` | Compare bins with a smooth density estimate | `Axes` |
| `sns.kdeplot(data=df, x='col')` | Show a smoothed distribution alone | `Axes` |
| `sns.kdeplot(..., bw_adjust=0.5)` | Narrower bumps: less smoothing, more detail | `Axes` |

### Code Snippet: Density Comparisons

```python
rng = np.random.default_rng(42)
normal_data = rng.normal(0, 1, 1000)  # mean 0, standard deviation 1
bimodal_data = np.concatenate([       # join two arrays end to end
    rng.normal(-2, 0.5, 500),
    rng.normal(2, 0.5, 500),
])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

pd.Series(normal_data).plot.density(ax=axes[0, 0], title='Normal Distribution')
axes[0, 0].grid(True, alpha=0.3)

sns.kdeplot(data=normal_data, ax=axes[0, 1])
axes[0, 1].set_title('Normal Distribution (seaborn)')
axes[0, 1].grid(True, alpha=0.3)

sns.kdeplot(data=bimodal_data, ax=axes[1, 0])
axes[1, 0].set_title('Bimodal Distribution')
axes[1, 0].grid(True, alpha=0.3)

sns.histplot(data=normal_data, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Histogram + Density')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

# LIVE DEMO!

# Edward Tufte's Principles of Data Visualization

*Good visualization is like good writing - it should be clear, honest, and serve the reader (or viewer) first.*

**"Above all else, show the data."** - Edward Tufte

A chart can get every number right and still mislead. Picture a hospital dashboard showing hand-hygiene compliance of 96% in March and 97% in April, drawn as bars on a y-axis that starts at 95%. The April bar is twice as tall, so readers see compliance double when it rose one point. Nothing in the data is wrong; the drawing is.

The visualization contract says which comparison the reader should make. Edward Tufte's principles check that the drawing lets them make it honestly: spend ink on data rather than decoration, keep visual size proportional to the numbers, and keep repeated panels comparable. They are also the vocabulary you will use to critique and redesign a chart in the assignment.

## Five Principles

### Data-Ink Ratio

The **data-ink ratio** is the proportion of ink (or pixels) used to present actual data compared to the total ink used in the entire display.

```
Data-Ink Ratio = Data-Ink / Total Ink Used
```

Tufte's goal is to maximize this ratio by removing ink that carries no data:

- Remove unnecessary gridlines (or make them subtle)
- Eliminate decorative elements that don't convey information
- Use direct labeling instead of legends when possible
- Avoid 3D effects and shadows that distort perception
- Remove redundant labels and tick marks

![Data-Ink Ratio Comparison](media/tufte_data_ink_ratio.png)

*Left: Low data-ink ratio with excessive decoration. Right: High data-ink ratio focusing on the data.*

### Chartjunk

**Chartjunk** includes any visual elements that do not convey information:

- Unnecessary 3D effects
- Heavy grid lines
- Decorative fills and patterns
- Excessive colors
- Redundant labels

### Lie Factor

The **lie factor** measures how much a visualization distorts the data:

```
Lie Factor = (Size of effect shown in graphic) / (Size of effect in data)
```

A lie factor close to 1.0 means no distortion. In the hand-hygiene dashboard, the April bar grows 100% (from 1 to 2 units above the 95% baseline) while compliance grows about 1% (96 → 97), so the lie factor is roughly 100 / 1.04 ≈ 96.

Common distortions to avoid:

- Axis limits that hide relevant context or exaggerate differences. Bars normally need a zero baseline because length encodes magnitude; line charts do not always need to start at zero, but their range and any axis break must be clear and appropriate to the question.
- 3D perspective that distorts area/volume comparisons
- Inconsistent scales
- Cherry-picked time ranges

### Small Multiples

Use small, repeated charts with the same scale to enable easy comparison across categories or time; `sharey=True` from the pandas section does this.

![Small Multiples Example](media/tufte_small_multiples.png)

*Small multiples enable quick visual comparison across multiple dimensions while maintaining consistent scales.*

### Show the Detail

Show as much detail as the data allows - don't oversimplify or aggregate unnecessarily. Tufte calls these high-resolution data graphics.

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

- **Sequential:** Use for ordered data (temperature, age, income) - single hue gradient
- **Diverging:** Use for data with meaningful zero/midpoint (profit/loss, correlation) - two contrasting hues
- **Qualitative:** Use for categories with no inherent order - distinct, unrelated colors
- **Accessibility:** Always test for colorblind accessibility using tools like [ColorBrewer](https://colorbrewer2.org/)

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

### Reference Card: Redundant Cues for Bars and Lines

- `ax.plot(x, y, color=..., marker='o', linestyle='-')`: Pair each line color with its own marker and line style.
- `x = np.arange(n)`: One position per category group (Lecture 03); `ax.set_xticks(x, labels)` names the positions.
- `ax.bar(x - width / 2, heights, width, label=..., hatch='//')`: Draw one set of side-by-side bars, shifted left by half a bar width. A fill pattern (**hatch**) such as `'//'` or `'..'` keeps groups distinguishable in grayscale; because it encodes the group, it is data ink rather than chartjunk.
- `ax.bar_label(bars, fmt='%d%%')`: Write each bar's value on it, such as `64%`; `bars` is what `ax.bar()` returns.
- `ax.set_ylim(0, 100)`: Start bar axes at zero, because bar length encodes magnitude.

### Code Snippet: Redundant Cues on a Line Chart

The text alternative above describes this chart. Color is reinforced with marker shape, line style, and direct labels, so the comparison does not depend on color or a hover interaction alone:

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

Expected output: two rising lines labeled at their right ends, orange dashed squares for Standard (ending at 72) and blue solid circles for Guided (ending at 79), with no legend needed.

### Code Snippet: Grouped Bars That Work in Grayscale

```python
import matplotlib.pyplot as plt
import numpy as np

seasons = ['2023-24', '2024-25']
north = [58, 64]
south = [55, 58]
x = np.arange(len(seasons))  # [0 1]: one position per season
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))
north_bars = ax.bar(x - width / 2, north, width, label='North', color='#0072B2', hatch='//')
south_bars = ax.bar(x + width / 2, south, width, label='South', color='#D55E00', hatch='..')
ax.bar_label(north_bars, fmt='%d%%')
ax.bar_label(south_bars, fmt='%d%%')
ax.set_xticks(x, seasons)
ax.set(ylim=(0, 100), xlabel='Flu season', ylabel='Adults vaccinated (%)')
ax.legend(title='Clinic', loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.show()
```

Expected output: two pairs of bars rising from 0, labeled 58% and 55% for 2023-24 and 64% and 58% for 2024-25. North's bars are striped and South's dotted, and the legend sits just outside the right edge.

# Altair: Declarative Charts and Interaction

Altair is **declarative**: you describe *what* the chart shows (which column goes on which visible property) and Altair works out *how* to draw it, like ordering from a menu instead of cooking. matplotlib is the opposite: you give drawing steps one at a time. An Altair chart is built as **data → mark → typed encodings** and becomes a **Vega-Lite specification**, a JSON document that a browser renders; that JSON is also what you save and share. Each encoded field gets a type letter matching the contract's data types: categorical → `:N` (nominal), ordinal → `:O`, quantitative → `:Q`, temporal → `:T`.

![Six sessions show reflection scores increasing with activities completed; color and shape distinguish independent and guided pathways. This tiny example demonstrates encodings, not a causal effect.](media/altair_study_reference.png)

## Data, Mark, and Typed Encodings

### Reference Card: Altair chart construction

| Task | Call | Purpose / arguments | Result |
| :--- | :--- | :--- | :--- |
| Build | `alt.Chart(study)` | Supply the source DataFrame | Chart to configure |
| Build | `.mark_point(filled=True, size=90)` | Choose filled points and their area; `.mark_bar()` and `.mark_line()` draw bars or lines | Chart with marks |
| Build | `.properties(title=..., width=360, height=260)` | Add a visible title and set size in pixels | Chart |
| Encode | `.encode(x='field:Q', color='group:N')` | Map quantitative and categorical fields to visible properties | Encoded chart |
| Encode | `alt.X('field:Q', title='Label (unit)')` | Set an axis title with its unit (also `alt.Y`) | Encoding channel |
| Encode | `alt.Color('group:N', sort=['A', 'B'])` | Fix category and legend order (also for `alt.Shape`); `legend=None` hides the legend | Encoding channel |
| Encode | `y='mean(field):Q'` | Let Altair average rows per x category; the unit becomes one summary per group | Aggregated encoding |
| Interact | `.encode(tooltip=['field:Q'])` | Choose values shown on hover | Chart with tooltips |
| Interact | `alt.Tooltip('field:Q', title=..., format='.1f')` | Name and format a tooltip value | Tooltip channel |
| Interact | `.interactive()` | Add scale-bound pan/zoom interaction | Interactive chart |
| Compose | `alt.hconcat(left, right)` | Place two charts side by side | Compound chart |

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

## Save the Chart and Its Record

Because an Altair chart is a Vega-Lite specification, saving it as JSON keeps the chart and its rows together; anyone with the file can render the same chart. The standard-library `json` module saves anything else you want to keep beside the chart, such as its contract and text alternative.

### Reference Card: Saving Charts and Records

| Call | Purpose / arguments | Result |
| :--- | :--- | :--- |
| `chart.save('chart.json')` / `chart.save('chart.html')` | Write the Vega-Lite spec (data embedded) or a web page | File |
| `chart.to_dict()` | Return the same spec as a Python dictionary | `dict` |
| `json.dump(obj, file, indent=2, ensure_ascii=False)` | Write a dict or list as readable JSON; `ensure_ascii=False` keeps characters such as é unescaped. It does not end the file with a newline; call `file.write('\n')` afterward if one is required | JSON file |
| `json.load(file)` | Read a JSON file back | `dict` or `list` |

### Code Snippet: Save the Chart Specification

```python
scatter.save('study_scatter.json')  # Vega-Lite JSON with the six rows embedded
spec = scatter.to_dict()            # the same specification as a Python dict
print(spec['mark'])                 # {'type': 'point', 'filled': True, 'size': 90}
```

### Code Snippet: Save a Chart Record as JSON

`json.dump()` writes any dictionary of strings, numbers, lists, and nested dictionaries, including the one `chart.to_dict()` returns. `open()` and `with` come from Lecture 02.

```python
import json

chart_record = {
    'question': 'Do guided sessions show higher reflection scores at similar activity counts?',
    'grain': 'one prepared learning session',
    'text_alternative': 'Scatter plot of reflection score (points) against activities completed for six prepared sessions.',
}
with open('study_record.json', 'w', encoding='utf-8') as file:
    json.dump(chart_record, file, indent=2, ensure_ascii=False)

with open('study_record.json', encoding='utf-8') as file:
    saved = json.load(file)
print(saved['grain'])  # one prepared learning session
```

![xkcd 1138: Heatmap](media/xkcd_1138.png)

*"Pet peeve #208: Geographic profile maps which are basically just population maps." Before mapping counts, ask whether the pattern is just where people live.*

# LIVE DEMO!
