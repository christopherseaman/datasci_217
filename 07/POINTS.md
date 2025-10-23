# Data Visualization: From Exploration to Communication

## The Visualization Ecosystem

Data visualization is the bridge between raw data and human understanding. The Python ecosystem has evolved from matplotlib's low-level control to modern libraries that make beautiful plots with minimal code. Understanding this ecosystem helps you choose the right tool for each task - from quick exploration to publication-ready figures.

- matplotlib is the foundation that everything else builds on - it's verbose but gives you complete control over every pixel
- pandas plotting provides quick exploration tools that work directly on your DataFrames without learning new syntax
- seaborn makes statistical plots beautiful by default while still being customizable when you need it
- Modern tools like altair and plotnine offer grammar-of-graphics approaches that scale from simple to complex visualizations
- The key insight: start simple with pandas, use seaborn for analysis, customize with matplotlib, and explore modern tools for specific needs

## Choosing the Right Tool

Every visualization library has its sweet spot. The choice isn't about which is "best" but which is most appropriate for your current task and audience. Quick exploration needs different tools than publication figures, and interactive dashboards require different approaches than static reports.

- pandas.plot() is perfect for "let me see what this data looks like" - it's fast, works on your existing DataFrames, and requires no new learning
- matplotlib shines when you need pixel-perfect control or are creating publication figures that need to match specific formatting requirements
- seaborn is your go-to for statistical analysis because it automatically handles color palettes, statistical annotations, and relationship plotting
- altair and plotnine are worth learning if you're building interactive dashboards or prefer grammar-of-graphics thinking
- Pro tip: don't try to learn everything at once - master one tool well, then expand your toolkit based on actual needs

# LIVE DEMO!

## matplotlib Fundamentals

matplotlib is the foundation of Python visualization - it's what everything else is built on. While it can be verbose, understanding its core concepts gives you the power to create any visualization you can imagine. The key is understanding that every plot lives within a Figure object, which can contain multiple subplots.

- Every matplotlib plot requires a Figure object - think of it as your canvas that can hold multiple individual plots (subplots)
- plt.subplots() is your most common starting point - it creates both the figure and a grid of subplot axes in one call
- The axes object is where you actually plot your data - it's the individual plot area within the larger figure
- Always use figsize=(width, height) to control your plot dimensions - default sizes are usually too small for presentations
- plt.tight_layout() automatically adjusts spacing between subplots to prevent overlapping labels

## Figures and Subplots

Understanding the figure-subplot relationship is crucial for creating multi-panel plots. A figure is your overall canvas, while subplots are individual plotting areas within that canvas. This separation allows you to create complex layouts with multiple related plots.

- plt.subplots(rows, cols) creates a figure with a grid of subplots and returns both the figure and an array of axes objects
- Access individual subplots using array indexing: axes[0,0] for top-left, axes[1,1] for bottom-right
- Each subplot is independent - you can have different plot types, scales, and styles on each one
- The figure object controls overall properties like size, while axes objects control individual plot properties
- Always save your figure with fig.savefig() before showing it - this ensures you capture exactly what you see on screen

## Customizing Plots

matplotlib's power comes from its extensive customization options. Every visual element can be controlled - colors, line styles, markers, fonts, spacing, and more. The key is knowing which properties belong to the figure versus the axes.

- ax.set_title(), ax.set_xlabel(), ax.set_ylabel() control text elements - these are axes-level properties
- ax.set_xlim() and ax.set_ylim() control axis ranges - crucial for focusing attention on relevant data ranges
- ax.grid(True, alpha=0.3) adds subtle grid lines that help readers estimate values without cluttering the plot
- ax.legend() displays the legend for plots with multiple series - only call this if you have multiple data series
- Color specification works with named colors ('red'), hex codes ('#FF5733'), or RGB tuples (0.1, 0.2, 0.5)

## Colors, Markers, and Line Styles

Visual elements like colors, markers, and line styles are the vocabulary of data visualization. They help distinguish between different data series and guide the reader's eye through your plot. The key is using these elements consistently and meaningfully.

- Named colors like 'red', 'blue', 'green' are easy to remember but limited - use hex codes for precise color control
- Line styles communicate different types of relationships: solid lines for actual data, dashed for projections, dotted for thresholds
- Markers help identify individual data points - use them sparingly to avoid visual clutter
- The format string 'o-' means circles connected by solid lines, 's--' means squares connected by dashed lines
- Always consider colorblind accessibility - avoid relying solely on red/green distinctions

# pandas Plotting: Quick Exploration

pandas plotting methods are built on matplotlib but provide a much simpler interface for common visualization tasks. They're perfect for the "let me see what this data looks like" phase of analysis, where speed and simplicity matter more than perfect formatting.

- df.plot() creates a line plot by default - perfect for time series or any data where you want to see trends
- The kind parameter controls plot type: 'bar', 'hist', 'scatter', 'box', 'pie' - each optimized for different data types
- pandas plotting automatically handles your DataFrame index as the x-axis, which is usually what you want
- Use subplots=True when you have multiple columns and want to see each one separately
- The ax parameter lets you plot on specific subplot axes when creating multi-panel figures

## DataFrame Plotting Options

pandas plotting methods accept many parameters that control both the plot appearance and the underlying matplotlib behavior. Understanding these options lets you create publication-ready plots without dropping down to matplotlib syntax.

- figsize=(width, height) controls the overall plot size - essential for presentations and papers
- title, xlabel, ylabel parameters add descriptive text that makes your plots self-explanatory
- legend=True shows a legend when you have multiple data series - set to False to hide it
- grid=True adds subtle grid lines that help readers estimate values
- The alpha parameter controls transparency - useful when plotting multiple overlapping series

# LIVE DEMO!

# seaborn: Statistical Visualization

seaborn makes statistical plots beautiful by default while still being customizable when you need it. It's built on matplotlib but provides higher-level functions for common statistical visualization tasks. The key insight is that seaborn automatically handles color palettes, statistical annotations, and relationship plotting.

- sns.set_style() changes the overall appearance - 'whitegrid' is clean and professional, 'darkgrid' works well for presentations
- seaborn functions expect your data in long format with separate columns for x, y, and grouping variables
- The data parameter lets you pass your entire DataFrame, then specify which columns to use for x, y, and color
- seaborn automatically handles color palettes and legends - you rarely need to specify colors manually
- Most seaborn functions return matplotlib axes objects, so you can still customize with matplotlib syntax

## Advanced seaborn Features

seaborn excels at showing relationships between variables and distributions within groups. These advanced features handle complex statistical visualizations that would require many lines of matplotlib code.

- sns.pairplot() creates a grid showing all pairwise relationships - perfect for understanding correlation structure
- sns.jointplot() combines scatter plots with marginal distributions - shows both the relationship and individual variable distributions
- sns.violinplot() shows the full distribution shape, not just summary statistics like box plots
- sns.stripplot() shows individual data points, useful for small datasets where you want to see every observation
- sns.catplot() is a general function for categorical plots - it can create box plots, violin plots, or strip plots depending on the kind parameter

# Density Plots and Distribution Visualization

Density plots show the shape of your data distribution more smoothly than histograms. They're particularly useful for comparing distributions between groups or identifying multiple modes in your data.

- Density plots use kernel density estimation (KDE) to create smooth curves that show distribution shape
- sns.kdeplot() creates pure density plots, while sns.histplot(kde=True) combines histograms with density overlays
- Density plots reveal distribution features that might be hidden in histogram bins - bimodal distributions, skewness, outliers
- Use different colors or linestyles to compare distributions between groups
- Density plots work best with continuous data - for categorical data, stick with bar charts or histograms

# Modern Visualization Libraries

The Python visualization ecosystem continues to evolve with new libraries that offer different approaches to creating plots. While matplotlib and seaborn remain the workhorses, modern tools like altair and plotnine offer exciting alternatives for specific use cases.

## vega-altair: Grammar of Graphics

altair implements the grammar of graphics approach, making it intuitive to build complex visualizations by combining simple building blocks. It's particularly powerful for interactive plots and web-based dashboards.

- altair uses a declarative syntax where you describe what you want rather than how to draw it
- The encode() method maps data columns to visual properties like position, color, size, and shape
- altair automatically handles legends, axes, and color scales - you focus on the data mapping
- Interactive features come for free - zooming, panning, and brushing work automatically
- altair plots can be exported to HTML for web sharing or converted to static images

## Other Modern Tools: plotnine, Bokeh, and Plotly

These libraries offer specialized approaches to visualization. plotnine brings R's ggplot2 syntax to Python, while Bokeh and Plotly focus on interactive web-based visualizations.

- plotnine uses ggplot2 syntax familiar to R users - if you know ggplot2, plotnine will feel natural
- Bokeh creates interactive web visualizations that work in browsers without additional setup
- Plotly specializes in interactive dashboards and web-based data exploration tools
- These tools excel at specific use cases but aren't replacements for matplotlib/seaborn in general data analysis
- Consider these tools when you need interactivity, web deployment, or specific syntax preferences

# Command Line: Persistent Sessions with tmux

When working with long-running data analysis or remote servers, you need sessions that survive network hiccups and accidental terminal closures. tmux provides persistent terminal sessions that continue running even when you disconnect.

- tmux creates persistent sessions that survive network disconnections and terminal closures
- Use tmux new-session -s session_name to create named sessions that are easy to identify and reconnect to
- Ctrl+b d detaches from a session while keeping it running in the background
- tmux attach-session -t session_name reconnects to a running session
- This is essential for remote computing where network interruptions can kill your analysis

## tmux Configuration

tmux can be customized through configuration files to improve the user experience. Basic configuration makes tmux more user-friendly and efficient for data analysis workflows.

- Create ~/.tmux.conf to customize tmux behavior and key bindings
- set -g mouse on enables mouse support for easier window and pane management
- set -g default-terminal "screen-256color" ensures proper color support
- Custom key bindings can make common operations faster and more intuitive
- Configuration changes require reloading the config file or restarting tmux sessions

# Visualization Best Practices

Good visualization is about communication - it should be clear, honest, and serve the reader first. The goal is to make your data tell a compelling story without misleading or confusing your audience.

# FIXME: Add before/after visualization examples showing good vs bad design

# FIXME: Add color palette examples for different data types

## The Right Chart for the Job

Different chart types are optimized for different types of data and questions. Choosing the right chart type is crucial for effective communication - the wrong chart can hide important patterns or mislead your audience.

- Line charts are perfect for time series data and trends over time - they show how things change continuously
- Bar charts work well for categorical comparisons and discrete values - they're easy to read and compare
- Scatter plots reveal relationships between two continuous variables - they show correlation and outliers
- Histograms show the distribution of a single variable - they reveal skewness, modes, and outliers
- Box plots summarize distributions with quartiles and outliers - great for comparing groups
- Heatmaps show patterns in 2D data - they reveal clusters and relationships in matrix data
- Pie charts should be used sparingly - they're hard to compare accurately and often mislead

## Design Principles

Effective visualization follows key design principles that make your plots both beautiful and informative. These principles help ensure your visualizations communicate clearly without misleading your audience.

- Clarity means your message should be obvious - use clear titles, labels, and legends
- Honesty means don't mislead with scale manipulation or design choices that distort the data
- Simplicity means removing unnecessary elements that don't contribute to understanding
- Consistency means using the same colors, fonts, and styles throughout your presentation
- Accessibility means considering colorblind users and providing alternative ways to distinguish data
- Always test your visualizations with colleagues who aren't familiar with your data

# LIVE DEMO!

# Key Takeaways

Data visualization is a crucial skill for any data scientist. The key is choosing the right tool for each task and following design principles that make your plots both beautiful and informative.

- Start with pandas for quick exploration, use seaborn for statistical analysis, customize with matplotlib when needed
- Choose the right chart type for your data and question - different charts reveal different patterns
- Follow design principles for effective communication - clarity, honesty, simplicity, consistency, accessibility
- Use tmux for persistent computing sessions that survive disconnections
- Explore modern tools like altair and plotnine for specific needs like interactivity or grammar-of-graphics approaches
- Remember that good visualization is about communication - make your data tell a compelling story

You now have the skills to create effective visualizations that tell compelling data stories. These are essential skills for any data scientist.
