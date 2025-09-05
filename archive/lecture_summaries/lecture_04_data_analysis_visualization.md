# Lecture 4: Data Analysis + Visualization

## Learning Objectives
By the end of this lecture, students will be able to:
- Perform comprehensive data cleaning and preprocessing for real-world datasets
- Conduct exploratory data analysis using statistical summaries and grouping operations
- Create effective data visualizations using matplotlib, seaborn, and pandas plotting
- Apply data wrangling techniques including reshaping, merging, and transforming data
- Understand and implement data visualization best practices and design principles
- Combine analysis and visualization to tell compelling data stories

## Content Consolidation Details

### Primary Sources (Current Lectures)
- **Lecture 06 (90%)**: Data wrangling, cleaning, reshaping, combining datasets, visualization basics
- **Lecture 07 (85%)**: Data visualization principles, matplotlib, seaborn, advanced plotting techniques
- **Lecture 05 (20%)**: Pandas operations, basic plotting
- **Lecture 08 (15%)**: Time series basics, statistical analysis introduction

### Secondary Integration
- **Lecture 09 (10%)**: Practical automation approaches for data processing

## Specific Topics Covered

### Advanced Data Cleaning and Preprocessing (40 minutes)
1. **Data Quality Assessment**
   - Comprehensive data exploration: `.info()`, `.describe()`, `.value_counts()`
   - Identifying data quality issues: missing values, duplicates, inconsistencies
   - Data profiling techniques and patterns recognition
   - Understanding data types and appropriate conversions

2. **Handling Missing Data Strategically**
   - Different types of missing data: MCAR, MAR, MNAR
   - Advanced missing data strategies: forward fill, backward fill, interpolation
   - When to drop vs impute missing values
   - Domain-specific imputation strategies

3. **Data Type Optimization and Conversion**
   - Categorical data: creating categories, ordered vs unordered
   - Date and time parsing: `pd.to_datetime()`, time zones, date arithmetic
   - Numerical conversions: handling errors with `pd.to_numeric()`
   - Memory optimization techniques

4. **Text and String Processing**
   - Advanced string operations: `.str.contains()`, `.str.extract()` with regex
   - Text cleaning: removing whitespace, standardizing case, handling special characters
   - String splitting and concatenation for data transformation
   - Handling encoding issues and special characters

5. **Outlier Detection and Treatment**
   - Statistical methods: Z-score, IQR method
   - Visual identification using box plots and scatter plots
   - Domain knowledge vs statistical approaches
   - Outlier treatment strategies: removal, transformation, capping

### Data Transformation and Reshaping (30 minutes)
1. **Data Reshaping Operations**
   - Melting data: converting wide to long format with `pd.melt()`
   - Pivoting data: converting long to wide format with `pd.pivot()`
   - Pivot tables: `pd.pivot_table()` with aggregation functions
   - Stacking and unstacking: multi-level index manipulation

2. **Combining Datasets**
   - Concatenation: `pd.concat()` with different axes and join types
   - Merging: `pd.merge()` with different join types (inner, outer, left, right)
   - Index-based joins: using indices for efficient merging
   - Handling duplicate keys and merge validation

3. **Grouping and Aggregation**
   - Basic grouping: `.groupby()` with single and multiple columns
   - Aggregation functions: built-in and custom functions
   - Named aggregation for cleaner output
   - Grouped transformations and filtering
   - Split-apply-combine methodology

4. **Feature Engineering Basics**
   - Creating new variables from existing data
   - Binning and discretization: `pd.cut()` and `pd.qcut()`
   - Date feature extraction: year, month, day, weekday
   - Mathematical transformations: log, square root, normalization

### Data Visualization Fundamentals (45 minutes)
1. **Visualization Libraries Overview**
   - Pandas plotting: quick and convenient for exploration
   - Matplotlib: fine-grained control and customization
   - Seaborn: statistical visualization with better defaults
   - When to use each library for different purposes

2. **Basic Plot Types and Use Cases**
   - Line plots: time series and continuous data
   - Scatter plots: relationships between variables
   - Bar plots: categorical data comparisons
   - Histograms: distribution visualization
   - Box plots: distribution summaries and outlier detection

3. **Matplotlib Essentials**
   - Figure and axes concepts: understanding the hierarchy
   - Basic plotting: `plt.plot()`, `plt.scatter()`, `plt.bar()`
   - Customization: colors, markers, line styles, labels
   - Subplots: creating multiple plots in one figure
   - Saving plots: different formats and resolution settings

4. **Seaborn for Statistical Visualization**
   - Setting styles and color palettes: `sns.set_style()`, `sns.set_palette()`
   - Statistical plots: `sns.scatterplot()`, `sns.lineplot()`, `sns.barplot()`
   - Distribution plots: `sns.histplot()`, `sns.boxplot()`, `sns.violinplot()`
   - Categorical plots: handling categorical variables effectively
   - Correlation visualization: `sns.heatmap()` for correlation matrices

5. **Advanced Visualization Techniques**
   - Faceting: creating multiple subplots with `sns.FacetGrid()`
   - Multi-variable visualization: using color, size, and shape
   - Statistical relationships: regression lines and confidence intervals
   - Custom color palettes and accessibility considerations

### Design Principles and Best Practices (20 minutes)
1. **Visualization Design Principles**
   - Edward Tufte's principles: maximize data-ink ratio, minimize chartjunk
   - Choosing appropriate chart types for data and message
   - Color theory: using color effectively and accessibly
   - Typography and labeling best practices

2. **Common Visualization Mistakes**
   - Misleading scales and axes manipulation
   - Inappropriate chart types for data
   - Overcomplication and unnecessary 3D effects
   - Poor color choices and accessibility issues

3. **Storytelling with Data**
   - Structuring analysis for clear communication
   - Creating narrative flow in visualizations
   - Highlighting key insights effectively
   - Audience considerations and context

4. **Practical Implementation**
   - Creating publication-ready plots
   - Consistent styling across multiple visualizations
   - Interactive elements (basic introduction)
   - Exporting and sharing visualizations

### Applied Data Analysis Workflow (25 minutes)
1. **Exploratory Data Analysis (EDA) Process**
   - Systematic approach to data exploration
   - Question formulation and hypothesis generation
   - Iterative analysis and discovery process
   - Documenting findings and insights

2. **Statistical Analysis Integration**
   - Descriptive statistics: central tendency, variability, distribution
   - Basic correlation analysis: Pearson and Spearman correlation
   - Cross-tabulation and chi-square tests (introduction)
   - Statistical significance vs practical significance

3. **Case Study Application**
   - Complete data analysis workflow from raw data to insights
   - Real-world dataset with multiple variables and challenges
   - Problem formulation, analysis, and presentation
   - Reproducible analysis documentation

## Content to Trim (20% reduction from source lectures)

### From Lecture 06
- **Remove (10 minutes)**: Complex regex patterns in string processing
- **Reduce (8 minutes)**: Advanced pivot table options - focus on essential use cases
- **Simplify (5 minutes)**: Complex nested data structure examples

### From Lecture 07
- **Remove (15 minutes)**: Advanced matplotlib customization details
- **Remove (12 minutes)**: Complex multi-panel visualization examples
- **Reduce (10 minutes)**: Extended color theory discussion - focus on practical application

### From Lecture 08
- **Remove (8 minutes)**: Advanced time series decomposition
- **Remove (5 minutes)**: Complex statistical model examples

## Practical Exercises and Hands-on Components

### Data Cleaning Workshop (25 minutes)
1. **Messy Dataset Challenge**
   - Work with deliberately messy real-world dataset
   - Identify and fix multiple data quality issues
   - Document cleaning decisions and rationale
   - Compare before/after data quality metrics

2. **Text Data Processing**
   - Clean customer feedback data or social media posts
   - Standardize formats and extract key information
   - Handle encoding issues and special characters
   - Create categorical variables from text analysis

### Data Transformation Lab (20 minutes)
1. **Reshaping Exercise**
   - Transform sales data from wide to long format
   - Create pivot tables for different analysis perspectives
   - Practice multiple grouping and aggregation scenarios

2. **Data Integration Project**
   - Merge multiple related datasets (customers, orders, products)
   - Handle different join scenarios and key mismatches
   - Create comprehensive analytical dataset

### Visualization Workshop (30 minutes)
1. **Exploratory Visualization**
   - Create comprehensive EDA report for new dataset
   - Use multiple plot types to explore different aspects
   - Apply design principles to improve clarity

2. **Statistical Storytelling**
   - Build narrative around data analysis findings
   - Create publication-ready visualizations
   - Practice explaining technical results to non-technical audience

3. **Critique and Improvement**
   - Analyze poorly designed visualizations
   - Identify problems and create improved versions
   - Apply accessibility principles and best practices

### Integrated Analysis Project (25 minutes)
1. **Complete Workflow Exercise**
   - Take raw dataset through full analysis pipeline
   - Clean, transform, analyze, and visualize
   - Generate actionable insights and recommendations

2. **Domain-Specific Analysis**
   - Choose from business, health, or environmental dataset
   - Apply appropriate analytical techniques
   - Create executive summary with key findings

## Prerequisites and Dependencies

### From Previous Lectures
- Proficiency with NumPy arrays and operations
- Solid understanding of Pandas DataFrames and Series
- Basic Python programming skills and data structures
- File operations and data reading/writing

### Technical Requirements
- Complete data science Python stack: numpy, pandas, matplotlib, seaborn
- Jupyter Notebook or equivalent interactive environment
- Real-world datasets for practice and assignments
- Statistical calculation libraries: scipy (basic functions)

### Preparation Materials
- Data visualization cheat sheets
- Color palette and accessibility guides
- Sample datasets with documented problems
- Plotting reference documentation

## Assessment Components

### Formative Assessment (During Class)
- Live coding exercises with immediate feedback
- Peer review of visualizations and code
- Interactive debugging of data quality issues
- Group discussion of design choices and trade-offs

### Summative Assessment (Assignment)
1. **Data Cleaning and Transformation Portfolio**
   - Clean and prepare multiple datasets with different challenges
   - Document all decisions and rationale
   - Show before/after comparisons and validate results

2. **Comprehensive Data Analysis Report**
   - Complete analysis from raw data to actionable insights
   - Include data quality assessment, cleaning documentation
   - Create multiple visualizations supporting key findings
   - Generate executive summary for non-technical stakeholders

3. **Visualization Design Challenge**
   - Recreate provided poor visualizations with best practices
   - Create original visualizations for complex multi-variable data
   - Demonstrate understanding of design principles and accessibility

4. **Peer Review and Critique**
   - Review and provide feedback on classmate's analysis
   - Suggest improvements for clarity and accuracy
   - Practice professional code review and collaboration skills

## Key Success Metrics
- Students can clean and prepare real-world messy datasets
- Students create effective visualizations that communicate insights clearly
- Students understand and apply data analysis workflow systematically
- Students can critique and improve data visualizations
- 80% of students successfully complete comprehensive analysis project

## Integration with Course Progression
This lecture prepares students for:
- **Lecture 5**: Advanced analytics, statistical modeling, and machine learning
- **Final projects**: Students can now handle complete data analysis workflows
- **Professional work**: Core skills for data analyst and data scientist roles

## Resources and References

### Essential Resources
- [The Visual Display of Quantitative Information](https://www.edwardtufte.com/tufte/books_vdqi) by Edward Tufte
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) by Claus Wilke
- [Pandas User Guide: Data Manipulation](https://pandas.pydata.org/docs/user_guide/merging.html)
- [Seaborn Tutorial Gallery](https://seaborn.pydata.org/tutorial.html)

### Interactive Learning
- [Data Visualization with Python](https://www.kaggle.com/learn/data-visualization)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)
- [Real Python: Data Visualization](https://realpython.com/python-data-visualization/)

### Design and Best Practices
- [ColorBrewer 2.0](https://colorbrewer2.org/) - Color advice for cartography and data visualization
- [Datawrapper Academy](https://academy.datawrapper.de/) - Data visualization knowledge base
- [Storytelling with Data](http://www.storytellingwithdata.com/) - Blog and resources

### Practice Datasets
- [Tidy Tuesday](https://github.com/rfordatascience/tidytuesday) - Weekly data project practice
- [FiveThirtyEight Data](https://github.com/fivethirtyeight/data) - Real datasets with stories
- [Our World in Data](https://ourworldindata.org/) - Global development data

### Tools and Extensions
- [Pandas Profiling](https://github.com/ydataai/pandas-profiling) - Automated EDA reports
- [Plotly](https://plotly.com/python/) - Interactive plotting library
- [Altair](https://altair-viz.github.io/) - Grammar of graphics for Python

### Community and Support
- [Python Graph Gallery](https://www.python-graph-gallery.com/) - Visualization examples
- [Matplotlib Examples](https://matplotlib.org/stable/gallery/index.html)
- [Stack Overflow: matplotlib, seaborn, pandas-plotting tags](https://stackoverflow.com/)
- Course forum for project sharing and peer feedback