# Data Visualization Assignment

In this assignment, you'll work with population data from the Gapminder dataset to create various visualizations using matplotlib and seaborn.

## Setup

The data is located in the `ddf--datapoints--population--by--country--age--gender--year` directory. Each file contains population data for a specific country across different years, ages, and genders. 2024 and years in the future are estimates. Technically, all the years are estimates.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Tips for the data:
- Use `df = pd.read_csv(FILENAME)` to read the data
- You can make a dict of dataframes as `data = {}; data[COUNTRY] = pd.read_csv(COUNTRY_FILENAME)`
- Combine dataframes from multiple countries using `pd.concat(DATAFRAME1, DATAFRAME2)`

Other common operations:

```python
# Group by year and sum population
yearly_total = df.groupby('year')['population'].sum()

# Filter specific age groups
youth = df[df['age'].isin(['0-4', '5-9', '10-14'])]

# Calculate gender ratios
gender_ratio = df.groupby('year').agg({
    'population': lambda x: x[df['gender']=='f'].sum() / x[df['gender']=='m'].sum()
})
```

## Submission

Please write up your solutions for #1 and #2 in a Jupyter notebook, `visualization.ipynb`, including markdown cells for text and python cells for data handling and generating graphs.

## Part 1: Matplotlib

1. Create a line plot comparing total population over time:
   - Choose 5 countries
   - Calculate the total population by country for each year
   - Create a line plot showing population trends
   - Include a legend identifying each country
   - Add appropriate axis labels and title
   - Use different line styles or colors to distinguish the countries

2. Investigate demographic shifts using a stacked bar chart:
   - Compare age distributions (0-14, 15-30, 31-45, 46-65, 66+) between 1950, 1985, and 2020
   - Create a stacked bar chart showing these changes
   - Calculate and display the percentage for each age group
   - Add text annotations showing key percentages
   - Include clear labels and a legend

Tips for matplotlib:
- Set figure size before creating the plot: plt.figure(figsize=(10, 6))
- Use meaningful colors: plt.cm.Set2 or plt.cm.tab10 for distinct colors
- Add grid lines with plt.grid() for better readability
- Adjust legend position if it overlaps with data

## Part 2: Seaborn

1. Create a heatmap showing the population distribution across age groups (0-14, 15-30, 31-45, 46-65, 66+) and country for 5 countries of your choice in a specific year.
   - Use seaborn's heatmap function
   - Add clear annotations and labels
   - Write a brief explanation of what the heatmap reveals about the population structure

2. Create a pair of violin plots comparing the age distributions between two countries in 2020, separated by gender.
   - Create side-by-side violin plots using seaborn
   - Add clear labels and a title
   - Write a brief comparison of what the plots reveal

3. Create a pairplot comparing population metrics across four countries:
   - Select four countries of your choice
   - Include all variables in the dataset
   - Use the country as the hue parameter to distinguish between countries
   - Write a brief summary of any patterns or relationships revealed by the pairplot

Tips for seaborn:
- Set the style before plotting: sns.set_style("whitegrid")
- Use built-in color palettes: sns.color_palette("husl", 8)

## Part 3: Open-ended, not required

Create a visualization using either plotnine, streamlit, or plotly dash. Here are some concrete examples:

### Using plotnine
Create a series of statistical plots that show:
- Population trends over time using smooth trend lines
- Faceted views comparing multiple countries or regions
- Clear themes and styling following the Grammar of Graphics

### Using streamlit
Build an interactive dashboard that lets users:
- Select 2-3 countries from a dropdown menu
- View their total populations over time as a line chart
- Show a data table with key statistics
- Add filters for specific years or age groups

### Using plotly dash
Create an interactive application that:
- Shows population distribution across age groups
- Lets users switch between different years using a slider
- Updates automatically when selections change

## Tips

- Pay attention to color choices and accessibility
- Make sure your visualizations are easy to understand
- Include appropriate titles and labels
- Consider the story your visualizations tell about the data
- For the bonus question, focus on creating an engaging user experience
