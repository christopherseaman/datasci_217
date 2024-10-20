---
marp: true
theme: sqrl
paginate: true
class: invert
---

# Lecture 07: Data Visualization for Health Data Science

## Section 1: Data Visualization with pandas, Matplotlib, and Seaborn

---

# Brief Recap: Essential Python Concepts

- **Importing Libraries**
  - Use `import` statements to include external libraries.
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    ```
    - *Think of libraries as the special moves in your coding arsenal—gotta import them to unleash their power!*
- **Working with DataFrames**
  - DataFrames are table-like data structures provided by pandas.
    ```python
    df = pd.read_csv('data.csv')
    ```
    - *Like Excel spreadsheets, but way cooler and code-friendly!*
- **Basic Plotting**
  - Call plotting functions from libraries to create visualizations.
    ```python
    plt.plot(x, y)
    ```
    - *Because a picture is worth a thousand rows of data!*
- **Displaying Plots**
  - Use `plt.show()` to render the visualization.
    - *Don't leave your plots feeling unseen—let them shine!*

*If you need a refresher on these concepts, please review previous lectures or resources.*

---

# Introduction to Data Visualization Tools

- **pandas**
  - High-level data manipulation library.
  - Built-in plotting methods that simplify creating visualizations from DataFrames.
  - *Pandas: Not just adorable bears, but also your data's best friend!*
- **Matplotlib**
  - Foundation library for data visualization in Python.
  - Offers detailed control over plots.
  - *Think of it as the "Swiss Army Knife" of plotting—versatile and handy!*
- **Seaborn**
  - Statistical data visualization library.
  - Provides attractive default styles and color palettes.
  - *Because your data deserves to look as fabulous as it feels!*

---

# Matplotlib: The Foundation

## Introduction

- **Matplotlib** is the most widely used library for plotting in Python.
  - *It's the OG of Python plotting—setting the stage since the early days!*
- Think of it as the core building block for most other Python visualization libraries.
  - *Other libraries stand on the shoulders of this giant!*

---

## Key Concepts in Matplotlib

- **Figures and Axes**
  - **Figure**: The overall container for all plot elements.
    - *The canvas where your masterpiece comes to life!*
  - **Axes**: The area where data is plotted (can be thought of as individual plots).
    - *The stage where your data takes center spotlight!*

---

## Important Methods in Matplotlib

### 1. `plot()`: Basic Line Plot

- **Explanation**: Creates a simple line plot connecting data points.
- **Structure**:
  ```python
  plt.plot(x, y, marker, linestyle, color, label)
  ```
  - **Required Arguments**:
    - `x`: Data for the x-axis.
    - `y`: Data for the y-axis.
  - **Optional Arguments**:
    - `marker`: Style of the data point markers (e.g., `'o'` for circles).
    - `linestyle`: Style of the line connecting data points (e.g., `'-'` for solid line).
    - `color`: Color of the line.
    - `label`: Label for the legend.

---

### Code Example: Line Plot

```python
import matplotlib.pyplot as plt

# Data
years = [2010, 2012, 2014, 2016, 2018, 2020]
patients = [150, 180, 200, 230, 260, 300]

# Create line plot
plt.plot(years, patients, marker='o', linestyle='-', color='b', label='Patients')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Number of Patients')
plt.title('Number of Patients Over Years')

# Add legend
plt.legend()

# Show plot
plt.show()
```

---

### Code Explanation

- **Importing Matplotlib**:
  ```python
  import matplotlib.pyplot as plt
  ```
  - *Summoning the plotting powers—assemble!*
- **Defining Data**:
  ```python
  years = [2010, 2012, 2014, 2016, 2018, 2020]
  patients = [150, 180, 200, 230, 260, 300]
  ```
  - *Years and patient counts—the dynamic duo of our plot!*
- **Creating the Plot**:
  ```python
  plt.plot(years, patients, marker='o', linestyle='-', color='b', label='Patients')
  ```
  - *Connecting the dots like a constellation in the data sky!*
- **Adding Labels and Title**:
  ```python
  plt.xlabel('Year')
  plt.ylabel('Number of Patients')
  plt.title('Number of Patients Over Years')
  ```
  - *Because every hero (plot) needs an introduction!*
- **Adding a Legend**:
  ```python
  plt.legend()
  ```
  - *Giving credit where it's due—legends aren't just for myths!*
- **Displaying the Plot**:
  ```python
  plt.show()
  ```
  - *Showtime! Let's see this plot steal the show!*

---

### Output Example

*An image showing a line plot of patients over years with labeled axes and title.*

#FIXME-{{Insert the actual line plot image showing the trend of patient numbers over the years}}

---

### Common Issues and Troubleshooting

- **No Plot Displayed**:
  - Ensure `plt.show()` is called after plotting commands.
    - *Even the best actors need the curtain to rise—don't forget `plt.show()`!*
- **Data Length Mismatch**:
  - Verify that `x` and `y` are of equal length.
    - *Mismatch is great in a rom-com, but not in data plotting!*
- **Import Errors**:
  - Install Matplotlib using `pip install matplotlib` if it's not installed.
    - *Because missing imports are the coding equivalent of forgetting your keys!*

---

### 2. `scatter()`: Scatter Plot

- **Explanation**: Creates a scatter plot of x vs. y, useful for showing relationships between variables.
- **Structure**:
  ```python
  plt.scatter(x, y, s, c, alpha)
  ```
  - **Required Arguments**:
    - `x`: Data for the x-axis.
    - `y`: Data for the y-axis.
  - **Optional Arguments**:
    - `s`: Size of markers.
    - `c`: Color of markers.
    - `alpha`: Transparency level of markers.

---

### Code Example: Scatter Plot

```python
import matplotlib.pyplot as plt

# Data
age = [25, 35, 45, 20, 30, 40, 50, 60]
blood_pressure = [120, 130, 125, 115, 135, 140, 150, 145]

# Create scatter plot
plt.scatter(age, blood_pressure, c='red', alpha=0.7)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.title('Blood Pressure vs Age')

# Show plot
plt.show()
```

---

### Code Explanation

- **Defining Data**:
  - `age`: List of ages.
  - `blood_pressure`: Corresponding blood pressure readings.
  - *Matching ages with blood pressures like pairing fine wine with cheese!*
- **Creating the Plot**:
  ```python
  plt.scatter(age, blood_pressure, c='red', alpha=0.7)
  ```
  - *Plotting points like throwing darts at the board—aiming for insights!*
- **Adding Labels and Title**:
  - Labels the axes and sets the title.
  - *Guiding the audience through your data journey!*
- **Displaying the Plot**:
  - Uses `plt.show()` to render the plot.

---

### Output Example

*An image showing a scatter plot of blood pressure vs age.*

#FIXME-{{Include the actual scatter plot image of blood pressure versus age}}

---

### Additional Plot Types in Matplotlib

- **Bar Plot**:
  ```python
  plt.bar(categories, values)
  ```
  - *Great for categorical data—because sometimes you just need to raise the bar!*
- **Histogram**:
  ```python
  plt.hist(data, bins=10)
  ```
  - *Unveiling the distribution secrets of your data—think of it as data's fingerprint!*
- **Box Plot**:
  ```python
  plt.boxplot(data)
  ```
  - *Spotting outliers faster than you can say "Houston, we have a problem!"*
- **Pie Chart**:
  ```python
  plt.pie(sizes, labels=labels)
  ```
  - *When you want to show proportions and make your data slice-of-life interesting!*

---

# pandas: Built-in Plotting

## Introduction

- pandas provides convenient plotting methods directly on DataFrames and Series.
- Simplifies the creation of plots without explicitly using Matplotlib commands.
- *It's like having a fast-pass to plotting—skip the lines and get straight to the fun!*

---

## Important Methods in pandas

### 1. `plot()`: Line Plot

- **Explanation**: Plots DataFrame or Series data as lines.
- **Structure**:
  ```python
  df.plot(x='column1', y='column2', kind='line', marker, title)
  ```
  - *Who says data can't be classy? Let pandas do the heavy lifting!*

---

### Code Example: pandas Line Plot

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('hospital_admissions.csv')

# Inspect data
print(df.head())
```

*Sample Output*:
```
   Year  Admissions
0  2016         500
1  2017         550
2  2018         600
3  2019         650
4  2020         700
```

---

### Continuing the Code Example

```python
# Plot data
df.plot(x='Year', y='Admissions', kind='line', marker='o', title='Yearly Hospital Admissions')

# Add labels
plt.xlabel('Year')
plt.ylabel('Number of Admissions')

# Show plot
plt.show()
```

---

### Code Explanation

- **Loading Data**:
  ```python
  df = pd.read_csv('hospital_admissions.csv')
  ```
  - *Because data isn't going to read itself—time to bring it into the game!*
- **Inspecting Data**:
  ```python
  print(df.head())
  ```
  - *Always good to peek under the hood before hitting the road!*
- **Creating the Plot**:
  ```python
  df.plot(x='Year', y='Admissions', kind='line', marker='o', title='Yearly Hospital Admissions')
  ```
  - *One line of code to plot—all the cool kids are doing it!*
- **Adding Labels**:
  - Sets x and y-axis labels.
- **Displaying the Plot**:
  - Shows the plot using `plt.show()`.

---

### Output Example

*An image showing a line plot of yearly hospital admissions.*

#FIXME-{{Include the actual line plot image of hospital admissions over the years}}

---

### Common Issues and Troubleshooting

- **Column Not Found**:
  - Ensure column names in `x` and `y` match exactly with the DataFrame.
    - *Check your spelling—autocorrect won't save you here!*
- **No Plot Displayed**:
  - If plots don't display in Jupyter notebooks, use `%matplotlib inline` at the beginning.
    - *A little magic command to make plots appear—abracadabra!*

---

### 2. `hist()`: Histogram

- **Explanation**: Plots a histogram of a single column or series.
- *Unleash your inner statistician and dive into the distribution!*

---

### Code Example: pandas Histogram

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('patient_ages.csv')

# Plot histogram
df['Age'].hist(bins=10, color='skyblue', alpha=0.7)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.title('Age Distribution of Patients')

# Show plot
plt.show()
```

---

### Code Explanation

- **Loading Data**:
  - Reads patient age data into a DataFrame.
- **Creating the Histogram**:
  ```python
  df['Age'].hist(bins=10, color='skyblue', alpha=0.7)
  ```
  - *Plotting that data like a boss—with colors and style!*
- **Adding Labels and Title**:
  - Provides context to your masterpiece.
- **Displaying the Plot**:
  - Renders the histogram.

---

### Output Example

*An image showing a histogram of patient age distribution.*

#FIXME-{{Include the actual histogram image of patient age distribution}}

---

# Seaborn: Statistical Data Visualization

## Introduction

- **Seaborn** enhances Matplotlib's functionality by providing high-level interfaces.
- Ideal for statistical plots and works well with pandas DataFrames.
- *It's like upgrading from a bicycle to a sports car—same principles, more speed and style!*

---

## Important Methods in Seaborn

### 1. `scatterplot()`: Scatter Plot

- **Explanation**: Creates enhanced scatter plots with additional functionalities.
- **Structure**:
  ```python
  sns.scatterplot(x='x_col', y='y_col', data=df, hue, size, style)
  ```
  - *Bring your scatter plots to life with colors and styles—because who wants a boring plot?*

---

### Code Example: Seaborn Scatter Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv('health_data.csv')

# Create scatter plot
sns.scatterplot(x='BMI', y='BloodPressure', hue='AgeGroup', data=df)

# Add title
plt.title('Blood Pressure vs BMI by Age Group')

# Show plot
plt.show()
```

---

### Code Explanation

- **Importing Libraries**:
  - Imports seaborn, matplotlib, and pandas.
- **Loading Data**:
  ```python
  df = pd.read_csv('health_data.csv')
  ```
  - *Data in, insights out—that's the way we roll!*
- **Creating the Plot**:
  ```python
  sns.scatterplot(x='BMI', y='BloodPressure', hue='AgeGroup', data=df)
  ```
  - *Adding a splash of color—because life is too short for monochrome plots!*
- **Adding Title**:
  - Sets the title of the plot.
- **Displaying the Plot**:
  - Renders the scatter plot.

---

### Output Example

*An image showing a scatter plot of Blood Pressure vs BMI colored by Age Group.*

#FIXME-{{Include the actual Seaborn scatter plot image with hue based on Age Group}}

---

### Interpreting the Plot

- **Color Coding**:
  - Different colors represent different age groups.
    - *Like sorting candies by flavor—visually satisfying and informative!*
- **Trend Analysis**:
  - Helps identify how BMI relates to Blood Pressure across age groups.
    - *Finding patterns faster than a detective on a TV show!*

---

### Additional Plot Types in Seaborn

- **Histogram and KDE Plot**:
  ```python
  sns.histplot(data=df, x='BMI', kde=True)
  ```
  - *Because sometimes you need more curves than a roller coaster!*
- **Box Plot**:
  ```python
  sns.boxplot(x='AgeGroup', y='Cholesterol', data=df)
  ```
  - *Unboxing the secrets of your data—no subscription required!*
- **Heatmap**:
  ```python
  sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
  ```
  - *Visualizing correlations hotter than the latest trends!*
- **FacetGrid**:
  ```python
  g = sns.FacetGrid(df, col='Gender')
  g.map(plt.hist, 'BMI')
  ```
  - *When one plot isn't enough—facet all the things!*

---

### Common Issues and Troubleshooting

- **Seaborn Version Compatibility**:
  - Ensure you have the latest version installed using `pip install seaborn --upgrade`.
    - *Because outdated libraries are so last season!*
- **Attribute Errors**:
  - Verify function names and parameters, as they may differ between versions.
    - *Double-check before you wreck—your code, that is!*

---

# Tips for Effective Visualization

- **Consistency**:
  - Use consistent styles and color schemes throughout your visualizations.
    - *Because even data likes to coordinate its outfit!*
- **Clarity**:
  - Label axes, include units of measurement, and provide legends when necessary.
    - *Don't make your audience play "Guess Who" with your plots!*
- **Simplicity**:
  - Avoid unnecessary decorations that don't add informational value.
    - *Remember, sometimes less is more—unless we're talking about tacos!*
- **Interpretation**:
  - Always consider how the audience will interpret your plots.
    - *Aim for "Aha!" moments, not head-scratching confusion!*

---

# Resources for Further Learning

- **Matplotlib Documentation**: [matplotlib.org](https://matplotlib.org/)
- **pandas Visualization Guide**: [pandas.pydata.org](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html)
- **Seaborn Tutorials**: [seaborn.pydata.org](https://seaborn.pydata.org/tutorial.html)
- **Troubleshooting Tips**: Search for errors on Stack Overflow or consult the official documentation.

---

# Practice Exercise

**Task**: Create a visualization using one of the libraries discussed.

- **Dataset**: Choose a health-related dataset (e.g., patient records, public health statistics).
- **Objective**:
  - Use pandas to read and explore the data.
  - Create at least one plot (line, scatter, histogram, etc.).
  - Customize the plot with titles, labels, and style adjustments.
- **Submission**: Share your code and the resulting visualization.

*Remember, with great plotting power comes great responsibility—may your plots be ever in your favor!*

---
