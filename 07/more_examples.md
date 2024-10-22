


# pandas: Built-in Plotting

## Introduction

- pandas provides convenient plotting methods directly on DataFrames and Series.
- Simplifies the creation of plots without explicitly using Matplotlib commands.

---

## Important Methods in pandas

### 1. `plot()`: Line Plot

- **Explanation**: Plots DataFrame or Series data as lines.
- **Structure**:
  ```python
  df.plot(x='column1', y='column2', kind='line', marker, title)
  ```

---

### Code Example: pandas Line Plot

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

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
  df = pd.read_csv('data.csv')
  ```
- **Inspecting Data**:
  ```python
  print(df.head())
  ```
- **Creating the Plot**:
  ```python
  df.plot(x='Year', y='Admissions', kind='line', marker='o', title='Yearly Hospital Admissions')
  ```
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
- **No Plot Displayed**:
  - If plots don't display in Jupyter notebooks, use `%matplotlib inline` at the beginning.

---

### 2. `hist()`: Histogram

- **Explanation**: Plots a histogram of a single column or series.

---

### Code Example: pandas Histogram

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Plot histogram
df['Age'].hist(bins=10, color='skyblue', alpha=0.7)
```

---

### Code Explanation

- **Loading Data**:
  - Reads patient age data into a DataFrame.
- **Creating the Histogram**:
  ```python
  df['Age'].hist(bins=10, color='skyblue', alpha=0.7)
  ```
---

### Output Example

*An image showing a histogram of patient age distribution.*

#FIXME-{{Include the actual histogram image of patient age distribution}}

---

# Matplotlib: The Foundation

## Introduction

- **Matplotlib** is the most widely used library for plotting in Python.
- Think of it as the core building block for most other Python visualization libraries.

---

## Key Concepts in Matplotlib

- **Figures and Axes**
  - **Figure**: The overall container for all plot elements.
  - **Axes**: The area where data is plotted (can be thought of as individual plots).

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

  ```python
  import matplotlib.pyplot as plt
  
  years = [2010, 2012, 2014, 2016, 2018, 2020]
  patients = [150, 180, 200, 230, 260, 300]
  
  # Creating the Plot
  plt.plot(years, patients, marker='o', linestyle='-', color='b', label='Patients')
  
  # Adding Labels and Title
  plt.xlabel('Year')
  plt.ylabel('Number of Patients')
  plt.title('Number of Patients Over Years')
  
  # Adding a Legend
  plt.legend()
  
  # Displaying the Plot**:
  plt.show()
  ```
  
---

### Output Example

*An image showing a line plot of patients over years with labeled axes and title.*

#FIXME-{{Insert the actual line plot image showing the trend of patient numbers over the years}}

---

### Common Issues and Troubleshooting

- **No Plot Displayed**:
  - Ensure `plt.show()` is called after plotting commands.
- **Data Length Mismatch**:
  - Verify that `x` and `y` are of equal length.
- **Import Errors**:
  - Install Matplotlib using `pip install matplotlib` if it's not installed.

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
- **Creating the Plot**:
  ```python
  plt.scatter(age, blood_pressure, c='red', alpha=0.7)
  ```
- **Adding Labels and Title**:
  - Labels the axes and sets the title.
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
- **Histogram**:
  ```python
  plt.hist(data, bins=10)
  ```
- **Box Plot**:
  ```python
  plt.boxplot(data)
  ```
- **Pie Chart**:
  ```python
  plt.pie(sizes, labels=labels)
  ```
- **Subplots**:
  ```python
  fig, axs = plt.subplots(2, 2)  # Creates a 2x2 grid of subplots
  axs[0, 0].plot(x, y)  # Plot in the first subplot
  ```
- **Image Display**:
  ```python
  plt.imshow(image_data)
  ```
- **Contour Plot**:
  ```python
  plt.contour(X, Y, Z)  # For line contours
  plt.contourf(X, Y, Z)  # For filled contours
  ```
- **Error Bars**:
  ```python
  plt.errorbar(x, y, yerr=error)
  ```

---

# Seaborn: Statistical Data Visualization

## Introduction

- **Seaborn** enhances Matplotlib's functionality by providing high-level interfaces.
- Ideal for statistical plots and works well with pandas DataFrames.

---

## Important Methods in Seaborn

### 1. `scatterplot()`: Scatter Plot

- **Explanation**: Creates enhanced scatter plots with additional functionalities.
- **Structure**:
  ```python
  sns.scatterplot(x='x_col', y='y_col', data=df, hue, size, style)
  ```

---

### Code Example: Seaborn Scatter Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv('data.csv')

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
  df = pd.read_csv('data.csv')
  ```
- **Creating the Plot**:
  ```python
  sns.scatterplot(x='BMI', y='BloodPressure', hue='AgeGroup', data=df)
  ```
- **Adding Title**:
  - Sets the title of the plot.
- **Displaying the Plot**:
  - Renders the scatter plot.

---

### Output Example

*An image showing a scatter plot of Blood Pressure vs BMI colored by Class.*

#FIXME-{{Include the actual Seaborn scatter plot image with hue based on Age Group}}

---

### Interpreting the Plot

- **Color Coding**:
  - Different colors represent different age groups.
- **Trend Analysis**:
  - Helps identify how BMI relates to Blood Pressure across age groups.

---

### Additional Plot Types in Seaborn

- **Histogram and KDE Plot**:
  ```python
  sns.histplot(data=df, x='BMI', kde=True)
  ```
- **Box Plot**:
  ```python
  sns.boxplot(x='AgeGroup', y='Cholesterol', data=df)
  ```
- **Heatmap**:
  ```python
  sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
  ```
- **FacetGrid**:
  ```python
  g = sns.FacetGrid(df, col='Gender')
  g.map(plt.hist, 'BMI')
  ```
- **Line Plot**:
  ```python
  sns.lineplot(x='Year', y='Value', data=df)
  ```
- **Bar Plot**:
  ```python
  sns.barplot(x='Category', y='Value', data=df)
  ```
- **Violin Plot**:
  ```python
  sns.violinplot(x='Category', y='Value', data=df)
  ```
- **Pair Plot**:
  ```python
  sns.pairplot(df, hue='Category')
  ```

---

### Customization in Seaborn

- **Setting Plot Style**:
  ```python
  sns.set_style("whitegrid")
  ```
- **Color Palettes**:
  ```python
  sns.set_palette("husl")
  ```
- **Figure Size**:
  ```python
  plt.figure(figsize=(10, 6))
  ```

---
## Plotly Dash: Building an App

### Code Example: Simple Dash App

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Initialize the app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Health Data Dashboard'),
    dcc.Dropdown(
        id='age-dropdown',
        options=[{'label': age, 'value': age} for age in df['AgeGroup'].unique()],
        value=df['AgeGroup'].unique()[0]
    ),
    dcc.Graph(id='bmi-bloodpressure-scatter')
])

# Define the callback
@app.callback(
    Output('bmi-bloodpressure-scatter', 'figure'),
    [Input('age-dropdown', 'value')]
)
def update_graph(selected_age):
    filtered_df = df[df['AgeGroup'] == selected_age]
    fig = px.scatter(
        filtered_df, x='BMI', y='BloodPressure',
        title=f'BMI vs Blood Pressure for Age Group {selected_age}'
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
```

---

### Code Explanation

- **Imports**:
  - `dash`, `dash_core_components` (`dcc`), `dash_html_components` (`html`), `dash.dependencies` for interactivity.
  - `plotly.express` for plotting.
- **Data Loading**:
  - Reads health data into a DataFrame.
- **App Initialization**:
  - Creates a Dash app instance.
- **Layout Definition**:
  - Contains a header, dropdown menu, and a graph component.
- **Callback Function**:
  - Updates the graph based on the selected age group from the dropdown.

---

### Output Description

- An interactive dashboard with:
  - A dropdown to select the age group.
  - A scatter plot of BMI vs. Blood Pressure that updates dynamically.

---

## Streamlit: Building an App

### Code Example: Simple Streamlit App

```python
import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('data.csv')

# App title
st.title('Health Data Explorer')

# Sidebar filters
age_group = st.sidebar.selectbox('Select Age Group', df['AgeGroup'].unique())

# Filtered data
filtered_df = df[df['AgeGroup'] == age_group]

# Display data
st.write(f'Data for Age Group: {age_group}')
st.write(filtered_df)

# Plot
fig = px.scatter(filtered_df, x='BMI', y='BloodPressure',
                 title=f'BMI vs Blood Pressure for Age Group {age_group}')
st.plotly_chart(fig)
```

---

### Code Explanation

- **Imports**:
  - `streamlit` for the app interface.
  - `pandas` and `plotly.express` for data handling and plotting.
- **Data Loading**:
  - Reads the health data into a DataFrame.
- **App Components**:
  - `st.title()` sets the title.
  - `st.sidebar.selectbox()` creates a dropdown in the sidebar.
  - `st.write()` displays text and data.
  - `st.plotly_chart()` renders the plot.
- **Interactivity**:
  - Selecting an age group filters the data and updates the display and plot.

---

### Output Description

- A simple web app with:
  - A sidebar for selecting the age group.
  - Display of filtered data.
  - An interactive scatter plot.

---

## Deployment Considerations

- **Local Deployment**:
  - Run the app on your local machine for testing and development.
- **Sharing Apps**:
  - **Plotly Dash**:
    - Deploy on platforms like Heroku or Dash Enterprise.
  - **Streamlit**:
    - Use Streamlit Sharing or deploy on a cloud platform.
- **Requirements**:
  - Package dependencies specified in `requirements.txt`.

---