---
marp: true
theme: sqrl
paginate: true
class: invert
---

# Lecture 07: Data Visualization for Health Data Science

## Section 3: Advanced Visualization Techniques

---

# Advanced Visualization Techniques in Health Data Science

## Introduction

- **Plotnine**
  - Python implementation of the **Grammar of Graphics**
  - Inspired by R's **ggplot2**
- **Command-Line Visualization**
  - Tools for visualizing data directly from the command line
  - Examples: **Mermaid.js**, **spark**
- **Interactive and BI Visualizations**
  - Tools for building interactive dashboards and applications
  - Examples: **Plotly Dash**, **Streamlit**

---

# Plotnine: Grammar of Graphics in Python

## Understanding the Grammar of Graphics

- **Theory**: A structured approach to data visualization that breaks down graphs into semantic components
  - **Data**: The dataset being visualized
  - **Aesthetics (aes)**: Mappings between data and visual properties (e.g., x, y, color)
  - **Geometries (geoms)**: Visual elements that represent data (e.g., points, lines)
  - **Facets**: Subsets of data shown in multiple plots
  - **Statistical Transformations (stats)**: Summarizing data (e.g., binning, smoothing)
  - **Scales**: Control mapping from data space to aesthetic space
  - **Coordinate Systems (coords)**: The space in which the data is represented (e.g., Cartesian, polar)

---

## Plotnine vs. ggplot2

- **Similar Syntax and Concepts**
  - Plotnine mirrors ggplot2's structure and functions
  - Beneficial for students familiar with R
- **Example Comparison**

  - **ggplot2 (R)**:
    ```R
    ggplot(data, aes(x, y)) + geom_point()
    ```
  - **Plotnine (Python)**:
    ```python
    (ggplot(data, aes('x', 'y')) + geom_point())
    ```

---

## Important Components in Plotnine

### 1. `ggplot()`: Initialize a Plot

- **Explanation**: Creates a new plot object with data and aesthetic mappings
- **Structure**:
  ```python
  ggplot(data=DataFrame, mapping=aes('x_var', 'y_var'))
  ```
  - **Required Arguments**:
    - `data`: DataFrame containing the data
    - `mapping`: Aesthetic mappings created with `aes()`

---

### Code Example: Basic Scatter Plot

```python
from plotnine import ggplot, aes, geom_point, ggtitle
import pandas as pd

# Sample data
df = pd.DataFrame({
    'BMI': [22, 25, 28, 24, 27],
    'BloodPressure': [120, 130, 125, 118, 135]
})

# Create plot
plot = (ggplot(df, aes(x='BMI', y='BloodPressure'))
        + geom_point(color='blue')
        + ggtitle('Blood Pressure vs BMI'))

print(plot)
```

---

### Code Explanation

- **Importing Libraries**:
  ```python
  from plotnine import ggplot, aes, geom_point, ggtitle
  import pandas as pd
  ```
  - Imports necessary components from Plotnine and pandas.
- **Defining Data**:
  - Creates a DataFrame `df` with 'BMI' and 'BloodPressure' columns.
- **Creating the Plot**:
  ```python
  plot = (ggplot(df, aes(x='BMI', y='BloodPressure'))
          + geom_point(color='blue')
          + ggtitle('Blood Pressure vs BMI'))
  ```
  - Initializes the plot with data and aesthetic mappings.
  - Uses `geom_point()` to add scatter plot points.
  - Adds a title with `ggtitle()`.
- **Displaying the Plot**:
  - The `print(plot)` statement renders the plot.

---

### Output Description

- A scatter plot showing Blood Pressure versus BMI.
- Each point represents a data entry from the DataFrame.
- The plot includes axis labels and a title.

---

### 2. Layering Geometries and Aesthetics

- **Adding Layers**: Use the `+` operator to add layers.
- **Example**: Adding a regression line
  ```python
  from plotnine import geom_smooth

  plot = (ggplot(df, aes(x='BMI', y='BloodPressure'))
          + geom_point(color='blue')
          + geom_smooth(method='lm')
          + ggtitle('Blood Pressure vs BMI with Regression Line'))
  ```

---

### Code Explanation

- **Adding `geom_smooth()`**:
  - Adds a smooth line (here, a linear regression line) to the plot.
  - `method='lm'` specifies a linear model.
- **Enhanced Visualization**:
  - Helps in identifying trends or relationships in the data.

---

### Output Description

- The scatter plot now includes a regression line.
- Visualizes the trend of Blood Pressure increasing with BMI.

---

### 3. Facetting: Creating Multiple Plots

- **Explanation**: Splits the data into subsets according to a variable and creates multiple plots.
- **Structure**:
  ```python
  + facet_wrap('~ variable')
  ```
- **Example**:
  ```python
  from plotnine import facet_wrap

  plot = (ggplot(df, aes(x='BMI', y='BloodPressure'))
          + geom_point(color='blue')
          + facet_wrap('~ AgeGroup')
          + ggtitle('Blood Pressure vs BMI by Age Group'))
  ```

---

### Code Explanation

- **Assuming `df` Includes 'AgeGroup'**:
  - The DataFrame has a categorical variable 'AgeGroup'.
- **Using `facet_wrap()`**:
  - Creates separate plots for each age group.
- **Purpose**:
  - Allows comparison across different subsets of the data.

---

### Output Description

- Multiple scatter plots, each corresponding to an age group.
- Facilitates analysis of patterns within subgroups.

---

# Command-Line Visualization

## Mermaid.js

### Introduction

- **Mermaid.js**: A JavaScript-based tool for generating diagrams and flowcharts from text definitions.
- **Advantages**:
  - Quick creation of diagrams without graphic design tools.
  - Integration with markdown documents and presentations.

---

### Use Cases

- **Documentation**: Embed diagrams in technical documents or code repositories.
- **Workflow Visualization**: Illustrate processes, algorithms, or decision trees.
- **Education**: Visual aid for teaching concepts.

---

### Creating Diagrams with Mermaid.js

- **Structure**:
  ```markdown
  ```mermaid
  [Diagram Definition]
  ```
  ```
- **Example**:

  ```mermaid
  graph LR
    A[Start] --> B{Is the patient symptomatic?}
    B -->|Yes| C[Conduct Tests]
    B -->|No| D[Monitor Patient]
    C --> E[Treatment Plan]
    D --> E
    E --> F[Follow-up]
  ```

---

### Diagram Explanation

- **Nodes**:
  - `A`, `B`, `C`, `D`, `E`, `F` represent steps or decisions.
- **Edges**:
  - Arrows define the flow between nodes.
  - Labels like `|Yes|` and `|No|` represent decision outcomes.

---

### Integrating Mermaid.js into Documents

- **Markdown Files**:
  - Supported in many markdown editors and viewers.
- **Presentations**:
  - Tools like **Marp** allow embedding Mermaid diagrams in slides.
- **Version Control**:
  - Diagrams are text-based, facilitating collaboration and version tracking.

---

## Step-by-Step Guide

1. **Install Mermaid Support**:
   - Depending on the platform (e.g., VSCode extension, Marp plugin).
2. **Write Diagram Definition**:
   - Use Mermaid's syntax within markdown code blocks.
3. **Render the Diagram**:
   - The tool or platform will process and display the diagram.

---

# Interactive and BI Visualizations

## Plotly Dash vs. Streamlit

### Overview

- **Plotly Dash**:
  - Focused on creating complex, customizable applications.
  - Uses Flask and React.js under the hood.
- **Streamlit**:
  - Designed for rapid development and simplicity.
  - Emphasizes minimal code to produce apps.

---

### Comparison

| Feature            | Plotly Dash                           | Streamlit                        |
|--------------------|---------------------------------------|----------------------------------|
| **Ease of Use**    | Moderate (requires understanding callbacks) | Easy (simple script-based apps)  |
| **Customization**  | High (extensive layout and styling options) | Moderate                         |
| **Best For**       | Complex dashboards, enterprise apps   | Quick prototypes, data exploration |

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
df = pd.read_csv('health_data.csv')

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
df = pd.read_csv('health_data.csv')

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

## Applications in Health Data Science

- **Interactive Data Exploration**:
  - Enable users to explore datasets dynamically.
- **Patient Data Dashboards**:
  - Visualize patient metrics for clinical decision support.
- **Educational Tools**:
  - Create apps to teach concepts using real data.

---

# Summary

- **Advanced Visualization Tools** offer powerful ways to represent and interact with health data.
- **Plotnine** brings the grammar of graphics to Python, allowing for elegant and complex static visualizations.
- **Command-Line Tools** like **Mermaid.js** enable quick diagram creation within documentation.
- **Interactive Frameworks** like **Plotly Dash** and **Streamlit** facilitate the development of data apps for deeper insights.

---

# Further Resources

- **Plotnine Documentation**: [plotnine.readthedocs.io](https://plotnine.readthedocs.io/)
- **Mermaid.js Documentation**: [mermaid-js.github.io](https://mermaid-js.github.io/)
- **Plotly Dash User Guide**: [dash.plotly.com](https://dash.plotly.com/)
- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io/)
- **Deployment Guides**:
  - Dash: [Deployment](https://dash.plotly.com/deployment)
  - Streamlit: [Sharing Apps](https://streamlit.io/sharing)

---
