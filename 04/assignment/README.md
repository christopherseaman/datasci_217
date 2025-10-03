# Assignment 4: Pandas Basics and Data Exploration

## Overview

In this assignment, you'll practice fundamental pandas operations including data loading, exploration, selection, and basic manipulation. You'll work with a real dataset to gain hands-on experience with pandas DataFrames and Series.

## Learning Objectives

By completing this assignment, you will:
- Load and explore datasets using pandas
- Perform data selection and filtering operations
- Handle missing data appropriately
- Apply basic data manipulation techniques
- Use Jupyter notebooks effectively for data analysis

## Dataset

You'll work with a dataset containing information about employees at a fictional company. The dataset includes:
- Employee names and IDs
- Demographics (age, gender, city)
- Job information (department, position, salary)
- Performance metrics

## Assignment Tasks

### Task 1: Data Loading and Initial Exploration (20 points)

1. Load the dataset from the provided CSV file
2. Display the first 5 rows of the data
3. Show the shape of the dataset (rows and columns)
4. Display the data types of each column
5. Check for missing values in the dataset

### Task 2: Data Selection and Filtering (25 points)

1. Select all employees from the 'Sales' department
2. Find employees with salary greater than $60,000
3. Get employees aged between 25 and 35 (inclusive)
4. Select only the 'Name', 'Age', and 'Salary' columns for all employees
5. Find the top 5 highest-paid employees

### Task 3: Data Analysis (25 points)

1. Calculate the average salary by department
2. Find the department with the highest average salary
3. Calculate the age distribution (min, max, mean, median)
4. Count the number of employees in each city
5. Find the correlation between age and salary

### Task 4: Data Cleaning (20 points)

1. Handle any missing values in the dataset appropriately
2. Remove any duplicate entries
3. Clean the 'Name' column (remove extra spaces, standardize format)
4. Convert the 'Salary' column to numeric type if needed
5. Create a new column 'Salary_Category' based on salary ranges

### Task 5: Data Export and Summary (10 points)

1. Export the cleaned dataset to a new CSV file
2. Create a summary report of your findings
3. Include at least 3 interesting insights about the data

## Requirements

- Use Jupyter notebook for this assignment
- Include markdown cells to explain your approach
- Use appropriate pandas methods for each task
- Handle errors gracefully
- Comment your code appropriately

## Submission

Submit your completed Jupyter notebook (.ipynb file) with:
- All code cells executed
- Clear explanations in markdown cells
- Results and visualizations included
- Summary of findings

## Grading Criteria

- **Code Quality (30%)**: Clean, readable, and well-commented code
- **Correctness (40%)**: Accurate implementation of pandas operations
- **Documentation (20%)**: Clear explanations and markdown cells
- **Insights (10%)**: Meaningful analysis and conclusions

## Tips

- Start with simple operations and build complexity gradually
- Use the pandas documentation for reference
- Test your code with small samples first
- Don't forget to handle edge cases (missing data, etc.)
- Use appropriate data types for better performance

## Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

Good luck with your assignment!