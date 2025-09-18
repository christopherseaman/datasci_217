# Data Cleaning Assignment: Population Dataset

## Overview

Please document your work on this assignment in `readme.md` (you'll have to create it).

In this assignment, you will work with a messy population dataset derived from the original "ddf--datapoints--population--by--income_groups--age--gender--year.csv" file. Your task is to identify the data quality issues present in the dataset and then implement appropriate cleaning techniques to resolve these issues.

You may use `example_readme.md` as a guide for some of what this could include. Feel free to add code snippets in your writeup, but still include your working data cleaning script in the repository.

## Submission Requirements

1. `readme.md`: Documentation of your process and findings
2. `clean_data.py`: Your data cleaning script
3. `cleaned_population_data.csv`: Your cleaned dataset

## Instructions

### Part 0: Creating the Messy Dataset


1. Ensure you have the original `ddf--datapoints--population--by--income_groups--age--gender--year.csv` file in your working directory.
2. Install packages needed by `dirty-data.py` (in a virtual environment of course). `pip install`
   - `pandas`
   - `numpy`
   - `argparse`
   - `tqdm`
3. Run the provided `dirty-data.py` script to create the messy dataset:
   ```
   python dirty-data.py
   ```
   This will generate `messy_population_data.csv`, which you'll use for the rest of the assignment.

### Part 1: Identifying Data Issues

1. Load the messy dataset ("messy_population_data.csv") using Python with pandas.
2. Perform an exploratory data analysis (EDA) to identify data quality issues.
3. Document each issue you discover in your `readme.md`, including:
   - Description of the issue
   - Column(s) affected
   - Example of the problematic data
   - Potential impact on analysis if left uncleaned

Use pandas functions like `info()`, `describe()`, `isnull().sum()`, `duplicated().sum()`, and `value_counts()` to aid your exploration.

### Part 2: Cleaning the Data

Write a Python script (`clean_data.py`) for cleaning the data and document the steps along the way.  The script `clean_data.py` should be a runnable Python script that takes messy_population_data.csv as input and outputs cleaned_population_data.csv

1. For each issue you identified, propose and implement a method to clean or correct the data.
2. Use appropriate pandas and numpy functions for cleaning.
3. Document each cleaning step with comments in your code.
4. Include error handling and logging where appropriate.
5. Document your cleaning process in your `readme.md`, including:
   - The technique used to address each issue
   - Justification for your approach
   - The impact of your cleaning on the dataset (e.g., number of rows affected/removed, changes in data distribution)
   - Any assumptions you made

### Part 3: Documenting Results

In your `readme.md`:

1. Describe your cleaned dataset and how it compares to the original dirty one.
2. Discuss any challenges faced and how you overcame them.
3. Reflect on what you learned from this process.
4. Suggest potential next steps or further improvements.

## Evaluation Criteria

- Thoroughness in identifying data issues
- Effectiveness and efficiency of cleaning methods
- Quality and clarity of documentation
- Code quality (readability, comments, error handling)
- Creativity in problem-solving
