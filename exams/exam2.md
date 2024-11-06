# DS-217 Final Exam: Multiple Sclerosis Analysis

## Instructions

- This exam consists of four interconnected questions analyzing MS patient data
- Submit your solutions as separate files with the names specified in each question
- Each question builds upon the previous ones - complete them in order
- If you get stuck, reach out to the TA for assistance. 

## Data Validation Checkpoints

After each step, verify your results:

- All questions: Summarize your efforts and results in `readme.md`
- Q1: Cleaned CSV should be named `ms_data.csv` and have no empty lines, comments, or extra commas
- Q2: All patients should have consistent insurance types across visits
- Q3: Check for outliers before running statistical tests
- Q4: Verify plot readability by viewing at different zoom levels

## Question 1: Data Preparation with Command-Line Tools (20 points)

File name: `prepare.sh`

You are provided with a raw CSV file containing longitudinal walking speed measurements. Create a shell script `prepare.sh` that performs data cleaning using command line tools.

Your tasks:

1. Run the script `generate_dirty_data.py` to create `ms_data_dirty.csv` which you will clean in the next steps (you may need to install the `numpy` library).

2. Clean the raw data file:
   - Remove comment lines
   - Remove empty lines
   - Remove extra commas
   - Extract essential columns: patient_id, visit_date, age, education_level, walking_speed
   - Save the file as `ms_data.csv`

3. Create a file, `insurance.lst` listing unique labels for a new variable, `insurance_type`, one per line (your choice of labels). Depending on how you want to import this file later with Python, you may or may not want to add a header row nameing the column as `insurance_type`.

4. Generate a summary of the processed data:
   - Count the total number of visits (rows, not including the header)
   - Display the first few records

Tips:

- Use `grep -v` to remove comment lines (starting with '#')
- Use `sed` to remove empty lines. Reminders about `sed`:
  - `sed -e 's/THIS/THAT/g'` to replace 'THIS' with 'THAT' everywhere in a line (that's what the `g` at the end means)
  - `^` - start of line
  - `$` - end of line
- Use `cut` to extract specific columns (use `-d` to specify delimiter, `-f` to specify columns)
- Use `wc -l` to count records
- Combine commands with pipes (`|`) for complex processing
- Expected output should have exactly 5 columns in this order: patient_id, visit_date, age, education_level, walking_speed
- Walking speeds should be between 2.0-8.0 feet/second
- Create at least 3 different insurance types: "Basic", "Premium", "Platinum" (you can choose your own names)

## Question 2: Data Analysis with Python (25 points)

File name: `analyze_visits.py`

Using the cleaned data and insurance category file from Question 1:

1. Load and structure the data:
   - Read the processed CSV file
   - Convert visit_date to datetime
   - Sort by patient_id and visit_date

2. Add insurance information:
   - Read insurance types from `insurance.lst`
   - Randomly assign (but keep consistent per patient_id)
   - Generate visit costs based on insurance type:
     - Different plans have different effects on cost
     - Add random variation

3. Calculate summary statistics:
   - Mean walking speed by education level
   - Mean costs by insurance type
   - Age effects on walking speed

Tips:

- Use pandas for data manipulation
  - `pd.read_csv()` to load data
  - `pd.to_datetime()` for dates
  - `.groupby()` for aggregations
- Expected data types:
  - `patient_id`: string
  - `visit_date`: datetime
  - `age`: float
  - `education_level`: string/category
  - `walking_speed`: float
- Handle missing data appropriately
- Consider seasonal variations in the data

## Question 3: Statistical Analysis (25 points)

File name: `stats_analysis.py`

Perform statistical analysis on both outcomes:

1. Analyze walking speed:
   - Multiple regression with education and age
   - Account for repeated measures
   - Test for significant trends

2. Analyze costs:
   - Simple analysis of insurance type effect
   - Box plots and basic statistics
   - Calculate effect sizes

3. Advanced analysis:
   - Education age interaction effects on walking speed
   - Control for relevant confounders
   - Report key statistics and p-values

Tips:

- Use scipy.stats for statistical tests
- Use statsmodels for regression analysis:
  - Report coefficients and confidence intervals

## Question 4: Data Visualization (30 points)

File name: `visualize.ipynb`

Create visualizations for both walking speed and cost analyses in a Jupyter notebook:

1. Walking speed analysis:
   - Scatter plot of age vs walking speed with regression line
   - Box plots by education level
   - Line plot showing education age interaction

2. Cost analysis:
   - Bar plot of mean costs by insurance type
   - Box plots showing cost distributions
   - Add error bars or confidence intervals

3. Combined visualizations:
   - Pair plot of key variables
   - Faceted plots by education/insurance
   - Time trends where relevant

Tips:

- Use seaborn for statistical visualizations:
  - `sns.lmplot()` for regression plots
  - `sns.boxplot()` for distributions
  - `sns.barplot()` for means with error bars
- Create clear, informative plots
- Add proper titles and labels
- Use appropriate color schemes
- Save high-quality figures

### Bonus Points (10 points)

- Implement advanced statistical methods
- Create interactive visualizations
- Analyze additional patterns
- Add command-line argument parsing
- 