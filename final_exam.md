# DS-217 Final Exam: Health Data Analysis Workflow

## Instructions

- This exam consists of four interconnected questions that will guide you through a health data analysis project
- Submit your solutions as separate files with the names specified in each question
- You may only use tools and techniques covered in the course outline
- Each question builds upon the previous ones - complete them in order

## Question 1: Data Preparation with Command-Line Tools (20 points)

File name: `prepare_health_data.sh`

You are provided with a raw CSV file containing health data. Create a shell script that prepares this dataset using command-line text processing tools.

Your tasks:

1. Create text files categorizing key variables:
   - Create a file for geographic regions
   - Create a file for insurance types

2. Clean the raw data file:
   - Remove comment lines and headers
   - Remove lines with extra or missing data
   - Extract specific columns of interest

3. Generate a summary of the processed data:
   - Count the number of records
   - Display the first few lines of the processed data

Tips:

- Use `grep -v` to remove comment lines and headers
- Use `sed` to handle extra commas or incomplete data
  - `sed 's/,,*/,/g'` replaces multiple commas with a single comma
  - `sed '/^$/d'` removes empty lines
- Use `cut -d',' -f1,2,3,4,5` to extract specific columns
- Use `wc -l` to count records
- Use `head -n 5` to show first few lines
- Combine commands with pipes (`|`) for complex processing

## Question 2: Data Munging with Python (25 points)

File name: `data_munge.py`

Using the processed data from Question 1 and the category files you created:

1. Load the category files

2. Load the processed health data

3. Randomly assign categorical columns:
   - Assign regions to patients
   - Assign insurance types to patients

4. Create two outcome variables:
   - Generate a blood pressure variable with some systematic variation
   - Generate a patient cost variable with some systematic variation

Tips:

- Use `random.choice()` to assign categorical variables
- Create dictionaries to add systematic variation
  - Example: `region_bp_offset = {'Northeast': 5, 'Southeast': -3}`
- Use NumPy for random number generation
  - `np.random.normal(mean, std_dev, size)` creates semi-random variables
- Add base value + variation + random noise
  - `blood_pressure = 120 + region_offset + age_factor + random_noise`
- Use pandas for data manipulation
  - `.map()` to apply dictionary-based transformations
  - `.to_csv()` to save processed data

## Question 3: Statistical Analysis (25 points)

File name: `analyze.py`

Perform statistical analysis on the munged data from Question 2:

1. Conduct a chi-square test to examine relationships between categorical variables

2. Perform a linear regression:
   - Predict an outcome variable based on other variables
   - Assess the model's statistical significance

3. Explore relationships between variables:
   - Look for statistically significant associations
   - Interpret your findings

Tips:

- Use `scipy.stats.chi2_contingency()` for categorical analysis
  - Create a contingency table with `pd.crosstab()`
- Use scikit-learn for linear regression
  - `LinearRegression().fit(X, y)` to create model
  - `.score()` to get R-squared
- Use statsmodels for detailed regression analysis
  - `sm.OLS()` provides comprehensive statistical output
- Extract and interpret key statistics:
  - Coefficients
  - P-values
  - R-squared
- Consider multiple features for prediction

## Question 4: Data Visualization (30 points)

File name: `visualize.py`

Create visualizations that help understand the data and analysis from previous questions:

1. Generate a pair plot showing relationships between key variables

2. Create histograms that show the distribution of an outcome variable across different categories

3. Produce a box plot comparing an outcome variable across different categorical groups

4. Create a scatter plot exploring the relationship between two continuous variables

Tips:

- Use seaborn for advanced visualizations
  - `sns.pairplot()` for multi-variable relationships
  - `sns.histplot()` with `hue` parameter for categorical distributions
  - `sns.boxplot()` to compare distributions
  - `sns.scatterplot()` to show correlations
- Use `plt.subplot()` to create multi-panel figures
- Add meaningful titles and labels
- Use `hue` parameter to add categorical information
- Save figures with `plt.savefig()`

### Bonus Points (10 points)

- Add error handling and input validation
- Create more advanced or interactive visualizations
- Implement additional statistical tests
- Add command-line argument parsing
