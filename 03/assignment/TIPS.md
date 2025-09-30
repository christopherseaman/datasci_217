# Assignment 03: Tips and Troubleshooting Guide

## Quick Start Checklist

1. Generate the health data: `python generate_health_data.py`
2. Complete Part 1 CLI tasks and save output to files
3. Create virtual environment: `python -m venv .venv`
4. Activate virtual environment and install NumPy
5. Complete TODOs in `analyze_health_data.py`
6. Test your implementation before submitting

## Common Issues and Solutions

### Part 1: CLI Data Tools

**Issue: Commands not working on CSV data**

- Solution: Remember to skip the header row using `tail -n +2`
- Example: `cut -d',' -f1 health_data.csv | tail -n +2 | sort | uniq | wc -l`

**Issue: Wrong delimiter in cut or awk**

- Solution: CSV files use commas, so use `-d','` for cut or `-F','` for awk
- Example: `cut -d',' -f3 health_data.csv`

**Issue: Output includes header or extra text**

- Solution: Always skip the header with `tail -n +2` after extracting columns
- Example: `cut -d',' -f6 health_data.csv | tail -n +2`

**Issue: "No such file or directory" error**

- Solution: Make sure you've run `python generate_health_data.py` first
- Check you're in the assignment directory

**Issue: Decimal formatting not matching expected**

- Solution: Use `awk` with `printf` for precise decimal formatting
- Example: `awk '{sum+=$1; count++} END {printf "%.2f\n", sum/count}'`

**Hint: Counting with grep or awk**

- To count lines matching a pattern: `grep "pattern" file | wc -l > output/filename.txt`
- Or use awk: `awk -F',' '$3 > 130' file | wc -l > output/filename.txt`

**Hint: Sorting numbers**

- Use `sort -n` for numeric sort
- Use `sort -nr` for reverse numeric sort
- Use `sort -t',' -k2 -nr` to sort CSV by column 2 in reverse

### Part 2: Virtual Environment

**Issue: "No module named venv"**

- Solution: On some systems, install python3-venv: `sudo apt install python3-venv`
- Or use: `python3 -m venv .venv`

**Issue: Virtual environment not activating**

- Mac/Linux: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`
- Git Bash on Windows: `source .venv/Scripts/activate`

**Issue: "command not found: activate"**

- Solution: Make sure you're using `source` on Mac/Linux
- Full path: `source .venv/bin/activate`

**Issue: Not sure if virtual environment is active**

- Solution: Your prompt should show `(.venv)` at the beginning
- Or check: `which python` (should point to `.venv/bin/python`)

**Issue: pip install fails**

- Solution: Make sure virtual environment is activated first, e.g.,  `source .venv/bin/activate`
- Try upgrading pip: `pip install --upgrade pip`
- Then: `pip install -r requirements.txt`

**Issue: NumPy version conflicts**

- Solution: The requirements specify `numpy>=1.24.0` which should work
- If issues persist, try: `pip install numpy==1.24.0`

### Part 3: NumPy Data Analysis

**Issue: "No module named numpy"**

- Solution: Activate your virtual environment first
- Then verify: `python -c "import numpy; print(numpy.__version__)"`

**Issue: np.genfromtxt() not loading data correctly**

- Solution: Make sure you're using these parameters:

  ```python
  dtype = [('patient_id', 'U10'), ('timestamp', 'U20'), 
           ('heart_rate', 'i4'), ('blood_pressure_systolic', 'i4'),
           ('blood_pressure_diastolic', 'i4'), ('temperature', 'f4'),
           ('glucose_level', 'i4'), ('sensor_id', 'U10')]
  data = np.genfromtxt('health_data.csv', delimiter=',', dtype=dtype, skip_header=1)
  ```

**Issue: Can't access columns in structured array**

- Solution: Use bracket notation with column name: `data['heart_rate']`
- Not: `data.heart_rate` (this won't work)

**Issue: Boolean indexing not working**

- Solution: Use parentheses for complex conditions
- Example: `high_hr = data[data['heart_rate'] > 90]`
- To count: `count = len(data[data['heart_rate'] > 90])`
- Or: `count = (data['heart_rate'] > 90).sum()`

**Issue: .mean() not working**

- Solution: Make sure you're calling it on a column, not the whole array
- Correct: `data['heart_rate'].mean()`
- Not: `data.mean()` (won't work on structured arrays)

**Issue: F-string formatting not showing one decimal place**

- Solution: Use `.1f` format specifier
- Example: `f"Average: {value:.1f}"`
- For two decimals: `f"{value:.2f}"`

**Issue: "FileNotFoundError" when loading data**

- Solution: Make sure `health_data.csv` exists in the same directory
- Check your working directory: run script from assignment folder
- Use relative path: `'health_data.csv'` not `'./health_data.csv'`

**Issue: Report file not being created**

- Solution: Make sure you're using 'w' mode: `open(filename, 'w')`
- Check the file path is correct
- Verify the save_report function is being called in main()

## Testing Your Work

### Test Part 1 CLI Outputs

```bash
# Check files exist
ls -la output/part1_*.txt

# View contents
cat output/part1_patient_count.txt
cat output/part1_high_bp_count.txt
cat output/part1_avg_temp.txt
cat output/part1_glucose_stats.txt

# Verify formats
# patient_count: should be a single number (~10000)
# high_bp_count: should be a single number
# avg_temp: should be a decimal with 2 places
# glucose_stats: should be 5 numbers, one per line
```

### Test Part 2 Virtual Environment

```bash
# Activate environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Check Python location
which python  # Should show .venv/bin/python

# Check NumPy installation
python -c "import numpy; print(numpy.__version__)"

# Should print version number like "1.24.0" or higher
```

### Test Part 3 NumPy Analysis

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Run the analysis script
python analyze_health_data.py

# Check output was created
ls -la output/analysis_report.txt

# View the report
cat output/analysis_report.txt

# Should contain:
# - Dataset summary with total readings
# - Average measurements (heart rate, BP, glucose)
# - Abnormal readings counts
```

### Run Automated Tests (Advanced)

```bash
# Install pytest in your virtual environment
pip install pytest

# Run all tests
pytest .github/test/test_assignment.py -v

# Run specific test
pytest .github/test/test_assignment.py::TestPart1CLI -v
```

## Command-Line Pipeline Examples

### Example: Count unique values

```bash
# Count unique patients (save to output/)
cut -d',' -f1 health_data.csv | tail -n +2 | sort | uniq | wc -l > output/part1_patient_count.txt
```

### Example: Filter and count

```bash
# Count high blood pressure readings (systolic > 130, save to output/)
awk -F',' 'NR>1 && $4 > 130' health_data.csv | wc -l > output/part1_high_bp_count.txt
```

### Example: Calculate average

```bash
# Calculate average temperature (column 6, save to output/)
cut -d',' -f6 health_data.csv | tail -n +2 | awk '{sum+=$1; count++} END {printf "%.2f\n", sum/count}' > output/part1_avg_temp.txt
```

### Example: Sort and extract top values

```bash
# Get top 5 glucose readings (column 7, save to output/)
cut -d',' -f7 health_data.csv | tail -n +2 | sort -nr | head -5 > output/part1_glucose_stats.txt
```

## NumPy Quick Reference

### Loading Data

```python
# Define dtype for structured array
dtype = [('patient_id', 'U10'), ('timestamp', 'U20'), 
         ('heart_rate', 'i4'), ...]

# Load CSV with NumPy
data = np.genfromtxt('health_data.csv', delimiter=',', dtype=dtype, skip_header=1)
```

### Accessing Columns

```python
# Access a column
heart_rates = data['heart_rate']

# Calculate statistics
avg_hr = data['heart_rate'].mean()
max_hr = data['heart_rate'].max()
min_hr = data['heart_rate'].min()
```

### Boolean Indexing

```python
# Filter data
high_hr = data[data['heart_rate'] > 90]

# Count matches
count = len(data[data['heart_rate'] > 90])
# Or
count = (data['heart_rate'] > 90).sum()
```

### Formatting Output

```python
# F-strings with decimal formatting
avg = 79.8234
print(f"Average: {avg:.1f}")  # Output: "Average: 79.8"
print(f"Average: {avg:.2f}")  # Output: "Average: 79.82"
```

## Emergency Scaffolds (Only if Really Stuck)

### CLI Command Starter for Task 1.1

```bash
# Count unique patients
cut -d',' -f1 health_data.csv | tail -n +2 | sort | uniq | wc -l > output/part1_patient_count.txt
```

### CLI Command Starter for Task 1.2

```bash
# Count high BP readings
awk -F',' 'NR>1 && $4 > 130' health_data.csv | wc -l > output/part1_high_bp_count.txt
```

### NumPy load_data() Starter

```python
def load_data(filename):
    dtype = [('patient_id', 'U10'), ('timestamp', 'U20'), 
             ('heart_rate', 'i4'), ('blood_pressure_systolic', 'i4'),
             ('blood_pressure_diastolic', 'i4'), ('temperature', 'f4'),
             ('glucose_level', 'i4'), ('sensor_id', 'U10')]
    data = np.genfromtxt(filename, delimiter=',', dtype=dtype, skip_header=1)
    return data
```

### NumPy calculate_statistics() Starter

```python
def calculate_statistics(data):
    stats = {
        'avg_heart_rate': data['heart_rate'].mean(),
        'avg_systolic_bp': data['blood_pressure_systolic'].mean(),
        'avg_glucose': data['glucose_level'].mean()
    }
    return stats
```

**Note**: These scaffolds are starting points. You still need to complete the remaining functions and understand what each part does!
