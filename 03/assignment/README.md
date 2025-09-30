# Assignment 03: NumPy Arrays & Virtual Environments - Health Sensor Data Analysis

## Requirements

This assignment has **three progressive parts** that build upon each other. Each part focuses on essential data science skills.

### Provided Files

- `generate_health_data.py` - Script to generate sample health sensor data
- `analyze_health_data.py` - Scaffold script with TODO comments for you to complete
- `requirements.txt` - Python dependencies (NumPy)
- `TIPS.md` - Troubleshooting guide for common issues
- `.github/test/test_assignment.py` - Automated tests for grading
- `expected_outputs/` - Reference examples of expected output files

## Assignment

### Part 0: Run the Script to Prepare Example Data

**Setup**:
First, generate the sample data by running:

```bash
python generate_health_data.py
```

This creates `health_data.csv` with **50,000 rows** containing:

**CSV Schema:**

```
patient_id,timestamp,heart_rate,blood_pressure_systolic,blood_pressure_diastolic,temperature,glucose_level,sensor_id
```

**Sample Data:**

```csv
patient_id,timestamp,heart_rate,blood_pressure_systolic,blood_pressure_diastolic,temperature,glucose_level,sensor_id
P00001,2024-01-15T08:23:45,72,120,80,98.6,95,S001
P00002,2024-01-15T08:24:12,68,115,75,98.4,88,S002
P00003,2024-01-15T08:25:33,75,135,85,98.9,102,S003
...
```

### Part 1: CLI Data Tools (7 points)

**Objective**: Use command-line tools to analyze a large CSV file containing health sensor data.

**Tasks**:

You must complete these tasks using command-line tools and save output to files in the `output/` directory.

**Task 1.1: Count Unique Patients** (1.5 points)

```bash
# Extract patient IDs from the first column and count unique values
# Expected tools: cut, sort, uniq, wc
# Output format: Single number (e.g., "10000")
# Save to: output/part1_patient_count.txt
```

**Task 1.2: High Blood Pressure Analysis** (2 points)

```bash
# Find and count readings where systolic BP > 130
# Expected tools: grep or awk, wc
# Output format: Single number (e.g., "8543")
# Save to: output/part1_high_bp_count.txt
```

**Task 1.3: Average Temperature** (2 points)

```bash
# Calculate average temperature across all readings
# Expected tools: cut, awk, tail
# Output format: Single number with 2 decimal places (e.g., "98.25")
# Save to: output/part1_avg_temp.txt
```

**Task 1.4: Glucose Statistics** (1.5 points)

```bash
# Find the top 5 highest glucose readings
# Expected tools: cut, sort, head
# Output format: Five numbers, one per line, sorted descending
# Save to: output/part1_glucose_stats.txt
```

**Hints**:

- Use `cut -d',' -f1` to extract the first column
- Use `tail -n +2` to skip the header row
- Use `sort | uniq` to find unique values
- Use `awk -F','` to process CSV with awk
- Chain commands with pipes (`|`) to create data processing pipelines

### Part 2: Virtual Environment Setup (5 points)

**Objective**: Set up a Python virtual environment using `venv` and install NumPy.

**Tasks**:

1. **Create a virtual environment** named `.venv` using Python's venv module
2. **Activate the virtual environment**
3. **Install NumPy** from requirements.txt
4. **Verify NumPy** is installed and accessible

**Instructions**:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Mac/Linux/WSL)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy; print(numpy.__version__)"

# Deactivate when done
deactivate
```

**What to Submit**:

- NOTHING!
- Successful completion of Part 3 (which requires NumPy) shows you've managed packages

**Why This Matters**:
Virtual environments ensure your project dependencies are isolated from other Python projects, making your analysis reproducible and avoiding version conflicts.

### Part 3: NumPy Data Analysis (8 points)

**Objective**: Complete the scaffold Python script that uses NumPy to analyze health sensor data.

**Tasks**:

The provided file `analyze_health_data.py` contains function signatures with TODO comments. It is a scaffold for you to build upon. You must implement the following functions:

**Function 1: `load_data(filename)`** (1.5 points)

The code for loading data with `np.genfromtxt()` is **provided for you** because this function is not covered in the lecture. You can use this function as-is.

```python
def load_data(filename):
    """Load CSV data using NumPy.
    
    Args:
        filename: Path to CSV file
        
    Returns:
        NumPy structured array with all columns
    """
    # This code is provided (np.genfromtxt not covered in lecture)
    dtype = [('patient_id', 'U10'), ('timestamp', 'U20'),
             ('heart_rate', 'i4'), ('blood_pressure_systolic', 'i4'),
             ('blood_pressure_diastolic', 'i4'), ('temperature', 'f4'),
             ('glucose_level', 'i4'), ('sensor_id', 'U10')]
    
    data = np.genfromtxt(filename, delimiter=',', dtype=dtype, skip_header=1)
    return data
```

**Function 2: `calculate_statistics(data)`** (2 points)

```python
def calculate_statistics(data):
    """Calculate basic statistics for numeric columns.
    
    Args:
        data: NumPy structured array
        
    Returns:
        Dictionary with statistics
        
    TODO: Calculate and return:
    - Average heart rate (use .mean())
    - Average systolic BP (use .mean())
    - Average glucose level (use .mean())
    - Return as dictionary
    - Format values with f-strings using .1f
    """
```

**Function 3: `find_abnormal_readings(data)`** (2 points)

```python
def find_abnormal_readings(data):
    """Find readings with abnormal values.
    
    Args:
        data: NumPy structured array
        
    Returns:
        Dictionary with counts
        
    TODO: Count readings where:
    - Heart rate > 90 (use boolean indexing)
    - Systolic BP > 130 (use boolean indexing)
    - Glucose > 110 (use boolean indexing)
    - Return dictionary with counts
    """
```

**Function 4: `generate_report(stats, abnormal)`** (1.5 points)

```python
def generate_report(stats, abnormal):
    """Generate formatted analysis report.
    
    Args:
        stats: Dictionary of statistics
        abnormal: Dictionary of abnormal counts
        
    Returns:
        Formatted string report
        
    TODO: Create a formatted report string using f-strings
    - Include all statistics with proper formatting
    - Use .1f for decimal numbers
    - Make it readable and well-formatted
    """
```

**Function 5: `save_report(report, filename)`** (0.5 points)

```python
def save_report(report, filename):
    """Save report to file.
    
    Args:
        report: Report string
        filename: Output filename
        
    TODO: Write the report to a file
    """
```

**Function 6: `main()`** (0.5 points)

```python
def main():
    """Main execution function.
    
    TODO: Orchestrate the analysis:
    1. Load the data from 'health_data.csv'
    2. Calculate statistics
    3. Find abnormal readings
    4. Generate report
    5. Save to 'analysis_report.txt'
    6. Print success message
    """
```

**Expected Output Format**:

Your completed script should generate `output/analysis_report.txt` with content similar to:

```text
Health Sensor Data Analysis Report
==================================

Dataset Summary:
- Total readings: 50000

Average Measurements:
- Heart Rate: 79.8 bpm
- Systolic BP: 125.2 mmHg
- Glucose Level: 99.5 mg/dL

Abnormal Readings:
- High Heart Rate (>90): 4523 readings
- High Blood Pressure (>130): 8234 readings
- High Glucose (>110): 9876 readings
```

**NumPy Operations Expected** (from lecture):

- ✅ Accessing array columns/fields
- ✅ Statistical operations (`.mean()`)
- ✅ Boolean indexing for filtering (`data[data['column'] > threshold]`)
- ✅ Counting with `.sum()` or `len()` on filtered arrays
- ✅ F-string formatting with `.1f`

**Note**: The `np.genfromtxt()` code is provided for you since it's not covered in the lecture.

**Running Your Script**:

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate  # Windows

# Run the analysis
python3 analyze_health_data.py

# Check the output
cat output/analysis_report.txt
```

## Common Issues to Avoid

1. **CLI Issues**:
   - Don't forget to skip the header row with `tail -n +2`
   - Remember to use comma delimiter with `-d','` or `-F','`
   - Pipe commands together to create complete pipelines

2. **Virtual Environment Issues**:
   - Always activate the environment before running Python scripts
   - Use `which python` to verify you're using the venv Python
   - Deactivate when switching to other projects

3. **NumPy Issues**:
   - Use `skip_header=1` in `np.genfromtxt()` to skip the CSV header
   - Access structured array fields with bracket notation: `data['column_name']`
   - Boolean indexing returns a filtered array: `data[data['heart_rate'] > 90]`
   - Use `.sum()` on boolean arrays to count True values
