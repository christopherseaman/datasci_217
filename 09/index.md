# Lecture 9: Practical Python & Command Line Automation

## Overview

In this final lecture, we'll bring together Python and command line tools to solve practical, everyday problems. We'll see how concepts from previous lectures combine to create useful automation solutions.

## REDCap Access

1. [Request REDCap access](https://ucsf.service-now.com/ucsfit?id=ucsf_sc_cat_item&sys_id=42b99b6cdb040c10f3eddff648961988&sysparm_category=40c0305b7b92d000e2dc8180984d4d9f) for upcoming guest lecture
2. Email me with "REDCap: YOUR_REDCAP_USERNAME" in the subject (replace with your REDCap user name)
3. Someone will grant you access to an example REDCap project
4. You will need to request API access to the project

## References

- [*Python for Data Analysis*, Wes McKinney](https://wesmckinney.com/book/) (basis for last few lectures)
- [*Automate the Boring Stuff with Python*, Al Sweigart](https://automatetheboringstuff.com/)
- [The Missing Semester](https://missing.csail.mit.edu/) (command line, git, data wrangling)
- [*The Linux Command Line*, William Shotts](https://linuxcommand.org/tlcl.php)
- [The Shell Scripting Tutorial](https://www.shellscript.sh/)

## Revisiting the Command Line

<!-- TODO: Include 1-command_line.md -->

## Everyday Python

<!-- TODO: Include 2-everyday_python.md -->

## Bringing It All Together

### Example 1: Data Processing Pipeline

```python
# Combining concepts from previous lectures:
# - File operations (Lecture 1)
# - Data structures (Lecture 3)
# - Pandas for data (Lectures 5-6)
# - Visualization (Lecture 7)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

# Get CSV files using shell command
cmd = "find data/ -name '*.csv' -type f"
files = subprocess.check_output(cmd, shell=True).decode().split('\n')

# Process each file
dfs = []
for file in files:
    if not file: continue
    df = pd.read_csv(file)
    # Data cleaning (Lecture 6)
    df = df.dropna()
    dfs.append(df)

# Combine and analyze
combined = pd.concat(dfs)
summary = combined.groupby('category').agg({
    'value': ['mean', 'std']
})

# Visualize (Lecture 7)
plt.figure(figsize=(10, 6))
summary.plot(kind='bar')
plt.title('Analysis Results')
plt.savefig('output/analysis.png')
```

### Example 2: Automated Report Generation

```python
# Using:
# - File handling (Lecture 1)
# - Functions (Lecture 4)
# - Data visualization (Lecture 7)
# - Basic ML for predictions (Lecture 8)

from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def generate_report():
    # Load and process data
    data = pd.read_csv('sales.csv')
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    data.plot(kind='line')
    plt.savefig('report_figures/trends.png')
    
    # Make predictions
    model = LinearRegression()
    # ... model training code ...
    
    # Generate HTML report
    html = f"""
    <html>
        <body>
            <h1>Daily Report - {datetime.now().strftime('%Y-%m-%d')}</h1>
            <img src="report_figures/trends.png">
            <h2>Predictions</h2>
            <p>Tomorrow's forecast: {prediction:.2f}</p>
        </body>
    </html>
    """
    
    with open('report.html', 'w') as f:
        f.write(html)

# Can be automated with cron (Lecture 3)
generate_report()
```

### Example 3: ...?

## Key Takeaways

1. Python + CLI = Powerful automation
2. Real solutions often combine multiple concepts
3. Automation saves time and reduces errors
4. Start simple, then expand functionality
5. Documentation and maintainability are crucial

## Resources for Further Learning

- Automate the Boring Stuff with Python
- Python Documentation
- Shell Scripting Guide
- Real Python Tutorials

## Practice Exercise

Create a script that:

1. Monitors a directory for new files
2. Processes files based on type
3. Generates a summary report
4. Sends notification on completion

This brings together:

- File operations (Lecture 1)
- Shell commands (Lecture 1)
- Python functions (Lecture 4)
- Data processing (Lectures 5-6)
- Visualization (Lecture 7)
