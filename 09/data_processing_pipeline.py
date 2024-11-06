import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

# Create output directory if it doesn't exist
Path('output').mkdir(parents=True, exist_ok=True)

# Get CSV files using shell command
cmd = "find data/ -name '*.csv' -type f"
files = subprocess.check_output(cmd, shell=True).decode().split('\n')

# Process each file
dfs = []
for file in files:
    if not file: continue
    df = pd.read_csv(file)
    # Data cleaning
    df = df.dropna()
    dfs.append(df)

# Combine and analyze
combined = pd.concat(dfs)
summary = combined.groupby('category').agg({
    'value': ['mean', 'std']
})

# Visualize
plt.figure(figsize=(10, 6))
summary.plot(kind='bar')
plt.title('Analysis Results')
plt.tight_layout()
plt.savefig('output/analysis.png')
plt.close()

print("Data processing complete. Results saved in output/analysis.png")
