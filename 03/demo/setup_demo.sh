#!/bin/sh
# Download the Lecture 03 demo files into a new folder, ~/03-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/03/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from. The course's tests set DEMO_BASE_URL to read local copies instead.
base_url="${DEMO_BASE_URL:-https://raw.githubusercontent.com/christopherseaman/datasci_217/main/03/demo}"

# mkdir without -p stops the script here if ~/03-demo already exists, so earlier work is never overwritten.
mkdir ~/03-demo
cd ~/03-demo

# One download per file; -o saves it under the name that follows.
curl -fsSL "$base_url/setup_demo.sh" -o setup_demo.sh
curl -fsSL "$base_url/demo1_cli_pipeline.sh" -o demo1_cli_pipeline.sh
curl -fsSL "$base_url/demo2_types_and_lists.py" -o demo2_types_and_lists.py
curl -fsSL "$base_url/demo2_numpy_performance.py" -o demo2_numpy_performance.py
curl -fsSL "$base_url/demo2_numpy_arrays.py" -o demo2_numpy_arrays.py
curl -fsSL "$base_url/demo3_bp_analysis.py" -o demo3_bp_analysis.py
curl -fsSL "$base_url/demo3_csv_summary.py" -o demo3_csv_summary.py
curl -fsSL "$base_url/encounters.csv" -o encounters.csv

echo "Made ~/03-demo with the Lecture 03 demo scripts and encounters.csv."
echo "Next: cd ~/03-demo"
