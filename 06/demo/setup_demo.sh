#!/bin/sh
# Download the Lecture 06 demo notebooks, data, and environment files into a new folder, ~/06-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/06/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from: the course repository's 06/demo folder on GitHub.
base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/06/demo"

# mkdir without -p stops the script here if ~/06-demo already exists, so earlier work is never overwritten.
mkdir ~/06-demo
cd ~/06-demo

# One download per file; -o saves it under the name that follows.
curl -fsSL "$base_url/setup_demo.sh" -o setup_demo.sh
curl -fsSL "$base_url/.python-version" -o .python-version
curl -fsSL "$base_url/pyproject.toml" -o pyproject.toml
curl -fsSL "$base_url/uv.lock" -o uv.lock
curl -fsSL "$base_url/demo1_merge_operations.ipynb" -o demo1_merge_operations.ipynb
curl -fsSL "$base_url/demo2_pivot_melt.ipynb" -o demo2_pivot_melt.ipynb
curl -fsSL "$base_url/demo3_concat_combine.ipynb" -o demo3_concat_combine.ipynb

# The CSV files the notebooks read go in data/.
mkdir data
for file in demo1_central_a1c.csv demo1_labs.csv demo1_patients.csv demo1_poc_a1c.csv demo1_registry.csv demo1_targets.csv demo1_visits.csv demo2_bp_wide.csv demo2_device_week4.csv demo3_adt_census.csv demo3_audits.csv demo3_census.csv demo3_experience.csv demo3_feb_admissions.csv demo3_finance_census.csv demo3_jan_admissions.csv demo3_mar_admissions.csv demo3_q1_2025.csv demo3_q1_2026.csv demo3_staffing.csv demo3_units.csv; do
  curl -fsSL "$base_url/data/$file" -o "data/$file"
done

echo "Made ~/06-demo with the Lecture 06 demo notebooks and their environment files."
echo "Next: cd ~/06-demo"
