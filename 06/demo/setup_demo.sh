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

# The Parquet files the notebooks read go in data/.
mkdir data
for file in demo1_central_a1c.parquet demo1_labs.parquet demo1_patients.parquet demo1_poc_a1c.parquet demo1_registry.parquet demo1_targets.parquet demo1_visits.parquet demo2_bp_wide.parquet demo2_device_week4.parquet demo3_adt_census.parquet demo3_audits.parquet demo3_census.parquet demo3_experience.parquet demo3_feb_admissions.parquet demo3_finance_census.parquet demo3_jan_admissions.parquet demo3_mar_admissions.parquet demo3_q1_2025.parquet demo3_q1_2026.parquet demo3_staffing.parquet demo3_units.parquet; do
  curl -fsSL "$base_url/data/$file" -o "data/$file"
done

echo "Made ~/06-demo with the Lecture 06 demo notebooks and their environment files."

# The piped script cannot change the student's shell directory, so it builds the environment here.
command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Install it from https://docs.astral.sh/uv/ and run this script again."; exit 1; }
uv venv --seed
uv sync

# VS Code selects the folder's .venv for notebooks.
if command -v code >/dev/null 2>&1; then
  code ~/06-demo
else
  echo "Open ~/06-demo in VS Code."
fi
