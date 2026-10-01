#!/bin/sh
# Download the Lecture 04 demo files into a new folder, ~/04-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/04/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from: the course repository's 04/demo folder on GitHub.
base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/04/demo"

# mkdir without -p stops the script here if ~/04-demo already exists, so earlier work is never overwritten.
mkdir ~/04-demo
cd ~/04-demo
mkdir data

# One download per file; -o saves it under the name that follows.
curl -fsSL "$base_url/setup_demo.sh" -o setup_demo.sh
curl -fsSL "$base_url/.python-version" -o .python-version
curl -fsSL "$base_url/pyproject.toml" -o pyproject.toml
curl -fsSL "$base_url/uv.lock" -o uv.lock
curl -fsSL "$base_url/demo1_jupyter_basics.ipynb" -o demo1_jupyter_basics.ipynb
curl -fsSL "$base_url/demo2_pandas_basics.ipynb" -o demo2_pandas_basics.ipynb
curl -fsSL "$base_url/demo3_data_io.ipynb" -o demo3_data_io.ipynb
curl -fsSL "$base_url/data/clinic_visits.csv" -o data/clinic_visits.csv

echo "Made ~/04-demo with the Lecture 04 demo notebooks, their environment files, and data/clinic_visits.csv."
echo "Next: cd ~/04-demo"
