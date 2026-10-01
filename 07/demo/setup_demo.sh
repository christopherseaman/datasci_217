#!/bin/sh
# Download the Lecture 07 demo notebooks and their environment records into a new folder, ~/07-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/07/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from: the course repository's 07/demo folder on GitHub.
base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/07/demo"

# mkdir without -p stops the script here if ~/07-demo already exists, so earlier work is never overwritten.
mkdir ~/07-demo
cd ~/07-demo

# One download per file; -o saves it under the name that follows.
curl -fsSL "$base_url/setup_demo.sh" -o setup_demo.sh
curl -fsSL "$base_url/.python-version" -o .python-version
curl -fsSL "$base_url/pyproject.toml" -o pyproject.toml
curl -fsSL "$base_url/uv.lock" -o uv.lock
curl -fsSL "$base_url/healthexp.csv" -o healthexp.csv
curl -fsSL "$base_url/demo1_matplotlib_basics.ipynb" -o demo1_matplotlib_basics.ipynb
curl -fsSL "$base_url/demo2_seaborn_statistical.ipynb" -o demo2_seaborn_statistical.ipynb
curl -fsSL "$base_url/demo3_pandas_altair.ipynb" -o demo3_pandas_altair.ipynb

echo "Made ~/07-demo with the Lecture 07 demo notebooks and their environment records."
echo "Next: cd ~/07-demo"
