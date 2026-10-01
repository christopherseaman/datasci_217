#!/bin/sh
# Download the Lecture 06 demo notebooks and their environment files into a new folder, ~/06-demo.
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
curl -fsSL "$base_url/demo3_concat_timeseries.ipynb" -o demo3_concat_timeseries.ipynb

echo "Made ~/06-demo with the Lecture 06 demo notebooks and their environment files."
echo "Next: cd ~/06-demo"
