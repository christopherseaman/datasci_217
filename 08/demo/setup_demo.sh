#!/bin/sh
# Download the Lecture 08 demo notebooks and their environment records into a new folder, ~/08-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/08/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from: the course repository's 08/demo folder on GitHub.
base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/08/demo"

# mkdir without -p stops the script here if ~/08-demo already exists, so earlier work is never overwritten.
mkdir ~/08-demo
cd ~/08-demo

# One download per file; -o saves it under the name that follows.
curl -fsSL "$base_url/setup_demo.sh" -o setup_demo.sh
curl -fsSL "$base_url/pyproject.toml" -o pyproject.toml
curl -fsSL "$base_url/uv.lock" -o uv.lock
curl -fsSL "$base_url/.python-version" -o .python-version
curl -fsSL "$base_url/demo1_groupby_operations.ipynb" -o demo1_groupby_operations.ipynb
curl -fsSL "$base_url/demo2_coverage_result_shapes.ipynb" -o demo2_coverage_result_shapes.ipynb
curl -fsSL "$base_url/demo3_remote_performance.ipynb" -o demo3_remote_performance.ipynb

echo "Made ~/08-demo with the Lecture 08 demo notebooks, pyproject.toml, and uv.lock."
echo "Next: cd ~/08-demo"
