#!/bin/sh
# Download the Lecture 05 demo notebooks and their environment files into a new folder, ~/05-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/05/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from: the course repository's 05/demo folder on GitHub.
base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/05/demo"

# mkdir without -p stops the script here if ~/05-demo already exists, so earlier work is never overwritten.
mkdir ~/05-demo
cd ~/05-demo

# One download per file; -o saves it under the name that follows.
curl -fsSL "$base_url/setup_demo.sh" -o setup_demo.sh
curl -fsSL "$base_url/.python-version" -o .python-version
curl -fsSL "$base_url/pyproject.toml" -o pyproject.toml
curl -fsSL "$base_url/uv.lock" -o uv.lock
curl -fsSL "$base_url/demo1_missing_data.ipynb" -o demo1_missing_data.ipynb
curl -fsSL "$base_url/demo2_transformations.ipynb" -o demo2_transformations.ipynb
curl -fsSL "$base_url/demo3_workflow.ipynb" -o demo3_workflow.ipynb

echo "Made ~/05-demo with the Lecture 05 demo notebooks and their environment files."
echo "Next: cd ~/05-demo"
