#!/bin/sh
# Download the Lecture 10 demo files into a new folder, ~/10-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/10/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from: the course repository's 10/demo folder on GitHub.
base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/10/demo"

# mkdir without -p stops the script here if ~/10-demo already exists, so earlier work is never overwritten.
mkdir ~/10-demo
cd ~/10-demo

# One download per file; -o saves it under the name that follows.
curl -fsSL "$base_url/setup_demo.sh" -o setup_demo.sh
curl -fsSL "$base_url/.python-version" -o .python-version
curl -fsSL "$base_url/pyproject.toml" -o pyproject.toml
curl -fsSL "$base_url/uv.lock" -o uv.lock
curl -fsSL "$base_url/demo1_statistical_modeling.ipynb" -o demo1_statistical_modeling.ipynb
curl -fsSL "$base_url/demo2_sklearn_prediction.ipynb" -o demo2_sklearn_prediction.ipynb
curl -fsSL "$base_url/demo3_trees_boosting_networks.ipynb" -o demo3_trees_boosting_networks.ipynb

echo "Made ~/10-demo with the Lecture 10 demo notebooks and their environment files."
echo "Next: cd ~/10-demo"
