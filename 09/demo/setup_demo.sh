#!/bin/sh
# Download the Lecture 09 demo notebooks and their environment records into a new folder, ~/09-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/09/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from: the course repository's 09/demo folder on GitHub.
base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/09/demo"

# mkdir without -p stops the script here if ~/09-demo already exists, so earlier work is never overwritten.
mkdir ~/09-demo
cd ~/09-demo

# One download per file; -o saves it under the name that follows.
curl -fsSL "$base_url/setup_demo.sh" -o setup_demo.sh
curl -fsSL "$base_url/.python-version" -o .python-version
curl -fsSL "$base_url/pyproject.toml" -o pyproject.toml
curl -fsSL "$base_url/uv.lock" -o uv.lock
curl -fsSL "$base_url/demo1_datetime_fundamentals.ipynb" -o demo1_datetime_fundamentals.ipynb
curl -fsSL "$base_url/demo2_indexing_resampling.ipynb" -o demo2_indexing_resampling.ipynb
curl -fsSL "$base_url/demo3_visualization_automation.ipynb" -o demo3_visualization_automation.ipynb

echo "Made ~/09-demo with the three Lecture 09 demo notebooks, .python-version, pyproject.toml, and uv.lock."
echo "Next: cd ~/09-demo"
