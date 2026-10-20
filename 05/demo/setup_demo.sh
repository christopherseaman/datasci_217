#!/bin/sh
# Download the Lecture 05 demo notebooks and their environment files into a new folder, ~/05-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/05/demo/setup_demo.sh | sh

# Plumbing: stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

# Where the files come from: the course repository's 05/demo folder on GitHub.
base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/05/demo"

# Stop if ~/05-demo already exists, so earlier work is never overwritten.
if [ -e ~/05-demo ]; then
  echo "~/05-demo already exists. Rename or remove it, then run this script again."
  exit 1
fi
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

# The piped script cannot change the student's shell directory, so it builds the environment here.
command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Install it from https://docs.astral.sh/uv/ and run this script again."; exit 1; }
uv venv --seed
uv sync

# VS Code selects the folder's .venv for notebooks.
if command -v code >/dev/null 2>&1; then
  code ~/05-demo
else
  echo "Open ~/05-demo in VS Code."
fi
