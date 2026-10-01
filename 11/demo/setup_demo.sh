#!/bin/sh
# Download the Lecture 11 demo notebooks and environment records into a new folder, ~/11-demo.
# Run it with:
#   curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo/setup_demo.sh | sh
# The notebooks download their data themselves the first time they run.

# Stop at the first command that fails (-e) or at an unset variable (-u).
set -eu

base_url="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo"

# mkdir without -p stops the script here if ~/11-demo already exists, so earlier work is never overwritten.
mkdir ~/11-demo
cd ~/11-demo

curl -fsSL "$base_url/.python-version" -o .python-version
curl -fsSL "$base_url/pyproject.toml" -o pyproject.toml
curl -fsSL "$base_url/uv.lock" -o uv.lock
curl -fsSL "$base_url/01_setup.ipynb" -o 01_setup.ipynb
curl -fsSL "$base_url/02_wrangling.ipynb" -o 02_wrangling.ipynb
curl -fsSL "$base_url/03_model_prep.ipynb" -o 03_model_prep.ipynb
curl -fsSL "$base_url/04_modeling.ipynb" -o 04_modeling.ipynb
curl -fsSL "$base_url/05_geo_bonus.ipynb" -o 05_geo_bonus.ipynb

echo "Made ~/11-demo with the Lecture 11 demo notebooks, pyproject.toml, and uv.lock."
echo "Next: cd ~/11-demo"
