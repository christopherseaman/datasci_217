Project Environment Management

Why Virtual Environments?

Python projects need specific versions of packages. Different projects need different packages. Virtual environments create isolated Python installations for each project.

Think: "Each project gets its own Python toolkit, separate from others."

Creating Virtual Environments

**Reference (using conda):**

```bash
conda create -n datasci-practice python=3.11
conda activate datasci-practice

```

**Reference (using venv):**

```bash
python -m venv datasci-practice
Mac/Linux: source datasci-practice/bin/activate
Windows: datasci-practice\\Scripts\\activate

```

**Brief Example:**

```bash
Create environment
conda create -n datasci-practice python=3.11

Activate environment
conda activate datasci-practice

Install packages
conda install pandas numpy matplotlib

Deactivate when done
conda deactivate

```

Package Installation Basics

**Reference:**

```bash
With conda (preferred for data science)
conda install pandas numpy matplotlib

With pip
pip install pandas numpy matplotlib

Install from requirements file
pip install -r requirements.txt

```

**Managing Dependencies:**
Always document which packages your project needs:

```bash
# Save current environment packages
conda list --export > requirements.txt

# Or create environment file (better for conda)
conda env export > environment.yml

# For pip-only environments
pip freeze > requirements.txt

```

**Environment File Example (environment.yml):**

```yaml
name: datasci-practice
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas>=1.5.0
  - numpy>=1.20.0
  - matplotlib>=3.5.0
  - jupyter
  - pip
  - pip:
    - requests>=2.28.0

```

**Virtual Environment Troubleshooting:**

- **Environment not activating?** Check you're using the right command for your system
- **Package not found?** Try different channels: `conda install -c conda-forge package-name`
- **Slow conda installs?** Use mamba: `conda install mamba -c conda-forge`
- **Mixed pip/conda issues?** Prefer conda, use pip only for packages unavailable in conda

**Professional Environment Workflow:**

1. Start new project → Create new environment
2. Install packages → Document in environment.yml
3. Commit environment.yml to Git
4. Teammates clone repo → Create environment from file
5. Update packages → Update environment.yml
6. Everyone stays synchronized

The key insight: Always work in virtual environments for projects!

Integrating Git and Environments

.gitignore File

Some files shouldn't be tracked by Git:

- Virtual environment folders
- Data files (often too large)
- Temporary files
- Secret keys/passwords

**Reference (.gitignore contents):**

```
Virtual environments
venv/
env/
datasci-practice/

Data files (often too large for Git)
*.csv
data/

Python generated files
__pycache__/
*.pyc

Jupyter notebook outputs
.ipynb_checkpoints/

Operating system files
.DS_Store
Thumbs.db

```

requirements.txt File

Track which packages your project needs:

**Reference:**

```bash
Create requirements file
pip freeze > requirements.txt

Install from requirements file (on new computer/teammate)
pip install -r requirements.txt

```

This lets anyone recreate your exact environment!

Collaboration Basics

[README.md](http://readme.md/) Files

Every repository should have a README explaining:

- What the project does
- How to set it up
- How to run it
- Who to contact for questions

**Brief Example:**

```markdown
My Data Science Practice

This repository contains practice exercises for DataSci 217.

Setup
1. Clone this repository
2. Create virtual environment: `conda create -n practice python=3.11`
3. Activate environment: `conda activate practice`
4. Install packages: `pip install -r requirements.txt`

Usage
Run analysis: `python analysis.py`

Contact
Alice Smith - alice.smith@ucsf.edu

```

Markdown is just formatted text - we'll learn more next week!

Key Takeaways

1. **Git tracks every change** to your project over time
2. **GitHub stores your work** in the cloud and enables collaboration
3. **VS Code's GUI** makes Git visual and approachable
4. **Virtual environments** keep projects separate and reproducible
5. **Good habits early** save massive headaches later

You now have professional-grade project management skills. Every change you make is tracked, your work is backed up, and you can collaborate with teammates seamlessly.

Next week: We'll dive deeper into Python data structures and file operations!

Practice Challenge

Before next class:

1. Create a second repository called `week-02-practice`
2. Add a Python file with a simple calculation
3. Create a requirements.txt file (even if empty)
4. Write a descriptive [README.md](http://readme.md/)
5. Make several commits with good commit messages
6. Check that everything appears correctly on GitHub

Remember: This workflow becomes automatic with practice. Professional data scientists do this hundreds of times per week!