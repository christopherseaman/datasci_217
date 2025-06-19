# DS-217: Introduction to Python & Data Science Tools - Refined Curriculum

Health Data Science Masters Program | Flexible 1-2 Credit Course

## Course Structure

**Half-Semester (1 Credit):** Lectures 1-5 cover essential foundations (github, vs code, shell, python)
**Full-Semester (2 Credits):** All lectures 1-10 for comprehensive coverage (mostly additional python)

Each lecture: 90-120 minutes with 20-30 min hands-on demo + 2h lab period

## 📋 **Streamlining Changes from Previous Year**

**This year's curriculum was overpacked. Key changes to reduce content load:**

### 🔻 **Topics Moved to Bonus Content (Optional Self-Study)**

- **Advanced Git Operations**: Command-line Git, branching/merging, conflict resolution, SSH keys
- **Advanced Shell Commands**: `cron`, shell scripting, `find`, `grep`, `sed`, `awk`, process management
- **Remote Computing**: SSH, remote Jupyter, HPC cluster submission, CUDA/GPU intro
- **Advanced Visualization**: Plotly, Plotnine (focus on Matplotlib + Seaborn only)
- **Alternative Environments**: Jupyter Lab, conda (standardize on VS Code + `python -m venv`)
- **Advanced Python**: Generator expressions, complex nested data structures, regex

### ❌ **Topics Dropped/Significantly Reduced**

- **Multiple Environment Managers**: Focus only on `python -m venv` (mention `uv`)
- **Multiple Git Interfaces**: Focus only on VS Code Git + GitHub (mention CLI)
- **System Administration**: Process management, advanced file permissions
- **HPC/GPU Computing**: Move to specialized elective or DS-223

### ✅ **Core Content Streamlined To**

- **Git**: VS Code + GitHub workflow only
- **Shell**: Essential navigation + basic pipes only  
- **Python**: Fundamentals through data manipulation
- **Visualization**: Matplotlib + Seaborn only
- **Environment**: VS Code + `python -m venv` only

## 🎯 **Core Foundation (Half-Semester Program)**

### Lecture 1: **Getting Started with Python & Command Line**

Essential foundations for data science workflow  
*Streamlined from previous overpacked Lecture 1*

**Core Content (Lecture):**

- Python installation and basic syntax (variables, data types, operators)
- Running Python interactively vs scripts
- **Essential command line navigation ONLY**: `cd`, `ls`, `pwd`, `mkdir`
- **Basic file operations ONLY**: `cp`, `mv`, `rm`, and simple pipes (`|`, `>`)
- Setting up virtual environments with `python -m venv` (mention `uv` as modern alternative)
- VS Code setup for Python development

**🎁 Bonus Material (Self-Study) - Moved from Core:**

- **Advanced shell commands**: `find`, `grep`, `sed`, `awk` *(Previously in Lecture 1 core)*
- **Output redirection**: `>>`, `2>`, process substitution
- **Process management**: `ps`, `top`, `kill`, background jobs
- **SSH and remote access**: Basic connection, key setup *(Previously in Lecture 4 core)*
- **Alternative environments**: Conda, Docker, system package managers
- **Shell scripting basics**: Variables, loops, conditionals *(Previously in Lecture 3 core)*

### Lecture 2: **Python Data Structures & Control Flow**

Building blocks for data manipulation

**Core Content (Lecture):**

- Lists, dictionaries, tuples - when to use each
- For loops and conditional statements
- Reading and writing files in Python
- Basic error handling with try/except
- List comprehensions (simple examples)

**Bonus Material (Self-Study):**

- Sets and advanced data structure operations
- Generator expressions and iterators
- Complex nested data structures
- Regular expressions for text processing

### Lecture 3: **Version Control Essentials with Git**

Collaborative development and reproducibility

**Core Content (Lecture):**

- Git conceptual model (repositories, commits, branches)
- GitHub workflow: clone, add, commit, push, pull
- Creating and managing repositories
- Basic collaboration (pull requests, issues)
- Markdown for documentation

**Bonus Material (Self-Study):**

- Command-line Git (beyond GitHub Desktop)
- Advanced branching and merging
- Git stash, rebase, and conflict resolution
- Setting up SSH keys and authentication

### Lecture 4: **Working with Data using Pandas**

Essential data manipulation for health data science

**Core Content (Lecture):**

- Loading data (CSV, Excel) with pandas
- DataFrame and Series basics
- Essential operations: filtering, selecting, grouping
- Handling missing data (dropna, fillna)
- Basic data cleaning and transformation
- Working with data in VS Code notebooks (.ipynb files)

**Bonus Material (Self-Study):**

- Advanced pandas operations (pivot, melt, merge)
- Time series data handling
- Performance optimization techniques
- Working with large datasets (chunking)
- Jupyter Lab as alternative environment

### Lecture 5: **Data Visualization Fundamentals**

Communicating insights effectively  
*Reduced from 4 libraries to 2 core libraries*

**Core Content (Lecture):**

- **Matplotlib basics**: line plots, scatter plots, histograms
- **Seaborn for statistical visualizations** - focus on health data examples
- Basic customization (labels, colors, styles)
- Creating publication-ready figures
- Working with plots in VS Code notebooks

**🎁 Bonus Material (Self-Study) - Significant Reduction:**

- **Interactive visualizations with Plotly** *(Previously in Lecture 7 core)*
- **Grammar of graphics with Plotnine** *(Previously in Lecture 7 core)*
- Advanced customization and complex subplots
- Animated visualizations and dashboards
- **Alternative environments**: Jupyter Lab plotting workflows
- **Advanced matplotlib**: Custom colormaps, 3D plotting, animations

## 🚀 **Extended Program (Full-Semester)**

### Lecture 6: **Functions, Modules & Code Organization**

Writing reusable, maintainable code

**Core Content:**

- Writing functions with parameters and return values
- Organizing code into modules and packages
- Understanding scope and namespaces
- Documentation with docstrings
- Virtual environments with `python -m venv` and pip (mention `uv`)

**Key Concepts:** Code reusability, project organization, dependency management

### Lecture 7: **Advanced Data Manipulation**

Scaling up your data processing capabilities

**Core Content:**

- Complex data cleaning workflows
- Merging and joining datasets
- Reshaping data (pivot tables, melt)
- Working with different file formats (JSON, SQL, APIs)
- Data validation and quality checks

**Key Concepts:** Data pipeline design, quality assurance, format conversion

### Lecture 8: **Introduction to Machine Learning**

Getting started with predictive modeling

**Core Content:**

- Scikit-learn basics: train/test splits, model fitting
- Simple classification and regression examples
- Model evaluation metrics
- Cross-validation concepts
- Feature engineering basics

**Key Concepts:** Supervised learning workflow, model evaluation, overfitting

### Lecture 9: **Automation & Workflow Management**

Streamlining repetitive tasks

**Core Content:**

- Writing Python scripts for automation
- Command-line argument parsing
- Scheduling tasks and batch processing
- Building data processing pipelines
- Error handling in production scripts

**Key Concepts:** Automation principles, robustness, monitoring

### Lecture 10: **Putting It All Together**

Capstone project and best practices

**Core Content:**

- End-to-end project workflow
- Code review and collaboration practices
- Documentation and reproducibility
- Sharing and deploying your work
- Next steps in data science learning

**Key Concepts:** Professional development practices, project management

## 💡 **Pedagogical Approach**

### Core Principles

- **Practical First:** Every concept introduced with immediate healthcare data examples
- **Minimal Viable Competence:** Focus on tools students will use daily
- **Progressive Complexity:** Build from simple to sophisticated applications
- **Real-World Context:** Use actual health datasets and scenarios

### Lab Structure

- **Guided Practice (30 min):** Work through lecture examples together
- **Independent Work (60 min):** Students complete weekly assignment
- **Help Desk (30 min):** Individual assistance and Q&A

### Assessment Strategy

- **Weekly Assignments:** Practical exercises using health data
- **Midterm Project (Half-Semester):** Complete data analysis workflow
- **Final Project (Full-Semester):** Independent analysis with presentation

## 📚 **Learning Objectives**

### Half-Semester Graduates Can

- Set up and manage Python data science environments
- Navigate command line for basic file operations
- Use Git/GitHub for version control and collaboration
- Load, clean, and analyze tabular data with pandas
- Create effective visualizations to communicate findings

### Full-Semester Graduates Can

- Write modular, reusable Python code
- Design and implement data processing pipelines
- Apply basic machine learning techniques
- Automate repetitive data science tasks
- Manage end-to-end data science projects

## 🔗 **Connection to DS-223**

This course serves as the essential prerequisite for DS-223, ensuring students have:

- Comfortable Python programming abilities
- Experience with pandas and data manipulation
- Version control workflow knowledge
- Basic visualization skills
- Understanding of reproducible research practices

## 📖 **Recommended Resources**

**Primary Textbooks:**

- *Python for Data Analysis* by McKinney (pandas focus)
- *Automate the Boring Stuff with Python* by Sweigart (practical automation)

**Online Resources:**

- MIT's *The Missing Semester* (command line skills)
- Real Python tutorials (ongoing learning)
- Kaggle Learn courses (hands-on practice)

**Tools:**

- Anaconda Python distribution
- Jupyter Lab/Notebooks
- GitHub Desktop (beginners) → Git CLI (advanced)
- VS Code or PyCharm for development
