# DataSci 217: Revised 11-Lecture Plan with Load Reduction

## Planning Philosophy

Based on comprehensive analysis of:
- **lectures_bkp/** - Original content with identified overload issues
- **McKinney Ch 1-11** - Systematic Python/pandas progression (NO capstone)
- **Missing Semester** - Modern CLI and development practices  
- **TLCL** - Systematic command line foundation

**KEY DESIGN PRINCIPLES:**
1. **True Load Reduction** - Move heavy content to bonus, keep lectures to 1.5 hours max
2. **CLI-First Approach** - Command line foundation, Jupyter introduced at Week 6
3. **Practical Daily Tools** - Focus on what data scientists actually use
4. **Your Teaching Style** - Explanation → Reference → Brief Example + xkcd humor
5. **Progressive Skill Building** - No prerequisite gaps, logical progression

---

## Course Structure Overview

### Foundational Toolkit (Weeks 1-5) - 1 Unit Option
**Goal:** Complete functional data science toolkit for basic analysis
- Command line proficiency
- Git collaboration
- Python fundamentals  
- Basic data manipulation
- Essential visualization

### Advanced Mastery (Weeks 6-11) - 2 Unit Completion
**Goal:** Professional data science skills and debugging
- Advanced data analysis
- Debugging techniques (explicit focus)
- Automation and scripting
- Advanced visualization
- Research applications

---

## Detailed Lecture Plan

### **Lecture 01: Python and Command Line Foundations**
*Goal: Get students functional in both Python basics and CLI navigation*

**Core Content (1.5 hours):**
- **CLI Essentials (30 min):**
  - Navigation: pwd, ls, cd, mkdir
  - File operations: cp, mv, rm, touch
  - Viewing: cat, head, tail
  - Getting help: man, --help
  
- **Python Basics (50 min):**
  - Running Python (command line vs scripts)
  - Variables and basic types (int, float, str, bool)
  - Basic operations and string formatting
  - Print statements and basic input
  
- **Integration Demo Break (10 min):** # LIVE DEMO!

**Reference Materials:**
- CLI: pwd, ls [-la], cd [path], mkdir [-p], cp/mv/rm
- Python: variable = value, print(f"text {variable}"), basic operators

**Brief Examples:**
- Navigate to create project folder, run simple Python calculation
- xkcd reference: "Real Programmers" (command line) or "Python" (import antigravity)

**Bonus Content (Optional):**
- Advanced ls options and formatting
- Python REPL advanced features
- Command history and shortcuts
- Python help() and dir() functions

**Assignment 01:**
- Simple GitHub/Python verification task
- Run provided script to hash UCSF email → specific output file
- Demonstrates: CLI navigation, Python execution, file output, git commit

---

### **Lecture 02: Version Control and Project Setup**
*Goal: Git workflow mastery with VS Code/GitHub GUI emphasis*

**Core Content (1.5 hours):**
- **Git Concepts (30 min):**
  - What is version control and why it matters
  - Repository, commit, branch concepts
  - Remote vs local repositories
  
- **GUI-First Git Workflow (45 min):**
  - VS Code git integration (staging, committing, pushing)
  - GitHub web interface (creating repos, PRs, issues)
  - Basic conflict resolution in VS Code
  - .gitignore basics
  
- **Environment Setup (15 min):**
  - Virtual environments concept
  - Basic conda/pip usage

**Reference Materials:**
- VS Code: Source Control panel, stage/commit/push buttons
- GitHub: New repo, clone link, PR creation
- Git concepts: Working directory → Staging → Repository → Remote

**Brief Examples:**
- Create repo on GitHub, clone, make changes, commit via VS Code
- xkcd reference: "Git" (this is git. it tracks collaborative work...)

**Bonus Content (Optional):**
- Command line git (for power users)
- Advanced branching strategies
- Git hooks and automation
- SSH key setup vs HTTPS

**Assignment 02:**
- Create personal data science project repo
- Practice VS Code git workflow
- Simple Python script with proper .gitignore

---

### **Lecture 03: Python Data Structures and File Operations**
*Goal: Master Python containers and file handling*

**Core Content (1.5 hours):**
- **Essential Data Structures (45 min):**
  - Lists: creation, indexing, methods (append, extend, remove)
  - Dictionaries: creation, access, methods (keys, values, items)
  - Basic sets and tuples
  - String operations and methods
  
- **File Operations (30 min):**
  - Reading and writing text files
  - CSV file handling basics
  - with statement (context managers)
  - os.path basics for file paths
  
- **Integration Examples (15 min):** # LIVE DEMO!

**Reference Materials:**
- Lists: [1,2,3], list.append(x), list[0], list[1:3]
- Dicts: {'key': 'value'}, dict['key'], dict.get('key')
- Files: with open('file.txt', 'r') as f: content = f.read()

**Brief Examples:**
- Read CSV data into list of dictionaries
- xkcd reference: "Python" (import antigravity) or data structures humor

**Bonus Content (Optional):**
- List/dict comprehensions
- Advanced string formatting
- Complex nested structures
- File encoding considerations

**Assignment 03:**
- Process simple dataset from CSV file
- Use dictionaries to organize and analyze data
- Output results to new file

---

### **Lecture 04: Command Line Text Processing and Python Functions**
*Goal: CLI data processing + Python function mastery*

**Core Content (1.5 hours):**
- **CLI Text Processing (45 min):**
  - Redirection and pipes (>, |)
  - grep for pattern searching
  - cut for column extraction
  - sort and uniq for data organization
  - head/tail for data sampling
  
- **Python Functions (30 min):**
  - Function definition and parameters
  - Return values and scope
  - Default parameters
  - Basic modules and imports
  
- **Integration Practice (15 min):** # LIVE DEMO!

**Reference Materials:**
- CLI: cat file.csv | cut -d',' -f2 | sort | uniq -c
- Python: def function_name(param1, param2=default): return value

**Brief Examples:**
- Process CSV with CLI tools, then Python function for calculation
- xkcd reference: "Automation" (spent more time automating...)

**Bonus Content (Optional):**
- Advanced grep with regex
- sed for text transformation
- *args and **kwargs in Python
- Complex pipeline construction

**Assignment 04:**
- CLI data exploration of provided dataset
- Python functions for analysis tasks
- Combine CLI and Python for complete workflow

---

### **Lecture 05: Python Libraries and Environment Management**
*Goal: Package ecosystem mastery and NumPy introduction*

**Core Content (1.5 hours):**
- **Package Management (30 min):**
  - conda vs pip usage
  - Virtual environments (conda create, activate)
  - requirements.txt basics
  - Installing packages (numpy, pandas, matplotlib)
  
- **NumPy Fundamentals (50 min):**
  - Array creation and basic operations
  - Indexing and slicing
  - Basic mathematical operations
  - Array attributes (shape, dtype, ndim)
  
- **Practical Applications (10 min):** # LIVE DEMO!

**Reference Materials:**
- conda: conda create -n name python=3.x, conda activate name, conda install package
- NumPy: np.array([1,2,3]), array[0:2], array.shape, np.mean(array)

**Brief Examples:**
- Create environment for project, install packages, basic array operations
- xkcd reference: "Dependencies" (my project has 37 dependencies...)

**Bonus Content (Optional):**
- Advanced NumPy operations
- Broadcasting details  
- Array memory layout
- Performance considerations

**Assignment 05:**
- Set up project environment properly
- NumPy array operations on provided dataset
- Document dependencies and setup process

---

### **Lecture 06: Pandas Fundamentals and Jupyter Introduction**
*Goal: Introduce pandas and transition to notebook environment*

**Core Content (1.5 hours):**
- **Jupyter Setup (15 min):**
  - When and why to use Jupyter vs CLI/scripts
  - Basic notebook operations
  - Markdown cells for documentation
  
- **Pandas Basics (60 min):**
  - Series and DataFrame creation
  - Reading CSV files (pd.read_csv)
  - Basic selection and filtering (.loc, .iloc, [])
  - Essential methods (.head(), .info(), .describe())
  
- **Data Exploration (15 min):** # LIVE DEMO!

**Reference Materials:**
- Pandas: pd.DataFrame(data), df['column'], df.loc[rows, columns]
- Jupyter: Cell types, Shift+Enter, Markdown syntax

**Brief Examples:**
- Load dataset, explore structure, basic filtering
- xkcd reference: "Data Pipeline" (data goes in, insights come out)

**Bonus Content (Optional):**
- Advanced pandas indexing
- Complex data structures
- Jupyter advanced features
- Performance optimization

**Assignment 06:**
- Complete data exploration in Jupyter notebook
- Document findings with markdown
- Basic data cleaning tasks

---

### **Lecture 07: Data Cleaning and Basic Visualization**
*Goal: Essential data preparation and simple plotting*

**Core Content (1.5 hours):**
- **Data Cleaning (45 min):**
  - Missing data handling (isna, fillna, dropna)
  - Data type conversion
  - Basic string operations (.str methods)
  - Duplicate detection and removal
  
- **Basic Visualization (30 min):**
  - pandas plotting (.plot() method)
  - Simple matplotlib figures
  - Line plots, histograms, scatter plots
  - Basic customization (titles, labels)
  
- **Debugging Introduction (15 min):** # LIVE DEMO!
  - Reading error messages
  - Basic debugging strategies

**Reference Materials:**
- Cleaning: df.isna().sum(), df.fillna(value), df.drop_duplicates()
- Plotting: df['col'].plot(kind='hist'), plt.title('Title'), plt.show()

**Brief Examples:**
- Clean messy dataset, create basic visualizations
- xkcd reference: "Data Quality" or "Correlation vs Causation"

**Bonus Content (Optional):**
- Advanced string processing
- Regular expressions in pandas
- Complex visualization customization
- Statistical visualization principles

**Assignment 07:**
- Data cleaning pipeline on real messy dataset
- Create basic exploratory visualizations
- Document data quality issues

---

### **Lecture 08: Data Analysis and Debugging Techniques**
*Goal: Analysis patterns and systematic debugging*

**Core Content (1.5 hours):**
- **Analysis Patterns (45 min):**
  - Groupby operations (basic aggregation)
  - Simple statistical analysis
  - Comparing groups and conditions
  - Basic hypothesis testing concepts
  
- **Debugging Mastery (30 min):**
  - Systematic debugging approach
  - Python debugger basics (pdb)
  - Common error patterns and solutions
  - Logging vs print debugging
  
- **Integration Practice (15 min):** # LIVE DEMO!

**Reference Materials:**
- Analysis: df.groupby('col').mean(), df.corr(), basic statistical tests
- Debugging: pdb.set_trace(), systematic error investigation

**Brief Examples:**
- Analyze groups in dataset, debug common issues
- xkcd reference: "Debugging" (changing random stuff until it works)

**Bonus Content (Optional):**
- Advanced statistical methods
- Complex debugging scenarios
- Performance profiling
- Advanced testing strategies

**Assignment 08:**
- Complete analysis with grouping and statistics
- Practice debugging provided broken code
- Document debugging process

---

### **Lecture 09: Automation and Advanced Data Manipulation**
*Goal: Scripting workflows and complex pandas operations*

**Core Content (1.5 hours):**
- **Workflow Automation (45 min):**
  - Creating analysis scripts
  - Command line arguments for scripts
  - Basic shell scripting for data tasks
  - Batch processing concepts
  
- **Advanced Pandas (30 min):**
  - Merging and joining datasets
  - Basic reshaping (pivot, melt)
  - Time series basics (if time permits)
  
- **Professional Practices (15 min):** # LIVE DEMO!
  - Code organization and functions
  - Documentation and comments

**Reference Materials:**
- Scripts: if __name__ == "__main__":, argparse basics
- Pandas: pd.merge(), df.pivot(), basic datetime operations

**Brief Examples:**
- Create reusable analysis script, process multiple files
- xkcd reference: "Automation" or "Good Code"

**Bonus Content (Optional):**
- Complex automation workflows
- Advanced pandas operations
- Shell scripting mastery
- Error handling in scripts

**Assignment 09:**
- Create automated analysis pipeline
- Process multiple related datasets
- Script that can be run from command line

---

### **Lecture 10: Advanced Visualization and Reporting**
*Goal: Professional visualization and communication*

**Core Content (1.5 hours):**
- **Advanced Plotting (50 min):**
  - Matplotlib fundamentals
  - Subplot creation and management
  - Color schemes and aesthetics
  - Statistical plots for communication
  
- **Reporting Integration (25 min):**
  - Jupyter for reports
  - Combining analysis and visualization
  - Export formats and sharing
  
- **Best Practices (15 min):** # LIVE DEMO!
  - Visualization design principles
  - Common pitfalls to avoid

**Reference Materials:**
- Matplotlib: plt.figure(), plt.subplot(), color and style options
- Design: Clear titles, appropriate chart types, accessible colors

**Brief Examples:**
- Create publication-ready figures, professional report
- xkcd reference: "Correlation vs Causation" or "Convincing"

**Bonus Content (Optional):**
- Seaborn advanced features
- Interactive visualization
- Advanced matplotlib customization
- Statistical visualization theory

**Assignment 10:**
- Create comprehensive analytical report
- Professional visualizations
- Export and sharing requirements

---

### **Lecture 11: Research Applications and Best Practices**
*Goal: Apply skills to research context, career insights*

**Core Content (1.5 hours):**
- **Research Data Workflows (45 min):**
  - Reproducible analysis principles
  - Data management best practices
  - Version control for research
  - Documentation and metadata
  
- **Clinical/Research Context (30 min):**
  - Research data types and challenges
  - Ethical considerations in data analysis
  - Collaboration patterns in research
  
- **Career and Next Steps (15 min):**
  - Skills assessment and growth paths
  - Resources for continued learning
  - Professional development

**Reference Materials:**
- Research: Reproducibility principles, ethical guidelines
- Tools: Integration of all previous skills

**Brief Examples:**
- Complete research workflow from raw data to publication
- xkcd reference: "Purity" (fields arranged by purity) or research humor

**Bonus Content (Optional):**
- Advanced research methods
- Specialized research tools
- Industry vs academic perspectives
- Advanced career paths

**Assignment 11:**
- Complete research-style analysis project
- Reproducible workflow documentation
- Peer review exercise

---

## Implementation Notes

### Content Reduction Strategies Applied
1. **Moved to Bonus:** Advanced NumPy, complex pandas operations, advanced git, detailed system administration, advanced visualization theory
2. **Simplified Approaches:** GUI-first git, basic statistics only, essential CLI tools only
3. **Integrated Topics:** Combined related skills within lectures rather than separate lectures
4. **Practical Focus:** Emphasized daily-use tools, reduced theoretical depth

### Teaching Style Integration
- **Explanation → Reference → Brief Example** structure throughout
- **xkcd integration** - specific comics suggested for each lecture
- **# LIVE DEMO!** callouts replace detailed written examples
- **Progressive complexity** - no prerequisite gaps

### Assignment Philosophy
- **Assignment 01:** Basic verification (can they use the tools?)
- **Assignments 02-05:** Skill building (one major skill per assignment)
- **Assignments 06-09:** Integration (combining multiple skills)
- **Assignments 10-11:** Professional application (research-style work)

### Load Reduction Validation
- Each lecture targets exactly 1.5 hours of content
- Heavy topics moved to bonus subdirectories
- No lecture combines more than 2 major topic areas
- Progressive difficulty without overwhelming jumps
- Optional content clearly marked and truly optional

This plan addresses the original feedback by:
✅ Reducing actual content load per lecture
✅ Moving advanced topics to bonus content
✅ Maintaining CLI-first approach
✅ Including debugging throughout 6-11
✅ Following your teaching style
✅ Creating practical, manageable assignments
✅ Integrating xkcd humor
✅ Ensuring 11-lecture structure with proper pacing