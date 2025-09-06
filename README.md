# DataSci-217

Introduction to Python & Data Science Tools ([canonical url](https://ds217.badmath.org), [github repo](https://github.com/christopherseaman/datasci_217))

## Course Structure

The course offers **two track options** to accommodate different student backgrounds:

### 5-Lecture Intensive Track
**Target**: Students with some programming experience  
**Time**: 5 weeks × 3.5 hours = 17.5 contact hours + independent work  
**Outcome**: Core data science competency for immediate application

1. **Command Line Foundations + Development Environment**
   - Command line navigation and file operations
   - Python installation and virtual environments
   - VS Code setup and basic workflows
   - Introduction to development tools

2. **Python Basics + Version Control**
   - Python syntax, variables, and data types (McKinney Ch 2)
   - Control structures and functions basics
   - Git through VS Code/GitHub (GUI focus)
   - Project organization and collaboration

3. **Python Data Structures**
   - Lists, tuples, dictionaries, and sets (McKinney Ch 3)
   - String manipulation and text processing
   - File I/O and data loading fundamentals
   - Error handling basics

4. **NumPy Fundamentals**
   - N-dimensional arrays and array creation (McKinney Ch 4)
   - Array operations and universal functions
   - Boolean indexing and fancy indexing
   - Basic mathematical operations

5. **Pandas Getting Started**
   - Series and DataFrame creation (McKinney Ch 5)
   - Data selection and filtering
   - Basic data cleaning operations
   - Reading from common file formats (CSV, Excel)

### 11-Lecture Extended Track  
**Target**: Complete beginners, comprehensive preparation  
**Time**: 11 weeks × 3.5 hours = 38.5 contact hours + independent work  
**Outcome**: Professional-level data science skills

1. **Command Line Foundations + Development Environment**
   - Command line navigation and file operations
   - Python installation and virtual environments
   - VS Code setup and basic workflows
   - Introduction to development tools

2. **Python Basics + Version Control**
   - Python syntax, variables, and data types (McKinney Ch 2)
   - Control structures and functions basics
   - Git through VS Code/GitHub (GUI focus)
   - Project organization and collaboration

3. **Python Data Structures**
   - Lists, tuples, dictionaries, and sets (McKinney Ch 3)
   - String manipulation and text processing
   - File I/O and data loading fundamentals
   - Error handling basics

4. **NumPy Fundamentals**
   - N-dimensional arrays and array creation (McKinney Ch 4)
   - Array operations and universal functions
   - Boolean indexing and fancy indexing
   - Basic mathematical operations

5. **Pandas Getting Started**
   - Series and DataFrame creation (McKinney Ch 5)
   - Data selection and filtering
   - Basic data cleaning operations
   - Reading from common file formats (CSV, Excel)

6. **Data Loading and File Formats**
   - Advanced file reading/writing operations (McKinney Ch 6)
   - Working with different data formats (JSON, HDF5, databases)
   - Handling messy and missing data
   - Web scraping basics

7. **Data Cleaning and Preparation**
   - Data transformation techniques (McKinney Ch 7)
   - String operations for data cleaning
   - Handling missing data strategies
   - Data validation and quality assessment

8. **Data Wrangling**
   - Merge, join, and concatenate operations (McKinney Ch 8)
   - Reshaping data (pivot, melt, stack/unstack)
   - Grouping and aggregation
   - Advanced data transformation patterns

9. **Plotting and Visualization**
   - Matplotlib fundamentals (McKinney Ch 9)
   - Pandas plotting interface
   - Seaborn for statistical visualization
   - Creating effective data visualizations

10. **Data Aggregation and Group Operations**
    - GroupBy mechanics and advanced techniques (McKinney Ch 10)
    - Aggregation functions and transformations
    - Pivot tables and cross-tabulation
    - Time series basics

11. **Time Series and Applied Analysis**
    - Time series data handling and analysis (McKinney Ch 11)
    - Date/time operations and resampling
    - Applied data analysis workflows
    - Clinical research applications and case studies

## Implementation Status

✅ **Ready for immediate deployment**
- Complete 5-lecture intensive track materials available in [`docs/5_lecture_intensive/`](docs/5_lecture_intensive/)
- Conversion infrastructure operational with scripts in [`scripts/`](scripts/)
- All original content backed up and verified (349 files)
- Templates and format guidelines established

### Quick Start
```bash
# View complete course structure and implementation details
cat docs/course_overview.md

# Convert existing materials to new format
python scripts/convert_lecture.py [lecture_number]

# Verify backup integrity
python scripts/verify_backup.py
```

See [`docs/README.md`](docs/README.md) for complete documentation structure and implementation guidance.

## Resources

### Python Reference
- _Python for Data Analysis_, McKinney - [website](https://wesmckinney.com/book/)
- _Automate the Boring Stuff with Python_, Sweigart - [website](https://automatetheboringstuff.com/)
- _Whirlwind Tour of Python_, VanderPlas - [website](https://jakevdp.github.io/WhirlwindTourOfPython/)
- _Think Python_, Downey - [Green Tea Press](https://greenteapress.com/wp/think-python/)
- [Official Python documentation](https://docs.python.org/3/)

### Python Courses
- [Exercism Python track](https://exercism.io/tracks/python)
- [Codecademy Python course](https://www.codecademy.com/learn/learn-python-3)
- [Real Python](https://realpython.com/)
- [DataCamp (Python + Data Science)](https://www.datacamp.com/)

### Command Line & Tools
- [The Missing Semester](https://missing.csail.mit.edu/) (command line, git, data wrangling)
- [The Linux Command Line book](http://linuxcommand.org/tlcl.php)
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)
- [Pro Git book](https://git-scm.com/book/en/v2)
- [Markdown Guide](https://www.markdownguide.org/)

### Free Development Options
- [Google Cloud Shell](https://cloud.google.com/free/docs/compute-getting-started)
- [GitHub Codespaces](https://cli.github.com/manual/gh_codespace_ssh)