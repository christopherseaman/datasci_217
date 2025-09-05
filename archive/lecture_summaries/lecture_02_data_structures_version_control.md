# Lecture 2: Data Structures + Version Control Basics

## Learning Objectives
By the end of this lecture, students will be able to:
- Use Python's core data structures (lists, dictionaries, tuples, sets) for data organization
- Manipulate and access data using indexing, slicing, and iteration techniques
- Set up and use Git for version control with GitHub integration
- Create and manage Python virtual environments for project isolation
- Write and format documentation using Markdown syntax
- Apply data structure concepts to solve practical data science problems

## Content Consolidation Details

### Primary Sources (Current Lectures)
- **Lecture 02 (75%)**: Git fundamentals, GitHub, Markdown basics, package management
- **Lecture 03 (85%)**: Python data structures (lists, dictionaries, sets, tuples), string manipulation
- **Lecture 04 (20%)**: Python modules and file organization

### Secondary Integration
- **Lecture 01 (15%)**: Environment setup continuation from previous lecture

## Specific Topics Covered

### Version Control with Git (50 minutes)
1. **Git Fundamentals**
   - Understanding version control concepts and benefits
   - Configuring Git with user information and email
   - Repository initialization: `git init` vs `git clone`
   - Basic workflow: add, commit, push, pull

2. **GitHub Integration**
   - Creating and managing repositories
   - Remote repository concepts
   - Collaborative workflows and best practices
   - Understanding branches and basic branching

3. **Essential Git Commands**
   - `git status` - checking repository state
   - `git add` - staging changes
   - `git commit` - creating snapshots
   - `git push` and `git pull` - synchronizing with remote
   - `git clone` - copying repositories

4. **Conflict Resolution**
   - Understanding merge conflicts
   - Basic conflict resolution strategies
   - Using VS Code for conflict resolution
   - Prevention strategies and best practices

### Python Data Structures (45 minutes)
1. **Lists: Ordered Collections**
   - Creating lists: `[]` syntax and `list()` constructor
   - Indexing and slicing: positive/negative indices, slice notation
   - List methods: `append()`, `extend()`, `insert()`, `remove()`, `pop()`
   - List operations: concatenation, repetition, membership testing
   - Nested lists and multi-dimensional data

2. **Dictionaries: Key-Value Pairs**
   - Creating dictionaries: `{}` syntax and `dict()` constructor
   - Accessing values: bracket notation and `.get()` method
   - Dictionary methods: `.keys()`, `.values()`, `.items()`
   - Adding, updating, and deleting key-value pairs
   - Nested dictionaries for complex data structures

3. **Sets: Unique Collections**
   - Creating sets: `{}` syntax and `set()` constructor
   - Set operations: union, intersection, difference
   - Adding and removing elements
   - Set comprehensions (basic introduction)

4. **Tuples: Immutable Sequences**
   - Creating tuples: `()` syntax and `tuple()` constructor
   - Tuple unpacking and multiple assignment
   - Use cases for immutable data
   - Named tuples (brief introduction)

### String Manipulation and Processing (20 minutes)
1. **String Methods**
   - Case conversion: `.lower()`, `.upper()`, `.strip()`
   - String searching: `.find()`, `.count()`, `.startswith()`, `.endswith()`
   - String replacement: `.replace()`, string formatting
   - String splitting and joining: `.split()`, `.join()`

2. **Working with Text Data**
   - Reading and processing text files
   - Basic text cleaning techniques
   - Regular expressions introduction (basic patterns)

### Development Environment Setup (25 minutes)
1. **Python Virtual Environments**
   - Understanding dependency isolation
   - Creating virtual environments: `python -m venv`
   - Activating and deactivating environments
   - Installing packages with `pip`
   - Requirements files: `requirements.txt`

2. **Package Management**
   - Understanding Python packages and modules
   - Installing packages: `pip install`
   - Listing installed packages: `pip list`
   - Upgrading and uninstalling packages

3. **Project Organization**
   - Structuring Python projects
   - Module imports and organization
   - Best practices for file naming and structure

### Markdown Documentation (15 minutes)
1. **Markdown Syntax**
   - Headers, emphasis, and formatting
   - Lists (ordered and unordered)
   - Links and images
   - Code blocks and inline code
   - Tables (basic syntax)

2. **Documentation Best Practices**
   - README files for projects
   - Documenting code and processes
   - GitHub Markdown features

## Content to Trim (30% reduction from source lectures)

### From Lecture 02
- **Remove (15 minutes)**: Advanced Git features (rebasing, advanced branching strategies)
- **Reduce (10 minutes)**: Detailed conflict resolution scenarios - focus on basic cases
- **Simplify (8 minutes)**: Multiple remote management - focus on origin remote

### From Lecture 03
- **Remove (12 minutes)**: Advanced list comprehensions (move to later lecture)
- **Remove (8 minutes)**: Complex nested data structure examples
- **Reduce (10 minutes)**: Detailed string processing - focus on essential methods

### From Lecture 04
- **Remove (5 minutes)**: Advanced module organization concepts

## Practical Exercises and Hands-on Components

### Git Workflow Practice (25 minutes)
1. **Repository Setup**
   - Create local repository
   - Connect to GitHub remote
   - Practice basic add-commit-push cycle

2. **Collaboration Simulation**
   - Fork a repository
   - Make changes and create pull request
   - Handle basic merge conflicts

3. **Project Initialization**
   - Create data science project structure
   - Initialize with proper .gitignore
   - Set up documentation framework

### Data Structure Workshop (30 minutes)
1. **Student Information System**
   - Create nested data structures for student records
   - Implement functions to add, update, and query data
   - Practice different data structure use cases

2. **Text Processing Challenge**
   - Process sample CSV data using string methods
   - Clean and transform text data
   - Create summary statistics using data structures

3. **Data Analysis Fundamentals**
   - Read data from files into appropriate structures
   - Perform basic analysis operations
   - Generate reports using string formatting

### Environment and Workflow Setup (15 minutes)
1. **Virtual Environment Creation**
   - Set up project-specific environment
   - Install common data science packages
   - Create and test requirements.txt

2. **Documentation Practice**
   - Write comprehensive README.md
   - Document code with comments
   - Create project documentation structure

## Prerequisites and Dependencies

### From Previous Lecture
- Command line navigation skills
- Basic Python syntax and concepts
- Text editor setup and basic usage
- GitHub account creation

### Technical Requirements
- Git installation and configuration
- Python virtual environment capability
- Access to GitHub (web and command line)
- Text editor with Markdown support

### Preparation Materials
- Git configuration checklist
- GitHub account setup guide
- Virtual environment setup instructions

## Assessment Components

### Formative Assessment (During Class)
- Interactive Git commands practice
- Live data structure manipulation
- Peer code review exercises
- Collaborative problem-solving activities

### Summative Assessment (Assignment)
1. **Version Control Competency**
   - Create repository with proper structure
   - Demonstrate commit history with meaningful messages
   - Show collaboration through pull requests

2. **Data Structure Programming**
   - Implement data processing system using all major data structures
   - Handle real-world data organization challenges
   - Demonstrate proper error handling and edge cases

3. **Project Documentation**
   - Create comprehensive project documentation
   - Use Markdown for formatting and structure
   - Include setup instructions and usage examples

4. **Environment Management**
   - Set up isolated development environment
   - Create reproducible installation process
   - Document dependencies and requirements

## Key Success Metrics
- Students can perform essential Git operations confidently
- Students can choose appropriate data structures for different problems
- Students can set up and manage Python development environments
- Students demonstrate collaborative development skills through GitHub
- 85% of students successfully complete practical data structure assignments

## Integration with Course Progression
This lecture provides essential foundations for:
- **Lecture 3**: NumPy arrays build on list concepts
- **Lecture 4**: Pandas DataFrames extend dictionary concepts
- **Lecture 5**: Project management and collaboration skills
- **Future assignments**: All projects use Git for version control

## Resources and References

### Essential Resources
- [Pro Git Book](https://git-scm.com/book) - Chapters 1-3
- [Python Data Structures Documentation](https://docs.python.org/3/tutorial/datastructures.html)
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/)
- [Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)

### Interactive Learning
- [Learn Git Branching](https://learngitbranching.js.org/) - Visual Git tutorial
- [Python List Exercises](https://www.w3schools.com/python/python_lists_exercises.asp)
- [GitHub Skills](https://skills.github.com/) - Interactive GitHub tutorials

### Troubleshooting Resources
- Git common errors and solutions
- Python virtual environment troubleshooting
- Cross-platform setup guides
- Command reference sheets

### Community Support
- Course discussion forum for Git/Python questions
- GitHub community guidelines and best practices
- Python community resources and documentation
- Study group formation for collaborative learning