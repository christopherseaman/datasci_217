# BLOCK 1: THE MISSING SEMESTER (Lectures 1-3)
## Foundation Skills for Data Science

### BLOCK OVERVIEW
This block provides essential computational skills often missing from traditional academic curricula. Students develop command-line proficiency, version control expertise, and remote computing capabilities that form the foundation for all subsequent data science work.

**Block Learning Objectives:**
- Master command-line navigation and file operations
- Implement version control workflows for collaborative projects  
- Configure and manage development environments
- Execute remote computing tasks and maintain persistent sessions

---

## LECTURE 1: Command Line Mastery
**Duration**: 90 minutes | **Content Reduction**: 15% from current Lecture 01

### Learning Objectives
By the end of this lecture, students will be able to:
- Navigate file systems using command-line interfaces
- Perform basic file operations (create, copy, move, delete)
- Use pipes and redirection for data processing
- Execute Python scripts from the command line
- Understand basic shell scripting concepts

### Content Sources & Integration
**Primary Sources:**
- Current Lecture 01: Command line basics (80% retention)
- Current Lecture 03: Advanced shell operations (30% retention)

**Content Trimming (15% reduction):**
- **REMOVE**: Detailed Python syntax review (moved to Block 2)
- **REMOVE**: In-depth control structures examples (moved to Block 2)  
- **CONDENSE**: Installation procedures (provide pre-configured environments)
- **SIMPLIFY**: Focus on essential commands only

### Detailed Content Structure

#### Opening (10 min)
- Course overview and expectations
- Why command line skills matter for data science
- Getting to know your terminal

#### Core Command Line Skills (45 min)
**Navigation & File System:**
- `pwd`, `ls`, `cd` - understanding file system hierarchy
- Absolute vs relative paths
- Special directories: `~`, `.`, `..`

**File Operations:**
- Creating: `mkdir`, `touch`
- Copying and moving: `cp`, `mv`  
- Viewing: `cat`, `head`, `tail`
- Removing: `rm` (with safety warnings)

**Text Processing Basics:**
- `grep` for pattern searching
- Basic pipes: `|` for chaining commands
- Redirection: `>` and `>>`

#### Python Integration (20 min)
- Running Python scripts: `python script.py`
- Command-line arguments with `sys.argv`
- Virtual environments introduction

#### Hands-on Practice (15 min)
**Practical Exercise:**
- Create project directory structure
- Practice file operations
- Simple text processing pipeline
- Run first Python script from command line

### Advanced Topics (Briefly Introduced)
- Environment variables concept
- Basic shell scripting structure
- Introduction to automation thinking

### Prerequisites
- Basic computer literacy
- Willingness to work in text-based interfaces

### Assessment Integration
Students complete practical exercises demonstrating:
- File system navigation
- Basic text processing
- Python script execution
- Simple automation task

---

## LECTURE 2: Git, GitHub, and Development Environment
**Duration**: 90 minutes | **Content Reduction**: 10% from current Lecture 02

### Learning Objectives
By the end of this lecture, students will be able to:
- Initialize and manage Git repositories
- Implement basic Git workflows (add, commit, push, pull)
- Collaborate using GitHub (clone, fork, pull requests)
- Set up and manage Python development environments
- Create and maintain documentation using Markdown

### Content Sources & Integration
**Primary Sources:**
- Current Lecture 02: Git and GitHub (90% retention)
- Current Lecture 02: Markdown (full retention)
- Current Lecture 02: Python environments (80% retention)

**Content Trimming (10% reduction):**
- **CONDENSE**: Git conflict resolution (cover basics, advanced scenarios in lab)
- **STREAMLINE**: GitHub interface tour (focus on essential features)
- **DEFER**: Advanced Git workflows (rebasing, cherry-picking to advanced topics)

### Detailed Content Structure

#### Version Control Fundamentals (30 min)
**Git Concepts:**
- Repository, staging area, working directory
- Basic workflow: `init`, `add`, `commit`
- Understanding Git history and logs
- Remote repositories concept

**Hands-on Git Practice:**
- Initialize first repository
- Make commits with meaningful messages
- Examine repository history

#### GitHub Collaboration (25 min)
**GitHub Basics:**
- Creating repositories
- Cloning vs forking
- Issues and project management
- Basic pull request workflow

**Collaboration Exercise:**
- Fork a repository
- Make changes and commit
- Create pull request
- Code review concepts

#### Development Environment Setup (20 min)
**Python Environment Management:**
- Virtual environments: why and how
- Package management with pip
- Requirements files
- Environment best practices

**IDE Integration:**
- VS Code setup and extensions
- Git integration in IDE
- Terminal integration

#### Documentation with Markdown (15 min)
**Markdown Essentials:**
- Headers, lists, links
- Code blocks and syntax highlighting
- Tables and formatting
- README best practices

### Advanced Topics Introduced
- Branch workflows (feature branches)
- Continuous integration concepts
- Open source contribution workflow

### Prerequisites
- Lecture 1: Command Line Mastery
- Basic understanding of file operations

### Assessment Integration
Students create and manage:
- Personal Git repository
- Collaborative project with pull requests
- Professional README documentation
- Properly configured development environment

---

## LECTURE 3: Shell Scripting and Remote Computing
**Duration**: 90 minutes | **Content Reduction**: 20% from current Lectures 03 & 04

### Learning Objectives
By the end of this lecture, students will be able to:
- Write and execute shell scripts for automation
- Access and work on remote systems using SSH
- Manage persistent sessions with screen/tmux
- Schedule automated tasks with cron
- Handle file compression and data transfer

### Content Sources & Integration
**Primary Sources:**
- Current Lecture 03: Shell scripting, environment variables (70% retention)
- Current Lecture 04: SSH, remote access, session management (80% retention)
- Current Lecture 03: File compression (full retention)

**Content Trimming (20% reduction):**
- **REMOVE**: Detailed file permissions discussion (cover essentials only)
- **CONDENSE**: Environment variable configuration (focus on practical usage)
- **DEFER**: Advanced shell scripting patterns
- **SIMPLIFY**: SSH key setup (provide simplified instructions)

### Detailed Content Structure

#### Shell Scripting Fundamentals (30 min)
**Script Structure:**
- Shebang lines and execution
- Variables and command substitution
- Basic control flow (if/then, loops)
- Command-line arguments: `$1`, `$2`, `$#`

**Practical Scripting:**
- File processing automation
- Simple data backup scripts
- Environment setup automation

#### Environment & Configuration Management (20 min)
**Environment Variables:**
- Setting and using variables
- PATH manipulation
- `.env` files for project configuration
- Security considerations

**Shell Configuration:**
- `.bashrc` and `.zshrc` basics
- Aliases and functions
- Persistent settings

#### Remote Computing (25 min)
**SSH Fundamentals:**
- Basic SSH connection
- File transfer with `scp`
- SSH keys (simplified setup)
- Security best practices

**Session Management:**
- Why persistent sessions matter
- `tmux` basics: sessions, windows, panes
- `screen` alternative
- Remote Jupyter notebooks

#### Automation & Scheduling (15 min)
**Task Automation:**
- `cron` job scheduling
- Syntax and examples
- Logging and monitoring
- Common automation patterns

**File Management:**
- Archive creation: `tar`, `zip`
- Compression strategies
- Bulk operations

### Advanced Topics Introduced
- Symbolic links and file system organization
- Regular expressions in shell context
- Basic system monitoring

### Prerequisites
- Lecture 1: Command Line Mastery
- Lecture 2: Git, GitHub, and Development Environment
- Basic understanding of file systems

### Assessment Integration
Students demonstrate:
- Working shell script for data processing
- Successful remote system access
- Automated task scheduling
- File compression and transfer operations

---

## BLOCK 1 INTEGRATION ASSESSMENT

### Capstone Project: Development Environment Setup
Students create a complete, reproducible development environment including:

1. **Repository Setup**
   - Git repository with proper structure
   - README with installation instructions
   - `.gitignore` for Python projects

2. **Automation Scripts**
   - Environment setup script
   - Data processing pipeline script
   - Backup automation

3. **Remote Computing Demo**
   - SSH connection to remote system
   - File transfer operations
   - Persistent session management

4. **Documentation**
   - Complete Markdown documentation
   - Script usage instructions
   - Troubleshooting guide

### Block Learning Outcomes Validation
- **Technical Proficiency**: Command-line fluency for data science tasks
- **Collaboration Skills**: Git/GitHub workflow implementation
- **System Administration**: Basic remote computing and automation
- **Professional Practice**: Documentation and reproducible environments

### Preparation for Block 2
Students now have the foundational computing skills necessary to:
- Work efficiently with Python development environments
- Manage code and data files professionally
- Execute data science workflows on local and remote systems
- Collaborate effectively on programming projects

---

## BLOCK 1 SUMMARY

**Total Duration**: 4.5 hours (3 × 90-minute lectures)
**Content Reduction**: 15% average across block
**Skills Emphasis**: Practical computing foundations
**Assessment Strategy**: Hands-on demonstrations and project work

This block transforms students from GUI-dependent users to command-line proficient practitioners, establishing the technical foundation necessary for advanced data science work. The progressive structure ensures skills build logically while maintaining practical relevance to data science applications.