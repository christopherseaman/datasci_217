# MIT Missing Semester - Relevant Topics for Data Science

## Overview
This document summarizes relevant topics from MIT's "Missing Semester" curriculum that apply to data science workflows. Focus is on practical daily-use tools rather than system administration.

## Lecture 1: Course Overview + Shell Basics (course-shell.md)
**Essential Shell Skills:**
- What is the shell and why use it
- Navigating the filesystem (pwd, ls, cd)
- Basic file operations (mv, cp, rm, mkdir)
- Connecting programs with pipes and redirection
- Root user and permissions (sudo)

**Key Tools:**
- man pages for documentation
- which/whereis for finding programs
- Basic PATH understanding

**Skills Level:** Beginner (Week 1)
**Relevance:** High - foundational for CLI-first approach

## Lecture 2: Shell Tools and Scripting (shell-tools.md)
**Shell Scripting Fundamentals:**
- Variables and string manipulation
- Functions and conditionals  
- Loops and control flow
- Special variables ($0, $1, $#, $?, $$)

**Essential Tools:**
- grep and regex patterns
- find for file searching  
- locate for fast searching
- history and reverse search

**Advanced Tools:**
- xargs for argument processing
- shellcheck for script validation

**Skills Level:** Intermediate (Week 3-4)
**Relevance:** High - essential for automation

## Lecture 3: Editors (Vim) (editors.md)
**Text Editing Concepts:**
- Modal editing philosophy
- Basic vim navigation and editing
- File operations in editors
- Configuration and customization

**Practical Skills:**
- Emergency vim usage (when stuck in vim)
- Basic file editing for scripts/config files
- Integration with development workflow

**Skills Level:** Beginner (Week 2)
**Relevance:** Medium - useful for remote work, config editing

## Lecture 4: Data Wrangling (data-wrangling.md)
**Text Processing Tools:**
- sed for stream editing
- Regular expressions in depth
- awk for pattern scanning and processing
- sort and uniq for data manipulation

**Data Pipeline Concepts:**
- Building command-line data pipelines
- Processing structured data (CSV, TSV)
- Working with JSON (jq tool)
- Binary data handling

**Advanced Techniques:**
- Multi-step data transformations
- Combining multiple tools effectively
- Performance considerations

**Skills Level:** Intermediate-Advanced (Week 5-6)
**Relevance:** Very High - core data science skill

## Lecture 5: Command-line Environment (command-line.md)
**Shell Configuration:**
- Customizing shell (bash, zsh configuration)
- Aliases and functions
- Shell history management
- Tab completion

**Terminal Multiplexing:**
- Screen basics
- tmux fundamentals
- Session management
- Remote session persistence

**Essential Skills:**
- Dotfiles management
- Environment setup
- Persistent sessions for long-running tasks

**Skills Level:** Intermediate (Week 2-3)
**Relevance:** High - essential for productive workflow

## Lecture 6: Version Control (Git) (version-control.md)
**Git Fundamentals:**
- Version control concepts and motivation
- Git data model (snapshots, commits, branches)
- Basic workflow (add, commit, push, pull)
- Working with remotes

**Collaboration:**
- Branching and merging
- Merge conflicts resolution
- Git best practices
- Integration with GitHub

**Advanced Topics:**
- Git internals understanding
- Advanced git commands
- Git hooks and automation

**Skills Level:** Intermediate (Week 2)
**Relevance:** Very High - essential collaboration tool

## Lecture 7: Debugging and Profiling (debugging-profiling.md)
**Debugging Concepts:**
- Printf debugging vs debugger usage
- Python debugger (pdb) basics
- Static analysis tools
- Profiling for performance

**Essential Debugging Skills:**
- Reading error messages effectively
- Systematic debugging approach
- Using logging effectively
- Performance bottleneck identification

**Python-Specific:**
- pdb debugger usage
- Python profiling tools
- Memory usage analysis

**Skills Level:** Intermediate-Advanced (Week 6-8)
**Relevance:** Very High - critical for development

## Lecture 8: Metaprogramming (metaprogramming.md)
**Automation Concepts:**
- Build systems and dependency management
- Continuous integration basics
- Testing frameworks
- Documentation generation

**Practical Applications:**
- Makefile basics for data science
- Simple automation scripts
- Testing data analysis code

**Skills Level:** Advanced (Week 9-10)
**Relevance:** Medium - useful for reproducible analysis

## Lecture 9: Security and Cryptography (security.md)
**Practical Security:**
- SSH keys and authentication
- Basic cryptographic concepts
- Password managers
- Two-factor authentication

**Data Science Applications:**
- Secure data transfer
- API authentication
- Protecting sensitive data
- Safe credential management

**Skills Level:** Intermediate (Week 4)
**Relevance:** Medium - important for data handling

## Lecture 10: Potpourri (potpourri.md)
**Miscellaneous Useful Tools:**
- Package managers (apt, brew, conda)
- Docker basics for reproducibility
- Virtual machines and containers
- APIs and web requests

**Skills Level:** Advanced (Week 10-11)
**Relevance:** Medium - useful for advanced workflows

## Content Organization for Data Science Course

### Essential for Core Lectures (High Priority)
1. **Shell Basics** - Foundation for CLI-first approach
2. **Data Wrangling** - Core data science skill set
3. **Version Control** - Essential collaboration
4. **Debugging Techniques** - Critical development skill
5. **Command-line Environment** - Productivity essentials

### Suitable for Bonus Content
1. **Advanced Shell Scripting** - Beyond basic automation
2. **Vim Mastery** - Advanced editor techniques  
3. **Metaprogramming** - Advanced automation concepts
4. **Security Deep-dive** - Beyond basic SSH/keys
5. **Container/Docker** - Advanced reproducibility

### Integration with DataSci 217 Structure

**Week 1-2: Foundation**
- Shell basics and navigation
- Basic editor usage
- Command-line environment setup

**Week 3-4: Development Tools**
- Shell scripting fundamentals  
- Git workflow integration
- SSH and remote access

**Week 5-6: Data Processing**
- Advanced data wrangling techniques
- Text processing pipelines
- Regular expressions mastery

**Week 7-9: Development Skills**
- Debugging methodologies
- Profiling and optimization
- Testing approaches

**Week 10-11: Advanced Topics**
- Automation and metaprogramming
- Security considerations
- Advanced tool integration

## Missing Semester Strengths for Data Science
1. **Practical focus** - Tools used daily by practitioners
2. **Command-line emphasis** - Aligns with CLI-first approach
3. **Real-world examples** - Authentic use cases
4. **Progressive complexity** - Good skill building
5. **Modern practices** - Current industry standards

## Integration Considerations
- Missing Semester assumes some programming background
- Less Python-specific content (more general tools)
- Strong complement to McKinney's pandas focus
- Excellent source for debugging and development practices
- Good foundation for reproducible research workflows

## Recommended Adaptations for DataSci 217
1. **Start with basics** - Don't assume command-line familiarity
2. **Integrate with Python** - Show how tools work with data analysis
3. **Add data science examples** - Replace generic examples with data scenarios
4. **Emphasize reproducibility** - Connect tools to research workflows
5. **Include humor** - Add xkcd references as requested