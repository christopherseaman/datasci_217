#!/usr/bin/env python3
"""
DataSci 217 - Lecture 02 Assignment
Git Workflow, CLI Automation, and Python Data Processing

This assignment demonstrates integration of Git workflows, CLI automation, 
and Python data processing with progressive difficulty across three parts.

Part 1: Git Workflow Mastery (7 points)
- Repository setup with branching and merging
- Professional documentation

Part 2: CLI Project Scaffold Script (6 points)  
- Automated project setup with shell scripting
- Directory structure and file generation

Part 3: Python Data Processing (7 points)
- Data analysis with file I/O
- Modular design and function usage
"""

import os
import sys
from pathlib import Path

def main():
    """
    Main function for Lecture 02 assignment
    Demonstrates the complete workflow integration
    """
    print("DataSci 217 - Lecture 02 Assignment")
    print("=" * 50)
    print("Topic: Git Workflow, CLI Automation, and Python Data Processing")
    print()
    
    # Check if we're in the right directory structure
    check_project_structure()
    
    # Demonstrate the integrated workflow
    print("Assignment Components:")
    print("1. Git Workflow: Branching, committing, merging")
    print("2. CLI Automation: Project scaffold script")
    print("3. Python Processing: Data analysis with file I/O")
    print()
    
    # Check for required files
    check_required_files()
    
    print("✅ Assignment structure validated!")
    print("Next steps:")
    print("- Complete the Git workflow (Part 1)")
    print("- Implement the CLI script (Part 2)")  
    print("- Develop the Python analysis (Part 3)")
    print("- Test your implementation with: python test_assignment.py")

def check_project_structure():
    """Check that the project has the required directory structure"""
    required_dirs = ['src', 'data', 'output']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"⚠️  Missing directories: {', '.join(missing_dirs)}")
        print("Run ./setup_project.sh to create the project structure")
    else:
        print("✅ Project structure is correct")

def check_required_files():
    """Check for required assignment files"""
    required_files = [
        'README.md',
        '.gitignore',
        'requirements.txt', 
        'setup_project.sh',
        'src/data_analysis.py',
        'src/data_analysis_functions.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing files: {', '.join(missing_files)}")
        print("Complete the assignment requirements to create these files")
    else:
        print("✅ All required files present")

if __name__ == "__main__":
    main()
