# GitHub Classroom Integration Guide for DataSci 217

*Complete setup and management guide for assignment distribution and automated grading*

## Overview

This guide provides comprehensive instructions for integrating DataSci 217 assignments with GitHub Classroom, enabling automated distribution, collection, and grading of student assignments with pytest-based testing frameworks.

## Table of Contents

1. [GitHub Classroom Setup](#github-classroom-setup)
2. [Assignment Repository Templates](#assignment-repository-templates)
3. [Automated Grading Configuration](#automated-grading-configuration)
4. [Student Workflow](#student-workflow)
5. [Instructor Management](#instructor-management)
6. [Troubleshooting](#troubleshooting)

## GitHub Classroom Setup

### Prerequisites

- GitHub Education account (free for instructors)
- Organization for your course (e.g., `datasci-217-fall-2024`)
- Admin access to the course organization

### Initial Configuration

1. **Create Course Organization**:
   ```
   Organization Name: datasci-217-fall-2024
   Description: Data Science 217 Course Assignments
   Type: Educational
   ```

2. **Enable GitHub Classroom**:
   - Visit [classroom.github.com](https://classroom.github.com)
   - Connect your GitHub account
   - Create new classroom linked to your organization

3. **Configure Classroom Settings**:
   ```
   Classroom Name: DataSci 217
   Organization: datasci-217-fall-2024
   Student Identifier: Student ID or GitHub username
   ```

### Student Roster Management

**Option 1: CSV Upload**
```csv
identifier,name
student001,Alice Johnson
student002,Bob Smith
student003,Charlie Brown
```

**Option 2: GitHub Integration**
- Students join via assignment invitation links
- Automatic roster population based on acceptances

## Assignment Repository Templates

### Template Structure

Each assignment should follow this standardized structure:

```
assignment-template/
├── README.md                    # Assignment instructions
├── .github/
│   └── workflows/
│       └── classroom.yml        # GitHub Actions for grading
├── .gitignore                   # Python gitignore
├── requirements.txt             # Dependencies
├── starter_code.py              # Student starting point
├── test_assignment.py           # Automated tests
├── assignment_config.json       # Grading configuration
└── solution/                    # Reference implementation (private)
    ├── complete_solution.py
    └── test_results.json
```

### Template Creation Process

1. **Create Template Repository**:
   ```bash
   # In your course organization
   Repository Name: assignment-01-template
   Description: Assignment 1: Python Fundamentals
   Visibility: Public (templates must be public)
   Template: ✓ Check "Template repository"
   ```

2. **Add Assignment Files**:
   - Copy files from `/assignments/lecture_01/assignment/`
   - Ensure all paths are relative and work in clean environment

3. **Configure GitHub Actions**:

Create `.github/workflows/classroom.yml`:

```yaml
name: GitHub Classroom Workflow

on: 
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

permissions:
  checks: write
  actions: read
  contents: read

jobs:
  run-autograding-tests:
    runs-on: ubuntu-latest
    if: github.actor != 'github-classroom[bot]'
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run structure tests
      run: |
        python -m pytest test_assignment.py::TestProjectStructure -v
        
    - name: Run functionality tests  
      run: |
        python -m pytest test_assignment.py::TestPythonScriptFunctionality -v
        
    - name: Run code quality tests
      run: |
        python -m pytest test_assignment.py::TestCodeQuality -v
        
    - name: Run error handling tests
      run: |
        python -m pytest test_assignment.py::TestErrorHandlingAndEdgeCases -v

  autograding:
    runs-on: ubuntu-latest
    if: github.actor != 'github-classroom[bot]'
    steps:
    - uses: actions/checkout@v4
    
    - name: Project Structure Check
      id: project-structure
      uses: classroom-resources/autograding-command-grader@v1
      with:
        test-name: 'Project Structure'
        setup-command: 'pip install -r requirements.txt'
        command: 'python -m pytest test_assignment.py::TestProjectStructure --tb=short'
        timeout: 5
        max-score: 25

    - name: Python Functionality Check  
      id: python-functionality
      uses: classroom-resources/autograding-command-grader@v1
      with:
        test-name: 'Python Functionality'
        setup-command: ''
        command: 'python -m pytest test_assignment.py::TestPythonScriptFunctionality --tb=short'
        timeout: 10
        max-score: 35

    - name: Code Quality Check
      id: code-quality
      uses: classroom-resources/autograding-command-grader@v1
      with:
        test-name: 'Code Quality'
        setup-command: ''
        command: 'python -m pytest test_assignment.py::TestCodeQuality --tb=short'  
        timeout: 5
        max-score: 25

    - name: Error Handling Check
      id: error-handling
      uses: classroom-resources/autograding-command-grader@v1
      with:
        test-name: 'Error Handling'
        setup-command: ''
        command: 'python -m pytest test_assignment.py::TestErrorHandlingAndEdgeCases --tb=short'
        timeout: 5
        max-score: 15

    - name: Autograding Reporter
      uses: classroom-resources/autograding-grading-reporter@v1
      env:
        PROJECT-STRUCTURE_RESULTS: "${{steps.project-structure.outputs.result}}"
        PYTHON-FUNCTIONALITY_RESULTS: "${{steps.python-functionality.outputs.result}}"
        CODE-QUALITY_RESULTS: "${{steps.code-quality.outputs.result}}"
        ERROR-HANDLING_RESULTS: "${{steps.error-handling.outputs.result}}"
      with:
        runners: project-structure,python-functionality,code-quality,error-handling
```

### Assignment Configuration

Create `assignment_config.json`:

```json
{
  "assignment": {
    "name": "Assignment 1: Python Fundamentals & Command Line Mastery",
    "points_possible": 100,
    "due_date": "2024-02-15T23:59:59",
    "auto_accept": false,
    "feedback": "automatic"
  },
  "grading": {
    "categories": [
      {
        "name": "Project Structure",
        "weight": 25,
        "tests": "TestProjectStructure"
      },
      {
        "name": "Python Functionality", 
        "weight": 35,
        "tests": "TestPythonScriptFunctionality"
      },
      {
        "name": "Code Quality",
        "weight": 25, 
        "tests": "TestCodeQuality"
      },
      {
        "name": "Error Handling",
        "weight": 15,
        "tests": "TestErrorHandlingAndEdgeCases" 
      }
    ]
  },
  "environment": {
    "python_version": "3.9",
    "timeout_minutes": 10,
    "memory_limit": "1GB"
  }
}
```

## Automated Grading Configuration

### Test Categories and Scoring

| Test Category | Points | Test Class | Focus Area |
|---------------|--------|------------|------------|
| Project Structure | 25 | `TestProjectStructure` | Directory structure, required files |
| Python Functionality | 35 | `TestPythonScriptFunctionality` | Core functions, logic correctness |
| Code Quality | 25 | `TestCodeQuality` | Documentation, style, patterns |
| Error Handling | 15 | `TestErrorHandlingAndEdgeCases` | Exception handling, edge cases |

### Grading Rubric Implementation

```python
# In test_assignment.py, add scoring decorators
import pytest

class GradingMixin:
    """Mixin to provide grading utilities for tests."""
    
    @staticmethod
    def award_points(points, max_points, description=""):
        """Award partial or full points for a test."""
        percentage = (points / max_points) * 100
        print(f"Points: {points}/{max_points} ({percentage:.1f}%) - {description}")
        return points

    @staticmethod  
    def partial_credit(condition, full_points, partial_points=0, description=""):
        """Award partial credit based on condition."""
        if condition:
            return GradingMixin.award_points(full_points, full_points, description)
        else:
            return GradingMixin.award_points(partial_points, full_points, f"Partial: {description}")

# Example enhanced test with partial credit
class TestProjectStructure(GradingMixin):
    def test_directory_structure_with_scoring(self):
        """Test directory structure with partial credit."""
        required_dirs = ['data/raw', 'scripts', 'tests', 'config']
        found_dirs = []
        
        for directory in required_dirs:
            if Path(directory).exists():
                found_dirs.append(directory)
        
        # Award partial credit based on completion
        score = len(found_dirs) / len(required_dirs) * 5  # 5 points max
        self.award_points(score, 5, f"Found {len(found_dirs)}/{len(required_dirs)} directories")
        
        assert len(found_dirs) > 0, "At least some directories should exist"
```

### Advanced Scoring with Weights

```python
# test_scoring.py - Advanced scoring system
class AssignmentScorer:
    def __init__(self):
        self.scores = {}
        self.weights = {
            'structure': 0.25,
            'functionality': 0.35, 
            'quality': 0.25,
            'error_handling': 0.15
        }
    
    def add_score(self, category, points, max_points):
        """Add score for a category."""
        if category not in self.scores:
            self.scores[category] = {'points': 0, 'max_points': 0}
        
        self.scores[category]['points'] += points
        self.scores[category]['max_points'] += max_points
    
    def calculate_final_score(self):
        """Calculate weighted final score."""
        final_score = 0
        
        for category, weight in self.weights.items():
            if category in self.scores:
                category_score = self.scores[category]
                percentage = category_score['points'] / category_score['max_points']
                weighted_score = percentage * weight * 100
                final_score += weighted_score
        
        return min(final_score, 100)  # Cap at 100%
```

## Student Workflow

### Assignment Acceptance Process

1. **Receive Assignment Link**:
   ```
   Example: https://classroom.github.com/a/abc123def456
   ```

2. **Accept Assignment**:
   - Click link to accept assignment
   - Authorize GitHub Classroom (first time only)
   - Repository automatically created: `assignment-01-username`

3. **Clone and Setup**:
   ```bash
   # Clone assignment repository
   git clone https://github.com/datasci-217-fall-2024/assignment-01-username.git
   cd assignment-01-username
   
   # Create virtual environment
   python -m venv assignment_env
   source assignment_env/bin/activate  # Linux/Mac
   # or assignment_env\Scripts\activate  # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run initial tests to verify setup
   python -m pytest test_assignment.py --tb=short
   ```

4. **Development Process**:
   ```bash
   # Work on assignment
   # Make regular commits
   git add .
   git commit -m "Implement data generation function"
   git push origin main
   
   # Check automated feedback
   # Visit repository on GitHub to see Actions results
   ```

### Student Testing Guide

Create `TESTING_GUIDE.md` for students:

```markdown
# Testing Your Assignment

## Quick Validation
```bash
# Run basic structure check
python test_assignment.py

# Run full test suite
python -m pytest test_assignment.py -v

# Run specific test category
python -m pytest test_assignment.py::TestProjectStructure -v
```

## Understanding Test Results

| Test Status | Meaning | Action |
|-------------|---------|--------|
| ✅ PASSED | Test successful | Continue development |
| ❌ FAILED | Test failed | Fix implementation |
| ⚠️ SKIPPED | Test skipped | Check prerequisites |
| 🔄 RUNNING | Test in progress | Wait for completion |

## Common Issues and Solutions

**ImportError: No module named 'xyz'**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**FileNotFoundError**  
- Check file paths and directory structure
- Ensure all required files exist

**AssertionError in tests**
- Read test failure message carefully
- Check function return types and values
```

## Instructor Management

### Assignment Creation Workflow

1. **Prepare Assignment Template**:
   ```bash
   cd assignments/lecture_01/assignment
   
   # Test template locally
   python -m pytest test_assignment.py -v
   
   # Create template repository
   # Upload files to GitHub template repo
   ```

2. **Create GitHub Classroom Assignment**:
   - Go to GitHub Classroom dashboard
   - Click "New Assignment" 
   - Configure assignment settings:

   ```
   Assignment Title: Assignment 1: Python Fundamentals
   Repository Prefix: assignment-01
   Template Repository: datasci-217-fall-2024/assignment-01-template
   Deadline: 2024-02-15 23:59 EST
   Points: 100
   Auto-accept: Disabled
   Feedback: Automatic via tests
   ```

3. **Enable Autograding**:
   - Add autograding tests from template
   - Configure point distribution
   - Set timeout and resource limits

4. **Distribute Assignment**:
   - Copy invitation link
   - Share via LMS or email
   - Monitor acceptance in dashboard

### Monitoring and Feedback

**GitHub Classroom Dashboard Features**:
- Student acceptance status
- Submission timestamps  
- Automated test results
- Grade distribution analytics

**Bulk Operations**:
```bash
# Clone all student repositories for local review
gh classroom clone student-repos assignment-01

# Run custom analysis across all submissions
python scripts/analyze_submissions.py --assignment assignment-01
```

**Providing Additional Feedback**:
```bash
# Add instructor comments to specific commits
gh repo view student-repo --json url
# Use GitHub's code review features
```

### Grade Export and LMS Integration

**Export Grades**:
```python
# scripts/export_grades.py
import csv
import requests

def export_classroom_grades(assignment_id, output_file):
    """Export grades from GitHub Classroom to CSV."""
    # Use GitHub Classroom API to fetch grades
    grades = fetch_grades_from_classroom(assignment_id)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['student_id', 'github_username', 'score', 'feedback'])
        writer.writeheader()
        
        for grade in grades:
            writer.writerow({
                'student_id': grade['identifier'],
                'github_username': grade['username'], 
                'score': grade['points_awarded'],
                'feedback': grade['feedback']
            })
```

**LMS Integration** (Canvas example):
```python
# scripts/canvas_integration.py
from canvasapi import Canvas

def upload_grades_to_canvas(csv_file, assignment_id):
    """Upload grades from GitHub Classroom to Canvas."""
    canvas = Canvas(CANVAS_URL, CANVAS_TOKEN)
    course = canvas.get_course(COURSE_ID)
    assignment = course.get_assignment(assignment_id)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            student = course.get_user(row['student_id'], 'sis_user_id')
            submission = assignment.get_submission(student.id)
            submission.edit(posted_grade=row['score'], comment=row['feedback'])
```

## Troubleshooting

### Common Student Issues

**Issue: "Tests failing immediately"**
```
Solution:
1. Check Python version compatibility
2. Verify virtual environment setup
3. Ensure all requirements installed
4. Check file permissions and paths
```

**Issue: "GitHub Actions not running"**
```
Solution:
1. Check .github/workflows/classroom.yml exists
2. Verify repository has Actions enabled
3. Check for syntax errors in YAML
4. Ensure push triggers are correct
```

**Issue: "Import errors in tests"**
```
Solution:
1. Verify Python path setup in tests
2. Check module structure matches expectations
3. Ensure __init__.py files where needed
4. Validate relative import paths
```

### Instructor Troubleshooting

**Issue: "Template repository not appearing"**
```
Solution:
1. Ensure template repo is public
2. Check "Template repository" setting enabled
3. Verify organization membership
4. Wait for GitHub sync (can take minutes)
```

**Issue: "Autograding not working consistently"**
```  
Solution:
1. Test GitHub Actions locally with act
2. Check timeout settings (increase if needed)
3. Verify test dependencies in requirements.txt
4. Monitor resource usage in Actions logs
```

**Issue: "Student repositories not accessible"**
```
Solution: 
1. Check organization membership
2. Verify repository permissions
3. Ensure Classroom access permissions
4. Check student acceptance status
```

### Debugging Test Failures

**Local Testing**:
```bash
# Debug test failures locally
python -m pytest test_assignment.py -v --tb=long

# Run specific failing test
python -m pytest test_assignment.py::TestClass::test_method -v -s

# Run tests with debugging output
python -m pytest test_assignment.py --pdb
```

**GitHub Actions Debugging**:
```yaml
# Add debugging steps to classroom.yml
- name: Debug Environment
  run: |
    echo "Python version: $(python --version)"
    echo "Working directory: $(pwd)"
    echo "Files present:"
    ls -la
    echo "Python path:"
    python -c "import sys; print('\n'.join(sys.path))"
```

## Best Practices

### Assignment Design
- Keep initial assignments simple with clear requirements
- Provide comprehensive starter code and examples
- Include both positive and negative test cases
- Test your template thoroughly before distribution

### Test Design  
- Write tests that provide meaningful feedback
- Include partial credit opportunities
- Test edge cases and error conditions
- Provide clear assertion messages

### Student Support
- Create detailed setup instructions
- Provide troubleshooting guides
- Use discussion forums for common issues
- Offer office hours for hands-on help

### Quality Assurance
- Test assignments in fresh environments
- Verify all dependencies are specified
- Check cross-platform compatibility
- Monitor assignment completion rates

## Security Considerations

### Repository Access
- Keep solution files in private repositories
- Use GitHub's branch protection rules
- Monitor for academic integrity violations
- Limit access to assignment templates

### Automated Grading Safety
- Set resource limits in GitHub Actions
- Validate input parameters in tests  
- Sandbox test execution environments
- Monitor for malicious code attempts

---

This integration guide provides a complete framework for using GitHub Classroom with DataSci 217 assignments. The automated testing and grading system reduces manual grading effort while providing immediate feedback to students, improving the learning experience and maintaining consistency in evaluation.