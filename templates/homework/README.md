# Homework Assignment Auto-Grading Framework

A comprehensive framework for creating, testing, and automatically grading homework assignments using GitHub Actions, pytest, and modern development practices.

## 🚀 Quick Start

### For Instructors

1. **Copy framework to your course repository:**
   ```bash
   cp -r templates/homework/* your-course-repo/
   ```

2. **Set up GitHub Classroom assignment:**
   - Use the provided assignment template
   - Configure auto-grading with the GitHub Actions workflows

3. **Customize for your assignment:**
   - Edit test templates to match your requirements
   - Modify the assignment instructions
   - Update grading rubrics and point distributions

### For Students

1. **Accept the assignment** via GitHub Classroom link
2. **Clone your repository** and run the setup verification:
   ```bash
   git clone [your-repo-url]
   cd [repo-name]
   python verify_setup.py
   ```

3. **Implement your solution** in `main.py`
4. **Test locally** before submitting:
   ```bash
   python -m pytest tests/ -v
   ```

5. **Submit by pushing** to the main branch - auto-grading runs automatically!

## 📁 Framework Structure

```
homework/
├── test_templates/              # Pytest test patterns
│   ├── base_test_template.py   # Common testing utilities
│   ├── function_test_template.py # Function-based assignments
│   ├── data_processing_test_template.py # Data science assignments
│   ├── file_io_test_template.py # File I/O operations
│   └── cli_test_template.py    # Command-line tools
├── github_workflows/            # GitHub Actions workflows
│   ├── autograder.yml          # Main auto-grading pipeline
│   ├── submission_validator.yml # Pre-submission validation
│   ├── advanced_grading.yml    # Advanced analysis and grading
│   └── git_workflow_check.yml  # Git workflow verification
├── assignment_templates/        # Assignment instruction templates
│   └── assignment_template.md  # Complete assignment template
├── starter_code/               # Student starter code
│   ├── main.py                # Main implementation template
│   ├── README.md              # Student documentation template
│   ├── requirements.txt       # Python dependencies
│   ├── verify_setup.py        # Setup verification script
│   └── pytest.ini           # Pytest configuration
└── examples/                  # Working examples
    ├── function_assignment/   # Function-based example
    ├── data_processing/      # Data processing example
    └── cli_tools/           # CLI tools example
```

## 🧪 Testing Framework Features

### Comprehensive Test Coverage

- **Function Testing**: Unit tests for individual functions
- **Edge Case Testing**: Boundary conditions and unusual inputs
- **Error Handling**: Exception handling and input validation
- **Integration Testing**: How components work together
- **Performance Testing**: Speed and memory usage validation

### Grading Categories

| Category | Default Points | Description |
|----------|---------------|-------------|
| **Function Tests** | 40 | Core functionality implementation |
| **Edge Cases** | 20 | Boundary conditions and robustness |
| **Error Handling** | 15 | Proper exception handling |
| **Code Quality** | 15 | Style, documentation, maintainability |
| **Documentation** | 10 | README, docstrings, comments |

### Built-in Utilities

```python
from base_test_template import TestUtils, assert_function_exists

# Safe module importing with error handling
student_module = TestUtils.safe_import('main', required_functions=['func1', 'func2'])

# Timeout protection for student code
result = TestUtils.timeout_test(lambda: student_function(args), timeout=5.0)

# Temporary workspace for file operations
with TestUtils.temp_directory() as temp_dir:
    # Test file I/O operations safely
```

## ⚙️ GitHub Actions Workflows

### 1. Submission Validator (`submission_validator.yml`)

**Triggers:** Push to any branch, pull requests
**Purpose:** Quick validation of submission structure

- ✅ File structure validation
- ✅ Python syntax checking
- ✅ Basic security scan
- ✅ Code quality checks (formatting, style)
- ✅ Jupyter notebook validation

### 2. Auto-Grader (`autograder.yml`)

**Triggers:** Push to main/submit branches
**Purpose:** Complete assignment grading

- ✅ Dependency installation
- ✅ Test execution with timeout protection
- ✅ Grade calculation and reporting
- ✅ Automatic feedback via PR comments
- ✅ Grade artifact generation

### 3. Advanced Grading (`advanced_grading.yml`)

**Triggers:** Push to submit branch, manual dispatch
**Purpose:** Comprehensive analysis and grading

- ✅ Code complexity analysis
- ✅ Test coverage reporting
- ✅ Performance benchmarking
- ✅ Memory usage analysis
- ✅ Security vulnerability scanning
- ✅ Multi-matrix testing (unit, integration, performance)

### 4. Git Workflow Check (`git_workflow_check.yml`)

**Triggers:** All pushes and pull requests
**Purpose:** Validate development practices

- ✅ Commit history analysis
- ✅ Commit message quality assessment
- ✅ Branch structure validation
- ✅ Development timing patterns
- ✅ Workflow best practices feedback

## 🔧 Customization Guide

### Adapting Tests for Your Assignment

1. **Copy the appropriate test template:**
   ```bash
   cp test_templates/function_test_template.py tests/test_homework.py
   ```

2. **Modify the required functions list:**
   ```python
   REQUIRED_FUNCTIONS = [
       'your_function_1',
       'your_function_2',
       # Add your functions here
   ]
   ```

3. **Update test cases:**
   ```python
   def test_your_function(self):
       func = self.student_module.your_function
       test_cases = [
           (input1, expected_output1),
           (input2, expected_output2),
       ]
       # Implement your test logic
   ```

4. **Adjust point distribution:**
   ```python
   TestConfig.POINTS = {
       'function_tests': 50,  # Adjust as needed
       'edge_cases': 25,
       'error_handling': 15,
       'code_quality': 10,
   }
   ```

### Customizing GitHub Actions

1. **Modify workflow triggers:**
   ```yaml
   on:
     push:
       branches: [ main, submit, your-branch ]
   ```

2. **Adjust timeout and resource limits:**
   ```yaml
   timeout-minutes: 15  # Adjust for complex assignments
   ```

3. **Add custom validation steps:**
   ```yaml
   - name: Custom validation
     run: |
       # Add your custom validation logic
   ```

4. **Configure grade thresholds:**
   ```yaml
   env:
     PASSING_THRESHOLD: 70  # Minimum passing grade
   ```

## 📚 Assignment Types and Examples

### 1. Function-Based Assignments

**Best for:** Introduction to programming, algorithm implementation
**Example:** Mathematical functions, string processing, basic data structures

```python
def calculate_average(numbers: List[float]) -> float:
    """Calculate arithmetic mean of numbers."""
    # Student implements this
```

**Test Pattern:**
- Basic functionality tests
- Edge cases (empty lists, single values)
- Error handling (invalid inputs, type checking)
- Performance with large datasets

### 2. Data Processing Assignments

**Best for:** Data science, pandas/numpy practice, file handling
**Example:** CSV analysis, data cleaning, statistical calculations

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data."""
    # Student implements data cleaning logic
```

**Test Pattern:**
- Data loading and validation
- Cleaning effectiveness
- Analysis accuracy
- File I/O operations
- Data integrity preservation

### 3. Command-Line Interface Assignments

**Best for:** System programming, argument parsing, file processing
**Example:** Text analyzer, file converter, data processor

```python
def main():
    parser = argparse.ArgumentParser()
    # Student implements CLI interface
```

**Test Pattern:**
- Argument parsing
- Help system functionality
- File processing capabilities
- Error handling and user feedback
- Output format validation

### 4. File I/O Assignments

**Best for:** File handling, data formats, persistence
**Example:** JSON/CSV processors, configuration parsers, data exporters

```python
def read_json_file(file_path: str) -> dict:
    """Read and parse JSON file."""
    # Student implements file reading
```

**Test Pattern:**
- File reading/writing operations
- Format validation
- Error handling (missing files, permissions)
- Data integrity checks
- Edge cases (empty files, large files)

## 🎯 Best Practices

### For Instructors

1. **Start Simple**: Begin with function-based assignments, progress to complex projects
2. **Clear Specifications**: Provide detailed function signatures and examples
3. **Comprehensive Testing**: Cover happy path, edge cases, and error conditions
4. **Immediate Feedback**: Use auto-grading for fast student feedback
5. **Iterative Improvement**: Analyze common student mistakes to improve tests

### For Students

1. **Read Carefully**: Understand requirements before coding
2. **Test Locally**: Run tests before submitting
3. **Commit Often**: Make frequent, descriptive commits
4. **Handle Errors**: Implement proper error handling
5. **Document Code**: Write clear docstrings and comments

### Assignment Design Tips

1. **Incremental Complexity**: Start with basic functions, build to complete applications
2. **Real-World Relevance**: Use practical examples and realistic data
3. **Multiple Solutions**: Allow different valid approaches
4. **Clear Rubrics**: Students should understand how they're graded
5. **Accessibility**: Ensure assignments work across different environments

## 🔍 Debugging and Troubleshooting

### Common Student Issues

**"Tests are failing but my code works"**
- Check function signatures exactly match specifications
- Verify return types and data formats
- Test with provided examples first

**"Import errors when running tests"**
- Ensure all required files are present
- Check file names match exactly
- Verify Python path configuration

**"Auto-grader not running"**
- Confirm push to correct branch (usually `main`)
- Check Actions tab for error messages
- Verify repository permissions

### Common Instructor Issues

**"Tests are too strict/lenient"**
- Adjust timeout values in test configuration
- Modify point distributions in TestConfig
- Add more comprehensive edge case testing

**"Students getting unexpected errors"**
- Test framework with various Python versions
- Verify dependency compatibility
- Check cross-platform file path handling

### Debugging Tools

1. **Local test execution:**
   ```bash
   python -m pytest tests/ -v --tb=long
   ```

2. **Coverage analysis:**
   ```bash
   python -m pytest tests/ --cov=main --cov-report=html
   ```

3. **Performance profiling:**
   ```bash
   python -m pytest tests/ --benchmark-only
   ```

## 📈 Advanced Features

### Custom Metrics

Add domain-specific grading criteria:

```python
def test_algorithm_efficiency(self):
    """Test algorithmic efficiency beyond just correctness."""
    large_input = list(range(10000))
    
    start_time = time.time()
    result = self.student_module.sort_function(large_input)
    execution_time = time.time() - start_time
    
    # Award extra points for efficient implementation
    if execution_time < 0.1:
        self.award_points('performance_bonus', 5)
```

### Integration with Learning Management Systems

Export grades in common LMS formats:

```python
def generate_lms_export(test_results: dict) -> str:
    """Generate CSV export for LMS integration."""
    return f"{student_id},{final_score},{max_score},{timestamp}"
```

### Plagiarism Detection Integration

Add similarity checking:

```yaml
- name: Check for plagiarism
  run: |
    # Integration with plagiarism detection tools
    python check_similarity.py --threshold 0.8
```

## 🤝 Contributing

We welcome contributions to improve this framework! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with clear description

### Development Setup

```bash
git clone https://github.com/your-repo/homework-framework
cd homework-framework
pip install -r dev-requirements.txt
python -m pytest tests/ -v
```

## 📄 License

This framework is released under the MIT License. See LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the examples and test templates
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join our community discussions
- **Email**: Contact maintainers at [email]

## 🏆 Acknowledgments

Built with contributions from educators and students who believe in:
- Automated, fair grading
- Immediate feedback for learning
- Best practices in software development
- Accessible education technology

---

**Happy Teaching and Learning! 🎓**