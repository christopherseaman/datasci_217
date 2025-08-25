# Instructor Guide: Homework Auto-Grading Framework

This guide provides comprehensive instructions for instructors on how to set up, customize, and deploy the homework auto-grading framework.

## 🚀 Quick Setup (15 minutes)

### Step 1: Repository Setup

1. **Create a new repository** for your assignment (or use existing one)
2. **Copy the framework** into your repository:
   ```bash
   cp -r templates/homework/* your-assignment-repo/
   ```
3. **Initialize GitHub Actions** by ensuring workflows are in `.github/workflows/`

### Step 2: GitHub Classroom Configuration

1. **Create new assignment** in GitHub Classroom
2. **Choose template repository** (your repo with the framework)
3. **Enable autograding** - GitHub will automatically detect the workflows
4. **Set deadline** and other assignment parameters

### Step 3: Test the Setup

1. **Accept your own assignment** to test student experience
2. **Push sample solution** to verify auto-grading works
3. **Check Actions tab** for successful workflow execution

## 🔧 Customization Workflow

### Assignment Type Selection

Choose the appropriate template based on your assignment type:

| Assignment Type | Use Template | Best For |
|----------------|--------------|----------|
| **Basic Functions** | `function_test_template.py` | Programming fundamentals, algorithms |
| **Data Analysis** | `data_processing_test_template.py` | pandas, numpy, statistics |
| **File Operations** | `file_io_test_template.py` | Reading/writing files, data formats |
| **CLI Tools** | `cli_test_template.py` | Command-line programs, argument parsing |

### Detailed Customization Steps

#### 1. Modify Test Template

```bash
# Copy appropriate template
cp test_templates/function_test_template.py tests/test_homework.py
```

Edit the test file:

```python
class TestHomework(TestHomeworkBase):
    # Update required functions for your assignment
    REQUIRED_FUNCTIONS = [
        'calculate_mean',      # Your function 1
        'find_outliers',       # Your function 2
        'generate_report'      # Your function 3
    ]
    
    def test_calculate_mean_basic(self):
        """Test the calculate_mean function."""
        func = self.student_module.calculate_mean
        
        # Define your test cases
        test_cases = [
            ([1, 2, 3, 4, 5], 3.0),        # Normal case
            ([10], 10.0),                   # Single element
            ([2.5, 3.5], 3.0),            # Floating point
        ]
        
        total_earned = 0
        points_per_case = TestConfig.POINTS['function_tests'] // len(test_cases)
        
        for inputs, expected in test_cases:
            try:
                result = TestUtils.timeout_test(lambda: func(inputs))
                assert_close_enough(result, expected)
                total_earned += points_per_case
            except Exception as e:
                pytest.fail(f"Failed for input {inputs}: {e}")
        
        self.award_points('function_tests', total_earned,
                         description="Basic mean calculation")
```

#### 2. Update Point Distribution

Modify the point allocation in your test file:

```python
class TestConfig:
    POINTS = {
        'function_tests': 50,    # Adjust based on complexity
        'edge_cases': 20,        # Important for robust code
        'error_handling': 15,    # Good programming practices
        'code_quality': 10,      # Code style and documentation
        'documentation': 5       # README and docstrings
    }
```

#### 3. Customize Assignment Instructions

Edit `assignment_templates/assignment_template.md`:

```markdown
# Assignment 3: Statistical Analysis Functions

## Due Date: [Your Date]
## Points: 100 points

## Overview
Implement functions for basic statistical analysis including mean, median, 
mode, and outlier detection.

## Required Functions

### `calculate_mean(numbers: List[float]) -> float`
Calculate the arithmetic mean of a list of numbers.

**Examples:**
```python
>>> calculate_mean([1, 2, 3, 4, 5])
3.0
```

**Requirements:**
- Handle empty lists by raising `ValueError`
- Validate that all inputs are numeric
- Return result as float

### [Continue with other functions...]
```

#### 4. Modify GitHub Actions (if needed)

Most assignments work with default workflows, but you can customize:

**Adjust timeout for complex assignments:**
```yaml
# In .github/workflows/autograder.yml
jobs:
  autograder:
    timeout-minutes: 20  # Increase for complex assignments
```

**Add assignment-specific validation:**
```yaml
- name: Check assignment-specific requirements
  run: |
    # Check for required imports
    if ! grep -q "import numpy" main.py; then
      echo "::error::Assignment requires numpy import"
      exit 1
    fi
```

**Modify grade thresholds:**
```yaml
- name: Set passing threshold
  run: |
    # Change passing grade from 70 to your preference
    if [ "$final_score" -lt 80 ]; then
      echo "❌ Assignment score below 80%"
      exit 1
    fi
```

#### 5. Update Starter Code

Provide helpful starter code in `starter_code/main.py`:

```python
def calculate_mean(numbers: List[float]) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Args:
        numbers: List of numeric values
    
    Returns:
        The arithmetic mean as a float
    
    Raises:
        ValueError: If the list is empty
        TypeError: If the list contains non-numeric values
    """
    # TODO: Implement this function
    pass
```

## 📊 Grading Strategy Design

### Point Distribution Philosophy

**Conservative Approach (Recommended for Beginners):**
- Function Tests: 60%
- Error Handling: 20%
- Edge Cases: 15%
- Documentation: 5%

**Balanced Approach (Intermediate Students):**
- Function Tests: 40%
- Edge Cases: 25%
- Error Handling: 20%
- Code Quality: 10%
- Documentation: 5%

**Advanced Approach (Advanced Students):**
- Function Tests: 35%
- Code Quality: 25%
- Edge Cases: 20%
- Error Handling: 15%
- Documentation: 5%

### Creating Effective Test Cases

#### 1. Basic Functionality Tests
```python
# Test the happy path - normal expected usage
test_cases = [
    ([1, 2, 3], 2.0),           # Odd number of elements
    ([1, 2, 3, 4], 2.5),        # Even number of elements
    ([5], 5.0),                 # Single element
]
```

#### 2. Edge Case Tests
```python
# Test boundary conditions
edge_cases = [
    ([], "empty list"),                    # Empty input
    ([0], "single zero"),                  # Zero values
    ([1e10, 1e10], "very large numbers"), # Large numbers
    ([1e-10, 1e-10], "very small numbers"), # Small numbers
]
```

#### 3. Error Handling Tests
```python
# Test that proper exceptions are raised
error_cases = [
    ("not a list", TypeError),
    ([1, 2, "three"], TypeError),
    ([1, 2, float('nan')], ValueError),
]

for invalid_input, expected_error in error_cases:
    with pytest.raises(expected_error):
        student_function(invalid_input)
```

## 🛠️ Advanced Customizations

### Adding Performance Requirements

```python
@pytest.mark.performance
def test_performance_requirements(self):
    """Test that functions meet performance requirements."""
    import time
    
    # Create large dataset
    large_dataset = list(range(100000))
    
    # Test execution time
    start_time = time.time()
    result = self.student_module.calculate_mean(large_dataset)
    execution_time = time.time() - start_time
    
    # Award points based on performance
    if execution_time < 0.1:
        self.award_points('performance', 10, description="Excellent performance")
    elif execution_time < 0.5:
        self.award_points('performance', 5, description="Good performance")
```

### Custom Validation Rules

```python
def test_coding_standards(self):
    """Test adherence to coding standards."""
    import ast
    import inspect
    
    # Check function length
    for func_name in self.REQUIRED_FUNCTIONS:
        func = getattr(self.student_module, func_name)
        source_lines = len(inspect.getsourcelines(func)[0])
        
        if source_lines > 50:
            self.award_points('code_quality', -5, 
                            description="Function too long")
    
    # Check for magic numbers
    with open('main.py', 'r') as f:
        tree = ast.parse(f.read())
    
    magic_numbers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Num) and node.n not in [0, 1, 2]:
            magic_numbers.append(node.n)
    
    if len(magic_numbers) > 5:
        self.award_points('code_quality', -2,
                         description="Too many magic numbers")
```

### Integration with External Tools

#### Code Quality Integration
```yaml
- name: Check code quality with additional tools
  run: |
    pip install bandit safety
    
    # Security scan
    bandit -r . -f json -o security_report.json
    
    # Dependency vulnerability scan
    safety check --json --output safety_report.json
```

#### Plagiarism Detection
```yaml
- name: Basic similarity check
  run: |
    # Simple check for suspicious patterns
    if grep -r "stackoverflow.com\|github.com" *.py; then
      echo "::warning::Found external references in code"
    fi
```

## 📋 Assignment Management Workflows

### Pre-Assignment Checklist

- [ ] Test framework copied and customized
- [ ] All required functions defined in tests
- [ ] Point distribution configured appropriately  
- [ ] Assignment instructions updated
- [ ] Starter code provides helpful template
- [ ] GitHub Actions workflows tested
- [ ] Sample solution tested end-to-end
- [ ] Due dates and submission requirements set

### During Assignment Period

**Monitor common issues:**
```bash
# Check for frequent test failures
gh run list --workflow=autograder --json | jq '.[] | select(.conclusion=="failure") | .html_url'

# Watch for student questions patterns
grep -r "error" student-repos/*/README.md
```

**Provide clarifications:**
- Post announcements for common student questions
- Update test cases if requirements were unclear
- Adjust point distributions if needed

### Post-Assignment Analysis

**Generate grade reports:**
```python
# Extract grades from all student repositories
def collect_grades():
    grades = []
    for repo in student_repos:
        # Get final grade from last workflow run
        grade_data = get_workflow_results(repo)
        grades.append(grade_data)
    return grades
```

**Analyze common issues:**
```python
def analyze_common_errors():
    """Analyze what students struggled with most."""
    error_patterns = {}
    
    for repo in student_repos:
        test_results = get_test_results(repo)
        for failed_test in test_results['failed']:
            test_name = failed_test['name']
            error_patterns[test_name] = error_patterns.get(test_name, 0) + 1
    
    return sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
```

## 🎓 Pedagogical Best Practices

### Progressive Difficulty

**Week 1-2: Simple Functions**
```python
# Start with basic input/output
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

**Week 3-4: Error Handling**
```python
# Add validation and error handling
def calculate_grade(points: int, total: int) -> float:
    if total <= 0:
        raise ValueError("Total points must be positive")
    return (points / total) * 100
```

**Week 5-6: Data Structures**
```python
# Work with lists and dictionaries
def analyze_scores(scores: List[int]) -> Dict[str, float]:
    return {
        'mean': sum(scores) / len(scores),
        'min': min(scores),
        'max': max(scores)
    }
```

### Scaffolded Learning

1. **Provide function signatures** - students focus on logic, not interface design
2. **Include docstring templates** - teach documentation habits early
3. **Give specific examples** - clear expectations reduce confusion
4. **Start with guided practice** - first assignments have more hints

### Assessment Alignment

**Formative Assessments:**
- Quick auto-graded exercises with immediate feedback
- Focus on specific skills (error handling, edge cases)
- Allow multiple attempts

**Summative Assessments:**
- Comprehensive projects combining multiple skills
- Include code quality and documentation requirements
- Single submission with detailed feedback

## 🐛 Troubleshooting Guide

### Common Setup Issues

**Framework not auto-grading:**
1. Check `.github/workflows/` directory exists in repository
2. Verify workflow files have correct permissions
3. Ensure GitHub Actions are enabled for the repository

**Tests failing unexpectedly:**
1. Test the framework with a known working solution
2. Check Python version compatibility (use 3.8+ in workflows)
3. Verify all dependencies are listed in `requirements.txt`

**Students can't run tests locally:**
1. Provide clear setup instructions in README
2. Include `verify_setup.py` script for environment checking
3. Create video walkthrough of local testing process

### Performance Issues

**Workflows timing out:**
1. Increase timeout in workflow configuration
2. Optimize test cases for speed
3. Use more efficient testing patterns

**Large artifact storage:**
1. Limit artifact retention days
2. Compress reports before uploading
3. Only save essential debugging information

### Student Experience Issues

**Confusing error messages:**
1. Write custom assertion messages
2. Provide links to help documentation
3. Use progressive disclosure (simple errors first)

**Unclear requirements:**
1. Include more examples in assignment instructions
2. Provide sample input/output pairs
3. Create FAQ document for common questions

## 📊 Analytics and Insights

### Grade Distribution Analysis

```python
def analyze_grade_distribution(grades):
    import matplotlib.pyplot as plt
    
    plt.hist(grades, bins=10, range=(0, 100))
    plt.xlabel('Grade (%)')
    plt.ylabel('Number of Students')
    plt.title('Grade Distribution')
    
    mean_grade = sum(grades) / len(grades)
    print(f"Mean grade: {mean_grade:.1f}%")
    print(f"Standard deviation: {statistics.stdev(grades):.1f}")
```

### Learning Outcome Assessment

Track which concepts students struggle with:

```python
def track_learning_outcomes():
    outcomes = {
        'error_handling': 0,
        'edge_cases': 0,
        'documentation': 0,
        'code_quality': 0
    }
    
    for student_result in all_results:
        for category, score in student_result.items():
            if score < 0.7:  # Less than 70% in category
                outcomes[category] += 1
    
    return outcomes
```

## 🔮 Advanced Features and Future Enhancements

### AI-Powered Code Review

```python
# Integration with code analysis AI
def ai_code_review(student_code):
    """Provide AI-generated feedback on code quality."""
    feedback = ai_service.analyze_code(student_code)
    return {
        'suggestions': feedback.suggestions,
        'best_practices': feedback.best_practices,
        'potential_bugs': feedback.potential_bugs
    }
```

### Adaptive Testing

```python
def adaptive_test_selection(student_history):
    """Select tests based on student's previous performance."""
    if student_history.get('error_handling_score', 0) < 0.7:
        return 'error_handling_focused_tests'
    elif student_history.get('performance_score', 0) < 0.7:
        return 'performance_focused_tests'
    else:
        return 'standard_tests'
```

### Real-time Collaboration Features

- Live coding sessions with shared auto-grading
- Peer review integration with auto-grading feedback
- Team project support with individual contribution tracking

---

## 📞 Support and Community

### Getting Help

- **Documentation Issues**: Open issue on GitHub
- **Feature Requests**: Submit enhancement proposal
- **Technical Support**: Email [support-email]
- **Community**: Join our educator Discord/Slack

### Contributing Back

Help improve the framework:

1. Share your successful assignment patterns
2. Contribute test templates for new domains
3. Report bugs and edge cases you discover
4. Write documentation improvements

---

**Happy Teaching! 🎓 Your students will appreciate the immediate feedback and clear expectations.**