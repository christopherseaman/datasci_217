# Homework Assignment Template

## Assignment Information

- **Course**: [Course Code and Name]
- **Assignment**: [Assignment Number and Title]
- **Due Date**: [Date and Time]
- **Points**: [Total Points] points
- **Submission Method**: GitHub Classroom

## Learning Objectives

By completing this assignment, you will:
- [ ] [Objective 1]
- [ ] [Objective 2]
- [ ] [Objective 3]

## Overview

[Brief description of what students will build/implement]

## Prerequisites

Before starting this assignment, ensure you have:
- [ ] Python 3.8+ installed
- [ ] Required libraries installed (see `requirements.txt`)
- [ ] Git configured with your credentials
- [ ] Understanding of [relevant concepts]

## Setup Instructions

### 1. Accept the Assignment
1. Click the GitHub Classroom link provided in [LMS/Email]
2. Accept the assignment to create your personal repository
3. Clone your repository to your local machine:
   ```bash
   git clone [your-repo-url]
   cd [repo-name]
   ```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Verify Setup
Run the setup verification script:
```bash
python verify_setup.py
```

## Requirements

### Part 1: [Requirement Title] (X points)

[Detailed description of what needs to be implemented]

**Function Signature:**
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    
    Returns:
        Description of return value
    
    Raises:
        SpecificError: When this error might occur
    """
    pass
```

**Requirements:**
- [ ] Implement the function according to the signature
- [ ] Handle edge cases (empty inputs, invalid types, etc.)
- [ ] Include proper error handling
- [ ] Add comprehensive docstrings
- [ ] Follow PEP 8 style guidelines

**Example Usage:**
```python
>>> result = function_name("example", 42)
>>> print(result)
Expected output here
```

### Part 2: [Requirement Title] (Y points)

[Continue with additional parts...]

## Testing Your Solution

### Running Tests Locally
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_part1.py -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

### Understanding Test Output
- ✅ **Green (PASSED)**: Test passed successfully
- ❌ **Red (FAILED)**: Test failed - check the error message
- ⚠️  **Yellow (SKIPPED)**: Test was skipped (usually dependency issues)

### Test Categories
1. **Basic Tests**: Test core functionality
2. **Edge Cases**: Test boundary conditions and unusual inputs
3. **Error Handling**: Test proper exception handling
4. **Integration Tests**: Test how components work together

## Submission Guidelines

### What to Submit
Your repository should contain:
- [ ] `main.py` - Your main implementation
- [ ] `README.md` - Documentation of your solution
- [ ] `requirements.txt` - Dependencies (if you added any)
- [ ] Any additional modules you created

### Submission Process
1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Complete homework assignment - [brief description]"
   ```

2. **Push to GitHub**:
   ```bash
   git push origin main
   ```

3. **Verify submission**:
   - Check that all files are visible on GitHub
   - Ensure the auto-grader runs successfully
   - Review the feedback in the Actions tab

### Late Submission Policy
[Insert course-specific late policy]

## Grading Rubric

| Category | Points | Criteria |
|----------|--------|----------|
| **Correctness** | [X] | Code produces correct outputs for all test cases |
| **Code Quality** | [Y] | Clean, readable code following Python conventions |
| **Error Handling** | [Z] | Appropriate handling of edge cases and errors |
| **Documentation** | [W] | Clear docstrings, comments, and README |
| **Testing** | [V] | All tests pass, good test coverage |

### Grade Breakdown
- **90-100%**: Exceptional work, exceeds expectations
- **80-89%**: Good work, meets all requirements
- **70-79%**: Satisfactory work, meets most requirements
- **60-69%**: Below expectations, missing key components
- **Below 60%**: Unsatisfactory, significant issues

## Common Issues and Troubleshooting

### Issue: "Module not found" error
**Solution**: Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Tests are failing
**Solution**: 
1. Read the test error messages carefully
2. Check that your function signatures match exactly
3. Verify your return types and values
4. Test with the provided examples

### Issue: Auto-grader not running
**Solution**:
1. Check that you've pushed to the `main` branch
2. Verify all required files are present
3. Check the Actions tab for error messages

### Issue: Style/formatting warnings
**Solution**:
```bash
# Auto-format your code
black *.py

# Check for style issues
flake8 *.py
```

## Getting Help

### Resources
- [Course documentation/textbook]
- [Python official documentation](https://docs.python.org/)
- [Relevant library documentation]

### Office Hours
- **When**: [Days and times]
- **Where**: [Location or online link]
- **How to prepare**: Come with specific questions and code examples

### Discussion Forum
- Post questions on [forum/platform]
- Search existing questions first
- Include relevant code snippets and error messages
- Be respectful and help others when you can

## Academic Integrity

This is an individual assignment. You may:
- ✅ Consult course materials and documentation
- ✅ Use online resources for general programming concepts
- ✅ Discuss general approaches with classmates
- ✅ Ask questions in office hours or forums

You may NOT:
- ❌ Copy code from classmates or online sources
- ❌ Share your complete solutions with others
- ❌ Use AI tools to generate code solutions
- ❌ Submit work that is not your own

Violations will result in academic penalties according to the course policy.

## Extension and Extra Credit Opportunities

### Optional Enhancements ([Bonus Points])
If you complete the basic requirements, consider these enhancements:
- [ ] [Enhancement 1]: [Description] (+X points)
- [ ] [Enhancement 2]: [Description] (+Y points)
- [ ] [Enhancement 3]: [Description] (+Z points)

**Note**: Bonus points are only awarded if the basic requirements are fully met.

## Reflection Questions

After completing the assignment, consider these questions:
1. What was the most challenging part of this assignment?
2. What did you learn that you didn't know before?
3. How would you improve your solution if you had more time?
4. What resources were most helpful to you?

Include your responses in your README.md file.

---

**Happy coding! 🚀**

*If you have questions about this assignment, please post them in the course forum or attend office hours.*