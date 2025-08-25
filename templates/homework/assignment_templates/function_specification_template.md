# Function Specification Template

## Assignment: [ASSIGNMENT_NAME]
**Due Date**: [DUE_DATE]  
**Points**: [TOTAL_POINTS]  
**Course**: Data Science 217

---

## 📋 Assignment Overview

[BRIEF_DESCRIPTION_OF_ASSIGNMENT]

### Learning Objectives
By completing this assignment, you will:
- [LEARNING_OBJECTIVE_1]
- [LEARNING_OBJECTIVE_2]
- [LEARNING_OBJECTIVE_3]

### Prerequisites
- [PREREQUISITE_1]
- [PREREQUISITE_2]

---

## 🛠️ Required Functions

You must implement the following functions in `main.py`. Each function has specific requirements for arguments, return values, and error handling.

### Function 1: `[FUNCTION_NAME_1]`

```python
def function_name_1(param1: type, param2: type) -> return_type:
    """
    [BRIEF_DESCRIPTION]
    
    Args:
        param1 (type): [DESCRIPTION]
        param2 (type): [DESCRIPTION]
    
    Returns:
        return_type: [DESCRIPTION]
    
    Raises:
        ErrorType: [WHEN_THIS_ERROR_OCCURS]
    
    Examples:
        >>> function_name_1(example_input1, example_input2)
        expected_output
        >>> function_name_1(edge_case_input1, edge_case_input2)
        expected_edge_case_output
    """
    # YOUR IMPLEMENTATION HERE
    pass
```

**Requirements:**
- [REQUIREMENT_1]
- [REQUIREMENT_2]
- [REQUIREMENT_3]

**Test Cases:**
- **Basic functionality**: `function_name_1(normal_input)` should return `expected_output`
- **Edge cases**: Handle empty inputs, boundary conditions
- **Error handling**: Raise appropriate exceptions for invalid inputs

---

### Function 2: `[FUNCTION_NAME_2]`

```python
def function_name_2(param1: type) -> return_type:
    """
    [BRIEF_DESCRIPTION]
    
    Args:
        param1 (type): [DESCRIPTION]
    
    Returns:
        return_type: [DESCRIPTION]
    
    Raises:
        ErrorType: [WHEN_THIS_ERROR_OCCURS]
    
    Examples:
        >>> function_name_2(example_input)
        expected_output
    """
    # YOUR IMPLEMENTATION HERE
    pass
```

**Requirements:**
- [REQUIREMENT_1]
- [REQUIREMENT_2]

---

## 📊 Data Processing Functions (if applicable)

### Function 3: `[DATA_PROCESSING_FUNCTION]`

```python
def process_data(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Process and clean the provided dataset.
    
    Args:
        data (pd.DataFrame): Input dataset to process
        config (dict): Configuration parameters for processing
    
    Returns:
        pd.DataFrame: Cleaned and processed dataset
    
    Raises:
        ValueError: If data is empty or config is missing required keys
        KeyError: If required columns are missing from data
    
    Examples:
        >>> df = pd.DataFrame({'col1': [1, 2, None], 'col2': ['a', 'b', 'c']})
        >>> config = {'remove_nulls': True, 'normalize': False}
        >>> result = process_data(df, config)
        >>> len(result)
        2
    """
    # YOUR IMPLEMENTATION HERE
    pass
```

**Requirements:**
- Handle missing values according to config
- Validate input data structure
- Return processed data in same format
- Preserve original data (don't modify in-place)

---

## 🔧 Command Line Interface (if applicable)

### Main Function with CLI

```python
def main():
    """
    Main function with command-line interface.
    
    Should support:
    - --input: Input file path
    - --output: Output file path  
    - --verbose: Enable verbose output
    - --help: Show usage information
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='[DESCRIPTION]')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # YOUR IMPLEMENTATION HERE
```

**CLI Requirements:**
- Handle command-line arguments appropriately
- Provide helpful error messages
- Support `--help` flag
- Validate file paths and permissions

---

## 📝 Implementation Guidelines

### Code Quality Standards
- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Include comprehensive docstrings with Args, Returns, Raises, and Examples
- **Error Handling**: Implement proper exception handling with descriptive messages
- **Code Style**: Follow PEP 8 style guidelines
- **Testing**: Write your code to be easily testable

### Error Handling Patterns
```python
# Input validation
if not isinstance(param, expected_type):
    raise TypeError(f"Expected {expected_type}, got {type(param)}")

# Value validation  
if param < 0:
    raise ValueError("Parameter must be non-negative")

# File operations
try:
    with open(file_path, 'r') as f:
        data = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {file_path}")
except PermissionError:
    raise PermissionError(f"Permission denied: {file_path}")
```

### Performance Considerations
- For large datasets, consider memory usage
- Use appropriate data structures for the task
- Optimize time complexity where possible
- Handle edge cases efficiently

---

## 🧪 Testing Your Implementation

Before submitting, test your functions thoroughly:

```bash
# Run the provided tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=main

# Test your main function
python main.py --help
```

### Manual Testing Checklist
- [ ] All functions implement the required signature
- [ ] Functions return correct types
- [ ] Error handling works as specified
- [ ] Edge cases are handled properly
- [ ] Documentation is complete and accurate
- [ ] Code follows style guidelines

---

## 📁 File Structure

Your submission should include:

```
assignment/
├── main.py              # Your implementation
├── README.md           # Documentation and reflection
├── requirements.txt    # Dependencies (if any)
└── tests/             # Test files (provided)
    ├── test_homework.py
    └── conftest.py
```

---

## 🎯 Grading Rubric

| Category | Points | Description |
|----------|--------|-------------|
| **Function Implementation** | 40 | Core functionality works correctly |
| **Edge Case Handling** | 20 | Handles boundary conditions and unusual inputs |
| **Error Handling** | 15 | Proper exceptions with descriptive messages |
| **Code Quality** | 15 | Style, documentation, maintainability |
| **Documentation** | 10 | README, docstrings, comments |

### Grade Scale
- **90-100%**: Excellent - Exceeds expectations
- **80-89%**: Good - Meets all requirements  
- **70-79%**: Satisfactory - Meets most requirements
- **60-69%**: Needs Improvement - Missing key elements
- **< 60%**: Unsatisfactory - Major deficiencies

---

## 🚀 Submission Instructions

1. **Implement all required functions** in `main.py`
2. **Test thoroughly** using the provided test suite
3. **Document your solution** in `README.md`
4. **Commit and push** to your assignment repository
5. **Create a pull request** to trigger auto-grading

### Auto-Grading Process
- Your code will be automatically tested upon push to `main` branch
- Results will be posted as PR comments
- You can resubmit by pushing additional commits
- Final grading occurs at the deadline

---

## ❓ Getting Help

### Common Issues
- **Import errors**: Check your file structure and module names
- **Test failures**: Read the test output carefully for clues
- **Type errors**: Ensure you're using correct types and type hints

### Resources
- [Python Documentation](https://docs.python.org/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Type Hints Guide](https://docs.python.org/3/library/typing.html)
- Course forums and office hours

### Contact
- **Instructor**: [INSTRUCTOR_EMAIL]
- **TA**: [TA_EMAIL]  
- **Office Hours**: [OFFICE_HOURS]

---

## 🎓 Learning Reflection

After completing this assignment, write a brief reflection in your `README.md`:

1. **What was the most challenging part of this assignment?**
2. **What did you learn about [RELEVANT_TOPIC]?**
3. **How did you approach testing and debugging your code?**
4. **What would you do differently if you started over?**

---

**Good luck with your assignment! 🌟**