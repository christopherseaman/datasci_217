#!/usr/bin/env python3
"""
Create missing assignment files for DataSci 217 lectures
Generates main.py and test_assignment.py files for each lecture
"""

import os
from pathlib import Path

def create_main_py(lecture_num, topic_description):
    """Create main.py file for a lecture assignment"""
    content = f'''#!/usr/bin/env python3
"""
DataSci 217 - Lecture {lecture_num:02d} Assignment
{topic_description}

Complete the functions below according to the assignment requirements.
"""

def main():
    """
    Main function for Lecture {lecture_num:02d} assignment
    Add your implementation here
    """
    print(f"DataSci 217 - Lecture {lecture_num:02d} Assignment")
    print(f"Topic: {topic_description}")

    # TODO: Implement assignment requirements
    pass

if __name__ == "__main__":
    main()
'''
    return content

def create_test_assignment_py(lecture_num, topic_description):
    """Create test_assignment.py file for a lecture"""
    content = f'''#!/usr/bin/env python3
"""
DataSci 217 - Lecture {lecture_num:02d} Assignment Tests
{topic_description}

Test cases for validating assignment completion.
"""

import pytest
import sys
from pathlib import Path

# Add the assignment directory to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import main
except ImportError:
    print("Error: Could not import main.py")
    sys.exit(1)

class TestLecture{lecture_num:02d}:
    """Test cases for Lecture {lecture_num:02d} assignment"""

    def test_main_function_exists(self):
        """Test that main function exists"""
        assert hasattr(main, 'main'), "main() function not found in main.py"

    def test_main_function_callable(self):
        """Test that main function is callable"""
        assert callable(main.main), "main() function is not callable"

    def test_main_runs_without_error(self):
        """Test that main function runs without errors"""
        try:
            main.main()
        except Exception as e:
            pytest.fail(f"main() function raised an exception: {{e}}")

# Additional test cases will be added based on specific assignment requirements

if __name__ == "__main__":
    pytest.main([__file__])
'''
    return content

def main():
    """Create all missing assignment files"""

    # Lecture topics and descriptions
    lecture_topics = {
        1: "CLI Foundations and Python Basics",
        2: "Git Version Control and Collaboration",
        3: "Python Functions and Data Structures",
        4: "CLI Text Processing and Advanced Functions",
        5: "NumPy Arrays and Environment Management",
        6: "Pandas DataFrames and Jupyter Introduction",
        7: "Data Cleaning and Visualization",
        8: "Data Analysis and Debugging Techniques",
        9: "Automation Workflows and Advanced Pandas",
        10: "Data Integration and Complex Analysis",
        11: "Reproducible Research and Professional Workflows"
    }

    created_files = []

    for lecture_num in range(1, 12):
        lecture_dir = Path(f"{lecture_num:02d}")
        assignment_dir = lecture_dir / "assignment"

        if not assignment_dir.exists():
            print(f"Warning: Assignment directory {assignment_dir} does not exist")
            continue

        topic = lecture_topics.get(lecture_num, "Data Science Topic")

        # Create main.py if missing
        main_py_path = assignment_dir / "main.py"
        if not main_py_path.exists():
            main_content = create_main_py(lecture_num, topic)
            with open(main_py_path, 'w') as f:
                f.write(main_content)
            created_files.append(str(main_py_path))
            print(f"✓ Created {main_py_path}")

        # Create test_assignment.py if missing
        test_py_path = assignment_dir / "test_assignment.py"
        if not test_py_path.exists():
            test_content = create_test_assignment_py(lecture_num, topic)
            with open(test_py_path, 'w') as f:
                f.write(test_content)
            created_files.append(str(test_py_path))
            print(f"✓ Created {test_py_path}")

    print(f"\nCreated {len(created_files)} assignment files:")
    for file_path in created_files:
        print(f"  • {file_path}")

    print("\nAll missing assignment files have been created!")

if __name__ == "__main__":
    main()