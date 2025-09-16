#!/usr/bin/env python3
"""
Create GitHub Classroom workflows for all DataSci 217 lectures
"""

import os
from pathlib import Path

# Define lecture-specific requirements and files
LECTURE_CONFIG = {
    "02": {"deps": "pandas numpy", "main_file": "main.py", "topic": "Version Control and Project Setup"},
    "03": {"deps": "pandas numpy", "main_file": "main.py", "topic": "Python Data Structures and File Operations"},
    "04": {"deps": "pandas numpy", "main_file": "starter_code.py", "topic": "Command Line Text Processing and Python Functions"},
    "05": {"deps": "pandas numpy matplotlib", "main_file": "assignment.py", "topic": "Python Libraries and Environment Management"},
    "06": {"deps": "pandas numpy matplotlib jupyter", "main_file": "main.py", "topic": "Pandas Fundamentals and Jupyter Introduction"},
    "07": {"deps": "pandas numpy matplotlib seaborn", "main_file": "main.py", "topic": "Data Cleaning and Basic Visualization"},
    "08": {"deps": "pandas numpy matplotlib seaborn", "main_file": "main.py", "topic": "Data Analysis and Advanced Debugging Techniques"},
    "09": {"deps": "pandas numpy matplotlib seaborn", "main_file": "main.py", "topic": "Automation and Advanced Data Manipulation"},
    "10": {"deps": "pandas numpy matplotlib seaborn scikit-learn", "main_file": "main.py", "topic": "Advanced Data Analysis and Integration"},
    "11": {"deps": "pandas numpy matplotlib seaborn scikit-learn", "main_file": "main.py", "topic": "Professional Applications and Research Integration"}
}

def create_workflow_content(lecture_num, config):
    """Create GitHub workflow content for a specific lecture"""

    workflow_content = f"""name: DataSci 217 - Lecture {lecture_num} Assignment Grader

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:  # Allow manual triggering

env:
  PYTHON_VERSION: '3.11'

jobs:
  autograder:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - name: Checkout student code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ env.PYTHON_VERSION }}}}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{{{ runner.os }}}}-pip-${{{{ hashFiles('**/requirements.txt') }}}}
        restore-keys: |
          ${{{{ runner.os }}}}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-html pytest-json-report pytest-timeout

        # Install student requirements if they exist
        if [ -f requirements.txt ]; then
          pip install -r requirements.txt
        fi

        # Install lecture-specific dependencies
        pip install {config['deps']}

    - name: Verify required files exist
      run: |
        echo "Checking for required files..."
        required_files=("{config['main_file']}")
        missing_files=""

        for file in "${{required_files[@]}}"; do
          if [[ ! -f "$file" ]]; then
            missing_files="$missing_files $file"
          fi
        done

        if [[ -n "$missing_files" ]]; then
          echo "❌ Missing required files:$missing_files"
          echo "MISSING_FILES=true" >> $GITHUB_ENV
        else
          echo "✅ All required files present"
          echo "MISSING_FILES=false" >> $GITHUB_ENV
        fi

    - name: Run syntax check
      if: env.MISSING_FILES == 'false'
      run: |
        echo "Running syntax check..."
        python -m py_compile {config['main_file']} || {{
          echo "❌ Syntax errors found in {config['main_file']}"
          echo "SYNTAX_ERRORS=true" >> $GITHUB_ENV
          exit 1
        }}
        echo "✅ No syntax errors found"
        echo "SYNTAX_ERRORS=false" >> $GITHUB_ENV

    - name: Run tests with grading
      if: env.MISSING_FILES == 'false' && env.SYNTAX_ERRORS == 'false'
      run: |
        echo "Running automated tests..."
        pytest test_assignment.py -v \\
          --html=test_report.html --self-contained-html \\
          --json-report --json-report-file=test_results.json \\
          --tb=short \\
          --timeout=30 \\
          -x  # Stop on first failure for faster feedback

    - name: Generate grade report
      if: always() && env.MISSING_FILES == 'false'
      run: |
        python -c "
        import json
        import os

        def generate_grade_report():
            total_score = 0
            max_score = 100
            details = []

            try:
                if os.path.exists('test_results.json'):
                    with open('test_results.json', 'r') as f:
                        results = json.load(f)

                    passed = results['summary']['passed']
                    failed = results['summary']['failed']
                    total_tests = passed + failed

                    if total_tests > 0:
                        total_score = int((passed / total_tests) * 100)

                    details.append(f'Tests passed: {{passed}}/{{total_tests}}')
                    details.append(f'Test success rate: {{total_score}}%')
                else:
                    details.append('No test results found')

            except Exception as e:
                details.append(f'Error reading test results: {{e}}')

            # Apply penalties
            penalties = []
            if os.getenv('MISSING_FILES') == 'true':
                penalties.append('Missing required files: -50 points')
                total_score = max(0, total_score - 50)

            if os.getenv('SYNTAX_ERRORS') == 'true':
                penalties.append('Syntax errors: -25 points')
                total_score = max(0, total_score - 25)

            # Generate report
            report = []
            report.append('# Lecture {lecture_num} Assignment - Grade Report')
            report.append('')
            report.append('## Topic: {config["topic"]}')
            report.append('')
            report.append(f'## Final Score: {{total_score}}/{{max_score}}')

            if total_score >= 90:
                report.append('🎉 Excellent work!')
            elif total_score >= 80:
                report.append('👍 Good job!')
            elif total_score >= 70:
                report.append('✅ Meets requirements')
            else:
                report.append('❌ Please review and resubmit')

            report.append('')
            report.append('## Details')
            for detail in details:
                report.append(f'- {{detail}}')

            if penalties:
                report.append('')
                report.append('## Penalties Applied')
                for penalty in penalties:
                    report.append(f'- {{penalty}}')

            return '\\\\n'.join(report), total_score

        report_content, final_score = generate_grade_report()

        with open('GRADE_REPORT.md', 'w') as f:
            f.write(report_content)

        print('GRADE REPORT')
        print('=' * 50)
        print(report_content)

        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write(f'FINAL_SCORE={{final_score}}\\\\n')
        "

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results-lecture-{lecture_num}
        path: |
          test_report.html
          test_results.json
          GRADE_REPORT.md
        retention-days: 30

    - name: Fail job if score too low
      if: always()
      run: |
        final_score=${{FINAL_SCORE:-0}}
        if [ "$final_score" -lt 70 ]; then
          echo "❌ Assignment score ($final_score/100) is below the passing threshold (70/100)"
          exit 1
        else
          echo "✅ Assignment passed with score: $final_score/100"
        fi"""

    return workflow_content

def main():
    """Create GitHub workflows for all lectures"""
    base_path = Path("/home/christopher/projects/datasci_217")

    for lecture_num, config in LECTURE_CONFIG.items():
        # Create .github/workflows directory
        workflow_dir = base_path / lecture_num / "assignment" / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)

        # Create workflow file
        workflow_file = workflow_dir / "classroom.yml"
        content = create_workflow_content(lecture_num, config)

        with open(workflow_file, 'w') as f:
            f.write(content)

        print(f"Created workflow for Lecture {lecture_num}: {workflow_file}")

if __name__ == "__main__":
    main()