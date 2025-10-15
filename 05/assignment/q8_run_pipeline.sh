# TODO: Add shebang line: #!/bin/bash
# Assignment 5, Question 8: Pipeline Automation Script
# Run the entire clinical trial data analysis pipeline

echo "Starting clinical trial data pipeline..." > reports/pipeline_log.txt

# NOTE: Q2 (q2_process_metadata.py) is a standalone Python fundamentals exercise, not part of the main pipeline
# NOTE: Q3 (q3_data_utils.py) is a library imported by the notebooks, not run directly

# TODO: Run analysis notebooks in order (q4-q7)
# Example notebook execution with error handling:
# jupyter nbconvert --execute --to notebook q4_exploration.ipynb || {
#     echo "ERROR: Q4 exploration failed" >> reports/pipeline_log.txt
#     exit 1
# }
# echo "SUCCESS: Q4 exploration completed" >> reports/pipeline_log.txt

echo "Pipeline complete!" >> reports/pipeline_log.txt
