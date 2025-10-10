#!/bin/bash
# Assignment 5, Question 8: Pipeline Automation Script
# Run the entire clinical trial data analysis pipeline

echo "Starting clinical trial data pipeline..." > reports/pipeline_log.txt

# TODO: Run process_metadata.py
# TODO: Check exit code ($?), log errors to reports/pipeline_log.txt
# Example:
# python q2_process_metadata.py
# if [ $? -ne 0 ]; then
#     echo "ERROR: Metadata processing failed" >> reports/pipeline_log.txt
#     exit 1
# fi

# TODO: Run other analysis scripts (q3-q7)

# TODO: Generate reports/quality_report.txt
#       This should summarize the pipeline run (what was processed, any issues, etc.)

# TODO: Save output/final_clean_data.csv
#       This should be the final cleaned dataset ready for analysis

echo "Pipeline complete!" >> reports/pipeline_log.txt
