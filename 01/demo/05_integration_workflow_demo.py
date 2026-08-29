#!/usr/bin/env python3
"""
Demo 3: Complete integration workflow.

This deliberately small script combines the Lecture 01 building blocks:
variables, lists, loops, conditionals, arithmetic, and formatted output.
Use shell commands to organize the project and redirect the report to a file.
"""

print("=" * 60)
print("DEMO 3: COMPLETE WORKFLOW INTEGRATION")
print("=" * 60)
print("Goal: combine command-line organization with a small Python analysis")
print()

# The command line can create a project folder before this script runs.
project_name = "student_analysis_project"
print(f"Project: {project_name}")

# A short, readable fixture keeps the focus on the workflow.
student_names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
grades = [87, 92, 78, 88, 95]
passing_grade = 70

print("\nStudent results:")
passing_count = 0
for index in range(len(student_names)):
    name = student_names[index]
    grade = grades[index]
    if grade >= passing_grade:
        status = "PASS"
        passing_count += 1
    else:
        status = "REVIEW"
    print(f"  {name}: {grade}% — {status}")

# Summarize the same list with the built-in operations from the lecture.
total_points = sum(grades)
average_grade = total_points / len(grades)
highest_grade = max(grades)
lowest_grade = min(grades)

print("\nClass summary:")
print(f"  Students: {len(student_names)}")
print(f"  Average: {average_grade:.1f}%")
print(f"  Highest: {highest_grade}%")
print(f"  Lowest: {lowest_grade}%")
print(f"  Passing: {passing_count}/{len(grades)}")

if average_grade >= passing_grade:
    print("  Summary: the class met the passing threshold.")
else:
    print("  Summary: the class needs additional support.")

print("\nWorkflow complete.")
print("Save this report from the command line with:")
print("  python3 05_integration_workflow_demo.py > results.txt")
