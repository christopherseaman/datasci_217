#!/usr/bin/env python3
"""Demo 2: replace repeated grade work with reusable functions."""

from student_tools import calculate_average, find_highest_grade, get_grades


students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78},
]

print("=== Demo 2: Functions ===")
print("Before: every script would repeat this loop.")
grades = []
for student in students:
    grades.append(student["grade"])
print(f"Before loop extracted: {grades}")

print("After: reuse helpers from student_tools.")
grades = get_grades(students)
print(f"After get_grades() extracted: {grades}")
print(f"Average grade: {calculate_average(grades):.1f}")
print(f"Highest grade: {find_highest_grade(grades)}")
