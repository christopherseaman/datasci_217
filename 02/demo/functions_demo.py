#!/usr/bin/env python3
"""Demo 2: turn a small inline analysis into reusable functions."""

from student_tools import (
    find_highest_grade,
    get_grades_list,
    load_students_from_csv,
    print_student_grades,
    safe_calculate_average,
    save_results_to_file,
    validate_student_data,
)


CSV_CONTENT = """name,age,grade,subject
Alice,20,85,Math
Bob,19,92,Science
Charlie,21,78,English
Diana,20,88,Math
Eve,22,95,Science
"""


def main():
    """Create sample data, then use helpers to summarize it."""
    print("=== Demo 2: Functions ===")
    students = [
        {"name": "  Alice  ", "age": 20, "grade": 85, "subject": "MATH"},
        {"name": "bob", "age": 19, "grade": 92, "subject": "science"},
    ]
    print("Inline records:")
    for student in students:
        print(student['name'].strip().title(), student['grade'])

    with open('sample_students.csv', 'w') as file:
        file.write(CSV_CONTENT)
    loaded_students = load_students_from_csv('sample_students.csv')
    if not validate_student_data(loaded_students):
        print("Invalid student data")
        return

    grades = get_grades_list(loaded_students)
    average = safe_calculate_average(grades)
    highest = find_highest_grade(grades)
    print_student_grades(loaded_students)
    print(f"Average grade: {average:.1f}")
    print(f"Highest grade: {highest}")
    save_results_to_file('analysis_results.txt', loaded_students, average, highest)


if __name__ == "__main__":
    main()
