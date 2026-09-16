#!/usr/bin/env python3
"""Demo 3: reuse imported helpers in a script that writes a report."""

from student_tools import (
    calculate_average,
    find_highest_grade,
    format_student_grades,
    get_grades,
)


def main():
    """Build, save, and verify a small grade report."""
    students = [
        {"name": "Alice", "grade": 85},
        {"name": "Bob", "grade": 92},
        {"name": "Charlie", "grade": 78},
    ]
    grades = get_grades(students)
    lines = format_student_grades(students)
    lines.append(f"Average grade: {calculate_average(grades):.1f}")
    lines.append(f"Highest grade: {find_highest_grade(grades)}")
    report_text = "\n".join(lines) + "\n"

    try:
        with open("grade_report.txt", "w", encoding="utf-8") as report:
            report.write(report_text)
        with open("grade_report.txt", encoding="utf-8") as report:
            saved_text = report.read()
    except OSError as error:
        print(f"Could not save the report: {error}")
        return

    print("Read back from grade_report.txt:")
    print(saved_text, end="")
    print(f"Saved report matches: {saved_text == report_text}")


if __name__ == "__main__":
    main()
