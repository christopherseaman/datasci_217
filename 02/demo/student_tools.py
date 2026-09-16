"""Small, import-safe grade helpers for the Lecture 02 demos."""


def calculate_average(grades):
    """Return the arithmetic mean, or zero for an empty list."""
    if not grades:
        return 0
    return sum(grades) / len(grades)


def find_highest_grade(grades):
    """Return the largest grade, or zero for an empty list."""
    if not grades:
        return 0
    return max(grades)


def get_grades(students):
    """Return the grades stored in student records."""
    grades = []
    for student in students:
        grades.append(student["grade"])
    return grades


def format_student_grades(students):
    """Return one display line for each student record."""
    lines = []
    for student in students:
        lines.append(f"{student['name']}: {student['grade']}")
    return lines
