"""Small, import-safe student-analysis helpers for the Lecture 02 demos."""


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


def print_student_grades(students):
    """Print each student's name and grade."""
    for student in students:
        print(f"{student['name']}: {student['grade']}")


def get_grades_list(students):
    """Extract grades from student records."""
    return [student['grade'] for student in students]


def safe_calculate_average(grades):
    """Calculate an average after checking that every grade is numeric."""
    if not grades:
        return 0
    for grade in grades:
        if not isinstance(grade, (int, float)):
            raise ValueError("grades must contain only numbers")
    return calculate_average(grades)


def validate_student_data(students):
    """Return whether each record has the fields used by these demos."""
    required_fields = ['name', 'age', 'grade', 'subject']
    for student in students:
        for field in required_fields:
            if field not in student:
                return False
    return True


def save_results_to_file(filename, students, average, highest):
    """Write a concise grade report and return whether it succeeded."""
    try:
        with open(filename, 'w') as file:
            file.write("Student Grade Analysis\n")
            file.write("=" * 30 + "\n\n")
            for student in students:
                file.write(f"{student['name']}: {student['grade']}\n")
            file.write(f"\nAverage grade: {average:.1f}\n")
            file.write(f"Highest grade: {highest}\n")
            file.write(f"Total students: {len(students)}\n")
    except OSError as error:
        print(f"Error saving file: {error}")
        return False
    return True


def load_students_from_csv(filename):
    """Load the four-column student CSV used by the demos."""
    try:
        with open(filename) as file:
            lines = file.readlines()
    except OSError as error:
        print(f"Error loading file: {error}")
        return []

    students = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) == 4:
            try:
                students.append({
                    'name': parts[0], 'age': int(parts[1]),
                    'grade': int(parts[2]), 'subject': parts[3],
                })
            except ValueError:
                print(f"Skipping malformed row: {line.strip()}")
    return students
