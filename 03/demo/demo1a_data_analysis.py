"""Basic student data analysis script."""
import os

def load_students(filename):
    """Load student data from CSV file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    students = []
    for line in lines[1:]:  # Skip header
        line = line.strip()
        if line:
            name, age, grade, subject = line.split(',')
            students.append({
                'name': name,
                'age': int(age),
                'grade': int(grade),
                'subject': subject
            })

    return students

def calculate_average_grade(students):
    """Calculate average grade from student data."""
    if not students:
        return 0.0

    total = sum(student['grade'] for student in students)
    return total / len(students)

def count_math_students(students):
    """Count students enrolled in Math."""
    return sum(1 for student in students if student['subject'] == 'Math')

def generate_report(students):
    """Generate formatted analysis report."""
    total = len(students)
    avg = calculate_average_grade(students)
    math_count = count_math_students(students)

    report = f"""Student Analysis Report
{'=' * 40}

Total Students: {total}
Average Grade: {avg:.1f}

Subject Distribution:
  Math: {math_count}
  Science: {total - math_count}
"""
    return report

def save_report(report, filename):
    """Save report to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(report)

def main():
    """Main execution function."""
    students = load_students('data/students.csv')
    report = generate_report(students)
    save_report(report, 'output/analysis_report.txt')
    print(report)

if __name__ == "__main__":
    main()
