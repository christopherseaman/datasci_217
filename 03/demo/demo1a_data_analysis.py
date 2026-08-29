"""Basic student data analysis script."""
import os

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

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

def count_by_subject(students):
    """Count students enrolled in each subject."""
    counts = {}
    for student in students:
        subject = student['subject']
        counts[subject] = counts.get(subject, 0) + 1
    return counts

def generate_report(students):
    """Generate formatted analysis report."""
    total = len(students)
    avg = calculate_average_grade(students)
    subject_counts = count_by_subject(students)

    report = f"""Student Analysis Report
{'=' * 40}

Total Students: {total}
Average Grade: {avg:.1f}

Subject Distribution:
"""
    for subject, count in subject_counts.items():
        report += f"  {subject}: {count}\n"
    return report

def save_report(report, filename):
    """Save report to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(report)

def main():
    """Main execution function."""
    students = load_students(os.path.join(DEMO_DIR, 'students.csv'))
    report = generate_report(students)
    output_file = os.path.join(DEMO_DIR, 'output', 'analysis_report.txt')
    save_report(report, output_file)
    print(report)

if __name__ == "__main__":
    main()
