"""Advanced student data analysis with modular design."""
import os

def load_data(filename):
    """Generic loader that checks file extension."""
    if filename.endswith('.csv'):
        return load_csv(filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def load_csv(filename):
    """Load CSV data using manual parsing."""
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

def count_by_subject(students):
    """Count students by all subjects."""
    subjects = {}
    for student in students:
        subject = student['subject']
        subjects[subject] = subjects.get(subject, 0) + 1
    return subjects

def analyze_grade_distribution(grades):
    """Analyze grade distribution by letter grade."""
    total = len(grades)
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}

    for grade in grades:
        if grade >= 90:
            counts['A'] += 1
        elif grade >= 80:
            counts['B'] += 1
        elif grade >= 70:
            counts['C'] += 1
        elif grade >= 60:
            counts['D'] += 1
        else:
            counts['F'] += 1

    percentages = {
        letter: (count / total * 100)
        for letter, count in counts.items()
    }

    return {'counts': counts, 'percentages': percentages}

def analyze_data(students):
    """Perform comprehensive analysis on student data."""
    grades = [s['grade'] for s in students]

    return {
        'total_students': len(students),
        'average_grade': sum(grades) / len(grades),
        'highest_grade': max(grades),
        'lowest_grade': min(grades),
        'subjects': count_by_subject(students),
        'distribution': analyze_grade_distribution(grades)
    }

def save_results(results, filename):
    """Save analysis results to file."""
    report = f"""Advanced Student Analysis Report
{'=' * 50}

Total Students: {results['total_students']}
Average Grade: {results['average_grade']:.1f}
Highest: {results['highest_grade']} | Lowest: {results['lowest_grade']}

Subject Distribution:
"""

    for subject, count in results['subjects'].items():
        report += f"  {subject}: {count}\n"

    report += "\nGrade Distribution:\n"

    for letter in ['A', 'B', 'C', 'D', 'F']:
        count = results['distribution']['counts'][letter]
        pct = results['distribution']['percentages'][letter]
        report += f"  {letter}: {count} students ({pct:.1f}%)\n"

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(report)

def main():
    """Main execution function."""
    # Load data using modular functions
    students = load_data('data/students.csv')

    # Perform analysis
    results = analyze_data(students)

    # Save results
    save_results(results, 'output/analysis_report.txt')

    # Also print to console
    with open('output/analysis_report.txt', 'r') as f:
        print(f.read())

if __name__ == "__main__":
    main()
