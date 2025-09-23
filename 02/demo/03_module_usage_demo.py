#!/usr/bin/env python3
"""
Module Usage Demo - Importing and using functions from another script
This script demonstrates how to import functions from the previous script
and use them in a new context, showing the power of modular design.
"""

# Import functions from the previous script
# Note: In a real project, you'd typically put functions in a separate module file
# For this demo, we'll import from the functions demo script

import sys
import os

# Add current directory to Python path so we can import our demo script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import specific functions from our demo script
from python_functions_demo import (
    calculate_average,
    find_highest_grade,
    print_student_grades,
    get_grades_list,
    safe_calculate_average,
    validate_student_data,
    save_results_to_file,
    load_students_from_csv
)

def analyze_grade_distribution(grades):
    """Analyze the distribution of grades."""
    if not grades:
        return {}
    
    # Count grades by ranges
    distribution = {
        'A (90-100)': 0,
        'B (80-89)': 0,
        'C (70-79)': 0,
        'D (60-69)': 0,
        'F (0-59)': 0
    }
    
    for grade in grades:
        if grade >= 90:
            distribution['A (90-100)'] += 1
        elif grade >= 80:
            distribution['B (80-89)'] += 1
        elif grade >= 70:
            distribution['C (70-79)'] += 1
        elif grade >= 60:
            distribution['D (60-69)'] += 1
        else:
            distribution['F (0-59)'] += 1
    
    return distribution

def find_top_performers(students, threshold=90):
    """Find students who scored above a threshold."""
    top_performers = []
    for student in students:
        if student['grade'] >= threshold:
            top_performers.append(student)
    return top_performers

def generate_detailed_report(students, filename):
    """Generate a comprehensive analysis report."""
    if not students:
        print("No data to analyze")
        return False
    
    # Validate data first
    if not validate_student_data(students):
        print("Invalid student data")
        return False
    
    # Get grades and basic statistics
    grades = get_grades_list(students)
    average = safe_calculate_average(grades)
    highest = find_highest_grade(grades)
    lowest = min(grades) if grades else 0
    
    # Analyze grade distribution
    distribution = analyze_grade_distribution(grades)
    
    # Find top performers
    top_performers = find_top_performers(students, 90)
    
    # Generate report
    try:
        with open(filename, 'w') as file:
            file.write("COMPREHENSIVE STUDENT ANALYSIS REPORT\n")
            file.write("=" * 50 + "\n\n")
            
            file.write(f"Report generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            file.write("BASIC STATISTICS\n")
            file.write("-" * 20 + "\n")
            file.write(f"Total students: {len(students)}\n")
            file.write(f"Average grade: {average:.1f}\n")
            file.write(f"Highest grade: {highest}\n")
            file.write(f"Lowest grade: {lowest}\n")
            file.write(f"Grade range: {highest - lowest}\n\n")
            
            file.write("GRADE DISTRIBUTION\n")
            file.write("-" * 20 + "\n")
            for grade_range, count in distribution.items():
                percentage = (count / len(students)) * 100 if students else 0
                file.write(f"{grade_range}: {count} students ({percentage:.1f}%)\n")
            file.write("\n")
            
            file.write("TOP PERFORMERS (90+)\n")
            file.write("-" * 20 + "\n")
            if top_performers:
                for student in top_performers:
                    file.write(f"{student['name']}: {student['grade']} ({student['subject']})\n")
            else:
                file.write("No students scored 90 or above\n")
            file.write("\n")
            
            file.write("INDIVIDUAL STUDENT RECORDS\n")
            file.write("-" * 30 + "\n")
            for student in students:
                file.write(f"Name: {student['name']}\n")
                file.write(f"  Age: {student['age']}\n")
                file.write(f"  Grade: {student['grade']}\n")
                file.write(f"  Subject: {student['subject']}\n")
                file.write("\n")
        
        print(f"Detailed report saved to {filename}")
        return True
    except Exception as e:
        print(f"Error generating report: {e}")
        return False

def compare_subjects(students):
    """Compare performance across different subjects."""
    subject_stats = {}
    
    for student in students:
        subject = student['subject']
        grade = student['grade']
        
        if subject not in subject_stats:
            subject_stats[subject] = []
        subject_stats[subject].append(grade)
    
    print("\nSubject Performance Comparison:")
    print("-" * 35)
    
    for subject, grades in subject_stats.items():
        avg = calculate_average(grades)
        highest = find_highest_grade(grades)
        count = len(grades)
        print(f"{subject}:")
        print(f"  Students: {count}")
        print(f"  Average: {avg:.1f}")
        print(f"  Highest: {highest}")
        print()

def main():
    """Main function demonstrating module usage."""
    print("=== Module Usage Demo ===")
    print("Importing and using functions from another script")
    print()
    
    # Load data using imported function
    print("Loading student data...")
    students = load_students_from_csv('sample_students.csv')
    
    if not students:
        print("No data loaded. Please run the functions demo first to create sample data.")
        return
    
    print(f"Loaded {len(students)} students")
    print()
    
    # Use imported functions
    print("Using imported functions:")
    print("-" * 25)
    
    # Basic analysis using imported functions
    grades = get_grades_list(students)
    average = safe_calculate_average(grades)
    highest = find_highest_grade(grades)
    
    print("Student grades:")
    print_student_grades(students)
    print(f"\nAverage grade: {average:.1f}")
    print(f"Highest grade: {highest}")
    
    # Advanced analysis using new functions
    print("\n" + "="*50)
    print("Advanced Analysis using new functions:")
    
    # Grade distribution
    distribution = analyze_grade_distribution(grades)
    print("\nGrade Distribution:")
    for grade_range, count in distribution.items():
        percentage = (count / len(students)) * 100
        print(f"{grade_range}: {count} students ({percentage:.1f}%)")
    
    # Subject comparison
    compare_subjects(students)
    
    # Top performers
    top_performers = find_top_performers(students, 90)
    print(f"Top Performers (90+): {len(top_performers)} students")
    for student in top_performers:
        print(f"  {student['name']}: {student['grade']} ({student['subject']})")
    
    # Generate comprehensive report
    print("\n" + "="*50)
    print("Generating comprehensive report...")
    generate_detailed_report(students, 'comprehensive_analysis.txt')
    
    # Save basic results using imported function
    save_results_to_file('module_analysis_results.txt', students, average, highest)
    
    print("\n" + "="*50)
    print("=== Module Demo Complete ===")
    print("Key concepts demonstrated:")
    print("1. Importing functions from other scripts")
    print("2. Building new functions that use imported functions")
    print("3. Modular design and code reuse")
    print("4. Combining multiple functions for complex analysis")
    print("5. Professional script organization")
    print("\nThis shows how functions can be shared and reused across projects!")

if __name__ == "__main__":
    main()
