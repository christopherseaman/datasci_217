#!/usr/bin/env python3
"""
Student Grade Analysis with NumPy
Demonstrates practical NumPy operations
"""

import numpy as np

def create_sample_data():
    """Create realistic student grade data."""
    np.random.seed(42)  # Reproducible results

    n_students = 100
    n_assignments = 5

    # Generate grades (70-100 range)
    grades = np.random.randint(70, 101, size=(n_students, n_assignments))

    print(f"Created data: {n_students} students, {n_assignments} assignments")
    print(f"Array shape: {grades.shape}")
    print(f"Data type: {grades.dtype}\n")

    return grades

def demo_basic_operations(grades):
    """Demonstrate basic NumPy operations."""
    print("=== Basic Operations ===")

    # Arithmetic operations
    print(f"First student's grades: {grades[0]}")
    print(f"Doubled: {grades[0] * 2}")
    print(f"Curved (+5): {grades[0] + 5}")
    print()

def demo_statistical_operations(grades):
    """Demonstrate statistical operations."""
    print("=== Statistical Operations ===")

    # Overall statistics
    print(f"Overall average: {grades.mean():.1f}")
    print(f"Overall std dev: {grades.std():.1f}")
    print(f"Highest grade: {grades.max()}")
    print(f"Lowest grade: {grades.min()}")
    print()

    # Axis-specific operations
    student_averages = grades.mean(axis=1)  # Average per student
    assignment_averages = grades.mean(axis=0)  # Average per assignment

    print("Student averages (first 5):")
    print(student_averages[:5])
    print(f"\nAssignment averages:")
    for i, avg in enumerate(assignment_averages, 1):
        print(f"  Assignment {i}: {avg:.1f}")
    print()

def demo_boolean_indexing(grades):
    """Demonstrate boolean indexing."""
    print("=== Boolean Indexing ===")

    # Calculate student averages
    student_averages = grades.mean(axis=1)

    # Find high performers
    high_performers = student_averages > 90
    print(f"Students with average > 90: {high_performers.sum()}")
    print(f"High performer averages: {student_averages[high_performers][:5]}")

    # Multiple conditions
    excellent = (student_averages >= 90) & (student_averages <= 100)
    good = (student_averages >= 80) & (student_averages < 90)
    fair = (student_averages >= 70) & (student_averages < 80)

    print(f"\nGrade distribution:")
    print(f"  Excellent (90-100): {excellent.sum()} students")
    print(f"  Good (80-89): {good.sum()} students")
    print(f"  Fair (70-79): {fair.sum()} students")
    print()

def demo_array_reshaping(grades):
    """Demonstrate array reshaping."""
    print("=== Array Reshaping ===")

    # Get first 12 grades
    sample = grades.flat[:12]
    print(f"Flattened sample (12 grades): {sample}")

    # Reshape to different dimensions
    reshaped = sample.reshape(3, 4)
    print(f"\nReshaped to 3x4:")
    print(reshaped)

    # Transpose
    print(f"\nTransposed (4x3):")
    print(reshaped.T)
    print()

def demo_practical_analysis(grades):
    """Demonstrate practical analysis workflow."""
    print("=== Practical Analysis Workflow ===")

    # Find the hardest assignment
    assignment_averages = grades.mean(axis=0)
    hardest_idx = assignment_averages.argmin()
    easiest_idx = assignment_averages.argmax()

    print(f"Hardest assignment: #{hardest_idx + 1} (avg: {assignment_averages[hardest_idx]:.1f})")
    print(f"Easiest assignment: #{easiest_idx + 1} (avg: {assignment_averages[easiest_idx]:.1f})")

    # Find top 5 students
    student_averages = grades.mean(axis=1)
    top_5_indices = np.argsort(student_averages)[-5:][::-1]

    print(f"\nTop 5 students:")
    for rank, idx in enumerate(top_5_indices, 1):
        print(f"  #{rank}: Student {idx:3d} - Average: {student_averages[idx]:.1f}")

    # Calculate improvement (first vs last assignment)
    improvement = grades[:, -1] - grades[:, 0]
    improved_students = (improvement > 0).sum()
    avg_improvement = improvement[improvement > 0].mean()

    print(f"\nImprovement analysis:")
    print(f"  Students who improved: {improved_students}")
    print(f"  Average improvement: {avg_improvement:.1f} points")
    print()

def main():
    """Run all NumPy demos."""
    print("Student Grade Analysis with NumPy")
    print("=" * 50)
    print()

    # Create data
    grades = create_sample_data()

    # Run demos
    demo_basic_operations(grades)
    demo_statistical_operations(grades)
    demo_boolean_indexing(grades)
    demo_array_reshaping(grades)
    demo_practical_analysis(grades)

    print("✅ NumPy analysis complete!")

if __name__ == "__main__":
    main()