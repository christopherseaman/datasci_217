#!/usr/bin/env python3
"""
Demo 2: Student Grade Analysis (LIVE DEMO)
Demonstrates NumPy operations with real student data
This matches the LIVE DEMO sections at lines 241 and 853 of Lecture 3
"""

import numpy as np
import time

def generate_student_data():
    """Generate realistic student data as mentioned in the lecture"""
    print("GENERATING STUDENT DATA")
    print("=" * 25)
    
    # Set seed for reproducible results (as mentioned in lecture)
    np.random.seed(42)
    n_students = 100
    n_assignments = 5
    
    print(f"Creating data for {n_students} students with {n_assignments} assignments each...")
    
    # Generate random grades (70-100 range) - as mentioned in lecture
    grades = np.random.randint(70, 101, size=(n_students, n_assignments))
    student_names = [f"Student_{i:03d}" for i in range(n_students)]
    
    print(f"✓ Generated {grades.shape[0]} students × {grades.shape[1]} assignments")
    print(f"✓ Grade range: {grades.min()}-{grades.max()}")
    print(f"✓ Data type: {grades.dtype}")
    print(f"✓ Memory usage: {grades.nbytes:,} bytes")
    
    return grades, student_names

def demonstrate_vectorized_operations(grades):
    """Demonstrate vectorized operations for statistical analysis"""
    print("\nVECTORIZED OPERATIONS FOR STATISTICAL ANALYSIS")
    print("=" * 50)
    
    print("Calculating statistics using vectorized operations...")
    
    # Calculate statistics using vectorized operations (as mentioned in lecture)
    student_averages = grades.mean(axis=1)  # Average per student
    assignment_averages = grades.mean(axis=0)  # Average per assignment
    overall_average = grades.mean()  # Overall class average
    
    print(f"\n✓ Student averages calculated: {len(student_averages)} values")
    print(f"✓ Assignment averages calculated: {len(assignment_averages)} values")
    print(f"✓ Overall average: {overall_average:.1f}")
    
    # Show some examples
    print(f"\nFirst 5 student averages: {student_averages[:5]}")
    print(f"Assignment averages: {assignment_averages}")
    
    return student_averages, assignment_averages, overall_average

def find_high_performers(grades, student_names, student_averages):
    """Find high performers using boolean indexing"""
    print("\nFINDING HIGH PERFORMERS WITH BOOLEAN INDEXING")
    print("=" * 50)
    
    print("Using boolean indexing to find students with average > 90...")
    
    # Find high performers (above 90 average) - as mentioned in lecture
    high_performers = student_averages > 90
    high_performer_names = np.array(student_names)[high_performers]
    high_performer_grades = student_averages[high_performers]
    
    print(f"✓ Found {len(high_performer_names)} high performers")
    
    if len(high_performer_names) > 0:
        print(f"\nHigh Performers:")
        for name, grade in zip(high_performer_names, high_performer_grades):
            print(f"  {name}: {grade:.1f}")
    else:
        print("  No students scored above 90 average")
    
    return high_performer_names, high_performer_grades

def statistical_analysis(grades, student_averages):
    """Perform comprehensive statistical analysis"""
    print("\nCOMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 40)
    
    print("Performing statistical analysis using NumPy operations...")
    
    # Statistical analysis (as mentioned in lecture)
    print(f"\nClass Statistics:")
    print(f"Overall average: {grades.mean():.1f}")
    print(f"Highest student average: {student_averages.max():.1f}")
    print(f"Lowest student average: {student_averages.min():.1f}")
    print(f"Standard deviation: {student_averages.std():.1f}")
    
    # Grade distribution analysis
    all_grades = grades.flatten()
    a_grades = np.sum(all_grades >= 90)
    b_grades = np.sum((all_grades >= 80) & (all_grades < 90))
    c_grades = np.sum((all_grades >= 70) & (all_grades < 80))
    
    total_grades = len(all_grades)
    print(f"\nGrade Distribution:")
    print(f"A grades (90+): {a_grades} ({a_grades/total_grades*100:.1f}%)")
    print(f"B grades (80-89): {b_grades} ({b_grades/total_grades*100:.1f}%)")
    print(f"C grades (70-79): {c_grades} ({c_grades/total_grades*100:.1f}%)")

def assignment_difficulty_analysis(assignment_averages):
    """Analyze assignment difficulty"""
    print("\nASSIGNMENT DIFFICULTY ANALYSIS")
    print("=" * 35)
    
    print("Analyzing assignment difficulty based on average scores...")
    
    # Assignment difficulty analysis (as mentioned in lecture)
    print(f"\nAssignment Averages:")
    for i, avg in enumerate(assignment_averages):
        if avg > 85:
            difficulty = "Easy"
        elif avg > 80:
            difficulty = "Medium"
        else:
            difficulty = "Hard"
        print(f"  Assignment {i+1}: {avg:.1f} ({difficulty})")
    
    # Find easiest and hardest assignments
    easiest_idx = np.argmax(assignment_averages)
    hardest_idx = np.argmin(assignment_averages)
    
    print(f"\nEasiest assignment: Assignment {easiest_idx+1} ({assignment_averages[easiest_idx]:.1f})")
    print(f"Hardest assignment: Assignment {hardest_idx+1} ({assignment_averages[hardest_idx]:.1f})")

def demonstrate_array_operations(grades):
    """Demonstrate various NumPy array operations"""
    print("\nNUMPY ARRAY OPERATIONS DEMONSTRATION")
    print("=" * 45)
    
    print("Demonstrating various NumPy array operations...")
    
    # Array properties
    print(f"\nArray Properties:")
    print(f"Shape: {grades.shape}")
    print(f"Dimensions: {grades.ndim}")
    print(f"Size: {grades.size}")
    print(f"Data type: {grades.dtype}")
    print(f"Memory usage: {grades.nbytes:,} bytes")
    
    # Array operations
    print(f"\nArray Operations:")
    print(f"Sum of all grades: {grades.sum():,}")
    print(f"Mean of all grades: {grades.mean():.2f}")
    print(f"Standard deviation: {grades.std():.2f}")
    print(f"Minimum grade: {grades.min()}")
    print(f"Maximum grade: {grades.max()}")
    
    # Axis operations
    print(f"\nAxis Operations:")
    print(f"Sum by student (axis=1): {grades.sum(axis=1)[:5]}...")
    print(f"Sum by assignment (axis=0): {grades.sum(axis=0)}")
    print(f"Mean by student (axis=1): {grades.mean(axis=1)[:5]}...")
    print(f"Mean by assignment (axis=0): {grades.mean(axis=0)}")

def demonstrate_indexing_and_slicing(grades):
    """Demonstrate NumPy indexing and slicing"""
    print("\nINDEXING AND SLICING DEMONSTRATION")
    print("=" * 40)
    
    print("Demonstrating NumPy indexing and slicing...")
    
    # Basic indexing
    print(f"\nBasic Indexing:")
    print(f"First student's grades: {grades[0]}")
    print(f"First assignment grades: {grades[:, 0]}")
    print(f"Grade at [0,0]: {grades[0, 0]}")
    
    # Slicing
    print(f"\nSlicing:")
    print(f"First 3 students: {grades[:3]}")
    print(f"First 2 assignments: {grades[:, :2]}")
    print(f"Students 5-10, assignments 1-3: {grades[5:10, 1:4]}")
    
    # Boolean indexing
    print(f"\nBoolean Indexing:")
    high_grades = grades > 95
    print(f"Grades > 95: {np.sum(high_grades)} occurrences")
    print(f"Students with any grade > 95: {np.any(grades > 95, axis=1).sum()}")
    
    # Fancy indexing
    print(f"\nFancy Indexing:")
    top_students = [0, 5, 10, 15, 20]
    print(f"Top 5 students' grades: {grades[top_students]}")

def performance_comparison():
    """Compare NumPy vs Python for the same operations"""
    print("\nPERFORMANCE COMPARISON: NUMPY VS PYTHON")
    print("=" * 45)
    
    # Generate sample data
    n_students, n_assignments = 1000, 5
    grades_numpy = np.random.randint(70, 101, size=(n_students, n_assignments))
    grades_python = grades_numpy.tolist()
    
    print(f"Testing with {n_students} students × {n_assignments} assignments...")
    
    # Test mean calculation
    start_time = time.time()
    python_mean = sum(sum(row) for row in grades_python) / (n_students * n_assignments)
    python_time = time.time() - start_time
    
    start_time = time.time()
    numpy_mean = grades_numpy.mean()
    numpy_time = time.time() - start_time
    
    print(f"\nMean Calculation:")
    print(f"Python: {python_time:.4f} seconds (result: {python_mean:.2f})")
    print(f"NumPy:  {numpy_time:.4f} seconds (result: {numpy_mean:.2f})")
    print(f"Speedup: {python_time/numpy_time:.1f}x")
    
    # Test finding maximum
    start_time = time.time()
    python_max = max(max(row) for row in grades_python)
    python_time = time.time() - start_time
    
    start_time = time.time()
    numpy_max = grades_numpy.max()
    numpy_time = time.time() - start_time
    
    print(f"\nMaximum Finding:")
    print(f"Python: {python_time:.4f} seconds (result: {python_max})")
    print(f"NumPy:  {numpy_time:.4f} seconds (result: {numpy_max})")
    print(f"Speedup: {python_time/numpy_time:.1f}x")

def main():
    """Main function demonstrating student grade analysis with NumPy"""
    print("STUDENT GRADE ANALYSIS - LIVE DEMO")
    print("=" * 50)
    print("This demo directly demonstrates the LIVE DEMO sections")
    print("from Lecture 3, showing NumPy operations with real data.")
    print("=" * 50)
    
    # Generate student data
    grades, student_names = generate_student_data()
    
    # Demonstrate vectorized operations
    student_averages, assignment_averages, overall_average = demonstrate_vectorized_operations(grades)
    
    # Find high performers
    high_performer_names, high_performer_grades = find_high_performers(grades, student_names, student_averages)
    
    # Statistical analysis
    statistical_analysis(grades, student_averages)
    
    # Assignment difficulty analysis
    assignment_difficulty_analysis(assignment_averages)
    
    # Demonstrate array operations
    demonstrate_array_operations(grades)
    
    # Demonstrate indexing and slicing
    demonstrate_indexing_and_slicing(grades)
    
    # Performance comparison
    performance_comparison()
    
    print(f"\nLIVE DEMO COMPLETE!")
    print("=" * 25)
    print("This demonstration showed:")
    print("✓ NumPy operations with real student data")
    print("✓ Vectorized operations for statistical analysis")
    print("✓ Boolean indexing to find high performers")
    print("✓ Assignment difficulty analysis")
    print("✓ Array operations and indexing")
    print("✓ Performance advantages over Python")
    print("\nNext: Command line tools for data processing!")

if __name__ == "__main__":
    main()