#!/usr/bin/env python3
"""
Demo 5: Complete Integration Workflow
Lecture 01 - Command Line + Python

This demo demonstrates the complete data science workflow:
CLI organization → Python analysis → File operations → Result saving

Usage: python 05_integration_workflow_demo.py

Author: Data Science 217 Course Materials
"""

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime

def demo_header():
    """Display demo introduction"""
    print("=" * 60)
    print("DEMO 5: COMPLETE WORKFLOW INTEGRATION")
    print("=" * 60)
    print("Goal: Combine everything - CLI, Python, and file operations")
    print("This is how professional data scientists work every day!")
    print()

def setup_project_structure():
    """Create a realistic data science project structure"""
    print("STEP 1: Setting Up Project Structure")
    print("-" * 40)
    
    project_name = "student_analysis_project"
    base_path = Path(project_name)
    
    # Create directory structure
    directories = [
        "data/raw",
        "data/processed", 
        "scripts",
        "results/figures",
        "results/reports",
        "logs"
    ]
    
    print(f"Creating project: {project_name}/")
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_path}/")
    
    print()
    return base_path

def create_sample_data(base_path):
    """Generate realistic sample data files"""
    print("STEP 2: Creating Sample Data Files")
    print("-" * 40)
    
    # Create student records CSV
    students_file = base_path / "data" / "raw" / "students.csv"
    students_data = [
        ["student_id", "name", "major", "year"],
        ["001", "Alice Johnson", "Computer Science", "3"],
        ["002", "Bob Smith", "Data Science", "2"],
        ["003", "Charlie Brown", "Statistics", "4"],
        ["004", "Diana Prince", "Computer Science", "2"],
        ["005", "Eve Anderson", "Data Science", "3"],
        ["006", "Frank Miller", "Statistics", "1"],
    ]
    
    with open(students_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(students_data)
    print(f"  ✓ Created: students.csv")
    
    # Create grades CSV with some problematic data
    grades_file = base_path / "data" / "raw" / "grades.csv"
    grades_data = [
        ["student_id", "course", "grade", "semester"],
        ["001", "Python", "87.5", "Fall 2024"],
        ["001", "Statistics", "92.0", "Fall 2024"],
        ["002", "Python", "78.3", "Fall 2024"],
        ["002", "Database", "", "Fall 2024"],  # Missing grade!
        ["003", "Python", "95.2", "Fall 2024"],
        ["004", "Python", "invalid", "Fall 2024"],  # Invalid grade!
        ["005", "Statistics", "88.7", "Fall 2024"],
        ["006", "Python", "73.5", "Fall 2024"],
    ]
    
    with open(grades_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(grades_data)
    print(f"  ✓ Created: grades.csv (with intentional data issues)")
    
    # Create configuration file
    config_file = base_path / "config.json"
    config_data = {
        "project": "Student Analysis",
        "version": "1.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "settings": {
            "passing_grade": 70,
            "grade_scale": {
                "A": 90,
                "B": 80,
                "C": 70,
                "D": 60
            },
            "output_format": "csv"
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"  ✓ Created: config.json")
    
    print()
    return students_file, grades_file, config_file

def demonstrate_data_loading_with_errors(base_path):
    """Load data with error handling"""
    print("STEP 3: Loading and Validating Data")
    print("-" * 40)
    
    # Load configuration
    config_file = base_path / "config.json"
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded configuration: {config['project']} v{config['version']}")
        passing_grade = config['settings']['passing_grade']
        print(f"  Passing grade threshold: {passing_grade}")
    except FileNotFoundError:
        print("❌ Config file not found! Using defaults.")
        passing_grade = 70
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config: {e}")
        passing_grade = 70
    
    print()
    
    # Load student data
    students_file = base_path / "data" / "raw" / "students.csv"
    students = {}
    
    print("Loading student records...")
    try:
        with open(students_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                students[row['student_id']] = {
                    'name': row['name'],
                    'major': row['major'],
                    'year': row['year']
                }
        print(f"  ✓ Loaded {len(students)} student records")
    except Exception as e:
        print(f"  ❌ Error loading students: {e}")
        return None, None, None
    
    # Load grades with validation
    grades_file = base_path / "data" / "raw" / "grades.csv"
    grades = []
    errors = []
    
    print("Loading and validating grades...")
    try:
        with open(grades_file, 'r') as f:
            reader = csv.DictReader(f)
            for line_num, row in enumerate(reader, 2):  # Start at 2 (header is line 1)
                try:
                    # Validate and convert grade
                    if row['grade'] == '':
                        errors.append(f"Line {line_num}: Missing grade for {row['student_id']}")
                        continue
                    
                    grade_value = float(row['grade'])
                    
                    if grade_value < 0 or grade_value > 100:
                        errors.append(f"Line {line_num}: Invalid grade {grade_value}")
                        continue
                    
                    grades.append({
                        'student_id': row['student_id'],
                        'course': row['course'],
                        'grade': grade_value,
                        'semester': row['semester']
                    })
                    
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid grade '{row['grade']}' for {row['student_id']}")
        
        print(f"  ✓ Loaded {len(grades)} valid grades")
        
        if errors:
            print("  ⚠️ Data quality issues found:")
            for error in errors:
                print(f"    - {error}")
    
    except Exception as e:
        print(f"  ❌ Error loading grades: {e}")
        return None, None, None
    
    print()
    return students, grades, passing_grade

def analyze_data(students, grades, passing_grade):
    """Perform data analysis"""
    print("STEP 4: Analyzing Student Performance")
    print("-" * 40)
    
    if not students or not grades:
        print("❌ Cannot analyze: missing data")
        return None
    
    # Calculate statistics per student
    student_stats = {}
    
    for student_id, student_info in students.items():
        student_grades = [g for g in grades if g['student_id'] == student_id]
        
        if student_grades:
            scores = [g['grade'] for g in student_grades]
            avg_grade = sum(scores) / len(scores)
            
            student_stats[student_id] = {
                'name': student_info['name'],
                'major': student_info['major'],
                'year': student_info['year'],
                'courses_taken': len(scores),
                'average_grade': avg_grade,
                'highest_grade': max(scores),
                'lowest_grade': min(scores),
                'passing': avg_grade >= passing_grade
            }
        else:
            student_stats[student_id] = {
                'name': student_info['name'],
                'major': student_info['major'],
                'year': student_info['year'],
                'courses_taken': 0,
                'average_grade': 0,
                'highest_grade': 0,
                'lowest_grade': 0,
                'passing': False
            }
    
    # Calculate summary statistics
    all_averages = [s['average_grade'] for s in student_stats.values() if s['courses_taken'] > 0]
    
    if all_averages:
        overall_stats = {
            'total_students': len(students),
            'students_with_grades': len(all_averages),
            'class_average': sum(all_averages) / len(all_averages),
            'highest_average': max(all_averages),
            'lowest_average': min(all_averages),
            'passing_count': sum(1 for s in student_stats.values() if s['passing']),
            'failing_count': sum(1 for s in student_stats.values() if not s['passing'] and s['courses_taken'] > 0)
        }
    else:
        overall_stats = None
    
    # Display analysis results
    print("Individual Student Performance:")
    for sid, stats in student_stats.items():
        if stats['courses_taken'] > 0:
            status = "✓ PASSING" if stats['passing'] else "❌ FAILING"
            print(f"  {stats['name']} ({stats['major']}, Year {stats['year']})")
            print(f"    Average: {stats['average_grade']:.1f} ({status})")
            print(f"    Courses: {stats['courses_taken']}, Range: {stats['lowest_grade']:.1f}-{stats['highest_grade']:.1f}")
    
    if overall_stats:
        print()
        print("Class Summary:")
        print(f"  Total students: {overall_stats['total_students']}")
        print(f"  Students with grades: {overall_stats['students_with_grades']}")
        print(f"  Class average: {overall_stats['class_average']:.1f}")
        print(f"  Passing: {overall_stats['passing_count']}")
        print(f"  Failing: {overall_stats['failing_count']}")
    
    print()
    return student_stats, overall_stats

def save_results(base_path, student_stats, overall_stats):
    """Save analysis results to files"""
    print("STEP 5: Saving Results")
    print("-" * 40)
    
    if not student_stats:
        print("❌ No results to save")
        return
    
    # Save detailed results as CSV
    results_file = base_path / "results" / "reports" / "student_analysis.csv"
    
    try:
        with open(results_file, 'w', newline='') as f:
            fieldnames = ['student_id', 'name', 'major', 'year', 'courses_taken', 
                         'average_grade', 'highest_grade', 'lowest_grade', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for sid, stats in student_stats.items():
                row = {
                    'student_id': sid,
                    'name': stats['name'],
                    'major': stats['major'],
                    'year': stats['year'],
                    'courses_taken': stats['courses_taken'],
                    'average_grade': f"{stats['average_grade']:.2f}",
                    'highest_grade': f"{stats['highest_grade']:.1f}",
                    'lowest_grade': f"{stats['lowest_grade']:.1f}",
                    'status': 'PASSING' if stats['passing'] else 'FAILING'
                }
                writer.writerow(row)
        
        print(f"  ✓ Saved detailed results: {results_file.name}")
    except Exception as e:
        print(f"  ❌ Error saving results: {e}")
    
    # Save summary report as text
    summary_file = base_path / "results" / "reports" / "summary.txt"
    
    try:
        with open(summary_file, 'w') as f:
            f.write("STUDENT ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            if overall_stats:
                f.write("Overall Statistics:\n")
                f.write(f"  Total Students: {overall_stats['total_students']}\n")
                f.write(f"  Students with Grades: {overall_stats['students_with_grades']}\n")
                f.write(f"  Class Average: {overall_stats['class_average']:.1f}\n")
                f.write(f"  Passing: {overall_stats['passing_count']}\n")
                f.write(f"  Failing: {overall_stats['failing_count']}\n")
                f.write("\n")
            
            f.write("Top Performers:\n")
            sorted_students = sorted(student_stats.items(), 
                                   key=lambda x: x[1]['average_grade'], 
                                   reverse=True)
            
            for i, (sid, stats) in enumerate(sorted_students[:3], 1):
                if stats['courses_taken'] > 0:
                    f.write(f"  {i}. {stats['name']}: {stats['average_grade']:.1f}\n")
        
        print(f"  ✓ Saved summary report: {summary_file.name}")
    except Exception as e:
        print(f"  ❌ Error saving summary: {e}")
    
    # Create log entry
    log_file = base_path / "logs" / "analysis.log"
    
    try:
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] Analysis completed successfully\n")
            f.write(f"  - Processed {len(student_stats)} students\n")
            if overall_stats:
                f.write(f"  - Class average: {overall_stats['class_average']:.1f}\n")
            f.write("\n")
        
        print(f"  ✓ Updated log file: {log_file.name}")
    except Exception as e:
        print(f"  ❌ Error writing to log: {e}")
    
    print()

def demonstrate_file_organization(base_path):
    """Show final project organization"""
    print("STEP 6: Final Project Organization")
    print("-" * 40)
    
    print("Project structure after analysis:")
    print()
    
    # Show directory tree
    def show_tree(path, prefix="", max_items=5):
        """Display directory tree structure"""
        items = list(path.iterdir())
        
        for i, item in enumerate(items[:max_items]):
            is_last = i == len(items) - 1 or i == max_items - 1
            
            current = "└── " if is_last else "├── "
            print(f"{prefix}{current}{item.name}")
            
            if item.is_dir() and len(list(item.iterdir())) > 0:
                extension = "    " if is_last else "│   "
                show_tree(item, prefix + extension, max_items=3)
        
        if len(items) > max_items:
            print(f"{prefix}└── ... ({len(items) - max_items} more items)")
    
    print(f"{base_path.name}/")
    show_tree(base_path)
    print()

def demonstrate_workflow_automation():
    """Show how to automate the workflow"""
    print("STEP 7: Workflow Automation")
    print("-" * 40)
    
    print("Creating an automated analysis script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Automated Student Analysis Pipeline
Run this script to perform complete analysis
"""

import sys
from pathlib import Path

def main():
    # Configuration
    DATA_DIR = Path("data/raw")
    RESULTS_DIR = Path("results/reports")
    
    print("Starting automated analysis...")
    
    # Step 1: Validate environment
    if not DATA_DIR.exists():
        print("ERROR: Data directory not found!")
        sys.exit(1)
    
    # Step 2: Load data
    print("Loading data...")
    # [Analysis code here]
    
    # Step 3: Process and analyze
    print("Analyzing...")
    # [Processing code here]
    
    # Step 4: Generate reports
    print("Generating reports...")
    # [Reporting code here]
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
'''
    
    print("Sample automation script structure:")
    print("```python")
    for line in script_content.split('\n')[:20]:
        print(line)
    print("# ... [rest of script]")
    print("```")
    print()
    
    print("Benefits of automation:")
    print("  • Consistent results every time")
    print("  • Reduce manual errors")
    print("  • Save time on repetitive tasks")
    print("  • Easy to schedule (cron, Task Scheduler)")
    print()

def cleanup_demo(base_path):
    """Offer cleanup options"""
    print("STEP 8: Cleanup Options")
    print("-" * 40)
    
    print(f"Demo project created: {base_path}/")
    print()
    print("You can:")
    print("1. Keep it to explore the complete workflow")
    print("2. Run the analysis scripts yourself")
    print("3. Delete it when done:")
    print(f"   rm -rf {base_path}  # Mac/Linux")
    print(f"   rmdir /s {base_path}  # Windows")
    print()

def main():
    """Run the complete integration workflow demo"""
    demo_header()
    
    # Execute complete workflow
    base_path = setup_project_structure()
    students_file, grades_file, config_file = create_sample_data(base_path)
    students, grades, passing_grade = demonstrate_data_loading_with_errors(base_path)
    student_stats, overall_stats = analyze_data(students, grades, passing_grade)
    save_results(base_path, student_stats, overall_stats)
    demonstrate_file_organization(base_path)
    demonstrate_workflow_automation()
    cleanup_demo(base_path)
    
    # Final summary
    print("=" * 60)
    print("INTEGRATION WORKFLOW DEMO COMPLETE!")
    print("=" * 60)
    print()
    print("You've seen the complete data science workflow:")
    print()
    print("1. PROJECT SETUP")
    print("   → Organized directory structure")
    print("   → Separation of raw/processed data")
    print()
    print("2. DATA MANAGEMENT")
    print("   → Loading from multiple sources")
    print("   → Handling missing/invalid data")
    print("   → Configuration-driven processing")
    print()
    print("3. ANALYSIS")
    print("   → Statistical calculations")
    print("   → Error handling throughout")
    print("   → Clear result presentation")
    print()
    print("4. OUTPUT")
    print("   → Multiple output formats (CSV, TXT)")
    print("   → Organized results storage")
    print("   → Logging for reproducibility")
    print()
    print("5. AUTOMATION")
    print("   → Repeatable workflows")
    print("   → Command-line execution")
    print("   → Professional project structure")
    print()
    print("This is how real data scientists work every day!")
    print()
    print("Next steps:")
    print("• Practice this workflow with your own data")
    print("• Experiment with the demo scripts")
    print("• Build your own analysis pipelines")
    print()
    print("Remember: Good organization → Efficient analysis → Reliable results")

if __name__ == "__main__":
    main()