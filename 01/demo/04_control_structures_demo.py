#!/usr/bin/env python3
"""
Demo 4: Control Structures and Operations
Lecture 01 - Command Line + Python

This demo covers control structures (if/elif/else, loops) and operations
with intentional mistakes to practice debugging, corresponding to the
"Control Structures" section of the lecture.

Usage: python 04_control_structures_demo.py

Author: Data Science 217 Course Materials
"""

def demo_header():
    """Display demo introduction"""
    print("=" * 60)
    print("DEMO 4: CONTROL STRUCTURES & OPERATIONS")
    print("=" * 60)
    print("Goal: Master decision-making and repetition in Python")
    print("Watch for logic errors and off-by-one mistakes!")
    print()

def demonstrate_comparison_operators():
    """Show comparison operators and common logical errors"""
    print("STEP 1: Comparison Operators")
    print("-" * 40)
    
    # Basic comparisons
    print("Basic comparisons:")
    a = 10
    b = 20
    c = 10
    
    print(f"  a = {a}, b = {b}, c = {c}")
    print(f"  a == b: {a == b}  (equal to)")
    print(f"  a == c: {a == c}  (equal to)")
    print(f"  a != b: {a != b}  (not equal)")
    print(f"  a < b:  {a < b}   (less than)")
    print(f"  a <= c: {a <= c}  (less than or equal)")
    print(f"  b > a:  {b > a}   (greater than)")
    print()
    
    # INTENTIONAL ERROR 1: Assignment vs comparison
    print("Common mistake - using = instead of ==:")
    score = 85
    print(f"Score: {score}")
    
    # This is assignment, not comparison!
    # if score = 90:  # This would cause SyntaxError
    #     print("Perfect score!")
    
    print("❌ Wrong: if score = 90  (This assigns 90 to score!)")
    print("✓ Right: if score == 90  (This compares score to 90)")
    
    if score == 85:
        print(f"  Score is exactly 85")
    print()
    
    # INTENTIONAL ERROR 2: String vs number comparison
    print("Type confusion in comparisons:")
    user_input = "100"  # This is a string!
    threshold = 50      # This is a number
    
    print(f'  user_input = "{user_input}" (string)')
    print(f"  threshold = {threshold} (integer)")
    
    # This comparison might not work as expected
    try:
        if user_input > threshold:
            print("  Input is greater")
    except TypeError as e:
        print(f"❌ ERROR: {e}")
        print("💡 FIX: Convert string to number first")
        if int(user_input) > threshold:
            print(f"✓ {int(user_input)} > {threshold} is True")
    print()

def demonstrate_if_statements():
    """Show if/elif/else with data science examples"""
    print("STEP 2: If/Elif/Else Decision Making")
    print("-" * 40)
    
    # Grade calculation example
    print("Grade calculation system:")
    
    def calculate_grade(score):
        """Calculate letter grade from numerical score"""
        print(f"  Score: {score}", end=" -> ")
        
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        print(f"Grade: {grade}")
        return grade
    
    # Test various scores
    test_scores = [95, 85, 75, 65, 55]
    for score in test_scores:
        calculate_grade(score)
    print()
    
    # INTENTIONAL ERROR 3: Order of conditions matters!
    print("Wrong order of conditions:")
    
    def bad_grade_calc(score):
        """Incorrectly ordered conditions"""
        if score >= 60:  # This catches everything >= 60!
            return "D"
        elif score >= 70:  # This will never be reached
            return "C"
        elif score >= 80:  # Neither will this
            return "B"
        elif score >= 90:  # Or this
            return "A"
        else:
            return "F"
    
    score = 95
    print(f"  Score {score} with bad logic: {bad_grade_calc(score)}")
    print("❌ Wrong: Checking broader conditions first")
    print("✓ Fix: Check more specific conditions first (90, then 80, etc.)")
    print()
    
    # Compound conditions
    print("Compound conditions with and/or:")
    age = 25
    has_experience = True
    degree = "BS"
    
    print(f"  Candidate: age={age}, experience={has_experience}, degree={degree}")
    
    if age >= 21 and has_experience:
        print("  ✓ Qualified for senior position")
    elif age >= 21 or degree == "BS":
        print("  ✓ Qualified for entry position")
    else:
        print("  ❌ Not qualified")
    print()

def demonstrate_for_loops():
    """Show for loops with common patterns and errors"""
    print("STEP 3: For Loops - Iteration Patterns")
    print("-" * 40)
    
    # Basic iteration
    print("Basic iteration over list:")
    fruits = ["apple", "banana", "orange"]
    for fruit in fruits:
        print(f"  Processing: {fruit}")
    print()
    
    # Using range
    print("Using range() for counting:")
    print("  range(5):", list(range(5)))  # 0, 1, 2, 3, 4
    print("  range(1, 6):", list(range(1, 6)))  # 1, 2, 3, 4, 5
    print("  range(0, 10, 2):", list(range(0, 10, 2)))  # 0, 2, 4, 6, 8
    print()
    
    # INTENTIONAL ERROR 4: Off-by-one error
    print("Common off-by-one error:")
    numbers = [10, 20, 30, 40, 50]
    
    print("  Wrong way (trying to use 1-based indexing):")
    for i in range(1, len(numbers) + 1):
        try:
            print(f"    Position {i}: {numbers[i]}")
        except IndexError:
            print(f"    Position {i}: ❌ IndexError!")
    
    print()
    print("  Correct way (0-based indexing):")
    for i in range(len(numbers)):
        print(f"    Position {i}: {numbers[i]}")
    print()
    
    # Enumerate for index and value
    print("Using enumerate() for index and value:")
    for i, num in enumerate(numbers):
        print(f"  Index {i}: {num}")
    print()
    
    # Data science example: calculating statistics
    print("Data analysis with loops:")
    scores = [87, 92, 78, 95, 88, 91]
    
    total = 0
    count = 0
    above_90 = 0
    
    for score in scores:
        total += score
        count += 1
        if score > 90:
            above_90 += 1
            print(f"  Score {score}: Above 90! ⭐")
        else:
            print(f"  Score {score}")
    
    average = total / count
    print(f"  Average: {average:.1f}")
    print(f"  Scores above 90: {above_90}/{count}")
    print()

def demonstrate_while_loops():
    """Show while loops with proper termination conditions"""
    print("STEP 4: While Loops - Conditional Repetition")
    print("-" * 40)
    
    # Basic while loop
    print("Basic while loop:")
    counter = 0
    while counter < 5:
        print(f"  Counter: {counter}")
        counter += 1  # Don't forget this!
    print()
    
    # INTENTIONAL ERROR 5: Infinite loop risk
    print("Dangerous: Forgetting to update loop variable")
    print("  BAD CODE (commented out to prevent infinite loop):")
    print("  # counter = 0")
    print("  # while counter < 5:")
    print("  #     print(counter)")
    print("  #     # Forgot: counter += 1")
    print("❌ This creates an infinite loop!")
    print("💡 Always ensure loop variable changes")
    print()
    
    # Data validation example
    print("Data validation with while loop:")
    
    def get_valid_score():
        """Simulate getting valid input"""
        # Simulating user inputs
        attempts = ["-5", "105", "abc", "85"]
        attempt_num = 0
        
        while attempt_num < len(attempts):
            user_input = attempts[attempt_num]
            print(f"  Input: '{user_input}'")
            
            try:
                score = float(user_input)
                if 0 <= score <= 100:
                    print(f"    ✓ Valid score: {score}")
                    return score
                else:
                    print(f"    ❌ Score must be 0-100")
            except ValueError:
                print(f"    ❌ Not a number")
            
            attempt_num += 1
        
        print("    Using default score: 0")
        return 0
    
    valid_score = get_valid_score()
    print()

def demonstrate_nested_structures():
    """Show nested loops and conditions"""
    print("STEP 5: Nested Structures")
    print("-" * 40)
    
    # Nested loops for table
    print("Creating a multiplication table:")
    for i in range(1, 4):
        for j in range(1, 4):
            result = i * j
            print(f"  {i} × {j} = {result:2}", end="  ")
        print()  # New line after each row
    print()
    
    # Nested conditions
    print("Complex decision tree:")
    
    def classify_student(gpa, credits):
        """Classify student based on GPA and credits"""
        print(f"  GPA: {gpa}, Credits: {credits}")
        
        if gpa >= 3.5:
            if credits >= 90:
                status = "Dean's List Senior"
            elif credits >= 60:
                status = "Dean's List Junior"
            else:
                status = "Dean's List Underclassman"
        elif gpa >= 2.0:
            if credits >= 90:
                status = "Senior"
            elif credits >= 60:
                status = "Junior"
            elif credits >= 30:
                status = "Sophomore"
            else:
                status = "Freshman"
        else:
            status = "Academic Probation"
        
        print(f"    Status: {status}")
        return status
    
    # Test cases
    test_students = [
        (3.8, 95),  # Dean's List Senior
        (3.6, 45),  # Dean's List Underclassman
        (2.5, 70),  # Junior
        (1.8, 50),  # Academic Probation
    ]
    
    for gpa, credits in test_students:
        classify_student(gpa, credits)
    print()

def demonstrate_list_comprehensions():
    """Show Pythonic list comprehensions vs loops"""
    print("STEP 6: List Comprehensions - Pythonic Style")
    print("-" * 40)
    
    # Traditional loop approach
    print("Traditional loop approach:")
    numbers = [1, 2, 3, 4, 5]
    squared_loop = []
    for n in numbers:
        squared_loop.append(n ** 2)
    print(f"  Squared (loop): {squared_loop}")
    
    # List comprehension
    print("List comprehension (more Pythonic):")
    squared_comp = [n ** 2 for n in numbers]
    print(f"  Squared (comprehension): {squared_comp}")
    print()
    
    # With condition
    print("List comprehension with filter:")
    evens = [n for n in range(10) if n % 2 == 0]
    print(f"  Even numbers: {evens}")
    
    # Data science example
    print()
    print("Data science application:")
    scores = [87, 92, 78, 95, 88, 65, 91, 73]
    
    # Filter passing scores
    passing = [s for s in scores if s >= 70]
    print(f"  Passing scores (≥70): {passing}")
    
    # Convert to letter grades
    grades = ['A' if s >= 90 else 'B' if s >= 80 else 'C' if s >= 70 else 'D' if s >= 60 else 'F' 
              for s in scores]
    print(f"  Letter grades: {grades}")
    
    # Calculate adjustments
    curved = [min(100, s + 5) for s in scores]  # Add 5 points, max 100
    print(f"  Curved scores (+5): {curved}")
    print()

def demonstrate_break_continue():
    """Show loop control with break and continue"""
    print("STEP 7: Loop Control - Break and Continue")
    print("-" * 40)
    
    # Using break
    print("Using 'break' to exit early:")
    print("  Finding first score above 90:")
    
    scores = [85, 88, 92, 87, 95, 89]
    for i, score in enumerate(scores):
        print(f"    Checking score {i}: {score}")
        if score > 90:
            print(f"    ✓ Found! Score {score} at position {i}")
            break
    print()
    
    # Using continue
    print("Using 'continue' to skip iterations:")
    print("  Processing only valid data:")
    
    data = [25, -5, 30, None, 45, "invalid", 35]
    total = 0
    count = 0
    
    for value in data:
        if value is None:
            print(f"    Skipping None value")
            continue
        if not isinstance(value, (int, float)):
            print(f"    Skipping invalid type: {value}")
            continue
        if value < 0:
            print(f"    Skipping negative: {value}")
            continue
        
        print(f"    Processing: {value}")
        total += value
        count += 1
    
    if count > 0:
        average = total / count
        print(f"  Average of valid values: {average:.1f}")
    print()

def demonstrate_practical_example():
    """Complete practical example combining all concepts"""
    print("STEP 8: Practical Example - Grade Analysis System")
    print("-" * 40)
    
    # Sample student data
    students = [
        {"name": "Alice", "scores": [88, 92, 85, 90]},
        {"name": "Bob", "scores": [75, 82, 78]},
        {"name": "Charlie", "scores": [92, 95, 88, 91, 94]},
        {"name": "Diana", "scores": [68, 72, 65, 70]},
        {"name": "Eve", "scores": []},  # No scores yet
    ]
    
    print("Student Performance Analysis:")
    print()
    
    # Process each student
    for student in students:
        name = student["name"]
        scores = student["scores"]
        
        print(f"{name}:")
        
        # Handle edge case: no scores
        if not scores:
            print(f"  ⚠️ No scores recorded")
            continue
        
        # Calculate statistics
        total = 0
        highest = scores[0]
        lowest = scores[0]
        passing_count = 0
        
        for score in scores:
            total += score
            
            if score > highest:
                highest = score
            if score < lowest:
                lowest = score
            
            if score >= 70:
                passing_count += 1
        
        average = total / len(scores)
        
        # Determine overall grade
        if average >= 90:
            letter_grade = "A"
            status = "Excellent! 🌟"
        elif average >= 80:
            letter_grade = "B"
            status = "Good work!"
        elif average >= 70:
            letter_grade = "C"
            status = "Satisfactory"
        elif average >= 60:
            letter_grade = "D"
            status = "Needs improvement"
        else:
            letter_grade = "F"
            status = "Failing - seek help"
        
        # Display results
        print(f"  Scores: {scores}")
        print(f"  Average: {average:.1f}")
        print(f"  Range: {lowest} - {highest}")
        print(f"  Grade: {letter_grade} ({status})")
        print(f"  Passing rate: {passing_count}/{len(scores)} assignments")
        
        # Warning for at-risk students
        if average < 70:
            print(f"  ⚠️ ALERT: Student needs academic support")
        
        print()
    
    # Class summary
    print("Class Summary:")
    all_averages = []
    for student in students:
        if student["scores"]:
            avg = sum(student["scores"]) / len(student["scores"])
            all_averages.append(avg)
    
    if all_averages:
        class_average = sum(all_averages) / len(all_averages)
        print(f"  Class average: {class_average:.1f}")
        print(f"  Highest student average: {max(all_averages):.1f}")
        print(f"  Lowest student average: {min(all_averages):.1f}")

def main():
    """Run all control structure demos"""
    demo_header()
    
    # Run all demonstration steps
    demonstrate_comparison_operators()
    demonstrate_if_statements()
    demonstrate_for_loops()
    demonstrate_while_loops()
    demonstrate_nested_structures()
    demonstrate_list_comprehensions()
    demonstrate_break_continue()
    demonstrate_practical_example()
    
    # Final summary
    print("=" * 60)
    print("CONTROL STRUCTURES DEMO COMPLETE!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("1. Use == for comparison, = for assignment")
    print("2. Order matters in if/elif chains - specific first!")
    print("3. Remember Python uses 0-based indexing")
    print("4. Always ensure while loops can terminate")
    print("5. List comprehensions are Pythonic and efficient")
    print("6. Use break/continue for loop control")
    print("7. Handle edge cases (empty lists, invalid data)")
    print()
    print("Next: Complete workflow integration!")

if __name__ == "__main__":
    main()