#!/usr/bin/env python3
"""
Demo 4: Control Structures and Operations
Lecture 01 - Command Line + Python

This demo practices comparisons, decisions, loops, and simple arithmetic.

Usage: python 04_control_structures_demo.py

Author: Data Science 217 Course Materials
"""

print("=" * 60)
print("DEMO 4: CONTROL STRUCTURES & OPERATIONS")
print("=" * 60)
print("Goal: Make decisions, repeat work, and control a loop")
print()

# STEP 1: Comparison Operators
print("STEP 1: Comparison Operators")
print("-" * 40)

a = 10
b = 20
print(f"a = {a}, b = {b}")
print(f"a == b: {a == b}")
print(f"a != b: {a != b}")
print(f"a < b:  {a < b}")
print(f"a <= b: {a <= b}")
print(f"a > b:  {a > b}")
print(f"a >= b: {a >= b}")
print("Remember: = assigns a value; == compares values.")
print()

# STEP 2: If/Elif/Else Decision Making
print("STEP 2: If/Elif/Else Decision Making")
print("-" * 40)

score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "Needs more practice"
print(f"A score of {score} earns: {grade}")

age = 25
has_experience = True
if age >= 21 and has_experience:
    print("The candidate meets both requirements.")
elif age >= 21 or has_experience:
    print("The candidate meets one requirement.")
else:
    print("The candidate meets neither requirement.")
print()

# STEP 3: For Loops and enumerate
print("STEP 3: For Loops and enumerate()")
print("-" * 40)

scores = [87, 92, 78, 95, 88]
print("Scores:")
for score in scores:
    print(f"  {score}")

print("Scores with their positions:")
for position, score in enumerate(scores, start=1):
    print(f"  Position {position}: {score}")

total = 0
for score in scores:
    total += score
average = total / len(scores)
print(f"Average score: {average:.1f}")
print()

# STEP 4: While Loops
print("STEP 4: While Loops")
print("-" * 40)

counter = 1
while counter <= 3:
    print(f"  Counter: {counter}")
    counter += 1
print("A while loop repeats while its condition is True.")
print("Always update the loop variable so the loop can finish.")
print()

# STEP 5: Nested Loops
print("STEP 5: Nested Loops")
print("-" * 40)

print("A small multiplication table:")
for row in range(1, 4):
    for column in range(1, 4):
        print(f"  {row} x {column} = {row * column}")
print()

# STEP 6: Break and Continue
print("STEP 6: break and continue")
print("-" * 40)

print("Stop at the first score above 90:")
for position, score in enumerate(scores, start=1):
    print(f"  Checking position {position}: {score}")
    if score > 90:
        print(f"  Found {score}; break stops the loop here.")
        break

print("Skip scores below 80:")
for score in scores:
    if score < 80:
        print(f"  Skipping {score}")
        continue
    print(f"  Processing {score}")
print()

# STEP 7: Practical Example - Grade Analysis
print("STEP 7: Practical Example - Grade Analysis")
print("-" * 40)

student_names = ["Alice", "Bob", "Charlie", "Diana"]
student_scores = [92, 76, 88, 64]
passing_count = 0
total = 0

for position, name in enumerate(student_names):
    score = student_scores[position]
    total += score

    if score >= 90:
        status = "excellent"
    elif score >= 70:
        status = "passing"
    else:
        status = "needs support"

    if score >= 70:
        passing_count += 1
    print(f"{name}: {score} ({status})")

class_average = total / len(student_scores)
print(f"Class average: {class_average:.1f}")
print(f"Students at or above 70: {passing_count}/{len(student_scores)}")
print()

print("=" * 60)
print("CONTROL STRUCTURES DEMO COMPLETE!")
print("=" * 60)
print("Key takeaways: compare, decide, repeat, and control your loops.")
print("Next: Complete workflow integration!")
