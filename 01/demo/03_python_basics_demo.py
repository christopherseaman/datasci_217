#!/usr/bin/env python3
"""Demo 3: the smallest useful Python building blocks."""

print("=" * 50)
print("DEMO 3: PYTHON BASICS")
print("=" * 50)
print("Variables, types, arithmetic, and strings.")
print()
print("1. Variables and types")
student_name = "Alice"
student_age = 22
average_score = 87.5
is_enrolled = True
print("name:", student_name, "type:", type(student_name))
print("age:", student_age, "type:", type(student_age))
print("score:", average_score, "type:", type(average_score))
print("enrolled:", is_enrolled, "type:", type(is_enrolled))
print()
print("2. Arithmetic")
hours_studied = 3
score_per_hour = 10
points_earned = hours_studied * score_per_hour
print("hours:", hours_studied)
print("points per hour:", score_per_hour)
print("points earned:", points_earned)
print("next score:", average_score + 2)
print()
print("3. Strings and length")
course = "Data Science 217"
welcome = "Welcome to " + course
print(welcome)
print("course length:", len(course))
print("upper case:", course.upper())
print("trimmed text:", "  ready  ".strip())
print()
print("4. A tiny calculation")
weight_kg = 70
height_m = 1.75
bmi = weight_kg / (height_m * height_m)
print("weight:", weight_kg, "kg")
print("height:", height_m, "m")
print("BMI:", bmi)
print()
print("PYTHON BASICS COMPLETE")
print("Next: control structures, loops, and debugging.")
