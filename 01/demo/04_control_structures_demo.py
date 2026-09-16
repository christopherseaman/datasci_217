#!/usr/bin/env python3
"""Demo 4: decisions, loops, and a small debugging workflow."""

print("=" * 50)
print("DEMO 4: CONTROL STRUCTURES AND DEBUGGING")
print("=" * 50)
print("1. Comparisons and decisions")
score = 85
print("score:", score)
if score >= 90:
    print("grade: A")
elif score >= 80:
    print("grade: B")
else:
    print("grade: keep practicing")
age = 25
has_experience = True
if age >= 21 and has_experience:
    print("candidate meets both requirements")
else:
    print("candidate needs another requirement")
print()
print("2. For loops over a list")
scores = [87, 92, 78, 95, 88]
total = 0
count = 0
for score in scores:
    print("score:", score)
    total = total + score
    count = count + 1
print("total:", total, "count:", count, "average:", total / count)
print()
print("3. enumerate() when a position helps")
for position, score in enumerate(scores, start=1):
    print("assignment", position, "score", score)
print()
print("4. while, break, and continue")
counter = 1
while counter <= 3:
    print("counter:", counter)
    counter = counter + 1
print("Stop at the first score above 90:")
for score in scores:
    if score > 90:
        print("found:", score)
        break
print("Skip scores below 80:")
for score in scores:
    if score < 80:
        continue
    print("processing:", score)
print()
print("5. Debugging: read the traceback, fix, save, rerun")
print("NameError: check the spelling and definition of a name.")
# Uncomment to see the NameError, then comment it again before rerunning.
# print(total_socre)
total_score = total
print("Corrected version:", total_score)
print("TypeError: check whether values are text or numbers.")
age_text = "25"
# Uncomment to see the TypeError, then comment it again before rerunning.
# print(age_text + 1)
age_number = int(age_text)
print("Corrected version:", age_number + 1)
print("ValueError: the value cannot be converted to the requested type.")
# Uncomment to see the ValueError, then comment it again before rerunning.
# invalid_number = int("hello")
valid_number = int("42")
print("Corrected version:", valid_number)
print("Inspect values with print() and type() before guessing.")
print()
print("CONTROL STRUCTURES COMPLETE")
print("Next: run the small end-to-end workflow.")
