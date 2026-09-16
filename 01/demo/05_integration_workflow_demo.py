#!/usr/bin/env python3
"""A compact end-to-end workflow for Lecture 01."""

print("=" * 50)
print("DEMO 4: COMPLETE WORKFLOW")
print("=" * 50)
print("Run this script from the shell after the controls demo.")
print()
scores = [92, 76, 88, 64]
passing_score = 70
total = 0
count = 0
passing = 0
for position, score in enumerate(scores, start=1):
    total = total + score
    count = count + 1
    if score >= passing_score:
        status = "PASS"
        passing = passing + 1
    else:
        status = "REVIEW"
    print("Student", position, "score:", score, status)
average = total / count
print()
print("total:", total)
print("count:", count)
print("average:", average)
print("passing:", passing)
print("Workflow complete.")
