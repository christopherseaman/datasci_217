#!/usr/bin/env python3
"""
Sample analysis script with intentional path issue
"""
import csv

def load_data():
    # This path will cause issues!
    data_file = '../data/raw/sample_data.csv'

    try:
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {data_file}")
        print("💡 Hint: Check your current working directory!")
        return []

if __name__ == "__main__":
    print("Loading student data...")
    students = load_data()

    if students:
        print(f"✓ Loaded {len(students)} students")
        for student in students:
            print(f"  {student['name']}: {student['score']}")
    else:
        print("❌ No data loaded")
