#!/bin/bash

echo "============================================================"
echo "DEMO 2: COMMAND LINE NAVIGATION & PROJECT SETUP"
echo "============================================================"
echo "Goal: Learn essential command line operations"
echo "This teaches real CLI skills for data science workflows."
echo

echo "STEP 1: Navigation Concepts"
echo "----------------------------------------"
echo "Where am I? (pwd)"
pwd
echo

echo "What's here? (ls)"
ls -la
echo

echo "Parent directory contents:"
echo "Current: $(pwd)"
echo "Parent: $(dirname "$(pwd)")"
ls -la "$(dirname "$(pwd)")"
echo

echo "STEP 2: Creating Data Science Project Structure"
echo "----------------------------------------"
echo "Creating project: my_data_analysis"
echo

# Create main project directory
mkdir -p my_data_analysis
echo "✓ Created main directory: my_data_analysis/"

# Create subdirectories
cd my_data_analysis
mkdir -p data/raw data/processed scripts results docs
echo "✓ Created: data/raw/ - Original, unmodified data"
echo "✓ Created: data/processed/ - Cleaned and transformed data"
echo "✓ Created: scripts/ - Analysis scripts"
echo "✓ Created: results/ - Output files, charts, reports"
echo "✓ Created: docs/ - Documentation and notes"
echo

echo "Let's see our project structure:"
ls -la
echo

echo "STEP 3: File Operations"
echo "----------------------------------------"

# Create sample data file
cat > data/raw/sample_data.csv << 'EOF'
name,age,city,score
Alice,25,San Francisco,87.5
Bob,30,New York,92.1
Charlie,28,Boston,78.3
Diana,26,Seattle,95.2
EOF
echo "✓ Created sample data: data/raw/sample_data.csv"

# Create analysis script with intentional path issue
cat > scripts/analyze_data.py << 'EOF'
#!/usr/bin/env python3
"""
Sample analysis script with intentional path issue
"""
import csv

def load_data():
    # This path will cause issues when run from project root!
    data_file = 'data/raw/sample_data.csv'  # Missing scripts/ prefix

    try:
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {data_file}")
        print("💡 Hint: Check your current working directory!")
        print(f"   Current directory: $(pwd)")
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
EOF
echo "✓ Created analysis script: scripts/analyze_data.py"

# Create README
cat > README.md << 'EOF'
# My Data Analysis Project

## Structure
- `data/raw/` - Original datasets
- `data/processed/` - Cleaned data
- `scripts/` - Analysis scripts
- `results/` - Output files
- `docs/` - Documentation

## Usage
Run analysis from project root:
```bash
python3 scripts/analyze_data.py
```
EOF
echo "✓ Created project README: README.md"
echo

echo "STEP 4: Common Path Problems & Solutions"
echo "----------------------------------------"
echo "Let's try to run our analysis script and see what happens..."
echo

echo "First, let's see where we are:"
pwd
echo

echo "Attempting to run: scripts/analyze_data.py"
echo "Running from current directory (project root)..."
python3 scripts/analyze_data.py
echo

echo "💡 Problem: The script looks for 'data/raw/sample_data.csv' from scripts/ directory"
echo "Let's try running from scripts/ directory:"
echo

cd scripts
echo "Current directory: $(pwd)"
python3 analyze_data.py
echo

echo "❌ Still fails! Now it can't find the data from scripts/ directory"
echo "💡 Solution: Fix the path in the script"
echo

echo "Let's fix the script to use correct relative path:"
cat > analyze_data_fixed.py << 'EOF'
#!/usr/bin/env python3
"""
Fixed analysis script with correct paths
"""
import csv
import os

def load_data():
    # Fix: Use path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, '..', 'data', 'raw', 'sample_data.csv')

    try:
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {data_file}")
        return []

if __name__ == "__main__":
    print("Loading student data with fixed paths...")
    students = load_data()

    if students:
        print(f"✓ Loaded {len(students)} students")
        for student in students:
            print(f"  {student['name']}: {student['score']}")
    else:
        print("❌ No data loaded")
EOF

echo "Running fixed script:"
python3 analyze_data_fixed.py
echo

# Go back to project root
cd ..
echo "Back to project root: $(pwd)"
echo

echo "STEP 5: Viewing Files (head/tail/cat)"
echo "----------------------------------------"
echo "Viewing file: data/raw/sample_data.csv"
echo

echo "Entire file contents (cat):"
cat data/raw/sample_data.csv
echo

echo "First 3 lines (head -n 3):"
head -n 3 data/raw/sample_data.csv
echo

echo "Last 2 lines (tail -n 2):"
tail -n 2 data/raw/sample_data.csv
echo

echo "File info (wc):"
wc data/raw/sample_data.csv
echo

echo "STEP 6: Directory Navigation Practice"
echo "----------------------------------------"
echo "Current location: $(pwd)"
echo

echo "Move to data directory:"
cd data
echo "Now in: $(pwd)"
echo "Contents: $(ls)"
echo

echo "Move to raw subdirectory:"
cd raw
echo "Now in: $(pwd)"
echo "Contents: $(ls -la)"
echo

echo "Go back to project root using absolute path:"
cd ../..
echo "Back to: $(pwd)"
echo

echo "Alternative: relative path navigation"
cd data/raw
echo "In data/raw: $(pwd)"
cd ../../
echo "Back to root: $(pwd)"
echo

echo "STEP 7: Cleanup (Optional)"
echo "----------------------------------------"
echo "Demo project exists at: $(pwd)"
echo "You can:"
echo "1. Keep it to explore the structure: cd my_data_analysis"
echo "2. Delete it to clean up: rm -rf my_data_analysis"
echo

# Go back to demo directory
cd ..
echo "Returned to demo directory: $(pwd)"
echo

echo "============================================================"
echo "COMMAND LINE NAVIGATION DEMO COMPLETE!"
echo "============================================================"
echo
echo "Key Takeaways:"
echo "1. pwd shows current directory"
echo "2. ls lists directory contents (-la for details)"
echo "3. mkdir creates directories (-p for nested)"
echo "4. cd changes directories (absolute vs relative paths)"
echo "5. cat/head/tail view file contents"
echo "6. Relative paths depend on current working directory"
echo "7. Always organize projects with clear structure"
echo
echo "Practice these commands - they're essential for data science!"