#!/bin/bash

# CLI Advanced Demo Script
# This script demonstrates advanced command line operations
# and builds a data processing pipeline step by step

echo "=== CLI Advanced Demo ==="
echo "Building a data processing pipeline step by step"
echo

# Step 1: Create project structure
echo "Step 1: Creating project structure..."
mkdir -p data/{raw,processed} scripts results logs
echo "Project directories created"
echo

# Step 2: Generate sample data
echo "Step 2: Creating sample data files..."
cat > data/raw/students.csv << 'EOF'
name,age,grade,subject
Alice,20,85,Math
Bob,19,92,Science
Charlie,21,78,English
Diana,20,88,Math
Eve,22,95,Science
EOF

cat > data/raw/courses.csv << 'EOF'
course_id,name,credits,instructor
MATH101,Calculus I,4,Dr. Smith
SCI201,Physics,3,Dr. Johnson
ENG101,Composition,3,Prof. Brown
MATH102,Calculus II,4,Dr. Smith
SCI202,Chemistry,3,Dr. Wilson
EOF

echo "Sample data files created"
echo

# Step 3: Basic file operations
echo "Step 3: Basic file operations..."
echo "Current directory structure:"
tree -L 3
echo

echo "File sizes:"
ls -lh data/raw/*.csv
echo

# Step 4: Text processing and search
echo "Step 4: Text processing and search..."
echo "First 3 lines of students.csv:"
head -3 data/raw/students.csv
echo

echo "Last 2 lines of courses.csv:"
tail -2 data/raw/courses.csv
echo

echo "Searching for 'Math' in all CSV files:"
grep -i "math" data/raw/*.csv
echo

# Step 5: Data processing pipeline
echo "Step 5: Building data processing pipeline..."
echo "Counting lines in each file:"
wc -l data/raw/*.csv
echo

echo "Extracting unique subjects:"
cut -d',' -f4 data/raw/students.csv | tail -n +2 | sort | uniq
echo

# Step 6: Advanced text processing
echo "Step 6: Advanced text processing..."
echo "Students with grades above 85:"
awk -F',' '$3 > 85 {print $1, $3}' data/raw/students.csv
echo

echo "Average grade calculation:"
awk -F',' 'NR>1 {sum+=$3; count++} END {print "Average grade:", sum/count}' data/raw/students.csv
echo

# Step 7: File operations with wildcards
echo "Step 7: File operations with wildcards..."
echo "Copying all CSV files to processed directory:"
cp data/raw/*.csv data/processed/
echo "Files copied to processed directory"
echo

echo "Creating backup with timestamp:"
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p backups
cp data/raw/*.csv backups/backup_${timestamp}/
echo "Backup created: backup_${timestamp}"
echo

# Step 8: Command chaining and redirection
echo "Step 8: Command chaining and redirection..."
echo "Creating summary report:"
{
    echo "# Data Processing Summary"
    echo "Generated on: $(date)"
    echo
    echo "## File Statistics"
    echo "Raw data files:"
    ls -la data/raw/
    echo
    echo "Processed files:"
    ls -la data/processed/
    echo
    echo "## Data Analysis"
    echo "Total students: $(tail -n +2 data/raw/students.csv | wc -l)"
    echo "Total courses: $(tail -n +2 data/raw/courses.csv | wc -l)"
    echo "Average grade: $(awk -F',' 'NR>1 {sum+=$3; count++} END {printf "%.1f", sum/count}' data/raw/students.csv)"
} > results/summary_report.txt

echo "Summary report created: results/summary_report.txt"
echo

# Step 9: Error handling and validation
echo "Step 9: Error handling and validation..."
echo "Checking if required files exist:"
for file in data/raw/students.csv data/raw/courses.csv; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
    fi
done
echo

echo "Validating CSV format (checking for proper headers):"
head -1 data/raw/students.csv | grep -q "name,age,grade,subject" && echo "✓ Students CSV format valid" || echo "✗ Students CSV format invalid"
head -1 data/raw/courses.csv | grep -q "course_id,name,credits,instructor" && echo "✓ Courses CSV format valid" || echo "✗ Courses CSV format invalid"
echo

# Step 10: Automation and logging
echo "Step 10: Automation and logging..."
echo "Creating processing log:"
{
    echo "$(date): Data processing started"
    echo "$(date): Files validated successfully"
    echo "$(date): Summary report generated"
    echo "$(date): Processing completed"
} >> logs/processing.log

echo "Processing log created: logs/processing.log"
echo

# Step 11: Final project structure
echo "Step 11: Final project structure:"
tree
echo

echo "=== Demo Complete ==="
echo "Key concepts demonstrated:"
echo "1. Directory creation and navigation"
echo "2. File operations with wildcards"
echo "3. Text processing (head, tail, grep, awk)"
echo "4. Command chaining and redirection"
echo "5. Error handling and validation"
echo "6. Automation and logging"
echo
echo "This script shows how CLI tools can automate complex data processing workflows!"
