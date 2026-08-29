#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "CLI NAVIGATION DEMO"
echo "=========================================="

echo "Where am I?"
pwd

echo -e "\nWhat's here?"
ls

echo -e "\nLet's create a project:"
mkdir data_project
cd data_project
echo "Now in: $(pwd)"

echo -e "\nCreate some structure:"
mkdir data scripts
echo "Created directories:"
ls

echo -e "\nMake a data file:"
echo "name,score" > data/grades.csv
echo "Alice,95" >> data/grades.csv
echo "Bob,87" >> data/grades.csv

echo -e "\nView the file:"
cat data/grades.csv

echo -e "\nTry to analyze it:"
cat > scripts/analyze.py << 'EOF'
with open('grades.csv', 'r') as f:
    # with open('data/grades.csv', 'r') as f:  # <- FIXED
    print(f.read())
EOF

echo "Running analysis script..."
cd scripts
if python3 analyze.py 2>../path_error.txt; then
    echo "Unexpected success: inspect the working directory and path."
else
    echo "Expected path error captured (the script continues so we can fix it):"
    cat ../path_error.txt
fi

cat > analyze.py << 'EOF'
with open('../data/grades.csv', 'r') as f:
    print(f.read())
EOF

echo "Running the corrected analysis script..."
python3 analyze.py

echo -e "\n💡 Fix: the data path is relative to the scripts/ directory"

echo -e "\nCleanup:"
cd ../..
echo "Back to: $(pwd)"
echo "To remove project after verifying your location: rm -r data_project"

echo -e "\n=========================================="
echo "Key commands: pwd, ls, mkdir, cd, cat"
echo "Remember: Check your location with pwd!"
echo "=========================================="
