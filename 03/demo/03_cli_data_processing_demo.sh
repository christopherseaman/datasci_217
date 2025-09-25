#!/bin/bash
# Demo 3: Command Line Data Processing
# Demonstrates shell tools for data manipulation
# This demonstrates the command line section starting at line 386 of Lecture 3

echo "COMMAND LINE DATA PROCESSING DEMO"
echo "=================================="
echo "This demo demonstrates the specific shell tools mentioned"
echo "in Lecture 3, showing data processing pipelines with real examples."
echo "=================================="

# Create demo directory
mkdir -p cli_demo
cd cli_demo

echo ""
echo "SETTING UP SAMPLE DATA"
echo "======================"

# Create comprehensive student dataset (as mentioned in lecture)
cat > students.csv << 'EOF'
name,age,major,grade_math,grade_science,grade_english,total_credits,gpa
Alice Johnson,20,Biology,85,92,78,45,3.2
Bob Smith,19,Chemistry,92,88,85,42,3.5
Charlie Brown,21,Physics,78,95,82,48,3.1
Diana Wilson,20,Biology,88,85,90,45,3.4
Eve Chen,22,Mathematics,95,88,92,51,3.8
Frank Miller,19,Chemistry,82,90,76,42,3.3
Grace Lee,21,Biology,90,85,88,45,3.6
Henry Davis,20,Physics,85,92,85,48,3.4
Iris Garcia,19,Mathematics,88,95,90,42,3.7
Jack Wilson,22,Chemistry,92,88,85,51,3.5
EOF

# Create sales data (as mentioned in lecture)
cat > sales.csv << 'EOF'
date,product,category,quantity,price,total,region
2024-01-01,Laptop,Electronics,5,1200,6000,West
2024-01-01,Book,Education,20,25,500,East
2024-01-02,Phone,Electronics,8,800,6400,West
2024-01-02,Notebook,Education,50,5,250,East
2024-01-03,Tablet,Electronics,3,600,1800,West
2024-01-03,Pencil,Education,100,1,100,East
2024-01-04,Laptop,Electronics,2,1200,2400,West
2024-01-04,Book,Education,15,25,375,East
2024-01-05,Phone,Electronics,6,800,4800,West
2024-01-05,Notebook,Education,30,5,150,East
EOF

# Create log data (as mentioned in lecture)
cat > system.log << 'EOF'
2024-01-15 10:30:15 INFO System startup complete
2024-01-15 10:31:22 ERROR Database connection failed
2024-01-15 10:32:45 INFO User login: alice@ucsf.edu
2024-01-15 10:33:12 WARNING High memory usage detected
2024-01-15 10:34:01 INFO Data processing started
2024-01-15 10:35:30 ERROR File not found: config.json
2024-01-15 10:36:15 INFO User logout: alice@ucsf.edu
2024-01-15 10:37:22 WARNING Disk space low
2024-01-15 10:38:45 INFO Backup completed successfully
2024-01-15 10:39:12 ERROR Network timeout
2024-01-15 10:40:01 INFO System shutdown initiated
EOF

echo "✓ Created students.csv (10 student records)"
echo "✓ Created sales.csv (10 sales records)"
echo "✓ Created system.log (system log entries)"

echo ""
echo "ESSENTIAL SHELL TOOLS FOR DATA SCIENCE"
echo "======================================"
echo "Demonstrating the specific tools mentioned in Lecture 3:"
echo ""

echo "1. TEXT PROCESSING WITH cut, sort, and grep"
echo "============================================="

echo ""
echo "Using cut to extract columns:"
echo "-----------------------------"
echo "Extract names and ages (columns 1 and 2):"
cut -d',' -f1,2 students.csv

echo ""
echo "Extract names and GPAs (columns 1 and 8):"
cut -d',' -f1,8 students.csv

echo ""
echo "Extract grade columns (columns 4, 5, 6):"
cut -d',' -f4,5,6 students.csv

echo ""
echo "Using sort to organize data:"
echo "---------------------------"
echo "Sort alphabetically by name:"
sort students.csv

echo ""
echo "Sort numerically by GPA (column 8):"
sort -t',' -k8 -n students.csv

echo ""
echo "Sort by age in reverse order:"
sort -t',' -k2 -nr students.csv

echo ""
echo "Using grep to search and filter:"
echo "-------------------------------"
echo "Find Biology students:"
grep "Biology" students.csv

echo ""
echo "Find high GPA students (>3.5):"
grep -E "3\.[6-9]|4\.[0-9]" students.csv

echo ""
echo "Find error messages in logs:"
grep "ERROR" system.log

echo ""
echo "Case-insensitive search for login events:"
grep -i "login" system.log

echo ""
echo "2. ADVANCED TEXT PROCESSING"
echo "==========================="

echo ""
echo "Using awk for complex data processing:"
echo "-------------------------------------"
echo "Students with GPA > 3.5:"
awk -F',' '$8 > 3.5 {print $1 ": " $8}' students.csv

echo ""
echo "Calculate average GPA by major:"
awk -F',' '
BEGIN {print "Major\tAverage GPA"}
NR>1 {
    major[$3] += $8
    count[$3]++
}
END {
    for (m in major) {
        avg = major[m] / count[m]
        printf "%s\t%.2f\n", m, avg
    }
}' students.csv

echo ""
echo "Calculate total sales by region:"
awk -F',' '
BEGIN {print "Region\tTotal Sales"}
NR>1 {
    region[$7] += $6
}
END {
    for (r in region) {
        printf "%s\t$%d\n", r, region[r]
    }
}' sales.csv

echo ""
echo "Using sed for text transformation:"
echo "--------------------------------"
echo "Replace 'Electronics' with 'Tech':"
sed 's/Electronics/Tech/g' sales.csv

echo ""
echo "Add prefix to student names:"
sed 's/^/Student: /' students.csv | head -3

echo ""
echo "Delete warning messages from logs:"
sed '/WARNING/d' system.log

echo ""
echo "3. BUILDING DATA PIPELINES"
echo "=========================="

echo ""
echo "Pipeline 1: Find Biology students with high GPA"
echo "-----------------------------------------------"
grep "Biology" students.csv | \
awk -F',' '$8 > 3.4 {print $1 ": " $8}'

echo ""
echo "Pipeline 2: Count students by major"
echo "----------------------------------"
cut -d',' -f3 students.csv | \
tail -n +2 | \
sort | \
uniq -c | \
sort -nr

echo ""
echo "Pipeline 3: Extract and format error timestamps"
echo "---------------------------------------------"
grep "ERROR" system.log | \
cut -d' ' -f1,2 | \
sed 's/^/Error at: /'

echo ""
echo "Pipeline 4: Process student grades (as mentioned in lecture)"
echo "-----------------------------------------------------------"
echo "Find top performers in Science:"
grep "Science" students.csv | \
cut -d',' -f1,2 | \
awk -F',' '$2 > 90 {print $1}'

echo ""
echo "4. COMPLEX DATA PROCESSING PIPELINE"
echo "==================================="

echo ""
echo "Comprehensive student analysis pipeline:"
echo "---------------------------------------"

# Step 1: Extract relevant data
grep -v "name,age,major" students.csv > temp_students.txt

# Step 2: Find top performers
awk -F',' '$8 >= 3.6 {print $1 "," $3 "," $8}' temp_students.txt | \
sort -t',' -k3 -nr > top_performers.csv

# Step 3: Calculate major statistics
awk -F',' '
BEGIN {print "Major,Count,Avg_GPA,Min_GPA,Max_GPA"}
NR>0 {
    major[$3] += $8
    count[$3]++
    if (min[$3] == "" || $8 < min[$3]) min[$3] = $8
    if (max[$3] == "" || $8 > max[$3]) max[$3] = $8
}
END {
    for (m in major) {
        avg = major[m] / count[m]
        printf "%s,%d,%.2f,%.1f,%.1f\n", m, count[m], avg, min[m], max[m]
    }
}' temp_students.txt | \
sort -t',' -k3 -nr > major_statistics.csv

# Step 4: Generate summary report
cat > analysis_summary.txt << 'EOF'
Student Performance Analysis
============================

Generated: $(date)

Top Performers (GPA >= 3.6):
EOF

cat top_performers.csv >> analysis_summary.txt

echo -e "\nMajor Statistics:" >> analysis_summary.txt
cat major_statistics.csv >> analysis_summary.txt

# Clean up temporary files
rm temp_students.txt

echo "✓ Analysis complete! Check analysis_summary.txt for results"

echo ""
echo "5. PERFORMANCE AND EFFICIENCY"
echo "============================="

echo ""
echo "Measuring command performance:"
echo "-----------------------------"
echo "Time different approaches for finding Biology students:"
echo "Method 1: grep + awk"
time grep "Biology" students.csv | awk -F',' '{print $1 ": " $8}' > /dev/null

echo "Method 2: awk only"
time awk -F',' '/Biology/ {print $1 ": " $8}' students.csv > /dev/null

echo ""
echo "6. DATA VALIDATION AND ERROR HANDLING"
echo "===================================="

echo ""
echo "Data validation examples:"
echo "------------------------"
echo "Check for missing values:"
awk -F',' 'NR>1 {for(i=1;i<=NF;i++) if($i=="") print "Missing value in row " NR ", column " i}' students.csv

echo ""
echo "Check for invalid GPAs:"
awk -F',' 'NR>1 && ($8 < 0 || $8 > 4.0) {print "Invalid GPA for " $1 ": " $8}' students.csv

echo ""
echo "Check for duplicate names:"
cut -d',' -f1 students.csv | tail -n +2 | sort | uniq -d

echo ""
echo "COMMAND LINE DEMO COMPLETE!"
echo "=========================="
echo "This demonstration showed:"
echo "✓ Essential shell tools (cut, sort, grep)"
echo "✓ Advanced text processing (awk, sed)"
echo "✓ Data processing pipelines"
echo "✓ Complex analysis workflows"
echo "✓ Performance considerations"
echo "✓ Data validation techniques"
echo ""
echo "Files created:"
echo "- students.csv (sample student data)"
echo "- sales.csv (sample sales data)"
echo "- system.log (sample log data)"
echo "- top_performers.csv (analysis results)"
echo "- major_statistics.csv (major statistics)"
echo "- analysis_summary.txt (comprehensive report)"
echo ""
echo "Next: Essential Python development skills!"

# Return to original directory
cd ..