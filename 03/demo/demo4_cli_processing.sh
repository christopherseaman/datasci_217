#!/bin/bash
# Command Line Data Processing Demo
# Demonstrates cut, sort, grep, tr, sed, awk, pipelines, and CLI graphing

echo "=== Command Line Data Processing Demo ==="
echo ""

# Setup: Create sample data files
echo "Creating sample data files..."

cat > students.csv << 'DATA'
name,age,grade,subject
Alice,20,85,Math
Bob,19,92,Science
Charlie,21,78,English
Diana,20,88,Math
Eve,22,95,Science
Frank,19,82,History
Grace,21,91,Math
Henry,20,76,Science
Isabel,19,89,English
Jack,22,84,Math
DATA

echo "✅ Sample data created"
echo ""

# Demo 1: cut - Extract columns
echo "=== Demo 1: cut - Extract Columns ==="
echo "Extract student names and grades:"
cut -d',' -f1,3 students.csv | head -5
echo ""

# Demo 2: sort - Sort data
echo "=== Demo 2: sort - Sort Data ==="
echo "Sort by grade (numerically):"
sort -t',' -k3 -n students.csv | head -5
echo ""

# Demo 3: uniq - Remove duplicates
echo "=== Demo 3: uniq - Count by Subject ==="
echo "Note: uniq requires sorted input!"
cut -d',' -f4 students.csv | sort | uniq -c
echo ""

# Demo 4: grep - Search and filter
echo "=== Demo 4: grep - Search and Filter ==="
echo "Find all Math students:"
grep "Math" students.csv
echo ""

# Demo 5: awk - Pattern processing
echo "=== Demo 5: awk - Calculate Average ==="
echo "Note: NR>1 skips the header row (NR = Row Number)"
awk -F',' 'NR>1 {sum+=$3; count++} END {print "Average grade:", sum/count}' students.csv
echo ""

# Demo 6: Complex pipeline
echo "=== Demo 6: Complex Pipeline - Top 3 Math Students ==="
echo "Pipeline: filter -> extract -> sort -> limit"
grep "Math" students.csv | \
  cut -d',' -f1,3 | \
  sort -t',' -k2 -nr | \
  head -n 3
echo ""

# Demo 7: CLI Graphing with sparklines
echo "=== Demo 7: CLI Graphing with sparklines ==="

# Check if sparklines is available
if command -v sparklines &> /dev/null; then
    echo "Student grade distribution (sparkline):"
    echo "Command: cut -d',' -f3 students.csv | tail -n +2 | sparklines"
    echo "         └─ Extract column 3  └─ Skip header  └─ Visualize"
    echo ""
    echo "Note: tail -n +2 means 'start output at line 2' (skips header line)"
    echo ""
    cut -d',' -f3 students.csv | tail -n +2 | sparklines
    echo ""
    
    echo "With statistics:"
    cut -d',' -f3 students.csv | tail -n +2 | sparklines --stat-min --stat-max --stat-mean
    echo ""
    
    echo "Compare sparkline to actual grades:"
    echo "Sparkline: $(cut -d',' -f3 students.csv | tail -n +2 | sparklines)"
    echo "Grades:    $(cut -d',' -f3 students.csv | tail -n +2 | tr '\n' ' ')"
    echo ""
else
    echo "⚠️  sparklines not installed"
    echo "Install: pip install sparklines"
    echo "Or: brew install sparklines (Mac)"
    echo ""
fi

# Demo 8: CLI Graphing with gnuplot
echo "=== Demo 8: CLI Graphing with gnuplot ==="

# Check if gnuplot is available
if command -v gnuplot &> /dev/null; then
    echo "Grade distribution histogram:"
    echo "Note: tail -n +2 skips the CSV header before creating histogram"
    cut -d',' -f3 students.csv | tail -n +2 > /tmp/grades.txt
    gnuplot -e "
        set terminal dumb size 60,15;
        set title 'Student Grades';
        set xlabel 'Grade';
        set ylabel 'Frequency';
        set style fill solid;
        binwidth=5;
        bin(x,width)=width*floor(x/width);
        plot '/tmp/grades.txt' using (bin(\$1,binwidth)):(1.0) smooth freq with boxes notitle
    "
    echo ""
    
    echo "Subject distribution (bar chart):"
    echo "Note: NR>1 in awk skips the header row"
    awk -F',' 'NR>1 {count[$4]++} END {for(s in count) print s, count[s]}' students.csv > /tmp/subjects.txt
    gnuplot -e "
        set terminal dumb size 50,12;
        set title 'Students per Subject';
        set style data histogram;
        set style fill solid;
        plot '/tmp/subjects.txt' using 2:xtic(1) notitle
    "
    echo ""
    
    rm /tmp/grades.txt /tmp/subjects.txt
else
    echo "⚠️  gnuplot not installed"
    echo "Install: brew install gnuplot (Mac)"
    echo "Or: apt install gnuplot (Linux)"
    echo ""
fi

# Demo 9: Generate summary report
echo "=== Demo 9: Generate Summary Report ==="

cat > summary_report.txt << 'REPORT'
Student Performance Summary
===========================

Top Performers (Grade > 85):
REPORT

awk -F',' '$3 > 85 {print $1, "-", $3}' students.csv >> summary_report.txt

echo "" >> summary_report.txt
echo "Subject Distribution:" >> summary_report.txt
cut -d',' -f4 students.csv | tail -n +2 | sort | uniq -c | sort -nr >> summary_report.txt

echo "Generated summary_report.txt:"
cat summary_report.txt
echo ""

# Cleanup
echo "=== Cleanup ==="
read -p "Remove sample files? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm students.csv summary_report.txt
    echo "✅ Sample files removed"
else
    echo "Sample files kept for exploration"
fi

echo ""
echo "✅ Demo complete!"
echo ""
echo "📚 Key Commands for Skipping Headers:"
echo "  • tail -n +2   : Start at line 2 (skip first line)"
echo "  • sed '1d'     : Delete first line"
echo "  • awk 'NR>1'   : Process rows where row number > 1"
echo ""
echo "Note: Install visualization tools for full demo:"
echo "  pip install sparklines"
echo "  brew install gnuplot (Mac) or apt install gnuplot (Linux)"
