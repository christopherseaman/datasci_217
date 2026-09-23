#!/usr/bin/env bash
set -euo pipefail

# A bounded, repeatable shell pipeline. Run from a disposable directory:
# every path below is created relative to the current directory.
echo "=== Lecture 03: bounded CLI pipeline ==="
mkdir -p data/raw logs results

cat > data/raw/students.csv <<'EOF'
name,age,grade,subject
Alice,20,85,Math
Bob,19,92,Science
Charlie,21,78,English
Diana,20,88,Math
Eve,22,95,Science
Frank,20,88,Math
EOF

echo "Student records: $(tail -n +2 data/raw/students.csv | wc -l)"
echo "Subjects (with counts):"
# Skip the header, select one field, sort it for uniq, count it, and bound
# the displayed result to five lines.
tail -n +2 data/raw/students.csv \
  | cut -d',' -f4 \
  | sort \
  | uniq -c \
  | head -n 5

# Capture one timestamp and reuse it for every output from this run.
timestamp=$(date +"%Y%m%d_%H%M%S")
summary="results/summary_${timestamp}.txt"
echo "run timestamp: ${timestamp}" > "$summary"
echo "records: $(tail -n +2 data/raw/students.csv | wc -l)" >> "$summary"
echo "subject counts:" >> "$summary"
tail -n +2 data/raw/students.csv \
  | cut -d',' -f4 \
  | sort \
  | uniq -c \
  | head -n 5 >> "$summary"

# Append concise status messages to a log; the same timestamp identifies the
# run without repeatedly calling date.
echo "${timestamp} pipeline started" >> logs/processing.log
echo "${timestamp} wrote ${summary}" >> logs/processing.log
echo "Summary written to ${summary}"
echo "=== Demo complete ==="
