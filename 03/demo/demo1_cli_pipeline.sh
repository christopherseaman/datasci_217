#!/usr/bin/env bash
set -euo pipefail

# A bounded, repeatable shell pipeline. Run from a disposable directory:
# every path below is created relative to the current directory.
echo "=== Lecture 03: bounded CLI pipeline ==="
mkdir -p data/raw logs results

cat > data/raw/encounters.csv <<'EOF'
patient_id,age,systolic_bp,clinic
P001,54,128,Cardiology
P002,39,118,Primary Care
P003,67,142,Nephrology
P004,45,131,Cardiology
P005,72,145,Nephrology
P006,58,126,Cardiology
EOF

echo "Encounter records: $(tail -n +2 data/raw/encounters.csv | wc -l)"
echo "Clinics (with counts):"
# Skip the header, select one field, sort it for uniq, count it, and bound
# the displayed result to five lines.
tail -n +2 data/raw/encounters.csv \
  | cut -d',' -f4 \
  | sort \
  | uniq -c \
  | head -n 5

# Capture one timestamp and reuse it for every output from this run.
timestamp=$(date +"%Y%m%d_%H%M%S")
summary="results/summary_${timestamp}.txt"
echo "run timestamp: ${timestamp}" > "$summary"
echo "encounters: $(tail -n +2 data/raw/encounters.csv | wc -l)" >> "$summary"
echo "clinic counts:" >> "$summary"
tail -n +2 data/raw/encounters.csv \
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
