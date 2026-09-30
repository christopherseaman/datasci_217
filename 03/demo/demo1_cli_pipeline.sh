#!/bin/bash
# Stop at the first failing command, unset variable, or failing pipeline stage.
set -euo pipefail

# A repeatable shell pipeline. Run it from ~/03-demo with
# `bash demo1_cli_pipeline.sh`: it creates data/, logs/, and results/ there.
echo "=== Lecture 03: CLI pipeline ==="
mkdir -p data/raw logs results

# Capture one timestamp and reuse it for every output from this run. Log the
# start before any work, so a run that stops partway still leaves a line.
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "${timestamp} pipeline started" >> logs/processing.log

# Write the lines between <<'EOF' and EOF into the file, unchanged.
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
# Skip the header, select one field, sort it for uniq, count it, and keep the
# first five lines. A trailing \ continues the command on the next line, so
# this is still one pipeline.
tail -n +2 data/raw/encounters.csv \
  | cut -d',' -f4 \
  | sort \
  | uniq -c \
  | head -n 5

summary="results/summary_${timestamp}.txt"
echo "run timestamp: ${timestamp}" > "$summary"
echo "encounters: $(tail -n +2 data/raw/encounters.csv | wc -l)" >> "$summary"
echo "clinic counts:" >> "$summary"
tail -n +2 data/raw/encounters.csv \
  | cut -d',' -f4 \
  | sort \
  | uniq -c \
  | head -n 5 >> "$summary"

# Log what the run wrote, under the same timestamp as its start.
echo "${timestamp} wrote ${summary}" >> logs/processing.log
echo "Summary written to ${summary}"
echo "=== Demo complete ==="
