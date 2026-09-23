#!/bin/bash

echo "=========================================="
echo "DEMO 2: COMMAND-LINE NAVIGATION"
echo "=========================================="
echo "The shell organizes files; Python will analyze data later."
echo
echo "Where am I?"
pwd
echo
echo "What is here?"
ls
echo
echo "Make a clinic project and enter it:"
mkdir clinic_practice
cd clinic_practice
pwd
echo "Create folders and an empty README:"
mkdir data scripts results
touch README.txt
ls
echo "Write a visits file, one line at a time:"
echo "patient_id,systolic_mmHg" > data/visits.csv
echo "P001,118" >> data/visits.csv
echo "P002,142" >> data/visits.csv
echo "P003,131" >> data/visits.csv
cat data/visits.csv
echo "The header and first row, then the last row:"
head -n 2 data/visits.csv
tail -n 1 data/visits.csv
echo "Copy and rename the visits file:"
cp data/visits.csv results/visits_backup.csv
mv results/visits_backup.csv results/visits_raw.csv
ls results
echo
echo "Move back to the starting directory:"
cd ..
pwd
echo "The practice folder is still here:"
ls clinic_practice
echo
echo "Key commands: pwd, ls, cd, mkdir, touch, echo, cat, head, tail, cp, mv"
echo "Tip: use pwd whenever you are unsure where a relative path starts."
