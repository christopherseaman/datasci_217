# DS-217 Final Exam

## Instructions
- This exam consists of four interconnected questions that will guide you through a health data analysis workflow
- Submit your solutions as separate files with the names specified in each question
- You may only use command line tools and Python features covered in the course outline
- Each question builds upon the previous ones - complete them in order

## Question 1: Data Directory Setup (20 points)
File name: `setup_health_data.sh`

Create a shell script that sets up a directory structure for analyzing patient health records. Your script should:

1. Create a directory structure:
```
health_records/
├── raw_data/
├── processed/
└── reports/
```

2. Create template data files in raw_data/:
   - Use `echo` and redirection to create CSV headers for three files:
     * patients.csv: "id,age,gender,condition"
     * vitals.csv: "patient_id,timestamp,heart_rate,blood_pressure"
     * medications.csv: "patient_id,medication,dosage,frequency"

3. Set appropriate permissions:
   - Make raw_data/ read-only using chmod
   - Make the script executable

Tips:
- Use commands covered in lecture: mkdir, echo, chmod
- Use redirection (>) to create files
- Remember to add shebang line: #!/bin/bash

### Example usage
```bash
chmod +x setup_health_data.sh
./setup_health_data.sh
```

## Question 2: Generate Health Records (25 points)
File name: `generate_records.sh`

Create a shell script that generates sample health records using command line tools. Your script should:

1. Generate patient records:
```bash
# Create base patient data
echo "1,67,M,hypertension
2,45,F,diabetes
3,52,M,hypertension
4,71,F,diabetes
5,39,M,asthma" > raw_data/patients.csv
```

2. Generate vital signs records with timestamps:
   - Use echo to create records
   - Use `date` to generate timestamps
   - Create multiple readings per patient
   - Include realistic vital sign ranges

3. Process the generated data:
   - Use grep to filter records
   - Use cut to extract columns
   - Use tr to standardize formatting
   - Use sed to clean up data

Tips:
- The `date` command isn't covered in lecture but is essential for timestamps
  Example: `date "+%Y-%m-%d %H:%M:%S"`
- Use pipes (|) to combine commands
- Use redirection (>, >>) to save results
- Test commands individually first

### Example usage
```bash
./generate_records.sh
```

Example output (raw_data/vitals.csv):
```
patient_id,timestamp,heart_rate,blood_pressure
1,2024-01-15 09:30:00,72,120/80
1,2024-01-15 15:45:00,75,125/82
2,2024-01-15 10:15:00,68,118/75
```

## Question 3: Process Health Records (25 points)
File name: `process_records.py`

Create a Python script that processes the generated health records. Your script should:

1. Read the CSV files created by the previous script
2. For each patient:
   - Calculate average vital signs
   - Count number of readings
   - Identify any missing values
3. Create summary files in processed/:
   - Use basic Python file operations
   - Format numbers appropriately
   - Handle potential errors

Tips:
- Use only Python features covered in lecture
- Use basic file operations (open, read, write)
- Use string methods for parsing
- Use lists and dictionaries for data storage
- Handle potential file errors with try/except

### Example usage
```bash
python process_records.py
```

## Question 4: Analyze and Report (30 points)
File name: `analyze_records.sh`

Create a shell script that analyzes the processed health records using command line tools. Your script should:

1. Generate condition statistics:
   - Use grep to count conditions
   - Use tr to standardize case
   - Use sed to format output
   ```bash
   grep "hypertension" raw_data/patients.csv | wc -l > reports/hypertension_count.txt
   ```

2. Analyze vital signs:
   - Use cut to extract measurements
   - Use grep to find concerning readings
   - Use sed to format output
   ```bash
   cut -d',' -f2,3 raw_data/vitals.csv | grep "120/[89][0-9]"
   ```

3. Create a final report:
   - Use cat to combine statistics
   - Use sed to format the report
   - Use tr to clean up formatting
   ```bash
   cat reports/*_count.txt | sed 's/^/  /' > reports/final_report.txt
   ```

Tips:
- Use only commands from course outline
- Combine commands with pipes
- Use text processing tools: grep, sed, tr, cut
- Format output for readability

### Example usage
```bash
./analyze_records.sh
```

Example output (reports/final_report.txt):
```
HEALTH RECORDS ANALYSIS
----------------------
Patient Statistics:
  Total Patients: 5
  By Condition:
    Hypertension: 2
    Diabetes: 2
    Asthma: 1

Vital Signs Summary:
  Readings: 15
  Concerning BP: 3
  High HR: 2
```

### Bonus Points (10 points)
- Add error checking to shell scripts
- Create a simple command line menu using echo
- Add data validation using grep patterns
- Generate formatted reports using sed
