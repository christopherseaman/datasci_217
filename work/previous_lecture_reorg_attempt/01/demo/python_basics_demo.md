# Lecture 01 Demo 2: Python Basics Integration

## Instructor Demo Guide

**Time:** 10 minutes  
**Goal:** Show how Python and command line work together in real workflow  
**Format:** Live demonstration with guided student participation

## Setup (Before Demo)
- Have terminal open in the `datasci_practice` folder from previous demo
- Have Python installed and accessible from command line
- Have a text editor ready (VS Code, nano, or whatever students will use)

## Demo Script

### Part 1: Creating Our First Analysis Script (3 minutes)

**Say:** "Now we'll create our first data analysis script and see how Python and command line work together."

```bash
# Confirm we're in our practice directory
pwd
ls
```

**Say:** "Good, we're in our workspace with our organized folders."

```bash
cd analysis
touch my_first_analysis.py
ls
```

**Explain:** "We're in our analysis folder and created a Python script file."

**Say:** "Now let's add some code. I'll use [editor name], but you can use any text editor."

```python
# Create/edit my_first_analysis.py with:
# Simple data analysis calculation

# Patient data  
patient_ages = [23, 45, 67, 34, 56, 29, 41]
patient_name = "Study Group Alpha"

# Calculate basic statistics
average_age = sum(patient_ages) / len(patient_ages)
oldest_age = max(patient_ages)
youngest_age = min(patient_ages)

# Display results
print(f"Analysis Results for {patient_name}")
print(f"Number of patients: {len(patient_ages)}")
print(f"Average age: {average_age:.1f} years")  
print(f"Age range: {youngest_age} to {oldest_age} years")

# Create a simple report line
report_line = f"{patient_name},{len(patient_ages)},{average_age:.1f}"
print(f"Report line: {report_line}")
```

### Part 2: Running Python from Command Line (2 minutes)

**Say:** "Now let's see our analysis in action!"

```bash
python my_first_analysis.py
```

**Explain as output appears:** "Look at that! Python calculated our statistics and formatted them nicely. This is the workflow - write Python code, run it from command line."

**Say:** "Let's see what files we have now:"

```bash
ls
cat my_first_analysis.py
```

**Explain:** "Cat lets us see our code without opening an editor - handy for quick checks."

### Part 3: Saving Results (3 minutes)

**Say:** "In real data science, we often want to save our results. Let's see how command line helps with that."

```bash
python my_first_analysis.py > results.txt
ls
cat results.txt
```

**Explain:** "The > symbol redirected our Python output to a file instead of the screen. This is incredibly useful for saving results."

**Say:** "Let's create a more complete workflow:"

```bash
# Run analysis and save results
python my_first_analysis.py > ../data/analysis_results.txt

# Check what we created
ls ../data/
cat ../data/analysis_results.txt
```

**Explain:** "We saved our results in the data folder - keeping our workspace organized."

### Part 4: Interactive Python (2 minutes)

**Say:** "Python also has an interactive mode for quick experiments."

```bash
python
```

**In Python interactive mode:**
```python
# Quick calculations
ages = [25, 30, 35]  
print(f"Average: {sum(ages)/len(ages)}")

# String operations
name = "Data Science"
print(f"Subject: {name}")
print(f"Length: {len(name)} characters")

# Exit back to command line
exit()
```

**Explain:** "Interactive Python is great for testing ideas quickly, then you put the good stuff in scripts."

## Student Interaction Points

### After Part 1:
**Ask:** "What do you think will happen when we run this script?"  
**Listen for:** Predictions about output  
**Affirm:** "Let's find out!"

### After Part 2:
**Ask:** "Who can guess what the `.1f` does in our print statement?"  
**Listen for:** "Formats to 1 decimal place"  
**Clarify:** "Exactly! It rounds to 1 decimal place - much cleaner than `45.666666`"

### After Part 3:
**Ask:** "Why might we want to save results to a file instead of just seeing them on screen?"  
**Listen for:** "To keep them," "To share," "To use later"  
**Affirm:** "All correct! Files persist - screen output disappears."

## Common Student Questions & Responses

**Q:** "How do I edit the Python file if I make a mistake?"  
**A:** "Great question! You can edit it with any text editor. VS Code is popular, or even `nano filename.py` from command line."

**Q:** "What if my Python code has an error?"  
**A:** "Python will tell you! It gives error messages that help you fix problems. Don't worry - errors are normal and helpful."

**Q:** "Do I always need to save Python code in files?"  
**A:** "For anything you want to keep or rerun, yes. Interactive mode is great for quick tests, files are for real work."

**Q:** "What does the `#!/usr/bin/env python` line do?"  
**A:** "That's called a 'shebang' - it tells the system this is a Python file. We'll learn about that later."

## Real-World Connection Points

**Say:** "This workflow - write code, run from command line, save results - is exactly what professional data scientists do hundreds of times per day."

**Emphasize:**
- "You organize your workspace with command line"
- "You write analysis logic with Python"  
- "You run everything together for results"
- "This scales from simple calculations to complex machine learning"

## Demo Variations

**If students seem comfortable:**
- Show how to pass the results to another command: `python analysis.py | head -2`
- Demonstrate running the same script on different data

**If students seem overwhelmed:**
- Focus on just the basic run: `python script.py`
- Emphasize that they don't need to memorize the syntax
- Relate back to familiar concepts: "Like double-clicking a program"

## Troubleshooting

**If Python command fails:**
- Check if Python is installed: `python --version`
- Try `python3` instead of `python`
- Have a backup screenshot of expected output

**If file editing is difficult:**
- Have the script pre-written to copy-paste
- Show multiple editor options
- Remind students this gets easier with practice

## Wrap-up (1 minute)

**Say:** "We just saw the complete data science workflow: organize with command line, analyze with Python, save results for sharing. This foundation will support everything we do this semester."

**Key takeaways:**
- Command line organizes your work
- Python does the calculations  
- Files preserve your results
- This workflow scales to complex projects
- Every data scientist uses this pattern daily

**Preview:** "Next class, we'll learn how to save and share this work with Git and GitHub - so you can collaborate with others and never lose your progress!"

## Materials Needed
- Python installation confirmed working
- Text editor accessible to students  
- Sample data ready to use
- Backup plan: pre-written files if live coding fails

## Success Indicators
- Students can see the connection between command line and Python
- At least some students try following along with the commands
- Questions focus on workflow rather than syntax details
- Students express interest in trying this themselves