# Assignment 01: Getting Started with Python and Command Line

## Overview
This assignment verifies that you can use the basic tools we'll need all semester: command line navigation, Python execution, and Git workflow. You'll run a provided script that processes your UCSF email address and commit the results.

**Due:** Before next class  
**Points:** 20 points  
**Skills practiced:** CLI navigation, Python execution, file output, Git workflow

## Learning Objectives
By completing this assignment, you will:
- Navigate directories using command line
- Execute Python scripts from command line  
- Understand basic file output redirection
- Practice Git add/commit workflow
- Verify your development environment is working

## Assignment Tasks

### Task 1: Repository Setup
1. Accept the GitHub Classroom assignment link (provided in class)
2. Clone your assignment repository to your local machine or open it in a GitHub Codespace
3. Navigate to the assignment directory using command line

### Task 2: Environment Verification  
1. Open terminal/command prompt in your assignment directory
2. Verify Python is working: `python --version` or `python3 --version`
3. List the files in your directory: `ls` (Mac/Linux) or `dir` (Windows)

### Task 3: Email Processing Script
1. Examine the provided script `process_email.py` (you don't need to understand all the code)
2. Run the script with your UCSF email: `python process_email.py your_email@ucsf.edu`
3. The script should create a file called `processed_email.txt`
4. Verify the output file was created: `ls` or `dir`

### Task 4: Result Submission
1. Add your results to Git using one of:
    - `git add processed_email.txt` 
    - Use the VS Code GUI
    - Upload to your GitHub repo on the web
2. Commit with a descriptive message using one of: 
    - `git commit -m "Add processed email results"`
    - GUI in VS Code  
3. Push to GitHub: `git push` 

### Task 5:  Questions
Edit the file `reflection.md` and let me know more about yourself!

## Files Provided
- `process_email.py` - Script that processes your email address
- `README.md` - This instruction file  
- `reflection.md` - File for your responses

## Expected Output
After running `python process_email.py alice@ucsf.edu`, you should see:
```
Processing email: alice@ucsf.edu
Extracting username: alice
Converting to lowercase: alice  
Creating hash: [some hash value]
Results saved to processed_email.txt
```

The `processed_email.txt` file should contain one line with your processed information.

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Script Execution | 8 | Script runs successfully, creates output file |
| File Content | 6 | Output file contains expected format and data |
| Git Workflow | 4 | Proper add/commit/push of results |
| Reflection | 2 | Complete answers to verification questions |

## Getting Help

**If Python command doesn't work:**
- Try `python3` instead of `python`
- Verify Python is installed: contact instructor if needed
- Check you're in the correct directory

**If script gives an error:**
- Check your email format (should be `yourname@ucsf.edu`)
- Make sure you included the `.py` in the command
- Try running: `python --version` to verify Python works

**If Git commands fail:**
- Make sure you cloned the repository correctly
- Check that you're in the assignment directory
- Verify you have internet connection for `git push`

**Still stuck?**
- Post in class forum with your error message
- Attend office hours
- Ask a classmate (collaboration on setup is encouraged!)

## Academic Integrity
- You may help each other with technical setup issues
- The email processing should be done individually
- Don't share your processed output with other students
- If you get help, acknowledge it in your reflection

## Tips for Success
1. **Read error messages carefully** - they usually tell you what's wrong
2. **Use tab completion** - press Tab to complete file/directory names  
3. **Check your current directory** - use `pwd` (Mac/Linux) or `cd` (Windows) if lost
4. **Start early** - technical setup can take time
5. **Ask for help** - everyone struggles with setup initially

## What This Assignment Tests
This assignment verifies you have the basic technical setup needed for the course:
- Working Python installation
- Command line navigation skills
- Git repository workflow
- Ability to run scripts and handle file output

These are fundamental skills you'll use in every future assignment!

## Next Steps
Once you complete this assignment, you'll be ready for more substantial Python programming and data analysis tasks. The workflow you practice here (clone → work → commit → push) will become second nature by the end of the semester.