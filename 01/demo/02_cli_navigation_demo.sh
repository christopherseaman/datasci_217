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
echo "Make a small project and enter it:"
mkdir cli_practice
cd cli_practice
pwd
echo "Create folders and an empty note:"
mkdir data scripts results
touch README.txt
ls
echo "Inspect the empty note file:"
cat README.txt
echo "Copy and rename the note:"
cp README.txt results/notes.txt
mv results/notes.txt results/lecture_notes.txt
ls results
echo
echo "Move back to the starting directory:"
cd ..
pwd
echo "The practice folder is still here:"
ls cli_practice
echo
echo "Key commands: pwd, ls, cd, mkdir, touch, cp, mv, cat"
echo "Tip: use pwd whenever you are unsure where a relative path starts."
