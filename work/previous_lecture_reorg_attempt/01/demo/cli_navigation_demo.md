# Lecture 01 Demo 1: Command Line Navigation

## Instructor Demo Guide

**Time:** 10 minutes  
**Goal:** Show students real-time CLI navigation and build confidence  
**Format:** Live demonstration with student participation

## Setup (Before Demo)
- Open terminal/command prompt
- Start in your home directory (`cd ~`)  
- Have a web browser ready to show file system graphically if needed

## Demo Script

### Part 1: Orientation (2 minutes)

**Say:** "Let's start by figuring out where we are and what's around us."

```bash
pwd
```
**Explain:** "This tells us our current location. Think of it as 'You are here' on a map."

```bash
ls
```
**Explain:** "This shows what's in our current location - like looking around a room."

```bash
ls -la  
```
**Explain:** "The -la flags give us more detail - file sizes, dates, hidden files. Don't worry about memorizing this syntax yet."

### Part 2: Moving Around (3 minutes)

**Say:** "Now let's move around like we're exploring a building."

```bash
# Show current location, then move
cd Documents        # or whatever folder exists
pwd                 # Confirm we moved
ls                  # See what's here
```

**Say:** "We just 'walked' into the Documents folder."

```bash
cd ..               # Go back up one level  
pwd                 # Show we're back where we started
```

**Explain:** "The .. means 'parent directory' - like going back to the hallway from a room."

```bash
cd ~                # Go to home directory
pwd                 # Confirm location
```

**Explain:** "The ~ symbol is a shortcut for your home folder - like having a 'home' button."

### Part 3: Creating Our Workspace (3 minutes)

**Say:** "Let's create a space for our data science work."

```bash
mkdir datasci_practice
ls                          # Show the new folder appeared
cd datasci_practice  
pwd                         # Confirm we're inside
```

**Say:** "We just created and entered our project folder."

```bash
mkdir data analysis scripts  
ls                           # Show multiple folders created
```

**Explain:** "We created three folders at once - data for our datasets, analysis for our work, scripts for our Python files."

```bash
touch data/sample_data.csv
touch analysis/first_analysis.py
ls data/
ls analysis/
```

**Say:** "Touch creates empty files - like putting placeholder documents in our folders."

### Part 4: Viewing Files (2 minutes)

**Say:** "Let's add some content and see how to view files."

```bash
echo "name,age,city" > data/sample_data.csv
echo "Alice,25,SF" >> data/sample_data.csv  
echo "Bob,30,NYC" >> data/sample_data.csv
```

**Explain:** "I'm creating a simple CSV file. One > creates/overwrites, two >> adds to the end."

```bash
cat data/sample_data.csv
```

**Say:** "Cat shows us the entire file contents."

```bash
head data/sample_data.csv
```

**Say:** "Head shows just the beginning - useful for large files where you just want a peek."

## Student Interaction Points

### After Part 1:
**Ask:** "Who can tell me what `pwd` stands for?"  
**Listen for:** "Print working directory"  
**Affirm:** "Exactly! It prints your current working directory."

### After Part 2:
**Ask:** "If I'm in `/Users/alice/Documents` and I want to go to `/Users/alice`, what command should I use?"  
**Listen for:** "`cd ..`"  
**If struggling:** "Remember, .. means go up one level."

### After Part 3:
**Ask:** "Who can predict what `ls scripts/` will show us?"  
**Demo:** Run the command to confirm it shows an empty directory.

## Common Student Questions & Responses

**Q:** "What if I make a typo in a command?"  
**A:** "Great question! The computer will tell you it doesn't understand. Just retype it - everyone makes typos constantly."

**Q:** "How do I know what folders exist?"  
**A:** "Use `ls` to look around before you `cd`. It's like checking a map before you walk."

**Q:** "What if I get lost?"  
**A:** "Use `pwd` to see where you are, and `cd ~` to get back home. Like having GPS!"

**Q:** "Can I use spaces in folder names?"  
**A:** "You can, but it's tricky - you need quotes. Better to use underscores: `my_project` instead of `my project`."

## Troubleshooting

**If command fails:**
- Check for typos
- Use tab completion to avoid typing full names
- Show that the computer is helpful - it often suggests what you meant

**If students look confused:**
- Relate to familiar concepts (folders in file explorer, GPS navigation)
- Emphasize that this is just another way to do what they already know
- Remind them they're not expected to memorize everything immediately

## Wrap-up (1 minute)

**Say:** "We just navigated around, created a workspace, and viewed files. This is the foundation - you organize your space, then you work in it. Next, we'll see how Python fits into this workflow."

**Key points to emphasize:**
- This is just like using file explorer, but with typing
- `pwd` = where am I?, `ls` = what's here?, `cd` = go somewhere
- Don't worry about memorizing - focus on the concepts
- Everyone starts here, including professional data scientists

## Materials Needed
- Terminal/command prompt access
- Projection/screen sharing capability
- Backup plan: screenshots of commands if live demo fails

## Success Indicators
- Students can explain what `pwd`, `ls`, and `cd` do
- At least half the students attempt to follow along
- Questions show engagement, not total confusion
- Students seem curious rather than overwhelmed