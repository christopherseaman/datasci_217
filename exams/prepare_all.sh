# 2. Clean the raw data file:
#    - Remove comment lines
#    - Remove empty lines
#    - Remove extra commas
#    - Extract essential columns: patient_id, visit_date, age, education_level, walking_speed
#    - Save the file as `ms_data.csv`

#!/bin/bash

# Input and output file paths
input_file="./ms_data_dirty.csv"
output_file="./ms_data.csv"

# Temporary file to store cleaned data
clean_file=$(mktemp)

# Remove comment lines, empty lines, and extra commas
grep -v '^#' "$input_file" | grep -v '^$' | sed 's/,,*/,/g' > "$clean_file"

# Extract header
header=$(head -n 1 "$clean_file")
echo "Header found: $header"

# Get column indices for the desired fields
columns=$(echo "$header" | tr ',' '\n' | nl -v 1)
echo "Columns with indices:"
echo "$columns"

patient_id_col=$(echo "$columns" | grep -i 'patient_id' | awk '{print $1}')
visit_date_col=$(echo "$columns" | grep -i 'visit_date' | awk '{print $1}')
age_col=$(echo "$columns" | grep -i 'age' | awk '{print $1}')
education_level_col=$(echo "$columns" | grep -i 'education_level' | awk '{print $1}')
walking_speed_col=$(echo "$columns" | grep -i 'walking_speed' | awk '{print $1}')

# Extract the required columns and save to the output file
awk -F',' -v p="$patient_id_col" -v v="$visit_date_col" -v a="$age_col" -v e="$education_level_col" -v w="$walking_speed_col" \
'NR==1 {print $p, $v, $a, $e, $w}
NR>1 {print $p, $v, $a, $e, $w}' OFS=',' "$clean_file" > "$output_file"

echo "Cleaned data saved to $output_file"


## 3. Create a file, `insurance.lst` listing unique labels for a new variable, `insurance_type`, one per line (your choice of labels). 
# Create insurance.lst with unique labels for `insurance_type`
insurance_file="insurance.lst"

# Add a header row and three unique insurance types
echo "insurance_type" > "$insurance_file"
echo "Private" >> "$insurance_file"
echo "Medicaid" >> "$insurance_file"
echo "Uninsured" >> "$insurance_file"

echo "File created: $insurance_file"

## 4. Generate a summary of the processed data:
processed_file="./ms_data.csv"

# Count the total number of visits (rows, not including the header)
total_visits=$(tail -n +2 "$processed_file" | wc -l)
echo "Total number of visits: $total_visits"

# Display the first few records 
echo "First few records:"
head -n 6 "$processed_file"
#!/bin/bash
python generate_dirty_data.py
#remove comment empty,extra commas
sed '/^#/d' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,*/,/g' > cleaned_data.csv

#Extract columns
cut -d ',' -f 1,2,4,5,6 cleaned_data.csv > filtered_data.csv

#Filter rows
awk -F ',' 'NR==1 || ($5 >= 2.0 && $5 <= 8.0)' OFS=',' filtered_data.csv > ms_data.csv
rm cleaned_data.csv filtered_data.csv
#Create insurance file
echo -e "insurance_type\nBronze\nSilver\nGold\nPlatinum" > insurance.lst

# Summarize
echo "Summary for Question 1" > readme.md
tail -n +2 ms_data.csv | wc -l >> readme.md
head -n 10 ms_data.csv >> readme.md

#!/bin/bash

# run generate_dirty_data.py
python3 generate_dirty_data.py

grep -v "#" ms_data_dirty.csv | sed -e '/^$/d' | sed -e "s/,,/,/g" | cut -d"," -f"1,2,4,5,6" > ms_data.csv

# remove comment lines
# grep -v "#" ms_data_dirty.csv > ms_data_ing.csv

# # remove empty lines
# sed -e '/^$/d' ms_data_ing.csv > ms_data_ing.csv

# # remove extra comma
# sed -e "s/,,/,/g" ms_data_ing.csv > ms_data_ing.csv

# # extract 
# cut -d"," -f"1,2,4,5,6" ms_data_ing.csv > ms_data.csv

## command in one line grep -v "#" ms_data_ing.csv | sed -e '/^$/d' > ms_data_ing.csv

# create insurance.lst
touch insurance.lst
echo Basic > insurance.lst
echo Premium >> insurance.lst
echo -n Platinum >> insurance.lst # append

# count the number of visits
echo "The total number of visits are $(tail -n+2 ms_data.csv | wc -l)"
tail -n+2 ms_data.csv | headpython -u "./generate_dirty_data.py"
# cat ms_data_dirty.csv | grep "#"
sed -i '' '/^#/d;/^$/d' ms_data_dirty.csv
sed -i '' 's/,,/,/g' ms_data_dirty.csv
cut -d',' -f1,2,4,5,6 ms_data_dirty.csv > ms_data.csv
echo "insurance_type
Basic
Premium
Platinum
Travel
Pet
Disability
Liability" > insurance.lst
tail -n +2 ms_data.csv | wc -l
head --lines 10 ms_data.csv#!/bin/zsh

# Generate the ms_data_dirty.csv
python3 generate_dirty_data.py

# Clean the data
# Remove comment lines
# Remove empty lines
# Remove extra commas
# Extract patient_id, visit_date, age, education_level, walking_speed columns (columns 1 tp 5)
# Save the output in the ms_data.csv file
cat ms_data_dirty.csv | grep -v '^#' | sed '/^$/d' | sed 's/,,*/,/g' | cut -d ',' -f1,2,4,5,6 > ms_data.csv 

# Create insurance.lst with unique insurance types
echo -e "Basic\nPremium\nPlatinum" > insurance.lst

# Generate summary
echo "# **Summary**" > readme.md
echo "## **Question 1**" >> readme.md
echo "**Total number of visits:**" >> readme.md
tail -n +2 ms_data.csv | wc -l >> readme.md
echo "" >> readme.md
echo -e "**First few records of ms_data.csv file:**" >> readme.md
echo "" >> readme.md
# Make a table with first 5 rows of the csv file
head -n 1 ms_data.csv | sed 's/^/| /g' | sed 's/$/ |/g' | sed 's/,/ | /g' >> readme.md
echo "|------------|------------|-----|-----------------|---------------|" >> readme.md
head -n 6 ms_data.csv | tail -n +2 | sed 's/^/| /g' | sed 's/$/ |/g' | sed 's/,/ | /g' >> readme.md
#!/bin/bash

# Step 1: Run the script to generate dirty data
python "./generate_dirty_data.py"

# Step 2: Clean the data
# Remove comment lines, empty lines, and extra commas
grep -v '^#' "./ms_data_dirty.csv" | sed '/^$/d' | sed 's/,,*/,/g' > "./temp.csv"

# Extract essential columns and filter walking speeds
awk -F, 'BEGIN {OFS=","} {if ($6 >= 2.0 && $6 <= 8.0) print $1, $2, $4, $5, $6}' "./temp.csv" > "./ms_data.csv"

# Step 3: Create insurance.lst with unique insurance types
echo -e "insurance_type\nBasic\nPremium\nPlatinum" > "./insurance.lst"

# Step 4: Generate a summary of the processed data
# Count the total number of visits
total_visits=$(wc -l < "./ms_data.csv")
total_visits=$((total_visits - 1)) # Subtract header row

# Display the first few records
head -n 5 "./ms_data.csv"

# Output the total number of visits
echo "Total number of visits: $total_visits"

# Clean up temporary file
rm "./temp.csv"
#generating ms_dirty_data.csv 
python3 generate_dirty_data.py

#input and output file names
input_file="ms_data_dirty.csv"
output_file="ms_data.csv"

#remove comment lines, empty lines, extra commas, and extract essential columns (patient_id, visit_date, age, education_level, walking_speed)
grep -v "^#" "$input_file" | sed '/^$/d' | sed 's/,,*/,/g' | cut -d ',' -f 1,2,4,5,6 > "$output_file"

#create insurance.lst file with a header
echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst

#total number of visits (excluding the header)
total_visits=$(tail -n +2 "$output_file" | wc -l)  
echo "Total number of visits: $total_visits"

#display the first few records
echo "First few records:"
head -n 5 "$output_file"#!/bin/bash

#Generate the raw dirty data
python generate_dirty_data.py

#Removing comment lines, empty lines, replace double commas with single commas, extract important columns, output to CSV
grep -v '^#' ms_data_dirty.csv | grep -v '^$' | sed 's/,,*/,/g' | cut -d ',' -f1,2,4,5,6 > ms_data.csv

#Create the insurance type file
echo 'insurance_type
Basic
Premium
Platinum' > insurance.lst

#Display summary in terminal
echo "Number of rows in cleaned data:"s
wc -l ms_data.csv

echo "Preview of cleaned data:"
head -n 8 ms_data.csv#!/bin/bash

#Step 1
python generate_dirty_data.py

#Step 2
input="ms_data_dirty.csv"   
output="ms_data.csv"
grep -v '^#' "$input" |         
sed '/^$/d' |                        
sed 's/,,*/,/g' > "$output"  

#Step 3
ins="insurance.lst"
echo "insurance_type" > "$ins"
echo -e "Basic\nPremium\nPlatinum" >> "$ins"

#Step 4
total_visits=$(tail -n +2 "$output" | wc -l)
echo "Total visits: $total_visits"
head "$output"grep -v '^#' ms_data_dirty.csv > ms_data_temp.csv
sed '/^[[:space:]]*$/d' ms_data_temp.csv > ms_data_temp2.csv
sed 's/,,*/,/g; s/,$//' ms_data_temp2.csv > ms_data_temp3.csv
head -n 1 ms_data_temp3.csv | cut -d',' -f1,2,4,5,6 > ms_data.csv
cut -d',' -f1,2,4,5,6 ms_data_temp3.csv > ms_data_temp4.csv
awk -F',' '$5 >= 2.0 && $5 <= 8.0' ms_data_temp4.csv >> ms_data.csv

rm ms_data_temp.csv ms_data_temp2.csv ms_data_temp3.csv ms_data_temp4.csv

echo -e 'insurance_type\nBasic\nPremium\nPlatinum\nNoInsurance' > insurance.lst

echo -e "# Summary of Q1" > readme.md
echo -e "Total number of visits: $(expr $(wc -l < ms_data.csv) - 1)<br>" >> readme.md
echo -e "First few rows of file:<br>" >> readme.md
head -10 ms_data.csv | while IFS= read -r line; do
  echo -e "$line<br>" >> readme.md
done

#!/bin/bash

# run the script
python3 generate_dirty_data.py

## clean the file 
# remove comments and empty lines
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | \

# remove extra commans and keep essential columns
sed 's/,,*/,/g' | cut -d ',' -f 1,2,4,5,6 | \

# walking speed between 2.0 and 8.0
awk -F ',' '$5 >= 2.0 && $5 <= 8.0' > ms_data.csv

## create insurance list file
# create insurance tiers in insurance.lst
echo -e "insurance_type\nLower\nMiddle\nHighest" > insurance.lst

## summary of clean data 
# count number of total visits not including rows, not including the header
echo "Total number of visits: $(tail -n +2 ms_data.csv | wc -l)"

# display first few records
echo "First few records:" 
head -n 5 ms_data.csv
#!/bin/bash 
grep -v '^#' ms_data_dirty.csv |
sed '/^$/d' |
sed 's/,,*/,/g' |
cut -d',' -f1,2,4,5,6 |
tee ms_data.csv > /dev/null
echo "Total rows not including excluding header:"
tail -n +2 ms_data.csv | wc -l

echo "First couple of rows:"
head ms_data.csv
#!/bin/bash

python3 generate_dirty_data.py

# cleaning
cat ms_data_dirty.csv | grep -v '^#' | grep -v -e '^$' | sed -e 's/,,*/,/g' | cut -d ',' -f1,2,4,5,6 > ms_data.csv

touch insurance.lst # create insurance file

echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst # add in different levels of insurance

visits=$(wc -l < ms_data.csv)

echo "Total number of visits: $((visits - 1))" # count visits without header

# show the first records
echo "First 4 records:"
head -n 5 ms_data.csv

chmod +x prepare.sh#!/bin/bash

input_file="ms_data_dirty.csv"
output_file="ms_data.csv"
lst_file="insurance.lst"

# awk -F ',' '
#     # skip comments
#     !/#/ && NF > 0 {
#         # replace ,, with , in each line
#         gsub(/,,/, ",")
#         # keep lines that have > 0 number of fields, print the line
#         if ($6 >=2 && $6 <= 8) print $1 "," $2 "," $4 "," $5 "," $6
#     }
# ' "$input_file" > "$output_file"

{
    grep -v '^#' "$input_file" |         
    sed '/^$/d' | #empty lines
    sed 's/,,/,/g' |
    awk -F ',' '
    NR == 1 {print $1 "," $2 "," $4 "," $5 "," $6}                
    NR > 1 {                             
        if ($6 >= 2.0 && $6 <= 8.0)
        {print $1 "," $2 "," $4 "," $5 "," $6}
    }' 
} > "$output_file"

echo "data cleaned and saved to $output_file"

echo "insurance_type" > lst_file
echo "Health" >> lst_file
echo "Dental" >> lst_file
echo "Vision" >> lst_file
echo "Car" >> lst_file
echo "Life" >> lst_file

echo "insurance labels added to $lst_file"

total_rows=$(tail -n +2 "$output_file" | wc -l) #skip header

echo "Total number of visits: $total_rows"
echo "First few records: "

head -n 5 $output_file
#!/bin/bash

#Get the directory where the script is located
script_dir="$(dirname "$0")"

#Paths for input and output files, using relative paths
input_file="$script_dir/ms_data_dirty.csv"
output_file="$script_dir/ms_data.csv"
insurance_file="$script_dir/insurance.lst"

#Run generate_dirty_data.py to create the raw data file
python3 "$script_dir/generate_dirty_data.py"

#Remove comment lines, empty lines, and fix extra commas
grep -v '^#' "$input_file" | sed '/^$/d' | sed 's/,,*/,/g' > "$script_dir/cleaned_temp.csv"

#Extract essential columns: patient_id, visit_date, age, education_level, walking_speed
awk -F, 'BEGIN {OFS=","} NR==1 || ($6 >= 2.0 && $6 <= 8.0) {print $1, $2, $4, $5, $6}' "$script_dir/cleaned_temp.csv" > "$output_file"

#Create an insurance.lst file with insurance types
echo -e "Value\nHMO\nPPO" > "$insurance_file"

#Generate a summary of the processed data
echo "Total number of visits:" $(tail -n +2 "$output_file" | wc -l)
echo "First 5 records of cleaned data:"
head -n 5 "$output_file"

#Remove temporary files
rm "$script_dir/cleaned_temp.csv"

echo "Data preparation complete."
echo "Processed data saved to $output_file"
echo "Insurance types saved to $insurance_file"
# create .csv
python3 generate_dirty_data.py

# remove comment lines, empty lines, replace double commas with single commas, extract important columns, output to csv
grep -v '^# ' ms_data_dirty.csv | grep -v '^$' | sed 's/,,/,/g' | cut -d ',' -f1,2,4,5,6 > ms_data.csv

# make insurance list
echo 'insurance_type
Basic
Premium
Platinum' > insurance.lst

# summary in terminal
wc -l ms_data.csv
head -n 8 ms_data.csvgrep -v "^#" ms_data_dirty.csv|  # Remove comment lines
sed -e '/^$/d'| # Remove empty lines 
sed -e 's/,,/,/g'| # Remove extra commas  
cut -d ',' -f 1,2,4,5,6 > ms_data.csv # Extract columns

touch insurance.lst # create insurance list
echo -e "Basic \nPremium \nPlatinum" > insurance.lst

echo "Total number of visits: $(( $(wc -l < ms_data.csv) - 1 ))" # Count total number of visits exclude header

head -n 6 ms_data.csv

# Q1



# Q1.1: Run generate_dirty_data.py.

# Q1.2: Clean ms_data_dirty.csv.

# Set working directory.
cd ./09-second-exam-irisk2050

# Remove comment lines.
# Remove empty lines
# Remove extra commas.
# Extract essential columns: patient_id, visit_date, age, education_level, walking_speed.
grep -v '^\s*#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,*/,/g' | cut -d "," -f1,2,4,5,6 > temp_file
# Walking speed should be between 2.0-8.0 feet/second.
awk -F, -v col=5 -v min=2.0 -v max=8.0 '
    NR == 1 { print }  # Print header row
    NR > 1 {
        value = $col + 0;  # Convert to numeric
        if (value >= min && value <= max) {
            print
        }
    }' temp_file > ms_data.csv

# Q1.3: Create insurance.lst, listing unique labels for a new variable 'insurance_type'. 
echo "Medicare\nMedicaid\nPrivate\nOther" > insurance.lst

# Q1.4: Generate a summary of the process data.

# Count the total number of visits (rows, excluding header)
wc -l ms_data.csv

# Display the first few records.
head ms_data.csv#!/bin/bash

# Go to the correct directory that contains the shell file
cd "$(dirname "$0")"

# Generate the dirty data
python3 generate_dirty_data.py

# Remove ms_data.csv first to make sure this script doesn't run multiple times for incorrect results
rm "ms_data.csv"

# Piping grep, awk, and cut together to pipe three commands together
# grep removes empty lines and lines with #
# awk goes line by line to remove the excessive commas
# cut extracts the columns we want
# redirect the output to the ms_data.csv instead of the terminal
grep -v -E '^\s*$|^\s*#' ms_data_dirty.csv | awk -F, '{
    for (i = 1; i <= NF; i++) {
        if ($i != "") {
            if(i == NF){
                printf ("%s\n", $i);
            }else{
                printf ("%s,", $i);
            }
            
        }
    }
}' | cut -f 1,2,4,5,6 -d ','> ms_data.csv

# create a new file named insrance.lst to store the insurance types
# Delete file first to ensure no duplication in case someone run this file multiple times
rm "insurance.lst"
echo "Basic" >> insurance.lst
echo "Premium" >> insurance.lst
echo "Platinum" >> insurance.lst
echo "Deluxe_Premium" >> insurance.lst
echo "Ultimate" >> insurance.lst

# Store number of lines into a variable then -1 to exclude the header
# Print out the number of visits, and display the first 10 visits
n_line=$(($(wc -l < ms_data.csv) -1)) 
echo ""
echo "------Summary of the Processed Data:------"
echo "there are $n_line visits/rows in the data(not including the headers)"
echo ""
echo "------First 10 Visits:------"
tail -n +1 ms_data.csv | head -10#!/bin/bash

# Step 1: Generate dirty data file
python3 generate_dirty_data.py

# Step 2: Remove comment lines and empty lines
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' > ms_data_cleaned.csv

# Step 3: Remove extra commas
sed 's/,,*/,/g' ms_data_cleaned.csv > ms_data_no_extra_commas.csv

# Step 4: Extract essential columns
cut -d',' -f1,2,3,4,5 ms_data_no_extra_commas.csv > ms_data_extracted.csv

# Step 5: Filter rows by walking speed between 2.0 and 8.0
awk -F',' 'NR==1 || ($5 >= 2.0 && $5 <= 8.0)' ms_data_extracted.csv > ms_data.csv

# Step 6: Create insurance type list
echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst

# Step 7: Generate summary of the processed data
echo "Total visits:"
tail -n +2 ms_data.csv | wc -l
echo "First few records:"
head -n 5 ms_data.csv
python \generate_dirty_data.py

# Clean file
pip install numpy
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,/,/g' > ms_data_cleaned.csv
cut -d',' -f1,2,4,5,6 ms_data_cleaned.csv > ms_data.csv
rm ms_data_cleaned.csv

# Set up insurance list
echo "Basic" > insurance.lst
echo "Average" >> insurance.lst
echo "Plus" >> insurance.lst
echo "Premium" >> insurance.lst

# Summary
wc -l ms_data.csv
head -n 10 ms_data.csv


#!/bin/bash
chmod +x prepare.sh
#creating dirty data file
python3 generate_dirty_data.py
#Part 2: Clean raw data file
#removing comments
grep -v "#" ./ms_data_dirty.csv > ms_no_comments.csv
#removing empty lines
sed -e '/^$/d' ./ms_no_comments.csv > ms_no_emptylines.csv
#extracting the 5 essential columns
cut -d ',' -f 1,2,4,5,6 ms_no_emptylines.csv > ms_5col.csv
#removing extra commas
sed -e 's/,,/,/g' ./ms_5col.csv > ms_data.csv
#Summary of processed data
row_count=$(wc -l < ms_data.csv)
echo $row_count
head -n 5 "ms_data.csv"

#creating insurance file
file_ins="insurance.lst"
#each label is added to the file on a new line
echo -e "Bronze\nSilver\nGold\nPlatinum" >> "$file_ins"
#!/bin/bash

python3 generate_dirty_data.py

input_file="ms_data_dirty.csv"
output_file="ms_data.csv"
insurance_file="insurance.lst"

# removing comments, empty lines, and extra commas
cleaned_data=$(grep -v '^#' "$input_file" | sed '/^$/d' | sed 's/,,*/,/g')

# keeping only the essential columns
essential_columns=$(echo "$cleaned_data" | cut -d',' -f1,2,4,5,6)

# keeping only rows with valid walking speeks
valid_data=$(echo "$essential_columns" | awk -F',' 'NR==1 || ($5 >= 2.0 && $5 <= 8.0)')

echo "$valid_data" > "$output_file"

# making insurance file
cat <<EOL > "$insurance_file"
insurance_type
Basic
Premium
Platinum
EOL

# Summary of the cleaned data
total_visits=$(echo "$valid_data" | wc -l)

let "total_visits-=1"

echo "Summary of the cleaned data:"
echo "Total number of visits: $total_visits"
echo "First few records:"
echo "$valid_data" | head -n 5#!/bin/bash

# step 1: generating dirty data
python3 generate_dirty_data.py

# step 2: cleaning dirty data
# remove comment lines, empty lines, and empty commas; extract essential columns 
grep -v '^#' ms_data_dirty.csv | sed -e '/^$/d' | sed -e 's/, ,/,/g' | cut -d ',' -f 1,2,4,5,6 | awk -F ',' 'NR==1 || $5 >= 2.0 && $5 <= 8.0' > ms_data.csv

# step 3: creating insurance file
echo -e "insurance_type\nBronze\nSilver\nGold" > insurance.lst

# step 4: summarizing processed data
rows=$(tail -n +2 ms_data.csv | wc -l) # ignore header
echo "Total number of visits: $rows"

echo "First 5 records:"
head -n 6 ms_data.csv#!/bin/bash

#Instructions: 
    #1) Run the script generate_dirty_data.py to create ms_data_dirty.csv which you will clean in the next steps (you may need to install the numpy library).
    #2) Clean the raw data file:
        # - Remove comment lines
        # - Remove empty lines
        # - Remove extra commas
        # - Extract essential columns: patient_id, visit_date, age, education_level, walking_speed
        # - Save the file as ms_data.csv
    #3) Create a file, insurance.lst listing unique labels for a new variable, insurance_type, one per line (your choice of labels). Depending on how you want to import this file later with Python, you may or may not want to add a header row nameing the column as insurance_type.
    #4) Generate a summary of the processed data:
        # - Count the total number of visits (rows, not including the header)
        # - Display the first few records

#1)
python3 generate_dirty_data.py #running the python script 

#2)
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,*/,/g' | cut -d, -f1,2,4,5,6  > cleaned_data.csv #removes lines with comments, empty lines, and extra commas from the dirty csv. extracts the five columns we need and appends it to csv called cleaned_data.csv
head -n 1 cleaned_data.csv > ms_data.csv #appends the first row (all the column names) to ms_data.csv
awk -F, '$5 >= 2.0 && $5 <= 8.0' cleaned_data.csv >> ms_data.csv #filters the walking speed from 2.0-8.0 and appends it to ms_data.csv.


#3)
echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst #creating a new list with the three insurance types 

#4) 
echo "Total number of visits:" 
tail -n +2 ms_data.csv | wc -l #counts the total number of rows in the dataset starting from the second row, which removes the headers 

echo "First 10 records:"  
head -n 10 ms_data.csv #printing first 10 records 
 
rm cleaned_data.csv 
echo "Removed cleaned_data.csv" #removing the cleaned_data.csv, which was a file created as an intermediate step in the data cleaning process 

#!/bin/bash

# Step 1: Run the data generation script
python3 generate_dirty_data.py

# Step 2: Clean the data
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,/,/g' | \
cut -d',' -f1,2,4,5,6 | awk -F',' '$5 >= 2.0 && $5 <= 8.0' > ms_data.csv

# Step 3: Create insurance type list
echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst

# Step 4: Summarize data
total_visits=$(tail -n +2 ms_data.csv | wc -l)
echo "Total number of visits (excluding header): $total_visits"
echo "First few records of the processed data:"
head -n 5 ms_data.csv
#!/bin/bash

python3 generate_dirty_data.py

#Remove comment lines, empty lines, extra commas + Extract essential columns
grep -v "#" ms_data_dirty.csv | sed -e '/^$/d' |  sed -e 's/,,/,/g' | cut -d',' -f1,2,4,5,6 > ms_data.csv 

#new file + new column insurance type
echo -e "Basic \nBetter \nBest" > insurance.lst

#Count the total number of visits/rows +  Display the first few records
wc -l ms_data.csv 
head ms_data.csv


#make script executable with: chmod +x prepare.sh
#run script: bash prepare.shsed '/^#/d' ms_data_dirty.csv | #remove comment lines
sed '/^$/d' | #remove empty lines
sed 's/,+"/,/g' | #replace more than one comma with just one comma
sed 's/ +"/ /g'| #replace any instance of multiple spaces with just one space
cut -d, -f1,2,4,5,6 > ms_data.csv #extract specific cols and save to new csv file

for i in Basic Premium Platinum; do #insurance levels
    echo $i >> insurance.lst
done

wc -l ms_data.csv #number of rows in dataset

python -c "import pandas as pd; print(pd.read_csv('ms_data.csv').head())" #head of dataset#!/bin/bash

# Generate dirty data (this step should create ms_data_dirty.csv)
python3 generate_dirty_data.py

# Debug: Check raw input file
echo "Raw input file (first 5 lines):"
head -n 5 ms_data_dirty.csv

# Step 1: Remove comment lines (lines starting with #) and empty lines
echo "After removing comments and empty lines (first 5 lines):"
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | head -n 5

# Step 2: Remove extra commas (if any)
echo "After removing extra commas (first 5 lines):"
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,*/,/g' | head -n 5

# Step 3: Extract relevant columns (patient_id, visit_date, age, education_level, walking_speed)
echo "After extracting relevant columns (first 5 lines):"
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,*/,/g' | cut -d',' -f1,2,4,5,6 | head -n 5

# Step 4: Filter walking speed values between 2.0 and 8.0
echo "After filtering walking speeds (first 5 lines):"
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,*/,/g' | cut -d',' -f1,2,4,5,6 | awk -F',' 'NR==1 || ($5 >= 2.0 && $5 <= 8.0)' | head -n 5

# Final Output to ms_data.csv
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,*/,/g' | cut -d',' -f1,2,4,5,6 | awk -F',' 'NR==1 || ($5 >= 2.0 && $5 <= 8.0)' > ms_data.csv

# Step 5: Create insurance.lst file
echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst

# Step 6: Display the total number of visits and the first few records
echo "Total visits:"
tail -n +2 ms_data.csv | wc -l
echo "First few records:"
head -n 5 ms_data.csv
#!/bin/bash

python3 generate_dirty_data.py

# remove comments, empty lines, and extra commas; extract columns
cat ms_data_dirty.csv | grep -v '^#' | grep -v -e '^$' | sed -e 's/,,*/,/g' | cut -d ',' -f1,2,4,5,6 > ms_data.csv

touch insurance.lst # create list file

echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst # create list

visits=$(wc -l < ms_data.csv)

echo "Total number of visits: $((visits - 1))" # count visits w/o header

# Display the first records
echo "First 4 records:"
head -n 5 ms_data.csv

chmod +x prepare.sh#!/bin/bash

python3 generate_dirty_data.py

# clean the file
# remove comments, empty lines, extra commas, and extract essential columns: 
# patient_id, visit_date, age, education_level, walking_speed
cat ms_data_dirty.csv | grep -v '^#' | sed '/^$/d' | sed -e 's/,,*/,/g' \
  | awk -F',' 'NR == 1 || ($6 >= 2.0 && $6 <= 8.0)' \
  | cut -d ',' -f1,2,4,5,6 > ms_data.csv

# create list file for insurance with tiers
echo -e "Basic\nPremium\nPlatinum" > insurance.lst # create list

visits=$(wc -l < ms_data.csv)

# summary of processed data
echo "Total number of visits: $((visits - 1))" # count visits w/o header

# display first few records
echo "First few records:" 
head -n 5 ms_data.csv

chmod +x prepare.sh# Step 1: Generate raw data
# Use the provided script to create a raw data file
python3 generate_dirty_data.py

# Step 2: Clean data
# Remove comment lines, empty lines, and extra commas
grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,*/,/g' > temp_data.csv
# Extract relevant columns: patient_id, visit_date, age, education_level, walking_speed
cut -d',' -f1,2,4,5,6 temp_data.csv > ms_data.csv

# Step 3: Generate insurance type file
# Create a file listing insurance types (one per line)
echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst

# Step 4: Summarize cleaned data
# Count the total number of visits (excluding the header)
echo "Total visits: $(wc -l < ms_data.csv | awk '{print $1 - 1}')"
# Display the first few rows of the cleaned data
echo "First few records:"
head -n 10 ms_data.csv

# Verify walking speed range
# Find the minimum and maximum walking speeds
min_speed=$(awk -F',' 'NR > 1 {if (min == "" || $5 < min) min = $5} END {print min}' ms_data.csv)
max_speed=$(awk -F',' 'NR > 1 {if (max == "" || $5 > max) max = $5} END {print max}' ms_data.csv)
echo "Minimum walking speed: $min_speed"
echo "Maximum walking speed: $max_speed"grep -v '^#' ms_data_dirty.csv | sed '/^$/d' | sed 's/,,*/,/g' | \
cut -d',' -f1,2,4,5,6 | \
awk -F',' '{if (NR==1 || ($5 >= 2.0 && $5 <= 8.0)) print}' > ms_data.csv
echo -e "insurance_type\nBasic\nPremium\nPlatinum" > insurance.lst
total_visits=$(tail -n +2 ms_data.csv | wc -l)
echo "Total number of visits: $total_visits"
echo "First few records:"
head -n 8 ms_data.csv#!/bin/bash

# The dirty dataset is generated by `generate_dirty_data.py`
# Clean the raw data file:
## Remove comment lines
grep -v '^#' ms_data_dirty.csv > ms_data_no_comments.csv

## Remove empty lines
sed '/^$/d' ms_data_no_comments.csv > ms_data_no_empty.csv

## Remove extra commas
sed -e 's/,,*/,/g' ms_data_no_empty.csv > ms_data_no_commas.csv

## Extract records within a specific range of walking speeds
awk -F',' '$6 >= 2.0 && $6 <= 8.0' ms_data_no_commas.csv > ms_data_limited_speed.csv

## Extract essential columns: patient_id, visit_date, age, education_level, walking_speed
# First, get the header from the original file
head -n 1 ms_data_no_commas.csv | cut -d',' -f1,2,4,5,6 > ms_data.csv

# Then, extract the data rows and append to ms_data.csv
tail -n +2 ms_data_limited_speed.csv | cut -d',' -f1,2,4,5,6 >> ms_data.csv

# Create insurance.lst with types A, B, and C
echo -e "insurance_type\nA\nB\nC" > insurance.lst

# Check the cleaned data
## Count the total number of visits (rows, not including the header)
total_rows=$(tail -n +2 ms_data.csv | wc -l)
echo "Total number of visits: $total_rows"

## Display the first few records in ms_data.csv
echo "First few records in ms_data.csv:"
head ms_data.csv

## Display the first few records in insurance.lst
echo "Insurance types:"
head insurance.lst

# Check for any records with walking speeds outside 2.0-8.0 feet/second range
echo "Records with walking speeds < 2.0 or > 8.0:"
out_of_range_records=$(awk -F',' '$5 < 2.0 || $5 > 8.0' ms_data.csv)

# Clean up temporary files
rm ms_data_no_comments.csv ms_data_no_empty.csv ms_data_no_commas.csv ms_data_limited_speed.csv