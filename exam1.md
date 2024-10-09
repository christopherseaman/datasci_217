# DS-217 Exam 1

## Instructions
- This exam consists of four interconnected questions that will guide you through a bioinformatics project workflow.
- Submit your solutions as separate files with the names specified in each question.
- Make sure your code is well-commented and follows good programming practices.
- You may use any built-in Python libraries or standard command-line tools, but no external packages.

## Question 1: Project Setup (15 points)

Save as a shell script with the name provided:
File name: `setup_project.sh`

Create a shell script that sets up a directory structure for a bioinformatics project. Your script should:

1. Create a main project directory called "bioinformatics_project".
2. Inside the main directory, create the following subdirectories:
   - data
   - scripts
   - results
3. In the scripts directory, create empty Python files named:
   - generate_fasta.py
   - dna_operations.py
   - find_cutsites.py
4. Create a README.md file in the main project directory with a brief description of the project structure.

### Task

Run the script and check the output (include it in your repository).

Example usage:
```
bash setup_project.sh
```

Expected output:
```
Project directory structure created successfully:
bioinformatics_project/
├── data/
├── scripts/
│   ├── generate_fasta.py
│   ├── dna_operations.py
│   └── find_cutsites.py
├── results/
└── README.md
```

## Question 2: Generate Random FASTA Data (25 points)
File name: `generate_fasta.py`

Create a Python script that generates a random DNA sequence and saves it in FASTA format. Your script should:

1. Generate a random DNA sequence of 1 million base pairs (using A, C, G, T).
2. Format the sequence with 80 base pairs per line.
3. Save the sequence in FASTA format in the data directory, with the filename "random_sequence.fasta".
4. The FASTA file should have a header line: ">Random_DNA_Sequence".

### Task

Run the script and check the output (include it in your repository).

Example usage:
```
python generate_fasta.py
```

Expected output:
```
Random DNA sequence generated and saved to bioinformatics_project/data/random_sequence.fasta
```

... (Questions 1 and 2 remain the same)

## Question 3: DNA Sequence Operations (30 points)
File name: `dna_operations.py`

Create a Python script that performs various operations on DNA sequences. Your script should:

1. Accept a DNA sequence as a command-line argument.
2. Implement the following functions:
   - `complement(sequence)`: Returns the complement of a DNA sequence (A -> T, C -> G, G -> C, T -> A).
   - `reverse(sequence)`: Returns the reverse of a sequence (e.g. "CCTCAGC" -> "CAGCCTC").
   - `reverse_complement(sequence)`: Returns the reverse complement of a DNA sequence (e.g. "CCTCAGC" -> "GAGCTTG"); i.e. the reverse of the complement (apply `complement` then `reverse`, or vice versa).
3. For the input sequence, print:
   - The original sequence
   - Its complement
   - Its reverse
   - Its reverse complement

### Task

Run the script on the sequence "CCTCAGC"

Example usage:
`````
python dna_operations.py GAATTC
`````

Expected output:
`````
Original sequence: GAATTC
Complement: CTTAAG
Reverse: CTTAAG
Reverse complement: GAATTC
`````

## Question 4: Find Distant Cutsites in FASTA Data (30 points)
File name: `find_distant_cutsites.py`

Create a Python script that finds pairs of restriction enzyme cut sites that are 80-120 kilobase pairs (kbp) apart in a given FASTA file. Your script should:

1. Accept two arguments: the FASTA file path and a cut site sequence (e.g., "GGATCC" for BamHI).
2. Read the FASTA file.
3. Find all occurrences of the cut site in the DNA sequence. Consider the start of the cut site as its location.
4. Find all pairs of cut sites that are 80-120 kbp apart.
5. Print the total number of cut site pairs found and the positions of the first 5 pairs.
6. Save a summary of the results in the results directory as "distant_cutsite_summary.txt".


### Task

Run the script on the random sequence you generated in Question 2 and with cut site sequence "ACCTGC" (BspMI)

Example usage:
`````
python find_distant_cutsites.py bioinformatics_project/data/random_sequence.fasta GGATCC
`````

Expected output:
`````
Analyzing cut site: GGATCC
Total cut sites found: 976
Cut site pairs 80-120 kbp apart: 1423
First 5 pairs:
1. 15231 - 101589
2. 15231 - 118956
3. 28764 - 109102
4. 28764 - 126471
5. 42198 - 122609

Results saved to bioinformatics_project/results/distant_cutsite_summary.txt
`````
