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
   - `data`
   - `scripts`
   - `results`
3. In the scripts directory, create empty Python files named:
   - `generate_fasta.py`
   - `dna_operations.py`
   - `find_cutsites.py`
4. In the results directory, create an empty file named "cutsite_summary.txt".
5. In the data directory, create an empty file named "random_sequence.fasta".
6. Create a README.md file in the main project directory with a brief description of the project structure.

Tips:
- Use `mkdir -p` to create directories and their parents if they don't exist.
- The `touch` command can be used to create empty files.
- Remember to make your script executable with `chmod +x setup_project.sh`.
- Use `echo` to add content to the README.md file.

### Task

Run the script and check the output (include all files in your assignmentrepository).

### Example usage
```
bash setup_project.sh
```

Expected output:
```
Project directory structure created successfully:
your_repository/
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

FASTA format is a text-based format for representing nucleotide or peptide sequences. FASTA files usually include a header line with the `> Sequence Name` before the sequence data, but this is not required for this assignment. Examples of FASTA files can be found on its [Wikipedia page](https://en.wikipedia.org/wiki/FASTA_format).

Create a Python script that generates a random DNA sequence and saves it in FASTA format. Your script should:

1. Generate a random DNA sequence of 1 million base pairs (using A, C, G, T).
2. Format the sequence with 80 base pairs per line.
3. Save the sequence in FASTA format in the "data" directory, with the filename "random_sequence.fasta".

Tips:
- Use Python's `random` module to generate random DNA sequences.
- Remember to open the file in write mode when saving the FASTA data.
- Use string joining for efficient concatenation of large sequences.
- Use a `for` loop to count characters when adding each line of the sequence to the file.
- (optional, advanced) The `textwrap` module can help you format the sequence into 80-character lines.

### Task

Run the script and check the output (include it in your repository).

### Example usage
```
python generate_fasta.py
```

Expected output:
```
Random DNA sequence generated and saved to bioinformatics_project/data/random_sequence.fasta
```

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

Tips:
- Create a dictionary mapping each base to its complement for easy lookup; e.g., `complement['A'] = 'T'`.
- String slicing with a step of -1 can be used to reverse a string efficiently.
- Remember to handle both uppercase and lowercase input.
- Use `sys.argv` or `argparse` to access command-line arguments in your script.
- (optional, advanced) Use `str.maketrans()` and `str.translate()` for efficient base substitution.

### Task

Run the script on the sequence "CCTCAGC"

### Example usage
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
File name: `find_cutsites.py`

In molecular biology, restriction enzymes cut DNA at specific sequences called restriction sites or cut sites. Finding pairs of cut sites that are a certain distance apart is important for various genetic engineering techniques. Cutsites are often represented with a vertical bar (|) in the cut site sequence, indicating where the enzyme cuts the DNA.

An example:
- Take cut site sequence "G|GATCC" for BamHI:
- In the sequence dna="AAGG|GATCCTT", the cut site starts at index 4
- The enzyme would cut between G and T, resulting in "AAGG" and "GATCCTT".
- So the cut would happen before dna[4], which we would count as it's location.

Create a Python script that finds pairs of restriction enzyme cut sites that are 80-120 kilobase pairs (kbp) apart in a given FASTA file. Your script should:

1. Accept two arguments: the FASTA file path (data/random_sequence.fasta) and a cut site sequence (e.g., "G|GATCC")
2. Read the FASTA file and save the DNA sequence to a variable omitting whitespace.
3. Find all occurrences of the cut site (specified below) in the DNA sequence.
4. Find all pairs of cut site locations that are 80,000-120,000 base pairs (80-120 kbp) apart.
5. Print the total number of cut site pairs found and the positions of the first 5 pairs.
6. Save a summary of the results in the results directory as "distant_cutsite_summary.txt".

Tips:
- When running the script, put the cut site sequence in quotes to prevent issues with the pipe character, e.g., "G|GATCC".
- Remember to remove the `|` character from the cut site sequence before searching for it in the DNA sequence.
- Consider using string methods like `.replace()` or `strip()` to remove whitespace from the FASTA sequence.
- (optional, advanced) The `re` module can be helpful for finding all occurrences of the cut site in the sequence.

### Task

Run the script on the random sequence you generated in Question 2 and with cut site sequence "G|GATCC" (BamHI)

### Example usage
`````
python find_distant_cutsites.py data/random_sequence.fasta "G|GATCC"
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
