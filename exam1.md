#draft
1. **Largest Product in a File** (50 points) Given a file containing a 20x20 grid of numbers, find the greatest product of four adjacent numbers in the same direction (up, down, left, right, or diagonally). Requirements:
    - Implement functions to read the grid from a file and to calculate the maximum product
    - Use `with` statements for file operations
    - Handle potential file I/O errors and invalid input with try/except
    - Use a Jupyter notebook to visualize the grid and highlight the four numbers with the greatest product
2.  **Longest Collatz Sequence** (40 points) For all starting numbers under a given limit (specified via command-line argument), calculate the length of the Collatz sequence. Write the results to a file, then read the file to find the starting number that produces the longest chain. Requirements:
    - Implement a function to generate the Collatz sequence
    - Use `with` statements for file operations
    - Handle potential errors (file I/O, invalid input) with try/except
    - Use a Jupyter notebook to plot the relationship between starting numbers and sequence lengths
3. Given a large FASTA file (text with ACTG base-pairs):
	1. Write python script that finds enzyme cutsites (specific short sequences and their complements)
	2. Count the "coverage" for each position in the sequence; e.g., the number of enzyme cutsite pairs 100-200k bp apart containing that position. Save a graph of it (this code will be given).
	3. Python script should take arguments for the cutsite sequence and FASTA file
	4. Write a shell script that: downloads FASTA.tgz from given URL, decompresses it, and runs the python script on it for several different cutsite sequences

A cutsite is a specific DNA sequence that a restriction enzyme recognizes and cuts. For example, the enzyme EcoRI recognizes the sequence "GAATTC". Importantly, DNA is double-stranded, and the enzyme will cut both strands. The complementary sequence on the opposite strand is the inverse and reverse of the original sequence. For "GAATTC", the complementary sequence would be "CTTAAG".

1. Create functions for complementing and reversing cutsites​​​​​​​​​​​​​​​​
2. Check if the FASTA file contains matched cutsites of any length for cutsite "GAATTC"​​​​​​​​​​​​​​​​
3. Find coverage of cutsites 10-20kbp apart in the fasta file for "GAATTC"​​​​​​​​​​​​​​​​
4. Find coverage of cutsites 10-20kbp apart for set of cutsites​​​​​​​​​​​​​​​​ {}
5. Plot the coverage of all cutsites using the provided graphing function


```python
plot_coverage(multi_coverage, 'Coverage of Multiple Cutsites')
```

This simplified version of the assignment breaks down the process into more manageable steps while still covering important concepts like:

- String manipulation (inverting and reversing sequences)
- File I/O (reading FASTA files)
- Basic algorithms (finding sequences, calculating coverage)
- Working with multiple datasets

1. Start by explaining cutsites and their biological significance.
2. Go through each function, explaining its purpose and how it works.
3. Demonstrate how to use these functions with a sample FASTA file.
4. Show how to combine these functions to analyze multiple cutsites.
5. Finally, demonstrate the plotting of results.

