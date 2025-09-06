1. **Daily Quote Generator:** Select a random quote for the day and prints it. Optional: The same quote should be generated for a given day.

	Your task:
	1. Complete the get_quote_of_the_day() function
	2. Print the crontab that would run this script daily at 8:00 AM and append the output to a file

	Hint: Look up `random.choice()` to select a random item from a list. You can use the `date` module to get the current date and set a seed for the random number generator.
2. **Word Frequency Counter:** Read a text file (example code included) and count the frequency of each word, ignoring case.
	Usage: `python word_frequency.py <input_file>`  
	
	Your task: Complete the word_frequency() function to count word frequencies sorted alphabetically. Run the script on 'alice_in_wonderland.txt'.
	
	Hints:
	- Use a dictionary to store word frequencies
	- Consider using the `lower()` method to ignore case
	- The `split()` method can be useful for splitting text into words
	- Decompress Alice in Wonderland before running the script on it
3. **Maximum Product of 13 Adjacent Digits:** Find the thirteen adjacent digits in the 1000-digit number that have the greatest product. What is the value of this product?
	
	Your task: Complete the find_greatest_product() function to solve the problem.
	
	Hints:
	- You can iterate through the string using a for loop and string slicing
	- Keep track of the maximum product as you go through the loooong number
	- (Optional) Convert characters to integers using `int()`

```
73167176531330624919225119674426574742355349194934
96983520312774506326239578318016984801869478851843
85861560789112949495459501737958331952853208805511
12540698747158523863050715693290963295227443043557
66896648950445244523161731856403098711121722383113
62229893423380308135336276614282806444486645238749
30358907296290491560440772390713810515859307960866
70172427121883998797908792274921901699720888093776
65727333001053367881220235421809751254540594752243
52584907711670556013604839586446706324415722155397
53697817977846174064955149290862569321978468622482
83972241375657056057490261407972968652414535100474
82166370484403199890008895243450658541227588666881
16427171479924442928230863465674813919123162824586
17866458359124566529476545682848912883142607690042
24219022671055626321111109370544217506941658960408
07198403850962455444362981230987879927244284909188
84580156166097919133875499200524063689912560717606
05886116467109405077541002256983155200055935729725
71636269561882670428252483600823257530420752963450
```
