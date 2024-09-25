# Command Line Live Demo Plan

## 1. Environment Setup
- Open a terminal
- Explain the prompt and basic navigation (`pwd`, `ls`, `cd`)
- Create a demo directory: `mkdir subdir`
- Create already existing directory `mkdir -p subdir && cd subdir`

## 2. Working with Files and Directories
- Create files: `touch file1.txt file2.txt`
- Move and copy files: `mv file1.txt subdir/` and `cp file2.txt subdir/file2_copy.txt`
- Use `cat`, `head`, and `tail` to display file contents

## 3. Environment Variables and .env Files
- Show existing environment variables: `env`
- `env` and grep
- Set a temporary variable: `export DEMO_VAR="Hello, Demo!"`
- Echo the variable: `echo $DEMO_VAR`
- Create a .env file: `echo "SECRET_KEY=mysecretkey" > .env`
- Load .env file using `source` and the `load_env` function
	```bash
	load_env () {
	    set -o allexport
	    source $1
	    set +o allexport
	}
	```

- Demonstrate accessing variables in Python

## 4. Shell Scripts
- Create a simple shell script:
  ```bash
  echo '#!/bin/bash
  echo "Current directory: $(pwd)"
  echo "Files in this directory:"
  ls -l
  echo "Today is $(date)"' > demo_script.sh
  ```
- Make it executable: `chmod +x demo_script.sh`
- Run the script: `./demo_script.sh`

## 5. Cron Jobs
- Explain crontab syntax
- Open crontab editor: `crontab -e`
- Add a simple cron job (e.g., to run every minute): `* * * * * echo "Cron test" >> ~/cron_log.txt`
- Show how to check if it's running: `tail -f ~/cron_log.txt`

## 6. Compression and Decompression
- Create sample files: `echo "File 1" > file1.txt && echo "File 2" > file2.txt`
- Compress with tar.gz: `tar -czvf archive.tar.gz file1.txt file2.txt`
	- Extract tar.gz: `mkdir extracted && tar -xzvf archive.tar.gz -C extracted`
	- Extract tar (not gzipped): `tar -xvf archive.tar` 
- Demonstrate zip: `zip archive.zip file1.txt file2.txt`
	- Extract zip: `unzip archive.zip -d unzipped`

## 7. Links
- Create a symbolic link: `ln -s /path/to/target symlink_name`
- Show link properties: `ls -l symlink_name`
- Demonstrate how changing the target affects the link

# Python Data Structures and Comprehensions

## Exercise 0: List operations

Given: `numbers = [0, 1, 2, 3, 4, 5]`

- Indexing:
    - First element: `numbers[0]` # 0
    - Last element: `numbers[-1]` # 5
    - Second-to-last: `numbers[-2]` # 4
- Slicing:
    - First three: `numbers[:3]` # [0, 1, 2]
    - All but first: `numbers[1:]` # [1, 2, 3, 4, 5]
    - Last three: `numbers[-3:]` # [3, 4, 5]
    - Reverse: `numbers[::-1]` # [5, 4, 3, 2, 1, 0]
- Operations:
    - Length: `len(numbers)` # 6
    - Min/Max: `min(numbers)`, `max(numbers)` # 0, 5
    - Sum: `sum(numbers)` # 15
- Modifying:
    - Append: `numbers.append(6)`
    - Insert: `numbers.insert(0, -1)`
    - Remove: `numbers.remove(3)`
    - Pop: `numbers.pop()`, `numbers.pop(0)`

## Exercise 1: Even Numbers List Comprehension

Create a list of even numbers from 0 to 20 using a list comprehension.

Solution:
```python
even_numbers = [x for x in range(21) if x % 2 == 0]
print(even_numbers)
# Output: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
```

---

## Exercise 2: Name Length Dictionary

Create a dictionary mapping names to their lengths for a list of names.

Solution:
```python
names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
name_lengths = {name: len(name) for name in names}
print(name_lengths)
# Output: {'Alice': 5, 'Bob': 3, 'Charlie': 7, 'David': 5, 'Eve': 3}
```

---

## Exercise 3: Unique Letters Set Comprehension

Use a set comprehension to find all unique letters in a sentence.

Solution:
```python
sentence = "The quick brown fox jumps over the lazy dog"
unique_letters = {letter.lower() for letter in sentence if letter.isalpha()}
print(unique_letters)
# Output: {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
#          'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
```

---

## Exercise 4a: Prime Numbers Generator Expression

Create a generator expression that yields prime numbers up to `n`.

Solution:
```python
def is_prime(n):
    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))

prime_gen = (x for x in range(2, 101) if is_prime(x))

print(list(prime_gen))
# Output: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
```

---

## Exercise 4b: Sieve of Eratosthenes

Implement the Sieve of Eratosthenes using a generator function and set comprehension.

Original Solution:
```python
def gen_primes(N):
    """Generate primes up to N"""
    primes = set()
    for n in range(2, N):
        if all(n % p > 0 for p in primes):
            primes.add(n)
            yield n

print(*gen_primes(70))
# Output: 2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67
```

### `all()` and `yield` in this example
In our `gen_primes` function:
```python
def gen_primes(N):
    primes = set()
    for n in range(2, N):
        if all(n % p > 0 for p in primes):
            primes.add(n)
            yield n
```

- `all(n % p > 0 for p in primes)`: Checks if `n` is not divisible by any known prime
- `yield n`: Pauses the function and returns `n` as the next prime number

This allows for efficient, on-demand generation of prime numbers.

