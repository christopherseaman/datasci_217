#!/usr/bin/env python3

def write_fruits(filename):
    fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    with open(filename, 'w') as file:
        for fruit in fruits:
            file.write(fruit + '\n')
    print(f"Wrote fruits to {filename}")

# if __name__ == "__main__":
#     write_fruits("fruits.txt")
#     print("I'm doing stuff!")