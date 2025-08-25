#!/usr/bin/env python3
"""
Project Euler Problem 1: Multiples of 3 or 5

If we list all the natural numbers below 10 that are multiples of 3 or 5,
we get 3, 5, 6, and 9. The sum of these multiples is 23.

Find the sum of all the multiples of 3 or 5 below 1000.
"""

def is_multiple_of_3_or_5(number):
    """
    Check if a number is a multiple of 3 or 5.
    
    Args:
        number (int): Number to check
        
    Returns:
        bool: True if multiple of 3 or 5, False otherwise
    """
    return number % 3 == 0 or number % 5 == 0

def solve_euler_1(limit):
    """
    Find sum of multiples of 3 or 5 below the given limit.
    
    Args:
        limit (int): Upper limit (exclusive)
        
    Returns:
        tuple: (sum, list_of_multiples)
    """
    total = 0
    multiples = []
    
    for number in range(1, limit):
        if is_multiple_of_3_or_5(number):
            multiples.append(number)
            total += number
    
    return total, multiples

def main():
    """
    Main function that demonstrates solving Project Euler Problem 1.
    
    This function shows the problem-solving approach: test with a small example
    first, then apply the solution to the actual problem. This methodology is
    fundamental to computational problem-solving in data science.
    """
    print("Project Euler Problem 1: Multiples of 3 or 5")
    print("=" * 45)
    
    # Test with small example first
    test_limit = 10
    test_sum, test_multiples = solve_euler_1(test_limit)
    print(f"\nTest case (below {test_limit}):")
    print(f"Multiples: {test_multiples}")
    print(f"Sum: {test_sum}")
    
    # Solve the actual problem
    actual_limit = 1000
    actual_sum, _ = solve_euler_1(actual_limit)
    print(f"\nActual problem (below {actual_limit}):")
    print(f"Sum of all multiples of 3 or 5: {actual_sum}")

if __name__ == "__main__":
    main()