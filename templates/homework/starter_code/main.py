#!/usr/bin/env python3
"""
[Assignment Title] - Main Implementation

Student: [Your Name]
Course: [Course Code]
Assignment: [Assignment Number]
Date: [Date]

Description:
[Brief description of what this module does]
"""

import sys
from typing import List, Dict, Any, Optional, Union


def function_template(param1: str, param2: int) -> str:
    """
    Template function showing proper structure and documentation.
    
    This is a template to show students the expected format for their functions.
    Replace this with your actual implementation.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter
    
    Returns:
        Description of what the function returns
    
    Raises:
        ValueError: When parameter validation fails
        TypeError: When parameters are of wrong type
    
    Example:
        >>> function_template("hello", 42)
        'hello_42'
    """
    # Input validation
    if not isinstance(param1, str):
        raise TypeError("param1 must be a string")
    if not isinstance(param2, int):
        raise TypeError("param2 must be an integer")
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
    
    # TODO: Replace with your implementation
    result = f"{param1}_{param2}"
    return result


def calculate_average(numbers: List[float]) -> float:
    """
    Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numeric values
    
    Returns:
        The arithmetic mean of the numbers
    
    Raises:
        ValueError: If the list is empty
        TypeError: If the list contains non-numeric values
    
    Example:
        >>> calculate_average([1, 2, 3, 4, 5])
        3.0
    """
    # TODO: Implement this function
    pass


def find_maximum(numbers: List[Union[int, float]]) -> Union[int, float]:
    """
    Find the maximum value in a list of numbers.
    
    Args:
        numbers: List of numeric values
    
    Returns:
        The largest number in the list
    
    Raises:
        ValueError: If the list is empty
        TypeError: If the list contains non-numeric values
    
    Example:
        >>> find_maximum([3, 1, 4, 1, 5, 9])
        9
    """
    # TODO: Implement this function
    pass


def count_occurrences(items: List[Any], target: Any) -> int:
    """
    Count how many times a target item appears in a list.
    
    Args:
        items: List of items to search through
        target: Item to count occurrences of
    
    Returns:
        Number of times target appears in items
    
    Example:
        >>> count_occurrences(['a', 'b', 'a', 'c', 'a'], 'a')
        3
    """
    # TODO: Implement this function
    pass


def validate_input(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that a dictionary contains all required keys.
    
    Args:
        data: Dictionary to validate
        required_keys: List of keys that must be present
    
    Returns:
        True if all required keys are present, False otherwise
    
    Raises:
        TypeError: If data is not a dictionary or required_keys is not a list
    
    Example:
        >>> validate_input({'name': 'Alice', 'age': 25}, ['name', 'age'])
        True
        >>> validate_input({'name': 'Bob'}, ['name', 'age'])
        False
    """
    # TODO: Implement this function
    pass


class DataProcessor:
    """
    Example class showing proper class structure and documentation.
    
    This class demonstrates how to structure classes with proper
    initialization, methods, and documentation.
    
    Attributes:
        data: The data being processed
        processed: Whether the data has been processed
    
    Example:
        >>> processor = DataProcessor([1, 2, 3, 4, 5])
        >>> result = processor.process()
        >>> print(result)
        [2, 4, 6, 8, 10]
    """
    
    def __init__(self, data: List[Any]):
        """
        Initialize the DataProcessor with data.
        
        Args:
            data: List of data items to process
        
        Raises:
            TypeError: If data is not a list
        """
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
        
        self.data = data
        self.processed = False
    
    def process(self) -> List[Any]:
        """
        Process the data according to assignment requirements.
        
        Returns:
            Processed data
        
        Raises:
            RuntimeError: If processing fails
        """
        # TODO: Implement processing logic
        try:
            # Example processing: double each number
            if all(isinstance(x, (int, float)) for x in self.data):
                result = [x * 2 for x in self.data]
            else:
                result = self.data.copy()
            
            self.processed = True
            return result
        except Exception as e:
            raise RuntimeError(f"Processing failed: {e}")
    
    def reset(self) -> None:
        """Reset the processor to its initial state."""
        self.processed = False
    
    def __str__(self) -> str:
        """Return string representation of the processor."""
        status = "processed" if self.processed else "unprocessed"
        return f"DataProcessor({len(self.data)} items, {status})"
    
    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"DataProcessor(data={self.data}, processed={self.processed})"


def main():
    """
    Main function demonstrating usage of the implemented functions.
    
    This function is called when the script is run directly and should
    demonstrate the functionality of your implemented functions.
    """
    print("Running homework assignment...")
    
    try:
        # Example usage of your functions
        print("1. Testing function_template:")
        result = function_template("test", 123)
        print(f"   Result: {result}")
        
        print("\n2. Testing calculate_average:")
        # TODO: Add test of your calculate_average function
        # numbers = [1, 2, 3, 4, 5]
        # avg = calculate_average(numbers)
        # print(f"   Average of {numbers}: {avg}")
        print("   TODO: Implement and test calculate_average")
        
        print("\n3. Testing find_maximum:")
        # TODO: Add test of your find_maximum function
        print("   TODO: Implement and test find_maximum")
        
        print("\n4. Testing count_occurrences:")
        # TODO: Add test of your count_occurrences function
        print("   TODO: Implement and test count_occurrences")
        
        print("\n5. Testing validate_input:")
        # TODO: Add test of your validate_input function
        print("   TODO: Implement and test validate_input")
        
        print("\n6. Testing DataProcessor:")
        processor = DataProcessor([1, 2, 3, 4, 5])
        print(f"   Created: {processor}")
        result = processor.process()
        print(f"   Processed: {result}")
        print(f"   Status: {processor}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the main function when script is executed directly
    sys.exit(main())