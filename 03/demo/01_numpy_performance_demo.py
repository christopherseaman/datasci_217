#!/usr/bin/env python3
"""
Demo 1: NumPy Performance Comparison
Demonstrates the 10-100x speed improvement of NumPy over Python lists
This directly demonstrates the performance comparison from Lecture 3, lines 19-34
"""

import numpy as np
import time

def demonstrate_python_performance():
    """Demonstrate Python list performance (SLOW)"""
    print("PYTHON LIST PERFORMANCE (SLOW)")
    print("=" * 40)
    
    # Create a million-element list
    print("Creating a list with 1,000,000 elements...")
    my_list = list(range(1_000_000))
    print(f"List created: {len(my_list):,} elements")
    
    # Time the multiplication operation
    print("\nMultiplying each element by 2...")
    start_time = time.time()
    result = [x * 2 for x in my_list]
    python_time = time.time() - start_time
    
    print(f"Python list multiplication: {python_time:.4f} seconds")
    print(f"Result: {len(result):,} elements")
    print(f"First 5 elements: {result[:5]}")
    
    return python_time

def demonstrate_numpy_performance():
    """Demonstrate NumPy array performance (FAST)"""
    print("\nNUMPY ARRAY PERFORMANCE (FAST)")
    print("=" * 40)
    
    # Create a million-element NumPy array
    print("Creating a NumPy array with 1,000,000 elements...")
    my_array = np.arange(1_000_000)
    print(f"Array created: {my_array.size:,} elements")
    print(f"Array shape: {my_array.shape}")
    print(f"Array dtype: {my_array.dtype}")
    
    # Time the multiplication operation
    print("\nMultiplying each element by 2...")
    start_time = time.time()
    result = my_array * 2
    numpy_time = time.time() - start_time
    
    print(f"NumPy array multiplication: {numpy_time:.4f} seconds")
    print(f"Result: {result.size:,} elements")
    print(f"First 5 elements: {result[:5]}")
    
    return numpy_time

def compare_performance():
    """Compare Python vs NumPy performance"""
    print("\nPERFORMANCE COMPARISON")
    print("=" * 30)
    
    # Test with different array sizes
    sizes = [10_000, 100_000, 1_000_000]
    
    print("Testing different array sizes:")
    print(f"{'Size':<12} {'Python (s)':<12} {'NumPy (s)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for size in sizes:
        # Python timing
        python_list = list(range(size))
        start_time = time.time()
        [x * 2 for x in python_list]
        python_time = time.time() - start_time
        
        # NumPy timing
        numpy_array = np.arange(size)
        start_time = time.time()
        numpy_array * 2
        numpy_time = time.time() - start_time
        
        speedup = python_time / numpy_time if numpy_time > 0 else float('inf')
        
        print(f"{size:<12,} {python_time:<12.4f} {numpy_time:<12.4f} {speedup:<10.1f}x")

def demonstrate_why_numpy_is_faster():
    """Explain why NumPy is faster"""
    print("\nWHY NUMPY IS FASTER")
    print("=" * 25)
    
    print("1. Memory Layout:")
    print("   - Python lists: Scattered objects in memory")
    print("   - NumPy arrays: Contiguous memory blocks")
    
    print("\n2. Data Types:")
    print("   - Python lists: Mixed types, dynamic")
    print("   - NumPy arrays: Homogeneous types, fixed")
    
    print("\n3. Operations:")
    print("   - Python lists: Interpreted loops")
    print("   - NumPy arrays: Compiled C code")
    
    print("\n4. Vectorization:")
    print("   - Python lists: Element-by-element processing")
    print("   - NumPy arrays: SIMD operations (Single Instruction, Multiple Data)")

def demonstrate_real_world_impact():
    """Show real-world impact of NumPy performance"""
    print("\nREAL-WORLD IMPACT")
    print("=" * 20)
    
    # Simulate data science operations
    print("Simulating common data science operations:")
    
    # Generate sample data
    data_size = 100_000
    python_data = [np.random.random() for _ in range(data_size)]
    numpy_data = np.random.random(data_size)
    
    # Statistical operations
    operations = [
        ("Mean calculation", lambda x: sum(x) / len(x), lambda x: x.mean()),
        ("Standard deviation", lambda x: (sum((xi - sum(x)/len(x))**2 for xi in x) / len(x))**0.5, lambda x: x.std()),
        ("Finding maximum", lambda x: max(x), lambda x: x.max()),
        ("Element-wise multiplication", lambda x: [xi * 2 for xi in x], lambda x: x * 2)
    ]
    
    print(f"\nTesting with {data_size:,} data points:")
    print(f"{'Operation':<25} {'Python (s)':<12} {'NumPy (s)':<12} {'Speedup':<10}")
    print("-" * 65)
    
    for op_name, python_func, numpy_func in operations:
        # Python timing
        start_time = time.time()
        python_result = python_func(python_data)
        python_time = time.time() - start_time
        
        # NumPy timing
        start_time = time.time()
        numpy_result = numpy_func(numpy_data)
        numpy_time = time.time() - start_time
        
        speedup = python_time / numpy_time if numpy_time > 0 else float('inf')
        
        print(f"{op_name:<25} {python_time:<12.4f} {numpy_time:<12.4f} {speedup:<10.1f}x")

def main():
    """Main function demonstrating NumPy performance"""
    print("NUMPY PERFORMANCE DEMONSTRATION")
    print("=" * 50)
    print("This demo directly demonstrates the performance comparison")
    print("from Lecture 3, showing why NumPy is essential for data science.")
    print("=" * 50)
    
    # Run performance demonstrations
    python_time = demonstrate_python_performance()
    numpy_time = demonstrate_numpy_performance()
    
    # Calculate speedup
    speedup = python_time / numpy_time if numpy_time > 0 else float('inf')
    
    print(f"\nSPEEDUP SUMMARY")
    print("=" * 20)
    print(f"Python list time: {python_time:.4f} seconds")
    print(f"NumPy array time: {numpy_time:.4f} seconds")
    print(f"NumPy is {speedup:.1f}x faster than Python!")
    
    # Additional demonstrations
    compare_performance()
    demonstrate_why_numpy_is_faster()
    demonstrate_real_world_impact()
    
    print(f"\nKEY TAKEAWAY")
    print("=" * 15)
    print("NumPy is 10-100x faster than pure Python for numerical operations.")
    print("This isn't just optimization - it's the difference between")
    print("'works for homework' and 'works for real data science.'")
    print("\nNext: See NumPy in action with real student data analysis!")

if __name__ == "__main__":
    main()