#!/usr/bin/env python3
"""
Lecture 05 Live Demo: NumPy Fundamentals and Environment Management
DataSci 217 - Hands-on demonstration of NumPy array operations and virtual environments

This demo shows practical NumPy usage for data analysis and proper environment setup.
"""

import numpy as np
import sys
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path

def demo_numpy_basics():
    """
    Demonstrate fundamental NumPy operations for data analysis
    """
    print("=== NUMPY FUNDAMENTALS DEMO ===")
    print()

    print("1. Array Creation and Basic Properties")
    print("=" * 40)

    # Creating arrays from different sources
    data_list = [1, 2, 3, 4, 5]
    arr_from_list = np.array(data_list)
    print(f"Array from list: {arr_from_list}")
    print(f"Shape: {arr_from_list.shape}, dtype: {arr_from_list.dtype}")
    print()

    # Different array creation methods
    zeros_arr = np.zeros(5)
    ones_arr = np.ones((3, 3))
    range_arr = np.arange(0, 10, 2)
    random_arr = np.random.random(5)

    print(f"Zeros array: {zeros_arr}")
    print(f"Ones matrix:\n{ones_arr}")
    print(f"Range array: {range_arr}")
    print(f"Random array: {random_arr}")
    print()

    print("2. Multi-dimensional Arrays (Matrices)")
    print("=" * 40)

    # Create a 2D array (matrix) representing student grades
    grades = np.array([
        [85, 92, 78, 90],  # Student 1: Math, Science, English, History
        [92, 88, 95, 87],  # Student 2
        [78, 85, 82, 89],  # Student 3
        [90, 94, 88, 92]   # Student 4
    ])

    print("Student Grades Matrix:")
    print(grades)
    print(f"Shape: {grades.shape} (4 students × 4 subjects)")
    print()

    # Array indexing and slicing
    print("Array Indexing Examples:")
    print(f"Student 1's grades: {grades[0]}")
    print(f"All Math grades (column 0): {grades[:, 0]}")
    print(f"First 2 students, first 2 subjects:\n{grades[:2, :2]}")
    print()

    print("3. Mathematical Operations")
    print("=" * 40)

    # Calculate statistics
    print("Grade Statistics:")
    print(f"Overall average: {grades.mean():.2f}")
    print(f"Student averages: {grades.mean(axis=1)}")
    print(f"Subject averages: {grades.mean(axis=0)}")
    print(f"Highest grade: {grades.max()}")
    print(f"Lowest grade: {grades.min()}")
    print()

    # Element-wise operations
    scaled_grades = grades * 1.05  # 5% bonus
    print("Grades with 5% bonus:")
    print(scaled_grades.round(1))
    print()

def demo_advanced_numpy():
    """
    Demonstrate advanced NumPy operations for data science
    """
    print("=== ADVANCED NUMPY OPERATIONS ===")
    print()

    print("1. Boolean Indexing and Filtering")
    print("=" * 40)

    # Create sample data: daily temperatures
    np.random.seed(42)  # For reproducible results
    temperatures = np.random.normal(20, 5, 30)  # 30 days, avg 20°C, std 5°C

    print(f"Temperature data (30 days): {temperatures[:10].round(1)}...")
    print()

    # Boolean filtering
    hot_days = temperatures > 25
    cold_days = temperatures < 15

    print(f"Hot days (>25°C): {np.sum(hot_days)} days")
    print(f"Cold days (<15°C): {np.sum(cold_days)} days")
    print(f"Hot day temperatures: {temperatures[hot_days].round(1)}")
    print()

    print("2. Array Reshaping and Broadcasting")
    print("=" * 40)

    # Sales data: products × months
    sales_data = np.array([
        [100, 150, 200, 180],  # Product A
        [80, 120, 160, 140],   # Product B
        [60, 90, 120, 110]     # Product C
    ])

    print("Monthly Sales Data (Products × Months):")
    print(sales_data)
    print()

    # Broadcasting: calculate percentage of total sales
    monthly_totals = sales_data.sum(axis=0)
    percentage_share = (sales_data / monthly_totals) * 100

    print("Monthly totals:", monthly_totals)
    print("Percentage share by product:")
    print(percentage_share.round(1))
    print()

    print("3. Linear Algebra Operations")
    print("=" * 40)

    # Simple linear regression using NumPy
    # Generate sample data: house size vs price
    np.random.seed(42)
    house_sizes = np.random.normal(1500, 300, 50)  # Square feet
    # Price = 100 * size + noise
    house_prices = 100 * house_sizes + np.random.normal(0, 10000, 50)

    # Calculate correlation
    correlation = np.corrcoef(house_sizes, house_prices)[0, 1]
    print(f"House size vs price correlation: {correlation:.3f}")

    # Simple linear fit (slope and intercept)
    # Using normal equations: slope = (X^T * X)^-1 * X^T * y
    X = np.column_stack([np.ones(len(house_sizes)), house_sizes])
    coefficients = np.linalg.lstsq(X, house_prices, rcond=None)[0]
    intercept, slope = coefficients

    print(f"Linear regression: Price = {slope:.2f} * Size + {intercept:.2f}")
    print()

def demo_numpy_data_analysis():
    """
    Demonstrate real-world data analysis with NumPy
    """
    print("=== NUMPY DATA ANALYSIS CASE STUDY ===")
    print()

    print("Scenario: Analyzing website traffic data")
    print("=" * 40)

    # Simulate website traffic data
    np.random.seed(42)
    days = 30
    hours_per_day = 24

    # Create traffic data: higher during business hours
    traffic_data = np.zeros((days, hours_per_day))

    for day in range(days):
        for hour in range(hours_per_day):
            # Base traffic
            base_traffic = 100

            # Higher during business hours (9-17)
            if 9 <= hour <= 17:
                base_traffic *= 2

            # Add some randomness
            traffic_data[day, hour] = np.random.poisson(base_traffic)

    print(f"Traffic data shape: {traffic_data.shape}")
    print(f"Total visitors in 30 days: {traffic_data.sum():,.0f}")
    print()

    # Analysis 1: Daily patterns
    hourly_averages = traffic_data.mean(axis=0)
    peak_hour = np.argmax(hourly_averages)
    lowest_hour = np.argmin(hourly_averages)

    print("Hourly Traffic Analysis:")
    print(f"Peak hour: {peak_hour}:00 (avg {hourly_averages[peak_hour]:.0f} visitors)")
    print(f"Lowest hour: {lowest_hour}:00 (avg {hourly_averages[lowest_hour]:.0f} visitors)")
    print()

    # Analysis 2: Daily totals and trends
    daily_totals = traffic_data.sum(axis=1)
    print("Daily Traffic Trends:")
    print(f"Average daily visitors: {daily_totals.mean():.0f}")
    print(f"Busiest day: Day {np.argmax(daily_totals) + 1} ({daily_totals.max():.0f} visitors)")
    print(f"Quietest day: Day {np.argmin(daily_totals) + 1} ({daily_totals.min():.0f} visitors)")
    print()

    # Analysis 3: Business vs non-business hours
    business_hours_mask = np.zeros(24, dtype=bool)
    business_hours_mask[9:18] = True  # 9 AM to 5 PM

    business_traffic = traffic_data[:, business_hours_mask].sum()
    total_traffic = traffic_data.sum()
    business_percentage = (business_traffic / total_traffic) * 100

    print(f"Business hours traffic: {business_percentage:.1f}% of total")
    print()

    # Create simple visualization data
    print("Hourly averages for visualization:")
    for i in range(0, 24, 4):
        hour_range = f"{i:02d}:00-{(i+3):02d}:00"
        avg_traffic = hourly_averages[i:i+4].mean()
        print(f"  {hour_range}: {avg_traffic:.0f} visitors")

def demo_environment_management():
    """
    Demonstrate Python environment management concepts
    """
    print("=== ENVIRONMENT MANAGEMENT DEMO ===")
    print()

    print("1. Current Environment Information")
    print("=" * 40)

    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"NumPy version: {np.__version__}")
    print()

    # Show current working directory and path
    print(f"Current working directory: {Path.cwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    print()

    print("2. Package Management Concepts")
    print("=" * 40)

    print("Virtual Environment Best Practices:")
    print("1. Create isolated environments for each project")
    print("2. Use requirements.txt to track dependencies")
    print("3. Activate environment before working")
    print("4. Keep environments lightweight and focused")
    print()

    print("Common Commands for Environment Management:")
    print("# Create virtual environment")
    print("python -m venv myproject_env")
    print()
    print("# Activate environment")
    print("# On Windows: myproject_env\\Scripts\\activate")
    print("# On Mac/Linux: source myproject_env/bin/activate")
    print()
    print("# Install packages")
    print("pip install numpy pandas matplotlib")
    print()
    print("# Save requirements")
    print("pip freeze > requirements.txt")
    print()
    print("# Install from requirements")
    print("pip install -r requirements.txt")
    print()

    print("3. Creating a Sample Requirements File")
    print("=" * 40)

    # Create a sample requirements.txt
    requirements_content = """# DataSci 217 - Lecture 05 Requirements
# Core data analysis packages
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0

# Additional useful packages
scipy>=1.7.0
scikit-learn>=1.0.0
jupyter>=1.0.0

# Development tools
pytest>=6.0.0
flake8>=3.9.0
"""

    with open("requirements_demo.txt", "w") as f:
        f.write(requirements_content)

    print("Sample requirements.txt created:")
    print(requirements_content)

def create_visualization_demo():
    """
    Create a simple visualization using NumPy data
    """
    print("=== BONUS: SIMPLE VISUALIZATION ===")
    print()

    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)

    # Create visualization data (for demonstration)
    print("Creating sample visualization data:")
    print(f"X values: {x[:5]}...{x[-5:]}")
    print(f"Y values: {y[:5].round(3)}...{y[-5:].round(3)}")
    print()

    print("This data could be plotted as:")
    print("plt.figure(figsize=(10, 6))")
    print("plt.plot(x, y, 'b-', alpha=0.7, label='Data')")
    print("plt.title('NumPy Generated Data')")
    print("plt.xlabel('X values')")
    print("plt.ylabel('Y values')")
    print("plt.legend()")
    print("plt.grid(True, alpha=0.3)")
    print("plt.show()")
    print()

    # Save data for later use
    np.savez("demo_data.npz", x=x, y=y)
    print("Data saved to demo_data.npz for later use")

def main():
    """
    Main demo execution function
    """
    print("Welcome to DataSci 217 - Lecture 05 Live Demo!")
    print("NumPy Fundamentals and Environment Management")
    print("=" * 60)
    print()

    # Run all demos
    demo_numpy_basics()
    print("\n" + "="*60 + "\n")

    demo_advanced_numpy()
    print("\n" + "="*60 + "\n")

    demo_numpy_data_analysis()
    print("\n" + "="*60 + "\n")

    demo_environment_management()
    print("\n" + "="*60 + "\n")

    create_visualization_demo()
    print("\n" + "="*60 + "\n")

    print("Demo complete!")
    print("\nKey takeaways:")
    print("1. NumPy arrays are the foundation of data science in Python")
    print("2. Vectorized operations are faster than loops")
    print("3. Boolean indexing enables powerful data filtering")
    print("4. Proper environment management prevents dependency conflicts")
    print("5. Always understand your data's shape and structure")

if __name__ == "__main__":
    main()