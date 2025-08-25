#!/usr/bin/env python3
"""
Lecture 1: Interactive Python Demonstrations
Python Fundamentals and Essential Command Line Integration

This script provides executable demonstrations of concepts from Lecture 1.
Students can run this script to see examples in action and experiment with
modifications to deepen their understanding.

Usage:
    python3 demo_lecture_01.py                    # Run all demonstrations
    python3 demo_lecture_01.py --section basics   # Run specific section
    python3 demo_lecture_01.py --interactive      # Interactive mode with prompts
"""

import sys
import os
import argparse
from datetime import datetime


# =============================================================================
# SECTION 1: PYTHON FUNDAMENTALS
# =============================================================================

def demonstrate_data_types():
    """
    Demonstrate Python's fundamental data types and their characteristics.
    
    This function shows how different data types work and why understanding
    them is crucial for data science applications.
    """
    print("=" * 60)
    print("SECTION 1: PYTHON DATA TYPES AND VARIABLES")
    print("=" * 60)
    
    # String data - common in datasets for categories, names, descriptions
    researcher_name = "Dr. Sarah Johnson"
    research_area = "climate science"
    
    print(f"Researcher: {researcher_name}")
    print(f"Area: {research_area}")
    print(f"Type of researcher_name: {type(researcher_name)}")
    
    # Numeric data - the backbone of quantitative analysis
    temperature_reading = 23.7
    sample_count = 1247
    is_significant = True
    
    print(f"\nTemperature Reading: {temperature_reading}°C")
    print(f"Number of samples: {sample_count}")
    print(f"Result is statistically significant: {is_significant}")
    
    # Demonstrate dynamic typing - variables can change types
    data_point = 42
    print(f"\ndata_point starts as: {data_point} (type: {type(data_point)})")
    
    data_point = "forty-two"
    print(f"data_point becomes: {data_point} (type: {type(data_point)})")
    
    data_point = 42.0
    print(f"data_point finally: {data_point} (type: {type(data_point)})")
    
    # String operations - essential for data cleaning
    messy_data = "  JOHN DOE  "
    clean_data = messy_data.strip().title()
    print(f"\nData cleaning example:")
    print(f"Original: '{messy_data}'")
    print(f"Cleaned:  '{clean_data}'")
    
    print("\nKey Insight: Different data types enable different operations.")
    print("Understanding types prevents errors in data analysis!")


def demonstrate_control_structures():
    """
    Show how control structures enable data analysis logic.
    
    Demonstrates conditional statements and loops in the context of
    processing scientific data and making decisions based on values.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: CONTROL STRUCTURES FOR DATA ANALYSIS")
    print("=" * 60)
    
    # Conditional logic - categorizing data based on values
    print("Temperature Analysis System")
    print("-" * 30)
    
    temperature_readings = [15.2, 28.7, 35.1, 8.9, 22.4, 31.8]
    
    for i, temp in enumerate(temperature_readings, 1):
        # Multiple condition checking - common in data classification
        if temp > 32:
            category = "Hot"
            action = "Activate cooling systems"
        elif temp > 25:
            category = "Warm"  
            action = "Monitor closely"
        elif temp > 15:
            category = "Moderate"
            action = "Normal operations"
        else:
            category = "Cold"
            action = "Check heating systems"
        
        print(f"Reading {i}: {temp:>5.1f}°C -> {category:>8} -> {action}")
    
    # Advanced conditional logic with multiple criteria
    print(f"\nAdvanced Analysis:")
    high_temp_readings = [temp for temp in temperature_readings if temp > 25]
    average_temp = sum(temperature_readings) / len(temperature_readings)
    
    print(f"High temperature readings: {high_temp_readings}")
    print(f"Average temperature: {average_temp:.1f}°C")
    
    if len(high_temp_readings) > len(temperature_readings) / 2:
        print("Alert: More than half of readings are above normal!")
    else:
        print("Status: Temperature readings within normal range.")
    
    # Demonstrate loop patterns common in data processing
    print(f"\nData Processing Patterns:")
    print("-" * 25)
    
    # Pattern 1: Data transformation
    fahrenheit_readings = [(temp * 9/5) + 32 for temp in temperature_readings]
    print("Celsius to Fahrenheit conversion:")
    for c, f in zip(temperature_readings, fahrenheit_readings):
        print(f"  {c:>5.1f}°C = {f:>5.1f}°F")
    
    # Pattern 2: Data filtering and counting
    extreme_count = sum(1 for temp in temperature_readings if temp > 30 or temp < 10)
    print(f"\nExtreme temperature readings: {extreme_count}")


def demonstrate_functions():
    """
    Show how functions create reusable data analysis tools.
    
    Functions are crucial in data science for creating modular,
    testable, and reusable analysis components.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: FUNCTIONS FOR REPRODUCIBLE ANALYSIS")
    print("=" * 60)
    
    # Use the global analyze_temperature_data function defined above
    # This demonstrates code reuse and modular design
    
    def format_analysis_report(analysis):
        """
        Format analysis results into a readable report.
        
        Separation of analysis and presentation is a best practice
        in data science - it allows the same analysis to be presented
        in different formats (console, web, PDF, etc.).
        """
        if "error" in analysis:
            return f"Analysis Error: {analysis['error']}"
        
        report_lines = [
            "Temperature Analysis Report",
            "=" * 28,
            f"Total Readings: {analysis['total_readings']}",
            f"Temperature Range: {analysis['min_temp']:.1f}°C to {analysis['max_temp']:.1f}°C",
            f"Average Temperature: {analysis['avg_temp']:.1f}°C",
            f"Temperature Spread: {analysis['temp_range']:.1f}°C",
            "",
            f"Readings above {analysis['threshold_used']}°C: {analysis['above_threshold']}",
            f"Percentage above threshold: {analysis['percentage_above']:.1f}%",
        ]
        
        return "\n".join(report_lines)
    
    # Demonstrate the functions with different datasets
    print("Analyzing Sample Data Sets:")
    print("-" * 30)
    
    # Dataset 1: Spring temperatures
    spring_temps = [18.2, 21.7, 19.8, 23.1, 17.9, 25.4, 22.8]
    spring_analysis = analyze_temperature_data(spring_temps, threshold=20.0)
    print("SPRING DATA:")
    print(format_analysis_report(spring_analysis))
    
    print("\n" + "-" * 50 + "\n")
    
    # Dataset 2: Summer temperatures
    summer_temps = [28.7, 32.1, 35.6, 29.8, 31.2, 33.9, 27.4, 30.1]
    summer_analysis = analyze_temperature_data(summer_temps, threshold=30.0)
    print("SUMMER DATA:")
    print(format_analysis_report(summer_analysis))
    
    # Demonstrate function reusability and modularity
    print("\n" + "=" * 40)
    print("Function Reusability Demonstration:")
    print("=" * 40)
    
    datasets = {
        "Morning": [16.2, 17.1, 15.8, 18.3, 16.9],
        "Afternoon": [24.7, 26.2, 25.1, 27.8, 25.9],
        "Evening": [21.3, 20.7, 19.8, 22.1, 20.4]
    }
    
    for period, temps in datasets.items():
        analysis = analyze_temperature_data(temps, threshold=22.0)
        print(f"\n{period} Analysis:")
        print(f"  Average: {analysis['avg_temp']:.1f}°C")
        print(f"  Above 22°C: {analysis['above_threshold']} readings ({analysis['percentage_above']:.1f}%)")


# =============================================================================
# SECTION 2: COMMAND LINE INTEGRATION
# =============================================================================

def analyze_temperature_data(readings, threshold=25.0):
    """
    Analyze temperature data and return summary statistics.
    
    This is a typical data science function that processes numerical
    data and returns structured results.
    
    Args:
        readings (list): List of temperature readings
        threshold (float): Threshold for categorizing readings
        
    Returns:
        dict: Analysis results including statistics and classifications
    """
    if not readings:
        return {"error": "No data provided"}
    
    analysis = {
        "total_readings": len(readings),
        "min_temp": min(readings),
        "max_temp": max(readings),
        "avg_temp": sum(readings) / len(readings),
        "above_threshold": len([r for r in readings if r > threshold]),
        "threshold_used": threshold
    }
    
    # Add derived insights
    analysis["percentage_above"] = (analysis["above_threshold"] / 
                                   analysis["total_readings"]) * 100
    analysis["temp_range"] = analysis["max_temp"] - analysis["min_temp"]
    
    return analysis


def demonstrate_command_line_integration():
    """
    Show how Python scripts can work with command line arguments and system operations.
    
    This demonstrates the bridge between command line skills and Python programming
    that's essential for automated data processing workflows.
    """
    print("\n" + "=" * 60)
    print("SECTION 4: COMMAND LINE AND PYTHON INTEGRATION")
    print("=" * 60)
    
    # Demonstrate command line argument processing
    print("Command Line Arguments:")
    print(f"Script name: {sys.argv[0]}")
    print(f"All arguments: {sys.argv}")
    print(f"Number of arguments: {len(sys.argv)}")
    
    if len(sys.argv) > 1:
        print(f"Additional arguments provided: {sys.argv[1:]}")
    else:
        print("No additional arguments provided")
    
    # Demonstrate file system operations
    print(f"\nFile System Information:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Create a sample data processing workflow
    print(f"\nSample Data Processing Workflow:")
    print("-" * 35)
    
    # Simulate reading data from a file (in practice, you'd use actual file I/O)
    simulated_data_files = [
        ("sensor_01.csv", [23.1, 24.7, 22.8, 25.2, 23.9]),
        ("sensor_02.csv", [21.8, 23.4, 22.1, 24.6, 23.2]),
        ("sensor_03.csv", [25.7, 27.1, 26.3, 28.2, 26.8])
    ]
    
    all_results = []
    
    for filename, data in simulated_data_files:
        print(f"\nProcessing {filename}...")
        analysis = analyze_temperature_data(data, threshold=24.0)
        
        # Add metadata
        analysis["source_file"] = filename
        analysis["processed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        all_results.append(analysis)
        
        # Show summary for this file
        print(f"  Readings: {analysis['total_readings']}")
        print(f"  Average: {analysis['avg_temp']:.1f}°C") 
        print(f"  Above threshold: {analysis['above_threshold']}")
    
    # Generate summary report
    print(f"\n" + "=" * 50)
    print("COMBINED ANALYSIS REPORT")
    print("=" * 50)
    
    total_readings = sum(r['total_readings'] for r in all_results)
    total_above_threshold = sum(r['above_threshold'] for r in all_results)
    overall_average = sum(r['avg_temp'] * r['total_readings'] for r in all_results) / total_readings
    
    print(f"Files processed: {len(all_results)}")
    print(f"Total readings: {total_readings}")
    print(f"Overall average temperature: {overall_average:.1f}°C")
    print(f"Readings above 24.0°C: {total_above_threshold} ({100*total_above_threshold/total_readings:.1f}%)")
    
    print(f"\nThis workflow demonstrates how Python scripts can:")
    print("• Process multiple data files automatically")
    print("• Generate standardized analysis reports") 
    print("• Combine results from different sources")
    print("• Run from command line with different parameters")


# =============================================================================
# SECTION 3: PROBLEM SOLVING DEMONSTRATION  
# =============================================================================

def demonstrate_problem_solving():
    """
    Walk through solving Project Euler Problem 1 step by step.
    
    This shows the problem-solving process that data scientists use:
    understanding the problem, breaking it down, implementing a solution,
    and testing/validating results.
    """
    print("\n" + "=" * 60)
    print("SECTION 5: COMPUTATIONAL PROBLEM SOLVING")
    print("=" * 60)
    
    print("Project Euler Problem 1: Multiples of 3 or 5")
    print("-" * 45)
    print("Problem: Find the sum of all multiples of 3 or 5 below 1000")
    print()
    
    # Step 1: Understand the problem with a small example
    print("Step 1: Understanding with a small example")
    print("Multiples of 3 or 5 below 10: ", end="")
    
    small_multiples = []
    for i in range(1, 10):
        if i % 3 == 0 or i % 5 == 0:
            small_multiples.append(i)
    
    print(small_multiples)
    print(f"Sum: {sum(small_multiples)}")
    
    # Step 2: Create a reusable function
    print(f"\nStep 2: Creating a general solution")
    
    def find_multiples_sum(limit, divisors=[3, 5]):
        """
        Find sum of multiples of given divisors below the limit.
        
        This function generalizes the problem so we can easily test
        different limits or different sets of divisors.
        
        Args:
            limit (int): Upper limit (exclusive)
            divisors (list): List of divisors to check for multiples
            
        Returns:
            tuple: (sum_of_multiples, list_of_multiples)
        """
        multiples = []
        
        for number in range(1, limit):
            if any(number % divisor == 0 for divisor in divisors):
                multiples.append(number)
        
        return sum(multiples), multiples
    
    # Step 3: Test with known example
    print("Testing our function with the example:")
    test_sum, test_multiples = find_multiples_sum(10)
    print(f"Multiples below 10: {test_multiples}")
    print(f"Sum: {test_sum} (should be 23)")
    
    # Step 4: Solve the actual problem
    print(f"\nStep 3: Solving for the actual problem (limit = 1000)")
    actual_sum, _ = find_multiples_sum(1000)
    print(f"Sum of multiples of 3 or 5 below 1000: {actual_sum}")
    
    # Step 5: Demonstrate extensibility
    print(f"\nStep 4: Demonstrating solution flexibility")
    print("Same approach works for different problems:")
    
    # Different limits
    for limit in [50, 100, 500]:
        result_sum, _ = find_multiples_sum(limit)
        print(f"  Sum below {limit}: {result_sum}")
    
    # Different divisors
    print(f"\nMultiples of 2 or 7 below 100:")
    alt_sum, alt_multiples = find_multiples_sum(100, [2, 7])
    print(f"  First 10 multiples: {alt_multiples[:10]}")
    print(f"  Total sum: {alt_sum}")
    
    print(f"\nKey Problem-Solving Steps Demonstrated:")
    print("1. Start with a simple, understandable example")
    print("2. Break down the logic into clear steps")
    print("3. Implement a general, reusable solution")
    print("4. Test with known results to verify correctness")
    print("5. Apply to the actual problem")
    print("6. Consider how the solution could extend to other problems")


# =============================================================================
# SECTION 4: COMMON PITFALLS AND DEBUGGING
# =============================================================================

def demonstrate_common_pitfalls():
    """
    Show common beginner mistakes and how to avoid or fix them.
    
    Understanding common errors helps students develop debugging skills
    and write more robust code from the start.
    """
    print("\n" + "=" * 60)
    print("SECTION 6: COMMON PITFALLS AND DEBUGGING")
    print("=" * 60)
    
    print("Learning from Common Mistakes:")
    print("-" * 32)
    
    # Pitfall 1: Type confusion
    print("\n1. Type Confusion:")
    temperature_str = "25.5"
    print(f"temperature_str = '{temperature_str}' (type: {type(temperature_str)})")
    
    # This would cause an error: temperature_str + 10
    print("❌ temperature_str + 10  # This would cause a TypeError")
    
    # Correct approach:
    temperature_num = float(temperature_str)
    result = temperature_num + 10
    print(f"✅ float(temperature_str) + 10 = {result}")
    
    # Pitfall 2: Off-by-one errors in ranges
    print(f"\n2. Range Boundaries:")
    print("Common confusion: range(1, 10)")
    print(f"   Produces: {list(range(1, 10))}")
    print("   Note: 10 is NOT included!")
    
    print("For inclusive ranges:")
    print(f"   range(1, 11) gives: {list(range(1, 11))}")
    
    # Pitfall 3: Indentation errors
    print(f"\n3. Indentation Consistency:")
    print("❌ Mixing tabs and spaces causes IndentationError")
    print("✅ Use consistent indentation (4 spaces recommended)")
    
    # Show what proper indentation looks like
    print("\nProper function indentation:")
    print("""
    def process_data(values):
        results = []
        for value in values:
            if value > 0:
                results.append(value * 2)
        return results
    """)
    
    # Pitfall 4: Variable scope confusion
    print("4. Variable Scope:")
    global_var = "I'm global"
    
    def scope_demo():
        local_var = "I'm local"
        print(f"  Inside function: {global_var}")  # Can access global
        print(f"  Inside function: {local_var}")   # Local variable
        return local_var
    
    result = scope_demo()
    print(f"  Outside function: {global_var}")      # Still accessible
    print(f"  Returned value: {result}")           # Returned from function
    # print(local_var)  # This would cause NameError
    
    print("\n5. Debugging Strategies:")
    print("   • Use print() statements to inspect values")
    print("   • Check data types with type()")
    print("   • Use descriptive variable names")
    print("   • Test with simple examples first")
    print("   • Read error messages carefully")


# =============================================================================
# MAIN EXECUTION AND INTERACTIVE FEATURES
# =============================================================================

def interactive_temperature_analyzer():
    """
    Interactive demonstration where users can input their own data.
    
    This provides hands-on experience with the concepts and lets
    students experiment with the code.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE SECTION: TEMPERATURE ANALYZER")
    print("=" * 60)
    
    print("Let's analyze some temperature data together!")
    print("Enter temperature readings one by one (press Enter without typing to finish):")
    
    temperatures = []
    while True:
        user_input = input(f"Temperature reading {len(temperatures) + 1}: ").strip()
        
        if not user_input:  # Empty input means we're done
            break
            
        try:
            temp = float(user_input)
            temperatures.append(temp)
            print(f"  Added {temp}°C")
        except ValueError:
            print("  Please enter a valid number")
    
    if not temperatures:
        print("No temperatures entered. Using sample data instead.")
        temperatures = [22.1, 25.6, 19.8, 27.3, 24.2]
    
    print(f"\nAnalyzing {len(temperatures)} temperature readings...")
    
    # Get user's threshold preference
    while True:
        try:
            threshold_input = input("Enter threshold temperature (default 25.0): ").strip()
            threshold = float(threshold_input) if threshold_input else 25.0
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Analyze the data using our functions
    analysis = analyze_temperature_data(temperatures, threshold)
    report = format_analysis_report(analysis)
    
    print(f"\n{report}")
    
    # Provide insights
    if analysis['percentage_above'] > 50:
        print(f"\n🔥 Over half your readings are above {threshold}°C!")
    elif analysis['percentage_above'] > 25:
        print(f"\n⚠️  About a quarter of your readings exceed {threshold}°C")
    else:
        print(f"\n✅ Most readings are below {threshold}°C threshold")


def main():
    """
    Main function that orchestrates the demonstration.
    
    This function shows how to structure a Python script with
    command line argument processing and modular execution.
    """
    parser = argparse.ArgumentParser(
        description="Interactive demonstrations for Lecture 1: Python Fundamentals and Command Line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 demo_lecture_01.py                    # Run all demonstrations
    python3 demo_lecture_01.py --section basics   # Run only basic concepts
    python3 demo_lecture_01.py --interactive      # Run with interactive examples
    python3 demo_lecture_01.py --help             # Show this help message
        """
    )
    
    parser.add_argument('--section', choices=['basics', 'control', 'functions', 'integration', 'problem', 'pitfalls'],
                        help='Run only specific section')
    parser.add_argument('--interactive', action='store_true',
                        help='Include interactive demonstrations')
    
    args = parser.parse_args()
    
    print("LECTURE 1: PYTHON FUNDAMENTALS AND COMMAND LINE INTEGRATION")
    print("Interactive Demonstrations")
    print("=" * 65)
    print(f"Demonstration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define sections
    sections = {
        'basics': demonstrate_data_types,
        'control': demonstrate_control_structures, 
        'functions': demonstrate_functions,
        'integration': demonstrate_command_line_integration,
        'problem': demonstrate_problem_solving,
        'pitfalls': demonstrate_common_pitfalls
    }
    
    # Run specific section or all sections
    if args.section:
        if args.section in sections:
            sections[args.section]()
        else:
            print(f"Unknown section: {args.section}")
            print(f"Available sections: {', '.join(sections.keys())}")
            sys.exit(1)
    else:
        # Run all demonstrations
        for section_func in sections.values():
            section_func()
    
    # Run interactive section if requested
    if args.interactive:
        interactive_temperature_analyzer()
    
    print("\n" + "=" * 65)
    print("DEMONSTRATION COMPLETED")
    print("=" * 65)
    print("\nKey Takeaways:")
    print("• Python's syntax is designed for readability and expressiveness")
    print("• Data types and control structures enable sophisticated data analysis")
    print("• Functions create modular, testable, reusable code")
    print("• Command line integration enables automated workflows")
    print("• Problem-solving involves breaking down complex tasks")
    print("• Understanding common pitfalls saves debugging time")
    
    print(f"\nNext Steps:")
    print("1. Practice these concepts with the provided exercises")
    print("2. Experiment by modifying the code in this demonstration")
    print("3. Try solving Project Euler problems for additional practice")
    print("4. Prepare for next lecture: Data Structures and Version Control")


if __name__ == "__main__":
    main()