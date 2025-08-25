#!/usr/bin/env python3
"""
Demo Script Template for Lecture {LECTURE_NUMBER}: {LECTURE_TITLE}

This script demonstrates the key concepts from the lecture through
executable code examples. Each section corresponds to a major
topic covered in the lecture narrative.

Usage:
    python demo_lecture_{LECTURE_NUMBER}.py

Author: Data Science 217 Course Materials
Date: {DATE}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# ============================================================================
# SECTION 1: {MAJOR_TOPIC_1}
# ============================================================================

def demonstrate_concept_1():
    """
    Demonstrates {specific concept} from the lecture.
    
    This function shows the practical implementation of {concept},
    which was explained in the lecture as {brief explanation}.
    
    Returns:
        dict: Results of the demonstration with explanatory keys
    """
    print("=" * 50)
    print("DEMONSTRATION 1: {MAJOR_TOPIC_1}")
    print("=" * 50)
    
    # Step 1: Set up the example
    print("\nStep 1: Setting up the example...")
    # {Implementation code}
    
    # Step 2: Apply the concept
    print("\nStep 2: Applying the concept...")
    # {Core demonstration code}
    
    # Step 3: Interpret results
    print("\nStep 3: Understanding the results...")
    # {Result interpretation code}
    
    print("\nKey Takeaway: {What students should learn from this demo}")
    return {}

# ============================================================================
# SECTION 2: {MAJOR_TOPIC_2}
# ============================================================================

def demonstrate_concept_2():
    """
    Demonstrates {specific concept} from the lecture.
    
    This builds on the previous demonstration by showing how
    {connection to previous concept}.
    
    Returns:
        dict: Results of the demonstration
    """
    print("\n" + "=" * 50)
    print("DEMONSTRATION 2: {MAJOR_TOPIC_2}")
    print("=" * 50)
    
    # Implementation following same pattern as above
    pass

# ============================================================================
# INTERACTIVE EXPLORATION
# ============================================================================

def interactive_exploration():
    """
    Provides an interactive way for students to explore the concepts.
    
    This function sets up scenarios that students can modify and
    re-run to see how different parameters affect the outcomes.
    """
    print("\n" + "=" * 50)
    print("INTERACTIVE EXPLORATION")
    print("=" * 50)
    
    print("\nTry modifying the parameters below and re-running to see different results:")
    
    # Parameters students can easily modify
    parameter_1 = 10  # {Description of what this affects}
    parameter_2 = 0.5  # {Description of what this affects}
    
    # Code that uses these parameters
    print(f"With parameter_1={parameter_1} and parameter_2={parameter_2}:")
    # {Interactive demonstration code}

# ============================================================================
# COMMON PITFALLS DEMONSTRATION
# ============================================================================

def demonstrate_pitfalls():
    """
    Shows common mistakes students make and how to avoid them.
    
    By seeing these errors in action, students learn to recognize
    and prevent them in their own work.
    """
    print("\n" + "=" * 50)
    print("COMMON PITFALLS AND SOLUTIONS")
    print("=" * 50)
    
    print("\nPitfall 1: {Common mistake}")
    try:
        # Code that demonstrates the mistake
        pass
    except Exception as e:
        print(f"Error: {e}")
        print("Solution: {How to fix it}")
    
    print("\nPitfall 2: {Another common mistake}")
    # {Another demonstration}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Data Science 217 - Lecture {LECTURE_NUMBER} Demo")
    print("Topic: {LECTURE_TITLE}")
    print("\nThis demo script walks through the key concepts from today's lecture.")
    print("Follow along and feel free to modify the code to explore further!\n")
    
    # Run all demonstrations in sequence
    demonstrate_concept_1()
    demonstrate_concept_2()
    interactive_exploration()
    demonstrate_pitfalls()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review the lecture notes to reinforce these concepts")
    print("2. Try the practice exercises with different parameters")
    print("3. Apply these techniques to the upcoming assignment")
    print("\nQuestions? Bring them to office hours or post on the discussion board!")

# ============================================================================
# EXERCISE TEMPLATES
# ============================================================================

def exercise_template():
    """
    Template for structured exercises that students can complete.
    
    Each exercise should:
    1. Have a clear objective
    2. Provide starter code
    3. Include hints for common challenges
    4. Show expected output format
    """
    print("\n" + "=" * 50)
    print("EXERCISE: {EXERCISE_NAME}")
    print("=" * 50)
    
    print("Objective: {Clear description of what students should accomplish}")
    print("Skills practiced: {Specific technical skills}")
    
    # TODO: Student code goes here
    print("\n# Your code here:")
    print("# Hint: Start by {helpful starting suggestion}")
    
    print("\nExpected output format:")
    print("# {Description of what the output should look like}")