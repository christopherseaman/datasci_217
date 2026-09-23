#!/usr/bin/env python3
"""Demo 2: inspect values and combine ordinary Python sequences."""


def main():
    """Run the short Python refresher before the NumPy examples."""
    print("Python Refresher: Introspection and Sequences")
    print("=" * 50)

    value = "42"
    print("Original value:", value, "Type:", type(value))
    if isinstance(value, str):
        value = int(value)
    print("Converted value:", value, "Type:", type(value))
    print("Strings have split:", "split" in dir("42"))

    names = ["Alice", "Bob", "Charlie"]
    grades = [85, 92, 78]
    print("\nNumbered names:")
    for number, name in enumerate(names, start=1):
        print(f"  Student {number}: {name}")
    print("Paired records:")
    for name, grade in zip(names, grades):
        print(f"  {name}: {grade}")
    print(f"Reverse order: {list(reversed(names))}")
    print(f"Sorted grades: {sorted(grades)}")


if __name__ == "__main__":
    main()
