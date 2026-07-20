"""Report the candidate Python and NumPy versions."""

import sys

import numpy as np


def main():
    """Print the candidate interpreter and direct dependency versions."""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    print(f"NumPy: {np.__version__}")


if __name__ == "__main__":
    main()
