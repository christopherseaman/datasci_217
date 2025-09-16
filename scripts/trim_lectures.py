#!/usr/bin/env python3
"""
Trim lectures to target word counts
- Lecture 09: 3311 -> 3000 (trim ~311 words)
- Lecture 10: 3510 -> 3000 (trim ~510 words)
"""

import re
from pathlib import Path

def trim_lecture_09():
    """Trim Lecture 09 by removing detailed logging and error handling"""
    file_path = Path("09/index.md")

    with open(file_path, 'r') as f:
        content = f.read()

    # Remove excessive logging setup section
    content = re.sub(
        r'Setup logging for script execution\nlogging\.basicConfig\(\n    level=logging\.INFO,\n    format=\'%\(asctime\)s - %\(levelname\)s - %\(message\)s\',\n    handlers=\[\n        logging\.FileHandler\(\'analysis\.log\'\),\n        logging\.StreamHandler\(sys\.stdout\)\n    \]\n\)\nlogger = logging\.getLogger\(__name__\)\n\n',
        '',
        content
    )

    # Simplify function documentation
    content = re.sub(
        r'    """\n    Load data file and perform basic validation\n    \n    Args:\n        file_path \(str\): Path to data file\n        \n    Returns:\n        pandas\.DataFrame: Validated data\n        \n    Raises:\n        FileNotFoundError: If file doesn\'t exist\n        ValueError: If data validation fails\n    """',
        '    """Load and validate sales data"""',
        content
    )

    # Remove detailed logging statements
    content = re.sub(r'    logger\.info\(f?"[^"]*"\)', '', content)
    content = re.sub(r'    logger\.error\(f?"[^"]*"\)', '', content)

    # Remove verbose error handling
    content = re.sub(
        r'    # Load data\n    try:\n        df = pd\.read_csv\(file_path\)\n        logger\.info\(f"Data loaded: \{df\.shape\[0\]\} rows, \{df\.shape\[1\]\} columns"\)\n    except Exception as e:\n        logger\.error\(f"Failed to load data: \{str\(e\)\}"\)\n        raise',
        '    df = pd.read_csv(file_path)',
        content
    )

    with open(file_path, 'w') as f:
        f.write(content)

def trim_lecture_10():
    """Trim Lecture 10 by removing detailed examples and verbose explanations"""
    file_path = Path("10/index.md")

    with open(file_path, 'r') as f:
        content = f.read()

    # Find and remove one of the longer code examples or verbose explanations
    # This is a targeted trim to remove about 510 words
    # Remove any overly detailed explanations in sections

    # Remove redundant examples in complex operations
    lines = content.split('\n')
    filtered_lines = []
    skip_next_example = False
    in_code_block = False
    code_block_count = 0

    for line in lines:
        if line.strip() == '```python' or line.strip() == '```':
            if line.strip() == '```python':
                in_code_block = True
                code_block_count += 1
                # Skip every 3rd code block to reduce content
                if code_block_count % 3 == 0:
                    skip_next_example = True
                    continue
            else:
                in_code_block = False
                if skip_next_example:
                    skip_next_example = False
                    continue

        if skip_next_example and in_code_block:
            continue

        # Remove overly verbose explanations
        if ('detailed breakdown' in line.lower() or
            'step-by-step walkthrough' in line.lower() or
            'comprehensive example' in line.lower()):
            continue

        filtered_lines.append(line)

    content = '\n'.join(filtered_lines)

    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Main trimming function"""
    print("Trimming Lecture 09...")
    trim_lecture_09()

    print("Trimming Lecture 10...")
    trim_lecture_10()

    print("Trimming complete. Checking word counts...")

    # Check new word counts
    import subprocess
    for lecture in ['09', '10']:
        result = subprocess.run(['wc', '-w', f'{lecture}/index.md'],
                               capture_output=True, text=True)
        print(f"Lecture {lecture}: {result.stdout.strip()}")

if __name__ == "__main__":
    main()