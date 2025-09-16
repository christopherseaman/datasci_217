#!/usr/bin/env python3
"""
Fix Notion-compatible heading format across all lecture files.
Convert standard Markdown headings to Notion format (one level down).
"""

import re
import os
from pathlib import Path

def fix_notion_headers(file_path):
    """Convert markdown headers to Notion-compatible format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Convert headers (reduce by one level)
    # #### becomes ###
    content = re.sub(r'^#### (.+)$', r'### \1', content, flags=re.MULTILINE)
    # ### becomes ##
    content = re.sub(r'^### (.+)$', r'## \1', content, flags=re.MULTILINE)
    # ## becomes #
    content = re.sub(r'^## (.+)$', r'# \1', content, flags=re.MULTILINE)
    # # becomes plain text (remove #)
    content = re.sub(r'^# (.+)$', r'\1', content, flags=re.MULTILINE)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Fixed headers in {file_path}")

def main():
    # Fix all index.md files in lecture directories
    base_path = Path('/home/christopher/projects/datasci_217')

    for lecture_num in range(1, 12):  # 01 through 11
        lecture_dir = base_path / f"{lecture_num:02d}"
        index_file = lecture_dir / "index.md"

        if index_file.exists():
            fix_notion_headers(index_file)
        else:
            print(f"Warning: {index_file} not found")

if __name__ == "__main__":
    main()