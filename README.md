# Introduction to Python & Data Science Tools

## Lecture authoring

Notion is the primary publishing surface. Write lectures and bonus pages with `#` for major sections, `##` for subsections, and `###` for deeper topics; multiple H1 sections are intentional. Keep each prose paragraph on one physical line, with no line-length limit. Preserve code-block formatting and list nesting. Do not add horizontal rules or prose that merely repeats a heading. Preserve humor.

When syncing, preserve heading levels exactly. Resolve local links and media URLs for Notion, preserve native child pages, and omit navigation links already represented by those child pages. YAML mapping metadata stays local. Notebook demos remain Markdown sources generated into `.ipynb` files, with Colab links targeting the notebooks.

From the repository root, run `python3 scripts/notion_publish.py SOURCE.md CURRENT_NOTION_CONTENT.md` to prepare a page from a fresh Notion content snapshot. It preserves headings and paragraphs; only links, native tables, child-page navigation, and metadata need publishing adaptation. The optional `notion.title_line` identifies the exact local title to omit because Notion already displays its page title. Review the output before publishing, then fetch the page again to verify its structure.

## Resources

- Canonical URL - https://not.badmath.org/ds217
- GitHub repo - https://github.com/christopherseaman/datasci_217

#### References

- [Python for Data Analysis](https://wesmckinney.com/book/) (rough basis for Python content)
- [The Missing Semester](https://missing.csail.mit.edu/) (command line, git, data wrangling)
- [The Linux Command Line book](http://linuxcommand.org/tlcl.php) (command line in-depth)
- [Markdown Guide](https://www.markdownguide.org/) (you can probably figure this one out)

#### Development Tools (free!)

- [VS Code](https://code.visualstudio.com/) (it’s pretty good)
- [Python](https://www.python.org/) (great docs & tutorials, too!)
- [GitHub Codespaces](https://cli.github.com/manual/gh_codespace_ssh) (free IDE in a browser)
- [Google Cloud Shell](https://cloud.google.com/free/docs/compute-getting-started) (practice command line anywhere)

## Assignments

Each lecture directory contains the source for its assignment. The instructor
supplies the term-specific assignment repository URLs separately.

Term-specific repository URLs: `#FIXME:ASSIGNMENT_URLS`

## Lectures

1. [1. Command Line + Python](01/README.md)
    - Command line navigation and file operations
    - Python installation and basic setup
    - VS Code setup and basic workflows
    - Introduction to development tools
2. [2. Python + Git](02/README.md)
    - Python syntax, variables, and data types
    - Control structures and functions basics
    - Git through VS Code/GitHub (GUI focus)
    - Project organization and collaboration
3. [3. NumPy + Virtual Environments](03/README.md)
    - N-dimensional arrays and array creation
    - Array operations and universal functions
    - Boolean indexing and fancy indexing
    - Basic mathematical operations
4. [4. Jupyter + Pandas](04/README.md)
    - Jupyter notebooks
    - Series and DataFrame creation
    - Data selection and filtering
    - Reading CSV, Excel, and JSON files; inspecting data types, missingness, and duplicate rows
5. [5. Data Cleaning](05/README.md)
    - Data transformation techniques
    - String operations for data cleaning
    - Handling missing data strategies
    - Data validation and quality assessment
6. [6. Joins, Combining + Reshaping](06/README.md)
    - Merge, join, and concatenate operations
    - Reshaping data (pivot, melt, stack/unstack)
    - DataFrame indexes and basic MultiIndex structures
    - Combining and reshaping data with explicit shape and key contracts
7. [7. Data Visualization](07/README.md)
    - Matplotlib fundamentals
    - Pandas plotting interface
    - Seaborn for statistical visualization
    - Creating effective data visualizations
8. [8. Aggregation + Group Operations](08/README.md)
    - GroupBy mechanics and advanced techniques
    - Aggregation functions and transformations
    - Pivot tables and cross-tabulation
    - Result-shape choices: aggregation, transform, filter, and apply
9. [9. Time Series Analysis](09/README.md)
    - Time series data handling and analysis
    - Date/time operations and resampling
10. [10. Statistics, Machine Learning + Deep Learning Models](10/README.md)
    - Statistical modeling, scikit-learn, gradient boosting, and neural-network foundations
    - Leakage-safe splitting, preprocessing, comparison, and evaluation
11. [11. From Question to Defensible Result](11/README.md)
    - A question-led, iterative capstone workflow
    - Data contracts, provenance, and defensible conclusions
