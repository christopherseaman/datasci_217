# DataSci-217
Introduction to Python & Data Science Tools ([canonical url](https://ds217.badmath.org), [github repo](https://github.com/christopherseaman/datasci_217))

## Lecture 1: Introduction to Python and Command Line Basics
([slides](01), [pdf](01/index.pdf), [assignment](?f=04/assignment))
- Python basics: syntax, data types, and control structures
- Running Python scripts from the command line
- Navigating the file system with `*sh`
- Basic file operations (creating, moving, copying, deleting)
- Pipes and command chaining

## Lecture 2: Version Control with Git and More Python
([slides](02), [pdf](02/index.pdf), [assignment](https://classroom.github.com/a/Z2sWwnXF))
- Git basics (init, clone, add, commit, push, pull)
- Branching and merging
- Collaboration workflows on GitHub
- Conflict resolution 
- Writing documents with Markdown and publishing to the web
- Virtual environments and dependencies with pip

## Lecture 3: Python Data Structures and Documentation
([slides](03), [pdf](03/index.pdf), [assignment](https://classroom.github.com/a/bTwHLV-s))
- Introduction to shell scripting, variables, and `cron`
- Python data structures (lists, tuples, dictionaries, sets)
- List comprehensions and generator expressions

## Lecture 4: Functions, Methods, and Remote Execution
([slides](04), [pdf](04/index.pdf), [assignment](https://classroom.github.com/a/m_U53ad8))
- Python functions, modules, and packages
- Interacting with files
- Using SSH for remote access
- Introduction to Jupyter Notebooks on remote servers
- CUDA and GPUs with Python (brief introduction)
- Submitting Python jobs to the university HPC cluster

## Lecture 5: Python for Data Management I
([slides](05), [pdf](05/index.pdf), [demo](?f=05/demo), [exam 1](https://classroom.github.com/a/S2smrp6e) due Oct 23rd)
- Remote Jupyter Notebooks
- Saving your place in ssh: `screen`, `tmux`, and `mosh`
- Introduction to NumPy & Pandas
- Working with data files
- DataFrame and Series objects
- Numpy arrays and operations
- Preview of data manipulation with Pandas

## Lecture 6: Python for Data Management II
([slides](06), [pdf](06/index.pdf), [demo](06/demo.ipynb), [assignment](https://classroom.github.com/a/u8FyG16T))
- Wrangling, cleaning, filtering, and transforming data
- Exploratory data analysis with Pandas
- Handling missing data
- Merging, joining, and reshaping data
- Scaling to larger datasets
- Example scripts: [dirty-data.py](06/pandas-dirty-data.py),  [data-cleaning.py](06/pandas-data-cleaning.py)

## Lecture 7: Python for Data Visualization
([slides](07), [pdf](07/index.pdf), [demo](07/demo.ipynb), [assignment](https://classroom.github.com/a/aqAaGXP3))
- Introduction to Matplotlib
- Creating basic plots (line, scatter, bar, histogram)
- Plotting with Seaborn for statistical visualizations
- Grammar of graphics with Plotnine (ggplot2-compatible)
- Interactive visualizations with Plotly

## Lecture 8: Machine Learning with Python
([slides](08), [pdf](08/index.pdf), [demo](08/demo.ipynb), [assignment](https://classroom.github.com/a/AOMngUYk))
- Containerization with Docker & Kubernetes for Python (brief introduction)
- Parallel processing in Python (multiprocessing, concurrent.futures)
- Introduction to scikit-learn for machine learning
- Data preprocessing and feature engineering
- Basic examples of regression and classification
- Model evaluation and validation
- Brief overview of ML libraries (PyTorch, TensorFlow, JAX, SciPy, statsmodels)

## Lecture 9: Algorithms, Data Structures, and Their Implementation in Python
- List comprehensions, generators, and `yield`
- Implementing common algorithms in Python (sorting, searching)
- Python data structures (lists, dictionaries, custom classes)
- Examples and practice problems
- Introduction to more advanced structures (trees, graphs)
- Big O notation and algorithm efficiency
- Dynamic programming and recursion
- Solving coding interview-style problems in Python

## Lecture 10: API Usage and Web Scraping with Python
- Understanding RESTful APIs
- Making HTTP requests with the `requests` library
- Parsing JSON responses
- Authentication and API keys
- Rate limiting and error handling
- Introduction to web scraping with BeautifulSoup
- Ethical considerations in web scraping and API usage
- Building a simple data pipeline using an API

# Resources

- [command line exercises](?f=shell_workout)

## Python

### Reference
- _Whirlwind Tour of Python_, VanderPlas - author’s [website](https://jakevdp.github.io/WhirlwindTourOfPython/)
- _Think Python_, Downey - purchase or read at [Green Tea Press](https://greenteapress.com/wp/think-python/)
- _Hitchhiker’s Guide to Python!_ - official [documentation](https://docs.python-guide.org/)
- [Introduction to Python](http://introtopython.org/)
- [_A Byte of Python_](https://python.swaroopch.com/)
- [Official Python documentation](https://docs.python.org/3/)
- [O’Reilly catalog via UCSF library](https://www.oreilly.com/library-access/)
	- _Introducing Python_
	- _Python Crash Course_

### Courses
- [Exercism Python track](https://exercism.io/tracks/python)
- [Codecademy Python course](https://www.codecademy.com/learn/learn-python-3)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
- [Real Python](https://realpython.com/)
- [DataCamp (Python + Data Science)](https://www.datacamp.com/)
- [Learn Python 3 @ codeacademy.com](https://www.codecademy.com/learn/learn-python-3)
- [Jetbrains' Learn PyCharm](https://www.jetbrains.com/pycharm/learn/)
- [Effective PyCharm Course](https://training.talkpython.fm/courses/explore_pycharm/mastering-pycharm-ide) (\$\$\$)

## Command Line
- [LinuxCommand.org](http://linuxcommand.org/lc3_learning_the_shell.php)
- [The Linux Command Line book](http://linuxcommand.org/tlcl.php)
- [The Missing Semester](https://missing.csail.mit.edu/)
- [Bash manual](https://www.gnu.org/software/bash/manual/)

## Git
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)
- [Pro Git book](https://git-scm.com/book/en/v2)
- [Learn Git Branching](https://learngitbranching.js.org/)
- [Official Git documentation](https://git-scm.com/doc)

## Markdown
- [Markdown Guide](https://www.markdownguide.org/)
- [Markdown Tutorial](https://www.markdowntutorial.com/)
- [CommonMark](https://commonmark.org/)

## Free SSH Options
- [Google Cloud Shell](https://cloud.google.com/free/docs/compute-getting-started)
- [GitHub Codespaces](https://cli.github.com/manual/gh_codespace_ssh)
