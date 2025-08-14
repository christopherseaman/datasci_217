# DataSci-217

Introduction to Python & Data Science Tools ([canonical url](https://ds217.badmath.org), [github repo](https://github.com/christopherseaman/datasci_217))

Remember to check the [list of resources](#resources) below!

## Lectures

### Lecture 1: Introduction to Python and Command Line Basics

([slides](01), [pdf](01/index.pdf), [assignment](?f=04/assignment))

- Python basics: syntax, data types, and control structures
- Running Python scripts from the command line
- Navigating the file system with `*sh`
- Basic file operations (creating, moving, copying, deleting)
- Pipes and command chaining

### Lecture 2: Version Control with Git and More Python

([slides](02), [pdf](02/index.pdf), [assignment](https://classroom.github.com/a/Z2sWwnXF))

- Git basics (init, clone, add, commit, push, pull)
- Branching and merging
- Collaboration workflows on GitHub
- Conflict resolution
- Writing documents with Markdown and publishing to the web
- Virtual environments and dependencies with pip

### Lecture 3: Python Data Structures and Documentation

([slides](03), [pdf](03/index.pdf), [assignment](https://classroom.github.com/a/bTwHLV-s))

- Introduction to shell scripting, variables, and `cron`
- Python data structures (lists, tuples, dictionaries, sets)
- List comprehensions and generator expressions

### Lecture 4: Functions, Methods, and Remote Execution

([slides](04), [pdf](04/index.pdf), [assignment](https://classroom.github.com/a/m_U53ad8))

- Python functions, modules, and packages
- Interacting with files
- Using SSH for remote access
- Introduction to Jupyter Notebooks on remote servers
- CUDA and GPUs with Python (brief introduction)
- Submitting Python jobs to the university HPC cluster

### Lecture 5: Python for Data Management I

([slides](05), [pdf](05/index.pdf), [demo](?f=05/demo), [exam 1](https://classroom.github.com/a/S2smrp6e) due Oct 23rd)

- Remote Jupyter Notebooks
- Saving your place in ssh: `screen`, `tmux`, and `mosh`
- Introduction to NumPy & Pandas
- Working with data files
- DataFrame and Series objects
- Numpy arrays and operations
- Preview of data manipulation with Pandas

### Lecture 6: Python for Data Management II

([slides](06), [pdf](06/index.pdf), [demo](06/demo.ipynb), [assignment](https://classroom.github.com/a/u8FyG16T))

- Wrangling, cleaning, filtering, and transforming data
- Exploratory data analysis with Pandas
- Handling missing data
- Merging, joining, and reshaping data
- Scaling to larger datasets
- Example scripts: [dirty-data.py](06/pandas-dirty-data.py),  [data-cleaning.py](06/pandas-data-cleaning.py)

### Lecture 7: Python for Data Visualization

([slides](07), [pdf](07/index.pdf), [demo](07/demo.ipynb), [assignment](https://classroom.github.com/a/aqAaGXP3))

- Introduction to Matplotlib
- Creating basic plots (line, scatter, bar, histogram)
- Plotting with Seaborn for statistical visualizations
- Grammar of graphics with Plotnine (ggplot2-compatible)
- Interactive visualizations with Plotly

### Lecture 8: Machine Learning with Python

([slides](08), [pdf](08/index.pdf), [demo](08/demo.ipynb), [time-series datasets](08/time_series_datasets.py), [assignment](https://classroom.github.com/a/AOMngUYk))

- Introduction to scikit-learn for machine learning
- Data preprocessing and feature engineering
- Basic examples of regression and classification
- Model evaluation and validation

### Lecture 9: Practical Python & Command Line Automation

([slides](09), [pdf](09/index.pdf), [python demo](09/demo.ipynb) & [shell demo](09/shell_demo.md), [exam 2](https://classroom.github.com/a/_RlVALv1))

- Revisiting powerful CLI commands and Python integration
- Everyday automation with Python (file organization, batch operations)
- Building practical data processing pipelines
- Automated report generation and system monitoring
- Real-world examples combining concepts from previous lectures

### Lecture 10: Guest Lecturer, Albert Lee
([resources](https://tiny.ucsf.edu/9SlN8f))

### Lecture 11: Guest Lecturer, Rian Bogley

([slides](11), [pdf](11/index.pdf), [demo](11/demo.ipynb))

### Lecture 12: Guest Lecturer, Sam Chen

## Resources

- [command line exercises](?f=shell_workout)
- [The Missing Semester](https://missing.csail.mit.edu/) (command line, git, data wrangling) - Similar course at MIT, discovered while creating this version

### Python Reference

- _Python for Data Analysis_, McKinney - author's [website](https://wesmckinney.com/book/)
- _Automate the Boring Stuff with Python_, Sweigart - author's [website]](https://automatetheboringstuff.com/)
- _Whirlwind Tour of Python_, VanderPlas - author's [website](https://jakevdp.github.io/WhirlwindTourOfPython/)
- _Think Python_, Downey - purchase or read at [Green Tea Press](https://greenteapress.com/wp/think-python/)
- _Hitchhiker's Guide to Python!_ - official [documentation](https://docs.python-guide.org/)
- [Introduction to Python](http://introtopython.org/)
- [_A Byte of Python_](https://python.swaroopch.com/)
- [Official Python documentation](https://docs.python.org/3/)
- [O'Reilly catalog via UCSF library](https://www.oreilly.com/library-access/)
  - _Introducing Python_
  - _Python Crash Course_

### Python Courses

- [Exercism Python track](https://exercism.io/tracks/python)
- [Codecademy Python course](https://www.codecademy.com/learn/learn-python-3)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
- [Real Python](https://realpython.com/)
- [DataCamp (Python + Data Science)](https://www.datacamp.com/)
- [Learn Python 3 @ codeacademy.com](https://www.codecademy.com/learn/learn-python-3)
- [Jetbrains' Learn PyCharm](https://www.jetbrains.com/pycharm/learn/)
- [Effective PyCharm Course](https://training.talkpython.fm/courses/explore_pycharm/mastering-pycharm-ide) (\$\$\$)

### Deep Learning

- [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Documentation](https://pytorch.org/docs/stable/tutorials/)
- [Keras Documentation](https://keras.io/guides/)
- [*Deep Learning with Python*, François Chollet](https://www.manning.com/books/deep-learning-with-python)
- [*Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

### Command Line

- [The Linux Command Line book](http://linuxcommand.org/tlcl.php)
- [The Shell Scripting Tutorial](https://www.shellscript.sh/)
- [The Missing Semester](https://missing.csail.mit.edu/)
- [Bash manual](https://www.gnu.org/software/bash/manual/)

### Git

- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)
- [Pro Git book](https://git-scm.com/book/en/v2)
- [Learn Git Branching](https://learngitbranching.js.org/)
- [Official Git documentation](https://git-scm.com/doc)

### Markdown

- [Markdown Guide](https://www.markdownguide.org/)
- [Markdown Tutorial](https://www.markdowntutorial.com/)
- [CommonMark](https://commonmark.org/)

### Free SSH Options

- [Google Cloud Shell](https://cloud.google.com/free/docs/compute-getting-started)
- [GitHub Codespaces](https://cli.github.com/manual/gh_codespace_ssh)
