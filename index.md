---
title: "Introduction to Python & Data Science Tools"
permalink: /
---

# Introduction to Python & Data Science Tools

## Resources

- Canonical URL — [not.badmath.org/ds217](https://not.badmath.org/ds217)
- GitHub repo — [github.com/christopherseaman/datasci_217](https://github.com/christopherseaman/datasci_217)

#### References

- [Python for Data Analysis](https://wesmckinney.com/book/) (rough basis for Python content)
- [The Missing Semester](https://missing.csail.mit.edu/) (command line, git, data wrangling)
- [The Linux Command Line book](http://linuxcommand.org/tlcl.php) (command line in-depth)
- [Markdown Guide](https://www.markdownguide.org/)

#### Development Tools (free!)

- [VS Code](https://code.visualstudio.com/)
- [Python](https://www.python.org/)
- [GitHub Codespaces](https://cli.github.com/manual/gh_codespace_ssh) (free IDE in a browser)
- [Google Cloud Shell](https://cloud.google.com/free/docs/compute-getting-started) (practice command line anywhere)

## Lectures

{% for lecture in collections.lectures -%}
- [{{ lecture.data.title }}]({{ lecture.url | url }})
{% endfor %}
