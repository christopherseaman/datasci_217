---
title: "Introduction to Python & Data Science Tools"
permalink: /
notion:
  role: course
  status: mapped
  page_id: "271d9fdd-1a1a-80c6-ab20-f6065b01e4e3"
  url: "https://app.notion.com/p/271d9fdd1a1a80c6ab20f6065b01e4e3"
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
