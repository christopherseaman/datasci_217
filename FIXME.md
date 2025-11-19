1. GitHub Education is the program that will get you some freebies for being associated with UCSF, mostly extra hours of Codespaces and access to Copilot
2. Will try to put more/simpler demos for lecture 1 on GitHub in the 01/demo folder by tomorrow or Friday
3. Made a mistake on the assignment which may make it difficult to git add processed_email.txt , so you have to add a "-f" flag to the command. Either git add -f processed_email.txt or git add processed_email.txt -f should work, then git commit -m "your commit message" and git push as normal. Will speak to this at the start of next lecture, which happens to be about git!
4. After pushing to GitHub you likely see a little red X that says "All checks have failed" when you hover or click. Don't worry about this, it is automated grading and I will re-run it later. Then you should get a satisfying green checkmark instead.
If you're having issues getting `brew` commands workings after installing homebrew (MacOS-specific), there might have been a step at the end of the install that you missed. It may vary depending on the version of MacOS you are running, but it would be something like this:

```bash
==> Next steps:
- Run these commands in your terminal to add Homebrew to your PATH:
    echo >> /Users/cseaman/.zprofile
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/cseaman/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
```

More explicit hw instructions; e.g., 7.2 heatmap and unclear correlation
