# Lecture 02 bonus: Command-line Git and deeper concepts

This material is optional. Required course work uses VS Code Source Control or GitHub Desktop. Lecture 03 does not assume command-line Git, Git internals, pull requests, or history rewriting.

Use these examples only in a disposable practice repository until you can predict which Git state each command changes. Never initialize a repository inside an existing assignment repository.

# Translate the state model to commands

The command line exposes the same repository, working-tree, staging, commit, branch, and remote states taught in the main lecture.

| Intention | Command |
|---|---|
| Inspect repository state | `git status` |
| Inspect unstaged changes | `git diff` |
| Inspect staged changes | `git diff --staged` |
| Stage one path | `git add path/to/file` |
| Create a commit | `git commit -m "Describe the change"` |
| Inspect concise history | `git log --oneline` |
| Download and integrate remote commits | `git pull` |
| Send local commits to the remote | `git push` |

Prefer naming the exact path you intend to stage. Broad staging can hide accidental data, credentials, generated output, or unrelated edits.

## One daily command-line cycle

```bash
git status
git diff
git add README.md
git diff --staged
git commit -m "Clarify project purpose"
git pull
git push
git status
```

Read the output after each command. If `git pull` reports a conflict, stop and resolve it before pushing.

# Branches and merging

`git switch` changes the active branch. The `-c` option creates a branch before switching to it.

```bash
git switch -c feature/summary-label
```

After editing and testing:

```bash
git diff
git add README.md
git commit -m "Clarify summary label"
git switch main
git pull
git merge feature/summary-label
git push
```

Delete the local feature branch only after confirming that its intended change is present on `main`:

```bash
git branch -d feature/summary-label
```

## Resolve a command-line merge conflict

Git writes three marker lines into a file when it cannot combine overlapping edits. The actual markers use seven less-than signs, seven equals signs, and seven greater-than signs:

```text
[begin current version: seven less-than signs followed by the current branch label]
text from the current branch
[separator: seven equals signs]
text from the branch being merged
[end incoming version: seven greater-than signs followed by the branch name]
```

Resolve the conflict by editing the file into its correct final form. Remove all markers, save, and inspect the result before staging it:

```bash
git status
git diff
git add README.md
git commit -m "Resolve summary-label conflict"
```

Do not choose a side solely because it is called `HEAD`, current, or incoming. The correct resolution may combine both changes.

# Repository setup commands

Most course repositories are already provisioned. These setup commands are useful outside that workflow.

Create a new repository in the current, intentionally selected directory:

```bash
pwd
ls
git init
```

Copy an existing remote repository into a new local directory:

```bash
git clone https://github.com/OWNER/REPOSITORY.git
```

Inspect configured remotes:

```bash
git remote -v
```

Before `git init` or `git clone`, confirm the current directory and target path. A nested repository is rarely what a beginner intends.

# A little Git internals

These terms explain why branches and commits behave as they do, but they are not needed for the required GUI workflow.

- A **blob** stores file content.
- A **tree** records filenames and directory structure that point to blobs and other trees.
- A **commit object** points to a tree, records metadata, and normally points to one parent commit.
- A **reference** is a human-readable name that points to a commit.
- A **branch** is a movable reference that advances when a new commit is made on that branch.
- `HEAD` identifies the currently checked-out branch or, in a detached state, a particular commit.

Inspect the commit graph:

```bash
git log --graph --oneline --decorate --all
```

Inspect one commit and its recorded change. Replace `COMMIT_ID` with an identifier shown by the log; do not type the placeholder literally:

```bash
git show COMMIT_ID
```

# Collaboration with pull requests

A **pull request** is a GitHub review workflow around a proposed branch merge. It is not the same as the `git pull` command.

A useful pull request normally has:

- one focused purpose;
- a descriptive title;
- a short explanation of what changed and why;
- evidence that the change was checked; and
- a diff small enough for another person to review.

The branch author and reviewer should discuss the intended result, not merely whether Git can merge it automatically.

# Safer recovery patterns

Recovery begins with inspection:

```bash
git status
git diff
git diff --staged
git log --oneline --decorate -10
```

## Unstage without discarding the edit

```bash
git restore --staged path/to/file
```

This moves the selected change out of the staging area while leaving the working-tree edit in place.

## Revert a published commit

```bash
git revert COMMIT_ID
```

`git revert` creates a new commit that reverses an earlier commit. It preserves the shared history, which generally makes it safer than rewriting a published branch.

## Discard an uncommitted file edit

```bash
git restore path/to/file
```

This command overwrites the uncommitted working-tree change in the named file. Inspect the diff and confirm that the path is exact before using it. If the work matters, make a copy outside the repository or ask for help first.

# History rewriting is an expert operation

Commands involving hard reset, interactive rebase, history filtering, or force push can remove reachable work or invalidate collaborators' branches. They are intentionally not provided as copy-and-paste recipes here.

For course repositories:

- do not force push `main`;
- do not rewrite history to make an assignment look cleaner;
- do not run a destructive recovery command suggested by an error message without understanding its target; and
- ask for help while the current repository state still contains the evidence needed for recovery.

# Further resources

- [Official Git reference](https://git-scm.com/docs)
- [Pro Git book](https://git-scm.com/book/en/v2)
- [GitHub documentation for pull requests](https://docs.github.com/en/pull-requests)
- [Learn Git Branching practice environment](https://learngitbranching.js.org/)
