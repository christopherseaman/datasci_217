# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.3.3", "pandas==3.0.5", "scikit-learn==1.9.0"]
# ///
"""Grade every student fork of an assignment with the course's own checks.

For teaching assistants, from a clone of this course repository:

    uv run scripts/grade_submissions.py 02
    uv run scripts/grade_submissions.py 02 --fork octocat/ds217-26f-02

The first form grades every fork of the assignment's course repository named
in assignments-26f.json; `--fork` (repeatable) grades only the forks named.
Each fork is cloned, or updated on later runs, under
scratch/submissions/NN/<github-user>/ so it can be opened afterwards.
Results go to scratch/submissions/NN/grades.csv, one row per fork, updated
after every fork: a fork graded again replaces its own row, and every other
row stays. A fork that cannot be regraded keeps its last score, marked with
the error, and the `checks` column records which version of the checks
produced each score.

Grades come from this repository's copy of the checks (NN/assignment_checks/
when it exists, otherwise NN/assignment/), never from the copy inside the
student's fork, and grading reads committed files only: no student code runs.
Committed symlinks are checked out as plain files, so a fork cannot point the
checks at files outside itself. A clone with local changes is reported and
left alone, so a TA can look around inside a submission; move or stash any
notes to regrade it.

Listing forks needs no login for these public repositories. Set GITHUB_TOKEN,
or log in with `gh auth login`, if GitHub's anonymous rate limit is reached.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request


REPO = Path(__file__).resolve().parents[1]
ASSIGNMENTS_FILE = REPO / "assignments-26f.json"
GRADE_TIMEOUT_SECONDS = 300
GIT_TIMEOUT_SECONDS = 600
GIT_ENVIRONMENT = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
LEADING_COLUMNS = ["github_user", "repository", "commit", "commit_date", "status", "score", "max_score", "checks"]


class SubmissionError(Exception):
    """A fork that could not be fetched or graded; reported, never scored."""


def load_assignment(number: str) -> dict:
    assignments = json.loads(ASSIGNMENTS_FILE.read_text(encoding="utf-8"))["assignments"]
    for assignment in assignments:
        if assignment["number"] == number:
            return assignment
    known = ", ".join(assignment["number"] for assignment in assignments)
    raise SystemExit(f"No assignment {number} in {ASSIGNMENTS_FILE.name}; choose one of {known}.")


def trusted_checks_dir(number: str) -> Path:
    """The course-owned checks CI fetches when present, else the handout's copy."""
    course_owned = REPO / number / "assignment_checks"
    if (course_owned / "check_assignment.py").is_file():
        return course_owned
    return REPO / number / "assignment"


def checks_version(checks: Path) -> str:
    """This repository's commit, flagged when the checks directory differs from it."""
    commit = git("rev-parse", "--short", "HEAD", cwd=REPO)
    changed = git("status", "--porcelain", "--", str(checks.relative_to(REPO)), cwd=REPO)
    return f"{commit}+local changes" if changed else commit


def github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token or shutil.which("gh") is None:
        return token
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def fork_listing_error(repository: str, error: urllib.error.HTTPError) -> str:
    if error.code == 404:
        return f"GitHub has no repository {repository}; check its entry in {ASSIGNMENTS_FILE.name}."
    if error.code == 401:
        return "GitHub rejected the token; unset GITHUB_TOKEN and GH_TOKEN or run `gh auth login` again."
    if error.code in (403, 429):
        return "GitHub refused to list forks, usually its rate limit; set GITHUB_TOKEN or run `gh auth login`."
    return f"GitHub answered {error.code} {error.reason} when listing forks of {repository}."


def list_forks(repository: str, token: str | None) -> list[dict]:
    """Every direct fork of `repository`, as {'user', 'repository', 'url'}."""
    forks, page = [], 1
    while True:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repository}/forks?per_page=100&page={page}",
            headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
        )
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                batch = json.load(response)
        except urllib.error.HTTPError as error:
            raise SystemExit(fork_listing_error(repository, error)) from None
        except urllib.error.URLError as error:
            raise SystemExit(f"Could not reach GitHub to list forks of {repository}: {error.reason}") from None
        if not batch:
            return sorted(forks, key=lambda fork: fork["user"].casefold())
        forks.extend(
            {"user": fork["owner"]["login"], "repository": fork["full_name"], "url": fork["clone_url"]}
            for fork in batch
        )
        page += 1


def named_fork(name: str) -> dict:
    """A fork given as GitHub's `user/repository`, or as any clone URL or path git accepts."""
    parts = name.split("/")
    if len(parts) == 2 and all(parts) and ":" not in name:
        return {"user": parts[0], "repository": name, "url": f"https://github.com/{name}.git"}
    tail = re.split(r"[/:]", name.rstrip("/").removesuffix(".git"))
    if len(tail) < 2 or not all(tail[-2:]):
        raise SystemExit(f"--fork {name!r} should look like github-user/ds217-26f-NN or a clone URL")
    user, repository = tail[-2:]
    return {"user": user, "repository": f"{user}/{repository}", "url": name}


def same_remote(first: str, second: str) -> bool:
    """Whether two clone URLs name one repository, ignoring `.git`, a trailing slash, and case."""
    def key(url: str) -> str:
        return re.sub(r"(\.git)?/*$", "", url.strip()).casefold()
    return key(first) == key(second)


def printable(text: str) -> str:
    """Text safe to print and save, even when it quotes a file name that is not UTF-8."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def git(*arguments: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=cwd, env=GIT_ENVIRONMENT, capture_output=True, text=True,
            errors="backslashreplace", timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise SubmissionError(f"git {arguments[0]} did not finish within {GIT_TIMEOUT_SECONDS} seconds") from error
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines() or ["no output"]
        reason = next((line for line in lines if line.startswith("fatal:")), lines[0])
        raise SubmissionError(f"git {arguments[0]} failed: {reason}")
    return result.stdout.strip()


def sync_clone(url: str, clone: Path) -> dict:
    """Clone the fork's default branch, or move an existing clean clone to its latest commit."""
    if not clone.exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        # Symlinks become plain files holding the link text, so a committed link
        # cannot point a check at another student's clone or anywhere else.
        git("clone", "--quiet", "--depth", "1", "--no-tags", "--config", "core.symlinks=false",
            "--", url, str(clone))
    else:
        if not (clone / ".git").is_dir():
            raise SubmissionError(f"{clone} exists but is not a clone; move it aside to grade this fork.")
        origin = git("config", "--get", "remote.origin.url", cwd=clone)
        if not same_remote(origin, url):
            raise SubmissionError(f"{clone} is a clone of {origin}, not {url}; move it aside to grade this fork.")
        if git("config", "--default", "true", "--get", "core.symlinks", cwd=clone) != "false":
            raise SubmissionError(f"{clone} was not cloned by this script; move it aside to grade this fork.")
        if git("status", "--porcelain", "--ignored", "--untracked-files=all", cwd=clone):
            raise SubmissionError(f"{clone} has local changes; move or stash them to regrade it.")
        git("fetch", "--quiet", "--depth", "1", "--no-tags", "origin", "HEAD", cwd=clone)
        git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=clone)
    commit, committed = git("log", "-1", "--format=%H%x09%cI", cwd=clone).split("\t")
    return {"commit": commit, "commit_date": committed}


def grade(checks: Path, clone: Path) -> dict:
    """Run the trusted checker on the clone's committed files and return its JSON report."""
    command = [sys.executable, "-B", "-E", "-s", str(checks / "check_assignment.py"), str(clone), "--json"]
    try:
        result = subprocess.run(
            command, cwd=clone, capture_output=True, text=True, errors="backslashreplace",
            timeout=GRADE_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise SubmissionError(f"checker did not finish within {GRADE_TIMEOUT_SECONDS} seconds") from error
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise SubmissionError(f"checker crashed: {detail[-1] if detail else 'no output'}") from error
    if isinstance(report, dict) and "error" in report:
        raise SubmissionError(f"checker could not grade: {report['error']}")
    tests = report.get("tests") if isinstance(report, dict) else None
    if result.returncode not in (0, 1) or not tests or not all(isinstance(test, dict) for test in tests):
        raise SubmissionError(f"checker exited {result.returncode} without a usable report")
    return report


def grade_fork(fork: dict, checks: Path, destination: Path, version: str) -> dict:
    row = {"github_user": fork["user"], "repository": fork["repository"], "checks": version}
    clone = destination / fork["user"].casefold()
    try:
        row |= sync_clone(fork["url"], clone)
        report = grade(checks, clone)
    except SubmissionError as error:
        return row | {"status": "error", "details": printable(str(error))}
    failed = [f"{test['test-name']}: {test.get('detail') or 'failed'}" for test in report["tests"] if not test["passed"]]
    return row | {
        "status": "graded",
        "score": report["score"],
        "max_score": report["max-score"],
        **{f"test: {test['test-name']}": test["score"] for test in report["tests"]},
        "details": printable(" | ".join(failed)),
    }


def write_grades(row: dict, path: Path) -> dict:
    """Put one fork's result into grades.csv without disturbing any other row; return what was saved."""
    latest = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for saved in csv.DictReader(handle):
                latest[saved["github_user"].casefold()] = {k: v for k, v in saved.items() if k is not None}
    previous = latest.get(row["github_user"].casefold(), {})
    if row["status"] == "error" and previous.get("score"):
        kept_from = previous["commit"][:7]
        row = previous | {"status": "error", "details": f"{row['details']} (score kept from commit {kept_from})"}
    else:
        # Columns a TA added, such as notes, stay; the script's own columns are replaced.
        added = {key: value for key, value in previous.items()
                 if key not in LEADING_COLUMNS and key != "details" and not key.startswith("test: ")}
        row = added | row
    latest[row["github_user"].casefold()] = row

    rows = sorted(latest.values(), key=lambda saved: saved["github_user"].casefold())
    keys = list(dict.fromkeys(key for saved in rows for key in saved))
    tests = [key for key in keys if key.startswith("test: ")]
    others = [key for key in keys if key not in LEADING_COLUMNS and key not in tests and key != "details"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADING_COLUMNS + tests + ["details"] + others)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("assignment", help="two-digit assignment number, such as 02")
    parser.add_argument("--fork", action="append", default=[], metavar="USER/REPO",
                        help="grade only this fork; repeat for several")
    parser.add_argument("--dest", type=Path, help="where clones and grades.csv go (default scratch/submissions/NN)")
    args = parser.parse_args(argv)

    number = args.assignment.zfill(2)
    assignment = load_assignment(number)
    checks = trusted_checks_dir(number)
    version = checks_version(checks)
    destination = (args.dest or REPO / "scratch" / "submissions" / number).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    grades = destination / "grades.csv"

    if args.fork:
        forks = [named_fork(name) for name in args.fork]
    else:
        forks = list_forks(assignment["repository"], github_token())
        print(f"{len(forks)} forks of {assignment['repository']}")
    print(f"Checks: {checks.relative_to(REPO)} at {version}  Clones: {destination}")

    errors = 0
    for fork in forks:
        row = write_grades(grade_fork(fork, checks, destination, version), grades)
        if row["status"] == "graded":
            print(f"{row['github_user']:<24} {row['score']:>3}/{row['max_score']:<3} {row['commit'][:7]}")
        else:
            errors += 1
            print(f"{row['github_user']:<24} ERROR  {row['details']}")
    print(f"Updated {grades}: {len(forks) - errors} graded, {errors} not graded")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
