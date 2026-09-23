# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.3.3", "pandas==3.0.5", "scikit-learn==1.9.0"]
# ///
"""Check the TA grading script, using Assignment 01 as it was handed out.

    uv run scripts/test_grade_submissions.py

Completes Assignment 01 by following its README literally, publishes that and
other submissions as local repositories standing in for forks, and grades them
through scripts/grade_submissions.py. Nothing here contacts GitHub.
"""

from contextlib import contextmanager
import csv
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
HANDOUT = REPO / "01" / "assignment"
sys.path.insert(0, str(REPO / "scripts"))
import grade_submissions  # noqa: E402

sys.path.insert(0, str(HANDOUT))
from _assignment_checks import EXPECTED_READINESS, ROSTER_HASHES  # noqa: E402

GIT_IDENTITY = ["-c", "user.name=Test Student", "-c", "user.email=student@example.com"]
ROSTER_HASH = sorted(ROSTER_HASHES)[0]
CLEAN = ["git", "status", "--porcelain", "--ignored", "--untracked-files=all"]


def run(command: list[str], cwd: Path, stdin: str | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, input=stdin, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    return result.stdout


@contextmanager
def patched(**values):
    """Temporarily replace attributes of the grading script."""
    saved = {name: getattr(grade_submissions, name) for name in values}
    for name, value in values.items():
        setattr(grade_submissions, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(grade_submissions, name, value)


def readme_blocks() -> dict[str, dict[str, list[str]]]:
    """The fenced blocks under each README heading, keyed by the heading's first word and fence language."""
    sections, heading = {}, None
    text = (HANDOUT / "README.md").read_text(encoding="utf-8")
    for match in re.finditer(r"^#{2,3} (\S+)|^```(\w*)\n(.*?)^```", text, re.M | re.S):
        if match.group(1) is not None:
            heading = match.group(1)
        else:
            sections.setdefault(heading, {}).setdefault(match.group(2) or "untagged", []).append(match.group(3))
    return sections


def block(sections: dict, task: str, language: str) -> str:
    found = sections.get(task, {}).get(language, [])
    assert len(found) == 1, f"README task {task} should hold exactly one ```{language} block; found {len(found)}"
    return found[0]


def printed(command: list[str], cwd: Path, expected: str, what: str) -> None:
    actual = run(command, cwd)
    assert actual == expected, f"{what} printed:\n{actual}\nThe README expects:\n{expected}"


def copy_tracked(number: str, destination: Path) -> Path:
    """The files a student's fork starts with: exactly what the course repository tracks."""
    tracked = run(["git", "ls-files", "-z", f"{number}/assignment"], REPO).split("\0")
    for name in filter(None, tracked):
        target = destination / Path(name).relative_to(f"{number}/assignment")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    return destination


def complete_as_written(root: Path) -> None:
    """Do every task the way the README says, checking each result the README quotes."""
    sections = readme_blocks()

    # Task 1.1 runs the README's own shell block.
    run(["bash", "-euo", "pipefail", "-c", block(sections, "1.1", "bash")], root)
    assert sorted(path.name for path in (root / "terminal-practice").iterdir()) == ["path-check.txt", "source.txt"]

    # Task 1.2 replaces only the three TODO lines, using the supplied variables.
    readiness = (root / "readiness.py").read_text()
    for old, new in (
        ('print("Python family: TODO")', 'print("Python family:", python_family)'),
        ('print("Project: TODO")', 'print("Project:", PROJECT_LABEL)'),
        ('print("Script: TODO")', 'print("Script:", script_filename)'),
    ):
        assert readiness.count(old) == 1, f"readiness.py no longer holds {old}"
        readiness = readiness.replace(old, new)
    (root / "readiness.py").write_text(readiness)
    printed([sys.executable, "readiness.py"], root, block(sections, "1.2", "text"), "readiness.py")

    # Task 2.1 follows the nine numbered steps with Lecture 01 tools only: no f-strings.
    summary = (root / "measurement_summary.py").read_text()
    scaffold = 'print("TODO: complete the measurement summary")\n'
    assert summary.endswith(scaffold), "measurement_summary.py no longer ends with its TODO line"
    (root / "measurement_summary.py").write_text(summary.removesuffix(scaffold) + (
        "review_threshold = int(review_threshold_text)\n"
        "total = 0\n"
        "review_count = 0\n"
        "for measurement in measurements:\n"
        "    total = total + measurement\n"
        "    if measurement >= review_threshold:\n"
        '        label = "review"\n'
        "        review_count = review_count + 1\n"
        "    else:\n"
        '        label = "within range"\n'
        '    print("Measurement:", measurement, label)\n'
        "mean = total / len(measurements)\n"
        'print("Count:", len(measurements))\n'
        'print("Total:", total)\n'
        'print("Mean:", mean)\n'
        'print("Review count:", review_count)\n'
    ))
    printed([sys.executable, "measurement_summary.py"], root, block(sections, "2.2", "text"), "measurement_summary.py")

    # Task 3.1: exactly three prepared errors, each surfaced by a traceback and fixed in turn.
    debug = root / "debug_report.py"
    for error, old, new in (
        ("IndentationError", '\nprint("Readiness: complete")', '\n    print("Readiness: complete")'),
        ("NameError", "participant_cout", "participant_count"),
        ("TypeError", "participant_count_text + 1", "participant_count + 1"),
    ):
        failed = subprocess.run([sys.executable, "debug_report.py"], cwd=root, capture_output=True, text=True)
        assert failed.returncode != 0 and failed.stderr.strip().splitlines()[-1].startswith(error), failed.stderr
        text = debug.read_text()
        assert text.count(old) == 1, f"debug_report.py no longer holds {old!r}"
        debug.write_text(text.replace(old, new))
    printed([sys.executable, "debug_report.py"], root, block(sections, "3.1", "text"), "debug_report.py")

    # Task 3.2: the supplied wrapper saves the three outputs in order, all 14 lines.
    run([sys.executable, "make_output.py"], root)
    saved = (root / "output" / "readiness.txt").read_text(encoding="utf-8")
    quoted = block(sections, "1.2", "text") + block(sections, "2.2", "text") + block(sections, "3.1", "text")
    assert saved == EXPECTED_READINESS == quoted and len(saved.splitlines()) == 14

    # Task 3.3: the helper saves only a hash; an email off the roster is rejected by the check,
    # so the test then stands in a roster hash for the student's own.
    run([sys.executable, "capture_identity.py"], root, stdin="Test.Student@ucsf.edu\n")
    identity = root / "output" / "student_identity.txt"
    assert re.fullmatch(r"[0-9a-f]{64}\n", identity.read_text())
    checked = subprocess.run([sys.executable, "-B", "check_assignment.py"], cwd=root, capture_output=True, text=True)
    assert checked.returncode == 1 and "Score: 20/100" in checked.stdout, checked.stdout
    identity.write_text(ROSTER_HASH + "\n", encoding="utf-8")

    # Check Your Work: the README's promised ending.
    checked = run([sys.executable, "-B", "check_assignment.py"], root)
    assert checked.endswith("Score: 100/100\nAll checks passed.\n"), checked


def publish_fork(forks: Path, user: str, files: Path, repository: str = "ds217-26f-01") -> str:
    """A local repository standing in for `user`'s fork on GitHub; returns its clone URL."""
    fork = forks / user / repository
    shutil.copytree(files, fork, symlinks=True)
    run(["git", "init", "--quiet", "--initial-branch", "main"], fork)
    push(fork, "Submit")
    return fork.as_uri()


def push(fork: Path, message: str) -> None:
    """Commit everything in a stand-in fork, as a student's push would."""
    run(["git", *GIT_IDENTITY, "add", "--all"], fork)
    run(["git", *GIT_IDENTITY, "commit", "--quiet", "-m", message], fork)


def head(url: str) -> str:
    return run(["git", "rev-parse", "HEAD"], Path(url.removeprefix("file://"))).strip()


def grade(destination: Path, *forks: str, assignment: str = "01") -> int:
    return grade_submissions.main([assignment, "--dest", str(destination), *(f"--fork={fork}" for fork in forks)])


def read_grades(destination: Path) -> dict[str, dict]:
    with (destination / "grades.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    users = [row["github_user"].casefold() for row in rows]
    assert len(users) == len(set(users)), f"grades.csv repeats a fork: {users}"
    return {row["github_user"]: row for row in rows}


def test_fork_names_and_listing() -> None:
    https = "https://github.com/alice/ds217-26f-01.git"
    assert grade_submissions.named_fork("alice/ds217-26f-01") == {
        "user": "alice", "repository": "alice/ds217-26f-01", "url": https}
    for url in (https, "git@github.com:alice/ds217-26f-01.git", "file:///srv/forks/alice/ds217-26f-01"):
        assert grade_submissions.named_fork(url)["user"] == "alice", url
    for bad in ("alice", "alice/", "/ds217-26f-01"):
        try:
            grade_submissions.named_fork(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"accepted {bad!r}")
    for same in ("https://github.com/Alice/ds217-26f-01", "https://github.com/alice/ds217-26f-01/", https):
        assert grade_submissions.same_remote(same, https), same
    assert not grade_submissions.same_remote("https://github.com/bob/ds217-26f-01.git", https)
    assert grade_submissions.printable("monitor\udcff.txt") == "monitor\\udcff.txt"

    # Grades come from the course-owned value checks where they exist, as CI fetches them.
    assert grade_submissions.trusted_checks_dir("01") == HANDOUT
    for number in ("02", "03"):
        assert grade_submissions.trusted_checks_dir(number) == REPO / number / "assignment_checks"
    assert grade_submissions.load_assignment("01")["repository"] == "UCSF-DataSci/ds217-26f-01"

    listing = {login: {"owner": {"login": login}, "full_name": f"{login}/ds217-26f-01",
                       "clone_url": f"https://github.com/{login}/ds217-26f-01.git"} for login in ("zoe", "Bob", "amir")}
    pages = {1: [listing["zoe"], listing["Bob"]], 2: [listing["amir"]], 3: []}
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(pages[int(re.search(r"[?&]page=(\d+)", request.full_url).group(1))]).encode())

    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        forks = grade_submissions.list_forks("UCSF-DataSci/ds217-26f-01", token="secret")
        assert forks == [{"user": login, "repository": f"{login}/ds217-26f-01",
                          "url": f"https://github.com/{login}/ds217-26f-01.git"} for login in ("amir", "Bob", "zoe")]
        # A fork named on the command line matches the same fork found by listing.
        assert grade_submissions.named_fork("zoe/ds217-26f-01") == forks[2]
        assert all(r.full_url.startswith("https://api.github.com/repos/UCSF-DataSci/ds217-26f-01/forks?")
                   for r in requests)
        assert all(r.get_header("Authorization") == "Bearer secret" for r in requests)
        requests.clear()
        grade_submissions.list_forks("UCSF-DataSci/ds217-26f-01", token=None)
        assert len(requests) == 3 and not any(r.has_header("Authorization") for r in requests)

        # GitHub's refusals become a sentence a TA can act on, not a traceback.
        for failure, advice in (
            (urllib.error.HTTPError("u", 403, "rate limit exceeded", {}, None), "GITHUB_TOKEN"),
            (urllib.error.HTTPError("u", 404, "Not Found", {}, None), "assignments-26f.json"),
            (urllib.error.HTTPError("u", 401, "Bad credentials", {}, None), "gh auth login"),
            (urllib.error.URLError("no network"), "Could not reach GitHub"),
        ):
            def refuse(request, timeout, failure=failure):
                raise failure
            urllib.request.urlopen = refuse
            try:
                grade_submissions.list_forks("UCSF-DataSci/ds217-26f-01", token=None)
            except SystemExit as exit:
                assert advice in str(exit.code), exit.code
            else:
                raise AssertionError(f"{failure} was not reported")
    finally:
        urllib.request.urlopen = real_urlopen


def test_every_assignment_through_the_script(work: Path) -> None:
    """Each assignment's untouched handout, graded as a fork, gives a zero row and leaves its clone clean."""
    for number in (f"{n:02}" for n in range(1, 12)):
        url = publish_fork(work / "forks", f"handout{number}", copy_tracked(number, work / f"files-{number}"),
                           repository=f"ds217-26f-{number}")
        destination = work / f"submissions-{number}"
        assert grade(destination, url, assignment=number) == 0, number
        row = read_grades(destination)[f"handout{number}"]
        assert row["status"] == "graded" and row["score"] == "0", (number, row)
        assert row["max_score"] == ("85" if number in ("05", "11") else "100"), (number, row)
        assert not re.search(r": failed( \||$)", row["details"]), (number, "a failing check gave no reason", row)
        assert run(CLEAN, destination / f"handout{number}") == "", (number, "the checks wrote into the clone")


def test_grading_forks(work: Path) -> None:
    forks, destination, markers = work / "forks", work / "submissions", work / "markers"
    markers.mkdir()
    completed = copy_tracked("01", work / "completed")
    complete_as_written(completed)
    urls = {"alice": publish_fork(forks, "alice", completed)}

    # Terminal practice only: 20 of 100, the split the Completion Contract states.
    partial = copy_tracked("01", work / "partial")
    shutil.copytree(completed / "terminal-practice", partial / "terminal-practice")
    urls["bob"] = publish_fork(forks, "bob", partial)

    urls["carol"] = publish_fork(forks, "carol", copy_tracked("01", work / "untouched"))

    # A fork whose own checker claims full marks, and where every Python file, including ones
    # named like the standard-library modules the checks import, leaves a trace if it runs.
    forged = work / "forged"
    shutil.copytree(completed, forged)
    (forged / "output" / "student_identity.txt").unlink()
    trap = f"open({str(markers / 'ran')!r}, 'w').write(__file__)\n"
    for name in ["sitecustomize.py", "json.py", "re.py", "argparse.py", "dataclasses.py", "pathlib.py",
                 *(path.name for path in forged.glob("*.py"))]:
        (forged / name).write_text(trap)
    fake_report = {"schema": "datasci217/grading-result/v1", "score": 100, "max-score": 100, "tests": []}
    (forged / "check_assignment.py").write_text(trap + f"print({json.dumps(json.dumps(fake_report))})\n")
    urls["mallory"] = publish_fork(forks, "mallory", forged)

    # Correct work saved with another Python loses nothing: the report's first line
    # records whichever version ran it, and that line is not graded.
    other_python = work / "other-python"
    shutil.copytree(completed, other_python)
    report = other_python / "output" / "readiness.txt"
    report.write_text(report.read_text().replace("Python family: 3.13\n", "Python family: 3.14\n", 1))
    urls["erin"] = publish_fork(forks, "erin", other_python)

    # A fork whose folders are symlinks into a classmate's clone beside it.
    borrowed = copy_tracked("01", work / "borrowed")
    for name in ("output", "terminal-practice"):
        shutil.rmtree(borrowed / name, ignore_errors=True)
        (borrowed / name).symlink_to(Path("..") / "alice" / name)
    urls["sam"] = publish_fork(forks, "sam", borrowed)

    assert grade(destination, *urls.values()) == 0
    grades = read_grades(destination)
    assert {user: row["score"] for user, row in grades.items()} == {
        "alice": "100", "bob": "20", "carol": "0", "erin": "100", "mallory": "20", "sam": "0"}
    assert all(row["status"] == "graded" and row["max_score"] == "100" for row in grades.values())
    assert grades["alice"]["test: terminal practice evidence"] == "20"
    assert grades["alice"]["test: committed readiness and identity artifacts"] == "80"
    assert grades["alice"]["details"] == ""
    assert "student_identity.txt" in grades["mallory"]["details"]
    assert len({row["checks"] for row in grades.values()}) == 1 and grades["alice"]["checks"]
    for user, url in urls.items():
        assert grades[user]["commit"] == head(url), user
    assert not list(markers.iterdir()), "a submission's own code ran"
    # The symlinks arrived as plain files, so nothing outside sam's fork was read.
    assert (destination / "sam" / "output").is_file() and not (destination / "sam" / "output").is_symlink()

    # Grading leaves every clone exactly as fetched, so the next run can update it.
    for user in urls:
        assert run(CLEAN, destination / user) == "", user

    # A student pushes the missing files; the next run moves the clone to the new commit
    # and leaves everyone else's row as it was.
    fork_bob = forks / "bob" / "ds217-26f-01"
    shutil.copytree(completed / "output", fork_bob / "output", dirs_exist_ok=True)
    push(fork_bob, "Add output")
    assert grade(destination, urls["bob"]) == 0
    regraded = read_grades(destination)
    assert regraded["bob"]["score"] == "100"
    assert regraded["bob"]["commit"] == head(urls["bob"]) != grades["bob"]["commit"]
    assert {u: r for u, r in regraded.items() if u != "bob"} == {u: r for u, r in grades.items() if u != "bob"}

    # The same fork named another way is still the same clone.
    for spelling in (urls["alice"] + "/", urls["alice"] + ".git"):
        assert grade(destination, spelling) == 0, spelling
    assert read_grades(destination)["alice"]["score"] == "100"

    # TA changes inside a clone of any kind are reported and preserved, never overwritten or graded,
    # and the student keeps the last score recorded for them.
    (destination / "carol" / "ta-notes.txt").write_text("untracked note\n")
    (destination / "alice" / "output" / "readiness.txt").write_text("edited by a TA\n")
    (destination / "mallory" / "__pycache__").mkdir()
    (destination / "mallory" / "__pycache__" / "ignored.pyc").write_bytes(b"ignored by .gitignore")
    missing = (forks / "nobody" / "ds217-26f-01").as_uri()
    assert grade(destination, urls["carol"], urls["alice"], urls["mallory"], missing, urls["erin"]) == 1
    rows = read_grades(destination)
    for user, kept in (("carol", "0"), ("alice", "100"), ("mallory", "20")):
        assert rows[user]["status"] == "error" and "local changes" in rows[user]["details"], rows[user]
        assert rows[user]["score"] == kept and "score kept from commit" in rows[user]["details"], rows[user]
    assert (destination / "carol" / "ta-notes.txt").read_text() == "untracked note\n"
    assert (destination / "alice" / "output" / "readiness.txt").read_text() == "edited by a TA\n"
    assert (destination / "mallory" / "__pycache__" / "ignored.pyc").exists()
    assert rows["nobody"]["status"] == "error" and rows["nobody"]["score"] == ""
    assert rows["nobody"]["details"].startswith("git clone failed: fatal:"), rows["nobody"]["details"]
    assert rows["erin"]["status"] == "graded" and rows["erin"]["score"] == "100"

    # A folder holding someone else's clone, or a clone this script did not make, is not graded.
    run(["git", "clone", "--quiet", urls["alice"], str(destination / "dave")], destination)
    urls["dave"] = publish_fork(forks, "dave", completed)
    urls["gil"] = publish_fork(forks, "gil", completed)
    run(["git", "clone", "--quiet", urls["gil"], str(destination / "gil")], destination)
    assert grade(destination, urls["dave"], urls["gil"]) == 1
    rows = read_grades(destination)
    assert "is a clone of" in rows["dave"]["details"]
    assert "was not cloned by this script" in rows["gil"]["details"]


def test_github_names_offline(work: Path) -> None:
    """GitHub names, a TA's URL rewriting, and login case, with GitHub replaced by local forks."""
    forks, destination = work / "gh-forks", work / "gh-submissions"
    publish_fork(forks, "frank", copy_tracked("01", work / "frank"))
    (forks / "frank" / "ds217-26f-01.git").symlink_to("ds217-26f-01")
    config = work / "gitconfig"
    # Many TAs rewrite GitHub URLs to SSH like this; here the rewrite points at the local forks.
    config.write_text(f'[url "{forks}/"]\n\tinsteadOf = https://github.com/\n')
    environment = {**grade_submissions.GIT_ENVIRONMENT, "GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1"}
    with patched(GIT_ENVIRONMENT=environment):
        assert grade_submissions.main(["01", "--dest", str(destination), "--fork", "frank/ds217-26f-01"]) == 0
        assert read_grades(destination)["frank"]["score"] == "0"
        shutil.copytree(HANDOUT / "output", forks / "frank" / "ds217-26f-01" / "output", dirs_exist_ok=True)
        (forks / "frank" / "ds217-26f-01" / "terminal-practice").mkdir()
        for name in ("source.txt", "path-check.txt"):
            (forks / "frank" / "ds217-26f-01" / "terminal-practice" / name).write_text("")
        push(forks / "frank" / "ds217-26f-01", "Practice files")
        # The second run, with the login in another case, updates the same clone and row.
        assert grade_submissions.main(["01", "--dest", str(destination), "--fork", "Frank/ds217-26f-01"]) == 0
    rows = read_grades(destination)
    assert len(rows) == 1 and next(iter(rows.values()))["score"] == "20", rows
    assert sorted(path.name for path in destination.iterdir() if path.is_dir()) == ["frank"]


def test_checker_failures(work: Path) -> None:
    """A checker that crashes, hangs, refuses, or answers badly gives an error row, never a score."""
    url = publish_fork(work / "failing-forks", "pat", copy_tracked("01", work / "pat"))
    for case, (body, timeout, expected) in enumerate((
        ("raise RuntimeError('boom')", 300, "checker crashed: RuntimeError: boom"),
        ("import time\ntime.sleep(5)", 1, "checker did not finish within 1 seconds"),
        ("print('{\"error\": \"InfrastructureError: versions differ\"}')\nraise SystemExit(2)", 300,
         "checker could not grade: InfrastructureError: versions differ"),
        ("print('{\"score\": 0, \"max-score\": 100, \"tests\": [{\"test-name\": \"t\", \"passed\": true, "
         "\"score\": 0, \"max-score\": 100}]}')\nraise SystemExit(3)", 300, "checker exited 3 without a usable report"),
        ("print('{\"score\": 0, \"max-score\": 100, \"tests\": []}')", 300, "checker exited 0 without a usable report"),
    )):
        checks = work / f"broken-checks-{case}"
        checks.mkdir()
        (checks / "check_assignment.py").write_text(body + "\n")
        destination = work / f"broken-{case}"
        with patched(trusted_checks_dir=lambda number, checks=checks: checks, GRADE_TIMEOUT_SECONDS=timeout):
            assert grade(destination, url) == 1, body
        row = read_grades(destination)["pat"]
        assert row["status"] == "error" and row["score"] == "" and row["details"] == expected, row


def test_grades_file(work: Path) -> None:
    """grades.csv survives a spreadsheet round trip, an interrupted run, and an empty fork listing."""
    forks, destination = work / "file-forks", work / "file-submissions"
    urls = [publish_fork(forks, user, copy_tracked("01", work / f"file-{user}")) for user in ("quinn", "ruth", "stan")]
    assert grade(destination, urls[0]) == 0

    # Saved from Excel as CSV UTF-8, with a column a TA added.
    grades = destination / "grades.csv"
    lines = grades.read_text(encoding="utf-8").splitlines()
    grades.write_text("﻿" + lines[0] + ",notes\n" + lines[1] + ",checked\n", encoding="utf-8")
    assert grade(destination, urls[0]) == 0
    assert read_grades(destination)["quinn"]["notes"] == "checked"
    assert not grades.read_bytes().startswith(b"\xef\xbb\xbf")

    # Stopping a run keeps every fork already graded.
    calls = []
    real_grade = grade_submissions.grade

    def interrupted(checks, clone):
        calls.append(clone.name)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_grade(checks, clone)

    with patched(grade=interrupted):
        try:
            grade(destination, urls[1], urls[2])
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("the interruption did not happen")
    assert sorted(read_grades(destination)) == ["quinn", "ruth"]

    # A listing that finds no forks changes nothing.
    before = grades.read_bytes()
    with patched(list_forks=lambda repository, token: [], github_token=lambda: None):
        assert grade_submissions.main(["01", "--dest", str(destination)]) == 0
    assert grades.read_bytes() == before


def main() -> None:
    assert sys.version_info[:2] == (3, 13), (
        f"The checks pin Python 3.13 and exact dependency versions; this is {sys.version.split()[0]}. "
        "Run with: uv run scripts/test_grade_submissions.py")
    test_fork_names_and_listing()
    (REPO / "scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=REPO / "scratch", prefix="grade-submissions-test-") as directory:
        work = Path(directory)
        test_grading_forks(work)
        test_github_names_offline(work)
        test_checker_failures(work)
        test_grades_file(work)
        test_every_assignment_through_the_script(work)
    print("grade_submissions: Assignment 01 completed as written scores 100/100; partial, untouched, forged, "
          "other-Python and symlinked forks score as intended from the course checks; TA edits, renamed URLs, "
          "checker failures, interrupted runs and grades.csv round trips behave; every assignment's handout "
          "grades to a zero row through the script.")


if __name__ == "__main__":
    main()
