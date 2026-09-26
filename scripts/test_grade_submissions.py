# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.3.3", "pandas==3.0.5", "scikit-learn==1.9.0"]
# ///
"""Check the TA grading script, using Assignment 01 as it was handed out.

    uv run scripts/test_grade_submissions.py

Completes Assignment 01 by following its README literally, publishes that and
other submissions as local repositories standing in for forks, and grades them
through scripts/grade_submissions.py, which uses the course-owned checks in
01/assignment_checks/. The handout ships a byte-identical copy of those checks,
so each fork's own local run is compared with them too. Nothing here contacts
GitHub.
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
COURSE_CHECKS = REPO / "01" / "assignment_checks"
sys.path.insert(0, str(REPO / "scripts"))
import grade_submissions  # noqa: E402

sys.path.insert(0, str(COURSE_CHECKS))
from _value_checks import EXPECTED_READINESS, ROSTER_HASHES  # noqa: E402

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


def value_report(root: Path, checks: Path = COURSE_CHECKS) -> dict:
    """The JSON report of the checks in `checks` for a submission: by default the course-owned
    copy, as CI and the TA script get it; given the submission itself, the student's local run."""
    result = subprocess.run([sys.executable, "-B", str(checks / "check_assignment.py"), str(root), "--json"],
                            cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode in (0, 1), (result.stdout, result.stderr)
    return json.loads(result.stdout)


def failing(report: dict) -> dict[str, str]:
    return {test["test-name"]: test["detail"] for test in report["tests"] if not test["passed"]}


def copy_tracked(number: str, destination: Path) -> Path:
    """The files a student's fork starts with: what the course repository tracks, with files added
    or deleted since the last commit counted, so the handout about to be committed is what is tested."""
    listed = run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", f"{number}/assignment"],
                 REPO).split("\0")
    for name in dict.fromkeys(filter(None, listed)):
        if not (REPO / name).is_file():
            continue  # deleted in the working tree, so the next commit drops it
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
        # As the README says: the IndentationError stops everything before any line runs, with no
        # Traceback header; the other two surface only after the lines above them have printed.
        if error == "IndentationError":
            assert "Traceback" not in failed.stderr and failed.stdout == "", (failed.stdout, failed.stderr)
        else:
            assert failed.stderr.startswith("Traceback (most recent call last)") and failed.stdout, failed
        text = debug.read_text()
        assert text.count(old) == 1, f"debug_report.py no longer holds {old!r}"
        debug.write_text(text.replace(old, new))
    printed([sys.executable, "debug_report.py"], root, block(sections, "3.1", "text"), "debug_report.py")

    # Task 3.2: the supplied wrapper saves the three outputs in order, all 14 lines.
    run([sys.executable, "make_output.py"], root)
    saved = (root / "output" / "readiness.txt").read_text(encoding="utf-8")
    quoted = block(sections, "1.2", "text") + block(sections, "2.2", "text") + block(sections, "3.1", "text")
    assert saved == EXPECTED_READINESS == quoted and len(saved.splitlines()) == 14

    # Task 3.3: the helper saves only a hash. This test's email is off the roster, so the checks,
    # locally and on GitHub alike, hold back the identity's 15 points and nothing else; the test
    # then stands in a roster hash for the student's own.
    run([sys.executable, "capture_identity.py"], root, stdin="Test.Student@ucsf.edu\n")
    identity = root / "output" / "student_identity.txt"
    assert re.fullmatch(r"[0-9a-f]{64}\n", identity.read_text())
    report = value_report(root)
    assert report == value_report(root, checks=root), "the local run and the course checks disagree"
    assert report["score"] == 85 and list(failing(report)) == ["identity hash on the roster"], report
    assert "course roster" in failing(report)["identity hash on the roster"], report
    identity.write_text(ROSTER_HASH + "\n", encoding="utf-8")

    # Check your work: the README's promised local ending, and full marks from the course checks.
    promised = block(sections, "Check", "text")
    checked = run([sys.executable, "-B", "check_assignment.py"], root)
    assert checked.endswith(promised), f"check_assignment.py printed:\n{checked}\nThe README promises:\n{promised}"
    assert value_report(root)["score"] == 100


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

    # Grades come from the course-owned checks where they exist, as CI fetches them.
    for number in ("01", "02", "03"):
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
        assert row["max_score"] == ("75" if number in ("05", "11") else "100"), (number, row)
        assert not re.search(r": failed( \||$)", row["details"]), (number, "a failing check gave no reason", row)
        assert run(CLEAN, destination / f"handout{number}") == "", (number, "the checks wrote into the clone")


def test_grading_forks(work: Path) -> None:
    forks, destination, markers = work / "forks", work / "submissions", work / "markers"
    markers.mkdir()
    completed = copy_tracked("01", work / "completed")
    complete_as_written(completed)
    urls = {"alice": publish_fork(forks, "alice", completed)}

    # Terminal practice only: 20 of 100, 10 for each practice file as the Completion Contract states.
    partial = copy_tracked("01", work / "partial")
    shutil.copytree(completed / "terminal-practice", partial / "terminal-practice")
    urls["bob"] = publish_fork(forks, "bob", partial)

    urls["carol"] = publish_fork(forks, "carol", copy_tracked("01", work / "untouched"))

    # A fork whose own checker claims full marks, and where every Python file, including ones
    # named like the standard-library modules the checks import, leaves a trace if it runs. Its
    # identity file is missing, so the course checks give 85, never the 100 it claims.
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

    # Plausible but wrong: a wrong total and a hash off the roster cost their own 5 and 15 points,
    # and the fork's local run already says so, exactly as the course checks do.
    plausible = work / "plausible"
    shutil.copytree(completed, plausible)
    report = plausible / "output" / "readiness.txt"
    report.write_text(report.read_text().replace("Total: 82\n", "Total: 83\n", 1))
    (plausible / "output" / "student_identity.txt").write_text("0" * 64 + "\n")
    local = value_report(plausible, checks=plausible)
    assert local == value_report(plausible) and local["score"] == 80, local
    assert sorted(failing(local)) == ["identity hash on the roster", "report: Total"], local
    urls["olga"] = publish_fork(forks, "olga", plausible)

    # A real mistake: the running total compared with the threshold instead of each measurement, and a
    # space inside each Task 2 label's quotes. Spacing is not graded, so only the two wrong lines cost points.
    running = work / "running-total"
    shutil.copytree(completed, running)
    summary = running / "measurement_summary.py"
    code = summary.read_text()
    for old, new in (("if measurement >= review_threshold:", "if total >= review_threshold:"),
                     *((f'print("{label}:", ', f'print("{label}: ", ')
                       for label in ("Measurement", "Count", "Total", "Mean", "Review count"))):
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    summary.write_text(code)
    run([sys.executable, "make_output.py"], running)
    saved = (running / "output" / "readiness.txt").read_text(encoding="utf-8").splitlines()
    assert saved[3:11] == ["Measurement:  18 within range", "Measurement:  21 review", "Measurement:  24 review",
                           "Measurement:  19 review", "Count:  4", "Total:  82", "Mean:  20.5", "Review count:  3"]
    local = value_report(running, checks=running)
    assert local == value_report(running) and local["score"] == 90, local
    assert failing(local)["report: Measurement 4"].startswith(
        "The line should read `Measurement: 19 within range`; yours reads `Measurement: 19 review`."), local
    assert sorted(failing(local)) == ["report: Measurement 4", "report: Review count"], local
    urls["tess"] = publish_fork(forks, "tess", running)

    # A fork whose folders are symlinks into a classmate's clone beside it.
    borrowed = copy_tracked("01", work / "borrowed")
    for name in ("output", "terminal-practice"):
        shutil.rmtree(borrowed / name, ignore_errors=True)
        (borrowed / name).symlink_to(Path("..") / "alice" / name)
    urls["sam"] = publish_fork(forks, "sam", borrowed)

    assert grade(destination, *urls.values()) == 0
    grades = read_grades(destination)
    assert {user: row["score"] for user, row in grades.items()} == {
        "alice": "100", "bob": "20", "carol": "0", "erin": "100", "mallory": "85", "olga": "80", "sam": "0",
        "tess": "90"}
    assert all(row["status"] == "graded" and row["max_score"] == "100" for row in grades.values())
    assert grades["alice"]["test: terminal-practice/source.txt"] == "10"
    assert grades["alice"]["test: report: Total"] == "5"
    assert grades["alice"]["test: identity hash on the roster"] == "15"
    assert len([column for column in grades["alice"] if column.startswith("test: ")]) == 16
    assert grades["alice"]["details"] == ""
    assert grades["bob"]["test: terminal-practice/path-check.txt"] == "10"
    assert grades["mallory"]["details"].startswith("identity hash on the roster: Run capture_identity.py"), grades
    assert "should read `Total: 82`; yours reads `Total: 83`" in grades["olga"]["details"], grades["olga"]["details"]
    assert grades["tess"]["test: report: Review count"] == "0" and grades["tess"]["test: report: Total"] == "5"
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
    for user, kept in (("carol", "0"), ("alice", "100"), ("mallory", "85")):
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


def test_course_repo_warnings(work: Path) -> None:
    """The script says when its checks may not be the ones on GitHub's main."""
    def git(*arguments, cwd):
        run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *arguments], cwd)
    remote, local = work / "course-remote.git", work / "course-local"
    run(["git", "init", "--quiet", "--bare", "--initial-branch=main", str(remote)], work)
    run(["git", "clone", "--quiet", str(remote), str(local)], work)
    git("commit", "--quiet", "--allow-empty", "-m", "one", cwd=local)
    git("push", "--quiet", "origin", "main", cwd=local)
    assert grade_submissions.course_repo_warnings(local) == []
    git("commit", "--quiet", "--allow-empty", "-m", "two", cwd=local)
    assert "1 local commit(s) are not on GitHub's main" in grade_submissions.course_repo_warnings(local)[0]
    git("push", "--quiet", "origin", "main", cwd=local)
    git("reset", "--quiet", "--hard", "HEAD~1", cwd=local)
    assert "run git pull" in grade_submissions.course_repo_warnings(local)[0]
    git("pull", "--quiet", "origin", "main", cwd=local)
    git("switch", "--quiet", "-c", "draft", cwd=local)
    assert "on draft, not main" in grade_submissions.course_repo_warnings(local)[0]
    git("remote", "set-url", "origin", str(work / "nowhere.git"), cwd=local)
    assert "could not compare with GitHub" in grade_submissions.course_repo_warnings(local)[-1]
    assert grade_submissions.local_time("not a date") == "not a date"


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
        test_course_repo_warnings(work)
        test_every_assignment_through_the_script(work)
    print("grade_submissions: Assignment 01 completed as written scores 100/100 from 01/assignment_checks and "
          "from its own local run; partial, untouched, forged, plausible-but-wrong, running-total, other-Python "
          "and symlinked forks score as intended from the course checks; TA edits, renamed URLs, "
          "checker failures, interrupted runs and grades.csv round trips behave; every assignment's handout "
          "grades to a zero row through the script.")


if __name__ == "__main__":
    main()
