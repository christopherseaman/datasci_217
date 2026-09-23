"""Run Lecture 02 demos in isolation and check the guide's expected output against real runs."""

from hashlib import sha256
from pathlib import Path
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "02" / "demo"
GUIDE = DEMOS / "DEMO_GUIDE.md"
LECTURE = ROOT / "02" / "README.md"
BONUS = ROOT / "02" / "BONUS.md"
LECTURE_01 = ROOT / "01" / "README.md"
SOURCES = ("functions_demo.py", "vitals_tools.py", "module_usage_demo.py", "clinic_vitals.csv")
PROMPT = "Flag systolic at or above (press Enter for 130): "
REPORT = (
    "P001: 128 mmHg\nP002: 142 mmHg\nP004: 136 mmHg\n"
    "Average systolic: 135.3 mmHg\nHighest systolic: 142 mmHg\n"
)
# What a student types between the guide's empty quotes.
IDENTITY = {"user.name": "Test Student", "user.email": "12345+tstudent@users.noreply.github.com"}
# A student pastes into an interactive shell. Bash, and Zsh (the macOS default) with its
# default options, in which `#` starts no comment in a typed or pasted line.
SHELLS = {
    "bash": ["bash", "--norc", "--noprofile", "--noediting", "-i", "-s"],
    "zsh": ["zsh", "-f", "+o", "promptsp", "-i", "-s"],
}
HASH = re.compile(r"^[0-9a-f]{7,40} ", re.M)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
MARK = "--- next command ---"


def fences(language):
    """Every fenced block of one language in DEMO_GUIDE.md, in document order."""
    guide = GUIDE.read_text(encoding="utf-8")
    return re.findall(r"^```%s\n(.*?)^```$" % language, guide, re.M | re.S)


def only(blocks, needle):
    """The one block containing needle; guide edits that duplicate it fail here."""
    found = [block for block in blocks if needle in block]
    assert len(found) == 1, f"{needle!r} matched {len(found)} blocks"
    return found[0]


def demo_sources():
    """Hash the tracked demo sources so a run cannot change them unnoticed."""
    return {
        path.relative_to(DEMOS).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(DEMOS.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and "output" not in path.parts
    }


def terminal(argv, cwd, typed=None, env=None, check=True, script=None):
    """Run a command under a pty; the transcript includes any typed echo.

    A script arrives through a pipe instead, so the transcript holds only what it prints.
    """
    master, slave = pty.openpty()
    stdin = slave if script is None else subprocess.PIPE
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=stdin, stdout=slave, stderr=slave)
    os.close(slave)
    if script is not None:                    # small enough for the pipe buffer
        process.stdin.write(script.encode())
        process.stdin.close()
    chunks, answered = [], typed is None
    while select.select([master], [], [], 30)[0]:
        try:
            data = os.read(master, 4096)
        except OSError:                       # the child closed the pty and exited
            break
        if not data:
            break
        chunks.append(data)
        if not answered and PROMPT.encode() in b"".join(chunks):
            answered = True                   # answer only once the prompt is on screen
            os.write(master, typed.encode())
    os.close(master)
    status = process.wait()
    assert status == 0 or not check, f"{argv} exited {status}"
    return b"".join(chunks).decode().replace("\r\n", "\n")


def pasted(block, cwd, env, shell):
    """Paste one guide bash block into an interactive shell; return (command, output) pairs.

    The terminal matters: it is what makes `git log` label branches for a student.
    """
    commands = [line for line in block.splitlines() if line.strip()]
    script = "".join(f"echo '{MARK}'\n{command}\n" for command in [*commands, ""])
    transcript = terminal(SHELLS[shell], cwd, env=env, check=False, script=script)
    outputs = ANSI.sub("", transcript).split(MARK + "\n")[1:]
    assert len(outputs) == len(commands) + 1, transcript   # the last is the shell leaving
    return list(zip(commands, outputs))


def output_of(steps, start):
    """The output of the one command that starts with start."""
    found = [output for command, output in steps if command.startswith(start)]
    assert len(found) == 1, f"{start!r} ran {len(found)} times"
    return found[0]


def lecture_block(heading):
    """The first text block under a Lecture 02 heading."""
    lecture = LECTURE.read_text(encoding="utf-8")
    return re.search(r"^%s\n.*?^```text\n(.*?)^```$" % re.escape(heading), lecture, re.M | re.S).group(1)


def annotated(block):
    """The output column of a text block that labels each output line with `<-`."""
    return [line.split("<-")[0].rstrip() for line in block.splitlines()]


def check_git_demo(scratch, shell):
    """Demo 1 in a fresh home with no Git identity, each block pasted in guide order."""
    guide = GUIDE.read_text(encoding="utf-8")
    for untaught in ("cd -", "git diff --", "Git: Merge Branch"):
        assert untaught not in guide, f"the guide still uses {untaught!r}"
    blocks = fences("bash")
    for line in "".join(blocks).splitlines():
        assert not re.search(r"(^|\s)#", line), f"Zsh passes # to the command, not a comment: {line!r}"
    setup = only(blocks, "git init")
    identity = only(blocks, "git config user.name")
    first = only(blocks, 'git commit -m "Start practice notes"')
    conflict = only(blocks, "Experiment: compare median systolic.")
    resolve = only(blocks, "git commit -m \"Merge branch 'experiment'\"")
    ignore = only(blocks, ".gitignore")
    order = [blocks.index(block) for block in (setup, identity, first, conflict, resolve, ignore)]
    assert order == sorted(order), "Demo 1's blocks are out of order"

    # The identity step is Lecture 01's two lines with the values left for the student.
    lecture_01 = [line for line in LECTURE_01.read_text(encoding="utf-8").splitlines()
                  if line.startswith("git config user.")]
    assert identity.splitlines() == [re.sub(r'"[^"]*"', '""', line) for line in lecture_01], identity

    def student(name):
        home = scratch / name
        home.mkdir()
        # useConfigOnly: fail on a missing identity even where the hostname would supply one.
        env = {"HOME": str(home), "PATH": os.environ["PATH"], "LC_ALL": "C", "TERM": "dumb",
               "PS1": "", "PS2": "", "GIT_PAGER": "cat", "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "user.useConfigOnly",
               "GIT_CONFIG_VALUE_0": "true"}
        return home / "ds217-practice", env

    def first_commit(name, identity_step=None):
        practice, env = student(name)
        pasted(setup, scratch, env, shell)
        if identity_step:
            pasted(identity_step, practice, env, shell)
        return output_of(pasted(first, practice, env, shell), 'git commit -m "Start')

    # Without the identity step the first commit can fail, as the guide warns; pasted
    # unedited, the empty name fails loudly instead of committing a placeholder.
    assert "Author identity unknown" in first_commit("no-identity")
    assert "fatal: empty ident name" in first_commit("unedited", identity)

    practice, env = student("home")
    pasted(setup, scratch, env, shell)
    filled = identity
    for key, value in IDENTITY.items():
        filled = filled.replace(f'{key} ""', f'{key} "{value}"')
    assert '""' not in filled, filled
    pasted(filled, practice, env, shell)

    # Working, staged, committed: the three short statuses the guide prints.
    steps = pasted(first, practice, env, shell)
    statuses = [output.rstrip("\n") for command, output in steps if command == "git status --short"]
    assert annotated(only(fences("text"), "<- working tree")) == statuses
    fast_forward = only(fences("text"), "(HEAD -> main, experiment)")
    assert HASH.sub("<hash> ", output_of(steps, "git log --oneline")) == fast_forward

    # The conflict is the lecture's example, marker for marker.
    steps = pasted(conflict, practice, env, shell)
    stops, unmerged = re.search(
        r"`git merge` stops with `([^`]+)`, and `git status --short` prints `([^`]+)`", guide
    ).groups()
    assert stops + "\n" in output_of(steps, "git merge")
    assert output_of(steps, "git status --short") == unmerged + "\n"
    markers = only(fences("text"), "<<<<<<< HEAD")
    assert markers == lecture_block("## Merge Conflicts")
    assert output_of(steps, "cat notes.md") == markers

    # The terminal resolution leaves what VS Code's Accept Incoming Change would.
    pasted(resolve, practice, env, shell)
    incoming = re.sub(r"<<<<<<< HEAD\n.*?=======\n(.*?)>>>>>>> experiment\n", r"\1", markers, flags=re.S)
    assert (practice / "notes.md").read_text(encoding="utf-8") == incoming
    steps = pasted("git status --short\ngit log --oneline\ngit log --format=%an", practice, env, shell)
    assert output_of(steps, "git status") == ""
    merged = only(fences("text"), "(HEAD -> main) Merge branch 'experiment'")
    assert HASH.sub("<hash> ", output_of(steps, "git log --oneline")) == merged
    assert set(output_of(steps, "git log --format").splitlines()) == {IDENTITY["user.name"]}

    # .gitignore hides the export with a pattern the lecture teaches; the file stays on disk.
    steps = pasted(ignore, practice, env, shell)
    statuses = [output.rstrip("\n") for command, output in steps if command == "git status --short"]
    assert annotated(only(fences("text"), "?? raw_vitals.csv")) == statuses
    pattern = re.search(r'^echo "(.+)" > \.gitignore$', ignore, re.M).group(1)
    lecture = LECTURE.read_text(encoding="utf-8")
    card = lecture[lecture.index("### Reference Card: Ignore Patterns"):]
    assert f"- `{pattern}`:" in card, f"{pattern} is not on the lecture's ignore card"
    assert (practice / "raw_vitals.csv").exists()

    # Publishing sends every commit on main, so GitHub's count matches the guide's.
    count = re.search(r"\(\*\*(\d+) Commits\*\*\) for the same", guide).group(1)
    steps = pasted("git rev-list --count HEAD", practice, env, shell)
    assert output_of(steps, "git rev-list") == count + "\n"

    # Less typing: `cat notes.md` shows the merged notes; `git diff notes.md` shows nothing.
    steps = pasted("cat notes.md\ngit diff notes.md", practice, env, shell)
    assert steps == [("cat notes.md", incoming), ("git diff notes.md", "")]


def check_debugger(demo, python):
    """The bonus page's breakpoint exercise pauses where it says, showing the values it names."""
    guide = BONUS.read_text(encoding="utf-8")
    number, code = re.search(r"click left of line (\d+), `([^`]+)`", guide).groups()
    script_lines = (demo / "module_usage_demo.py").read_text(encoding="utf-8").splitlines()
    assert script_lines[int(number) - 1].strip() == code, f"line {number} is not {code}"
    for shown in ("`patient_id: 'P001'`", "`raw_systolic: '128'`", "`systolic: 128`", "`'P002'`"):
        assert shown in guide
    # pdb steps like VS Code: pause before the line, F10 is `next`, F5 is `continue`.
    steps = f"b {number}\nc\np patient_id, raw_systolic, 'systolic' in locals()\nn\np systolic\nc\np patient_id\nq\n"
    paused = python("-m", "pdb", "module_usage_demo.py", stdin=steps).stdout
    printed = re.findall(r"^\(Pdb\) (?!>|Breakpoint)(.+)$", paused, re.M)   # values, not pdb's notices
    assert printed == ["('P001', '128', False)", "128", "'P002'"], paused
    # The debugger runs from the folder open in VS Code; with the clone's root open, it never pauses.
    clone_root = demo / "clone-root"
    clone_root.mkdir()
    elsewhere = python("-m", "pdb", str(demo / "module_usage_demo.py"), cwd=clone_root,
                       stdin=f"b {number}\nc\nq\n").stdout
    assert "Cannot find clinic_vitals.csv" in elsewhere and code not in elsewhere


def quotes_real_code(block, source):
    """True when the block is a run of source lines with only narration prints left out.

    Excerpting a `print()` keeps the transcript readable; leaving out anything else
    changes what the quoted code does, which is what the guide promises it shows.
    """
    lines = source.splitlines()
    position = -1
    for line in block.splitlines():
        if not line.strip():
            continue
        try:
            found = lines.index(line, position + 1)
        except ValueError:
            return False
        gap = lines[position + 1:found] if position >= 0 else []
        if any(skipped.strip() and not skipped.strip().startswith("print(") for skipped in gap):
            return False
        position = found
    return position >= 0


def check_guide_code(sources):
    """Every block the guide quotes as demo code is code the demo really runs."""
    edit = only(fences("python"), ".upper()")
    original = edit.replace(".upper()", "")
    # Contiguous and indentation-exact, so a student can paste the block as printed.
    # A published page can flatten the indent of a block that starts mid-nesting, so the
    # guide quotes enough context for its first line to sit at its own indent level.
    assert original.rstrip("\n") in sources["module_usage_demo.py"], (
        f"the guide prints the checkpoint edit as code the script does not have:\n{edit}"
    )
    for block in fences("python"):
        if block == edit:                     # the intentional edit, checked above
            continue
        variants = [
            "\n".join(indent + line if line.strip() else line for line in block.splitlines())
            for indent in ("", "    ", "        ")   # excerpts are quoted without their nesting
        ]
        if not any(quotes_real_code(variant, source)
                   for variant in variants for source in sources.values()):
            raise AssertionError(f"the guide quotes code no demo runs as written:\n{block}")
    return edit, original


def check_excerpts_run(sources, demo, python, summary):
    """The Demo 2 excerpts run in the order the guide prints them, defining every name they use.

    A student reading down the guide meets these blocks in this order, so a block that
    uses a name no earlier block builds is a block they cannot run.
    """
    imports = "".join(line for line in sources["functions_demo.py"].splitlines(keepends=True)
                      if line.startswith(("import ", "from ")))
    excerpts = [block for block in fences("python")
                if quotes_real_code(block, sources["functions_demo.py"])]
    (demo / "guide_excerpts.py").write_text(imports + "\n" + "\n".join(excerpts), encoding="utf-8")
    run = python("guide_excerpts.py", stdin="120\n", check=False)
    assert run.returncode == 0, f"the guide's excerpts do not run in order:\n{run.stderr}"
    printed = run.stdout.replace(PROMPT, "").splitlines()
    transcript = summary.splitlines()
    for line in printed:
        assert line in transcript, f"the excerpts printed a line the transcript lacks:\n{line}"
    assert printed[-1] == transcript[-1], "the excerpts stop before the flagged comparison"


def run():
    before = demo_sources()
    sources = {name: (DEMOS / name).read_text(encoding="utf-8") for name in SOURCES}
    edit, original = check_guide_code(sources)

    summary = only(fences("text"), "=== Bayview Clinic 2026-09-18")
    cutoff_again = only(fences("text"), "Flagged (130 mmHg and above)")
    csv_preview = only(fences("text"), "P003,not recorded\n")
    demo_three = only(fences("text"), "Checkpoint passed:")
    mismatch = re.search(r"`(Saved report matches: False)`", GUIDE.read_text(encoding="utf-8")).group(1)
    failure = only(fences("text"), "AssertionError:")
    blank_row = only(fences("text"), "Skipping a blank row.")
    too_many_fields = only(fences("text"), "Skipping a row with 3 fields")
    missing = only(fences("text"), "Cannot find clinic_vitals.csv")
    skipped = demo_three.splitlines()[0] + "\n"

    assert csv_preview == sources["clinic_vitals.csv"], "the guide's CSV is not the shipped CSV"

    with tempfile.TemporaryDirectory(dir=ROOT / "scratch", prefix="lecture02-") as temporary:
        demo = Path(temporary)
        for name in SOURCES:
            shutil.copy2(DEMOS / name, demo / name)

        def python(*arguments, cwd=demo, check=True, stdin=""):
            return subprocess.run(
                [sys.executable, *arguments], cwd=cwd, check=check,
                input=stdin, capture_output=True, text=True,
            )

        # Demo 1: the Git workflow, conflict, and .gitignore, as a student pastes them.
        for shell in SHELLS:
            if not shutil.which(shell):
                print(f"{shell} is not installed: Demo 1 was not pasted into {shell}.")
                continue
            git = demo / f"git-{shell}"
            git.mkdir()
            check_git_demo(git, shell)

        # Demo 3's import step is silent: no output, no report.
        assert python("-c", "import vitals_tools; import module_usage_demo").stdout == ""
        assert not (demo / "output").exists()

        # Demo 2 in a terminal: the guide's transcript is what a student sees, echo included.
        assert terminal([sys.executable, "functions_demo.py"], demo, "120\n") == summary
        assert terminal([sys.executable, "functions_demo.py"], demo, "\n") == (
            summary[:summary.index(PROMPT)] + cutoff_again
        )

        # Piped instead of typed: the same run, minus the echo the terminal adds, and no files.
        piped = python("functions_demo.py", stdin="\n").stdout
        assert piped == (summary[:summary.index(PROMPT)] + cutoff_again).replace(
            PROMPT + "\n", PROMPT
        )
        assert python("functions_demo.py", stdin="\n").stdout == piped
        assert not (demo / "output").exists()

        # Read down the guide instead: the Demo 2 excerpts run on their own, in order.
        check_excerpts_run(sources, demo, python, summary)

        # The helpers say "no result" with None, and 0.0 is a real average, not a missing one.
        python("-c", "from vitals_tools import mean_reading, highest_reading, get_systolic; "
               "assert mean_reading([]) is None; assert highest_reading([]) is None; "
               "assert mean_reading([0, 0]) == 0.0; assert get_systolic([]) == []")

        # Demo 3 skips the unusable reading, saves the report, and repeats identically.
        report = demo / "output" / "vitals_report.txt"
        for _ in range(2):
            assert python("module_usage_demo.py").stdout == demo_three
            assert report.read_text(encoding="utf-8") == REPORT

        # A blank row is reported and skipped, wherever the export put it.
        rows = sources["clinic_vitals.csv"]
        for csv_text in (rows + "\n", rows.replace("P003", "\nP003", 1)):
            (demo / "clinic_vitals.csv").write_text(csv_text, encoding="utf-8")
            output = python("module_usage_demo.py").stdout
            assert "Skipping a blank row.\n" in output
            assert output.replace("Skipping a blank row.\n", "") == demo_three
            assert report.read_text(encoding="utf-8") == REPORT
        (demo / "clinic_vitals.csv").write_text(rows + "\n", encoding="utf-8")
        assert python("module_usage_demo.py").stdout.startswith(blank_row)

        # A row with one comma too many is reported like the unusable grade, not a crash.
        (demo / "clinic_vitals.csv").write_text(rows + "P005,134,extra\n", encoding="utf-8")
        extra_field = python("module_usage_demo.py").stdout
        assert extra_field.startswith(too_many_fields)
        assert extra_field.replace(too_many_fields.splitlines(keepends=True)[1], "") == demo_three
        assert report.read_text(encoding="utf-8") == REPORT

        # Blank rows alone leave nothing to average, and the first assert says so.
        (demo / "clinic_vitals.csv").write_text("patient_id,systolic\n\n\n", encoding="utf-8")
        empty = python("module_usage_demo.py", check=False)
        assert empty.returncode == 1
        assert empty.stderr.splitlines()[-1] == (
            "AssertionError: no usable readings in clinic_vitals.csv"
        )
        (demo / "clinic_vitals.csv").write_text(rows, encoding="utf-8")

        check_debugger(demo, python)

        # Run from a folder without the data file: one message, no traceback.
        elsewhere = demo / "wrong-folder"
        elsewhere.mkdir()
        for name in ("module_usage_demo.py", "vitals_tools.py"):
            shutil.copy2(demo / name, elsewhere / name)
        assert python("module_usage_demo.py", cwd=elsewhere).stdout == missing

        # The guide's intentional error: saving different text prints False and trips the assert.
        broken = demo / "broken"
        broken.mkdir()
        for name in SOURCES:
            shutil.copy2(demo / name, broken / name)
        (broken / "module_usage_demo.py").write_text(
            sources["module_usage_demo.py"].replace(original, edit), encoding="utf-8",
        )
        failed = python("module_usage_demo.py", cwd=broken, check=False)
        assert failed.returncode == 1
        assert failed.stdout.startswith(
            skipped + "Read back from output/vitals_report.txt:\nP001: 128 MMHG\n"
        )
        assert failed.stdout.endswith(mismatch + "\n")
        assert failed.stderr.splitlines()[-1] == failure.rstrip("\n")

    assert demo_sources() == before, "a demo run changed the course sources"
    print("Lecture 02: Git workflow, conflict, .gitignore, guide transcripts, excerpts, bad rows, "
          "debugger pause, saved report, and checkpoints passed.")


if __name__ == "__main__":
    run()
