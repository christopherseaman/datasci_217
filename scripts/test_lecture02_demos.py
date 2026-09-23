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
SOURCES = ("functions_demo.py", "vitals_tools.py", "module_usage_demo.py", "clinic_vitals.csv")
PROMPT = "Flag systolic at or above (press Enter for 130): "
REPORT = (
    "P001: 128 mmHg\nP002: 142 mmHg\nP004: 136 mmHg\n"
    "Average systolic: 135.3 mmHg\nHighest systolic: 142 mmHg\n"
)


def fences(language):
    """Every fenced block of one language in DEMO_GUIDE.md, in document order."""
    guide = (DEMOS / "DEMO_GUIDE.md").read_text(encoding="utf-8")
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


def terminal(argv, cwd, typed):
    """Run an interactive script under a pty; the transcript includes the typed echo."""
    master, slave = pty.openpty()
    process = subprocess.Popen(argv, cwd=cwd, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    chunks, answered = [], False
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
    assert process.wait() == 0
    return b"".join(chunks).decode().replace("\r\n", "\n")


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
    # Line-exact, so the guide cannot print an edit at an indentation a student cannot paste.
    assert original in sources["module_usage_demo.py"].splitlines(keepends=True), (
        f"the guide prints the checkpoint edit as a line the script does not have:\n{edit}"
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
    mismatch = only(fences("text"), "Saved report matches: False")
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
        assert failed.stdout.endswith(mismatch)
        assert failed.stderr.splitlines()[-1] == failure.rstrip("\n")

    assert demo_sources() == before, "a demo run changed the course sources"
    print("Lecture 02: guide transcripts, excerpts, bad rows, saved report, and checkpoints passed.")


if __name__ == "__main__":
    run()
