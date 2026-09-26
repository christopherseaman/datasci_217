"""Checks for Assignment 01.

The course owns this file in 01/assignment_checks/, and the handout ships a
byte-identical copy, so `python3 check_assignment.py` gives a student the same
checks and the same advice as GitHub Actions. Every Actions run downloads the
course's current copy, and scripts/grade_submissions.py grades every fork with
it, so a correction reaches every student on their next push.

The checks read committed artifacts only: the two files in
`terminal-practice/` and the two in `output/`. Student source code is never
read, imported, or executed. Each check stands on its own: one per practice
file, one per graded line of the readiness report, and one for the identity
hash, so a mistake costs only the points for what it got wrong. Report lines
are matched in order with whitespace ignored, so a blank, extra, or missing
line never moves the lines after it out of place.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import difflib
from pathlib import Path
import re


PRACTICE_DIR = Path("terminal-practice")
PRACTICE_FILES = ("source.txt", "path-check.txt")
OUTPUT_DIR = Path("output")
READINESS_FILE = OUTPUT_DIR / "readiness.txt"
IDENTITY_FILE = OUTPUT_DIR / "student_identity.txt"

# A SHA-256 hash as capture_identity.py saves it: 64 hexadecimal digits.
IDENTITY_HASH = re.compile(r"[0-9a-f]{64}")


ROSTER_HASHES = frozenset(
    {
        "e9645171577dbd0d10eaee45d71a3ee4a392b3eaf8c582fa8c16909e2120e5d5",
        "cc1ab2723b2d220cb650406e0344706bebd86e12507507618e5673e8a4d436ad",
        "76204d1fff5c01c977cd22aa29e19b6136600c8ab9943df99b7214c80ba88762",
        "919b5177debfaa9db39d1cfa54eb38770be99c312cd76a048dc7e9bdfc344513",
        "07f66adfe2a38fb3a84a2ab2f55d40bdb5cabd85355ecb9f65290385c5cc76b8",
        "1021e732596c04d56e2aa68709e701bb7ea4136bd24c536a3c51997882dd43f2",
        "10ece8c62bc81e6caa64b78a1040b29a1604f84761ba6df6e4495bd15a463b2e",
        "18bb6954a841ec03028a44e250628fc3575e911bde64b5fd8611238d66e65a2b",
        "24a520cb993f3b75c6741cab1e0ff53376e96370a64191a559afb99faa80a6fe",
        "289dba23316b3e2b10b677f966b3ec29a5277ced8af256bb54ea3d14b0b5e00e",
        "39ab2539040fa52910bcff9b9ede5dd578adf9a4c21edd7af70fd236de85ff0b",
        "40bc50894e9238057d3f61d96c89c43fde656a0daa4e57cc297b7a4b41ea9822",
        "4231dafea7f15471053e731f9f583447cf1a97fa98edf01ec95e9681321c6270",
        "49cc715f310f0ba0e0fc33a23db7dcc9656ac312c24d32f9424dc8c333a22f29",
        "5cbf859665a8ea51a585d25a929b4d0ecceb619e17e2413d4ad29fa866dcb88e",
        "5eaf3d8fedcbaa1a0230e3886444a36b0c2c8e35fa519862e1def1ca829717fb",
        "62b4a37e8eb0ba7d6f6a8c97cacc3460c4fc0344ed3437ebc5b893e77b854676",
        "62cea3fcba60444e3c3d22fada8814ede08a4576e2294d7668986280719d5384",
        "65744a984caca19adf80e29930bab1d0edfb3009e54f9427ec9747c25073d84f",
        "724c02d046a6f4c13cd76d29a9297cc1fbe27775e0e7ba6680b151e48e634bd5",
        "7ae0d4645e632ac8e54f92bb38adf30b819e82dff1f71476d7b795cc8ae84223",
        "819dd0b63f9c1cb9cc1dc20e11d099e8b7d06c499314000c27600024f5e35220",
        "8c7aacd0f592fbff2ce69a5cfd51bbd136133752c6b11960fe9849e3dfff0aff",
        "919b5177debfaa9db39d1cfa54eb38770be99c312cd76a048dc7e9bdfc344513",
        "94fb8f3cc19b9e6599824fb7a6ed055a2be650221a8289af1d10971fb33ab0c3",
        "97983c85fbb08e26a8e7c05829d614313c94b46bb8d3453b76dee3fb1cb408b8",
        "9cfe029fece611598ed5d758b7a537c9d9d65d182c21a2a06fe82273a5e77b8e",
        "9d6815a23b72f3770f107a5371926d63a01d09e5ce13eb8651d8bc7c76a2ff68",
        "9e40278b71e32198b63bd6e91b33257f6b39cc49fa8368478c65b1c9e70bf086",
        "9fd76c7d1eafa348c5c325d9b1f6c1b91091875444c0b0d6514be57be272fa2b",
        "a128b3dd4426613a4c9ecd0e29b27d2a26bd51d4e713605e36495f02b079e6b1",
        "aacfa06ad5191c6ca7c27bf740e5522ff1f696547c34730e0278459036f7e5bb",
        "abcd16e55a833d6423e31cc2d5821dadb6faa5bcfb0e932f4549d4f8570395b1",
        "ac21180378214fde49624883c961affef8b76e6e3ee17934fb32182a65e4df17",
        "aff17118cad18b223ff2d3e6483d55a0fc0436812463a310e237b4e6d65e6de5",
        "b1968e7d557ab56b575533fe2250a0e85634c70cd7de9b966b4e94097e39b6a2",
        "b21f05d20a6f9a585ab42ffce153a3209943dee30a6f1e3b58c110c719113a54",
        "c0a3423603e25aee7189168c4117b0e619c79f43b95cbf27877dcd1078605f7b",
        "cabb9a5e2202f47f3da834e7a69c59b238cdf438c153d14d4920454a32fc6ef4",
        "d6d383fdcf137c98e15d60afd2312076168e3ddd6428a478c6b46f63e3c276ba",
        "dc0dbc8c08708f765945d5f06bb6fc73f8d92234ff291652ec645c8548e467a5",
        "e6ca7c71e53b865114afa8c65561b40edabb8ff1d6972b6e4a85d49cd3263ef3",
        "f79c0262a92f5def6ae726c8850be7b29b6d63977a8e6dbddf90c8756efd8226",
        "fee59a908616085e5e356ae776eb1955ba84a4bcd97be649c7749be4ea294346",
    }
)


EXPECTED_READINESS = """Python family: 3.13
Project: DataSci 217 Assignment 01
Script: readiness.py
Measurement: 18 within range
Measurement: 21 review
Measurement: 24 review
Measurement: 19 within range
Count: 4
Total: 82
Mean: 20.5
Review count: 2
Readiness: complete
Participant count: 4
Next checkpoint: 5
"""

# The first line records whichever Python ran readiness.py and is never graded; every other line is.
GRADED_LINES = tuple(EXPECTED_READINESS.splitlines()[1:])
# The script that prints each graded line, in report order, and the task that completes each script.
LINE_SOURCES = (
    ("readiness.py",) * 2 + ("measurement_summary.py",) * 8 + ("debug_report.py",) * 3
)
SCRIPT_TASKS = {"readiness.py": "Task 1.2", "measurement_summary.py": "Task 2.1", "debug_report.py": "Task 3.1"}
# A saved line is shown in feedback up to this many characters.
SHOWN_LENGTH = 80


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]
    # The file the check reads and the check's name within it, for the report's `Left to fix` line.
    artifact: str = ""
    label: str = ""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _shown(text: str) -> str:
    return text if len(text) <= SHOWN_LENGTH else text[: SHOWN_LENGTH - 3] + "..."


def _is_regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _file_state(path: Path) -> str:
    """What stands where a file belongs, in plain words."""
    if path.is_symlink():
        return "is a link, not a file"
    if path.is_dir():
        return "is a folder, not a file"
    return "is not a regular file" if path.exists() else "is missing"


def _folder_state(path: Path) -> str:
    """What stands where a folder belongs, in plain words."""
    if path.is_symlink():
        return "is a link, not a folder"
    return "is a file, not a folder" if path.exists() else "is missing"


def _look_alikes(root: Path, relative: Path) -> list[str]:
    """Files of the same type beside the wanted one, or at the assignment root, named almost the same."""
    wanted = root / relative
    found = []
    for folder in dict.fromkeys((wanted.parent, root)):
        if not folder.is_dir() or folder.is_symlink():
            continue
        for path in sorted(folder.iterdir()):
            if (
                path != wanted
                and _is_regular_file(path)
                and path.suffix.casefold() == wanted.suffix.casefold()
                and difflib.SequenceMatcher(None, relative.name.casefold(), path.name.casefold()).ratio() >= 0.8
            ):
                found.append(path.relative_to(root).as_posix())
    return found


def _not_saved(root: Path, relative: Path, save: str) -> str:
    """Say what stands where an artifact belongs, and how to save it; `save` is the step that saves it."""
    path, name = root / relative, relative.as_posix()
    if path.exists() or path.is_symlink():
        return f"{name} {_file_state(path)}. Delete it, then {save}, and commit the new file."
    found = _look_alikes(root, relative)
    if found:
        return f"{name} is missing. Found {_join(found)}; rename or move it to {name}, then commit it."
    return f"{name} is missing. {save[0].upper()}{save[1:]}, then commit it."


def _read_text(root: Path, relative: Path, missing: Callable[[], str], save: str) -> str:
    """Read a committed artifact as UTF-8 text; `missing()` says why when it is not a regular file."""
    path = root / relative
    if not _is_regular_file(path):
        raise AssertionError(missing())
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise AssertionError(
            f"{relative.as_posix()} is not saved as UTF-8 text, so it cannot be read. "
            f"{save[0].upper()}{save[1:]} again, then commit it."
        ) from None
    except OSError as error:
        raise AssertionError(
            f"{relative.as_posix()} cannot be opened ({error.strerror or 'unreadable'}). "
            f"{save[0].upper()}{save[1:]} again, then commit it."
        ) from None


def _output_dir(root: Path) -> None:
    output = root / OUTPUT_DIR
    if output.is_dir() and not output.is_symlink():
        return
    state = _folder_state(output)
    if state == "is missing":
        raise AssertionError(
            "output/ is missing. Run make_output.py (Task 3.2) and capture_identity.py (Task 3.3), "
            "which create it, then commit the two files they save."
        )
    raise AssertionError(
        f"output/ {state}. Delete it, then run make_output.py (Task 3.2) and capture_identity.py "
        "(Task 3.3), which create the folder, and commit the two files they save."
    )


REPORT_SAVE = "run make_output.py (Task 3.2) to save the report"


def _readiness_report(root: Path) -> str:
    _output_dir(root)
    report = _read_text(root, READINESS_FILE, lambda: _not_saved(root, READINESS_FILE, REPORT_SAVE), REPORT_SAVE)
    _assert(
        bool(report.strip()),
        "output/readiness.txt is empty. Run make_output.py (Task 3.2) once all three scripts run cleanly, "
        "then commit the report it saves.",
    )
    return report


def _without_whitespace(line: str) -> str:
    """The line with every whitespace character removed, which is how report lines are compared."""
    return "".join(line.split())


def _label(line: str) -> str:
    """The text before a line's first colon, in any letter case, which pairs a wrong line with its expected one."""
    return _without_whitespace(line).partition(":")[0].casefold()


def _unmatched_lines(report: str) -> dict[int, tuple[str | None, bool]]:
    """Each graded line the report lacks, by index, mapped to the report's line to show in its place.

    The report's non-blank lines are aligned in order with the graded lines, whitespace ignored.
    A graded line passes when the alignment matches it, so an extra line (the Python version, a
    blank line, a leftover debug print) costs nothing and a missing line costs only itself.

    A line that fails is shown beside a line from the stretch of report between the same matched
    lines: first the line with the same label, marked True; otherwise, when the rest of the
    stretch holds exactly as many lines as are left to show and does not start the report, the
    line in the same place, marked False. With neither, the line is paired with None.
    """
    yours = [" ".join(line.split()) for line in report.split("\n")]
    yours = [line for line in yours if line]
    matcher = difflib.SequenceMatcher(
        None,
        [_without_whitespace(line) for line in GRADED_LINES],
        [_without_whitespace(line) for line in yours],
        autojunk=False,
    )
    unmatched: dict[int, tuple[str | None, bool]] = {}
    for tag, first, last, your_first, your_last in matcher.get_opcodes():
        if tag == "equal":
            continue
        candidates = yours[your_first:your_last]
        unlabelled = []
        for index in range(first, last):
            label = _label(GRADED_LINES[index])
            shown = next((line for line in candidates if _label(line) == label), None)
            if shown is not None:
                candidates.remove(shown)
                unmatched[index] = (shown, True)
            else:
                unlabelled.append(index)
        paired = your_first > 0 and len(candidates) == len(unlabelled)
        for position, index in enumerate(unlabelled):
            unmatched[index] = (candidates[position] if paired else None, False)
    return unmatched


def _between(index: int) -> str:
    """Where a graded line belongs, named by the graded lines around it."""
    before = GRADED_LINES[index - 1] if index > 0 else None
    after = GRADED_LINES[index + 1] if index + 1 < len(GRADED_LINES) else None
    if before and after:
        return f"between `{before}` and `{after}`"
    return f"before `{after}`" if after else f"after `{before}`"


IDENTITY_SAVE = "run capture_identity.py (Task 3.3) with your course roster email"


def _identity_hash(root: Path) -> str:
    """The saved hash with surrounding whitespace and letter case ignored."""

    def missing() -> str:
        path = root / IDENTITY_FILE
        if path.exists() or path.is_symlink():
            return _not_saved(root, IDENTITY_FILE, IDENTITY_SAVE)
        found = _look_alikes(root, IDENTITY_FILE)
        return (
            "Run capture_identity.py (Task 3.3) with your course roster email, then commit the file it saves: "
            f"{IDENTITY_FILE.as_posix()} is missing."
            + (f" Found {_join(found)}; rename or move it to {IDENTITY_FILE.as_posix()}." if found else "")
        )

    text = _read_text(root, IDENTITY_FILE, missing, IDENTITY_SAVE)
    identity_hash = text.strip().lower()
    if IDENTITY_HASH.fullmatch(identity_hash) is None:
        saved = text.strip()
        if not saved:
            found = "but the file is empty"
        elif "@" in saved:
            found = "but it holds an email address instead; keep your email out of your files and commits"
        elif len(saved.splitlines()) > 1:
            found = f"but it holds {len(saved.splitlines())} lines"
        else:
            found = f"but it holds `{_shown(saved)}`"
        raise AssertionError(
            f"{IDENTITY_FILE.as_posix()} should hold one SHA-256 hash, the 64 characters capture_identity.py "
            f"saves, {found}. Rerun capture_identity.py (Task 3.3) with your course roster email, then commit "
            "the file it saves."
        )
    return identity_hash


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def practice_file_check(name: str) -> Callable[[Path], None]:
    relative = PRACTICE_DIR / name
    commands = "the Task 1.1 commands from the folder that holds README.md"

    def check(root: Path) -> None:
        practice = root / PRACTICE_DIR
        if not practice.is_dir() or practice.is_symlink():
            state = _folder_state(practice)
            start = "Run" if state == "is missing" else "Delete it, then run"
            raise AssertionError(
                f"terminal-practice/ {state}. {start} {commands} to create it with source.txt and "
                "path-check.txt, then commit both files."
            )
        path = root / relative
        if _is_regular_file(path):
            return
        if path.exists() or path.is_symlink():
            raise AssertionError(
                f"{relative.as_posix()} {_file_state(path)}. Delete it, then run {commands} again and commit "
                "the empty file they create."
            )
        found = _look_alikes(root, relative)
        if found:
            raise AssertionError(
                f"{relative.as_posix()} is missing. Found {_join(found)}; rename or move it to "
                f"{relative.as_posix()}, then commit it."
            )
        raise AssertionError(
            f"{relative.as_posix()} is missing. Run {commands} again, then commit the empty file they create."
        )

    return check


def report_line_check(index: int) -> Callable[[Path], None]:
    expected, source = GRADED_LINES[index], LINE_SOURCES[index]
    fix = (
        f"Fix {source} ({SCRIPT_TASKS[source]}), which prints this line, then rerun make_output.py "
        "(Task 3.2) and commit output/readiness.txt."
    )

    def check(root: Path) -> None:
        report = _readiness_report(root)
        unmatched = _unmatched_lines(report)
        if index not in unmatched:
            return
        yours, same_label = unmatched[index]
        if yours is None:
            elsewhere = _without_whitespace(expected) in map(_without_whitespace, report.split("\n"))
            raise AssertionError(
                f"output/readiness.txt has `{expected}`, but out of order: it belongs {_between(index)}. {fix}"
                if elsewhere
                else f"output/readiness.txt has no line reading `{expected}` {_between(index)}. {fix}"
            )
        place = "yours reads" if same_label else "in its place yours reads"
        raise AssertionError(f"The line should read `{expected}`; {place} `{_shown(yours)}`. {fix}")

    return check


def check_identity(root: Path) -> None:
    _output_dir(root)
    _assert(
        _identity_hash(root) in ROSTER_HASHES,
        f"{IDENTITY_FILE.as_posix()} holds a hash that is not on the course roster. Rerun capture_identity.py "
        "(Task 3.3) with the email address the course roster lists for you, then commit the file it saves; "
        "contact the course team if it still does not match.",
    )


def _line_names() -> list[str]:
    """`report: <label>` for each graded line, numbering the four measurements."""
    names, measurements = [], 0
    for line in GRADED_LINES:
        label = line.partition(":")[0]
        if label == "Measurement":
            measurements += 1
            label = f"Measurement {measurements}"
        names.append(f"report: {label}")
    return names


CHECKS = (
    *(
        Check(f"terminal-practice/{name}", practice_file_check(name),
              (PRACTICE_DIR / name).as_posix(), (PRACTICE_DIR / name).as_posix())
        for name in PRACTICE_FILES
    ),
    *(
        Check(name, report_line_check(index), READINESS_FILE.as_posix(), name.removeprefix("report: "))
        for index, name in enumerate(_line_names())
    ),
    Check("identity hash on the roster", check_identity, IDENTITY_FILE.as_posix(), "identity hash on the roster"),
)


def _unreadable(root: Path, error: OSError) -> str:
    """An error opening a file, in plain words."""
    name = error.filename
    try:
        name = Path(name).relative_to(root).as_posix()
    except (TypeError, ValueError):
        pass
    return f"{name or 'A file'} cannot be opened ({error.strerror or 'unreadable'}); save it again, then commit it."


def run_checks(root: Path) -> list[tuple[str, str | None]]:
    results = []
    for check in CHECKS:
        try:
            check.action(root)
        except AssertionError as error:
            results.append((check.name, str(error)))
        except OSError as error:
            results.append((check.name, _unreadable(root, error)))
        else:
            results.append((check.name, None))
    return results
