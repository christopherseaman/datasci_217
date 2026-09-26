"""Checks for Assignment 02.

Every expected value is recomputed from the supplied encounter file, so there is
no answer key here. The course owns this file in 02/assignment_checks/, the
handout ships a byte-identical copy so students can run the checks locally, and
the GitHub Actions run downloads the course's current copy on every push.

The checks read committed artifacts only: `README.md`, `.gitignore`,
`data/clinic_encounters.csv`, and the two files in `output/`. Student source
code is never read, imported, or executed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import difflib
from pathlib import Path
import hashlib
import re


DATA_FILE = Path("data") / "clinic_encounters.csv"
# The supplied encounter file, ignoring line-ending and trailing-whitespace
# differences. A changed file cannot be summarized into a known answer.
DATA_FINGERPRINT = "2ff169aa160fb1d0e4157aae93164ff59bd6c7cac0f0a8115db5b24f5c0e21df"

README_FILE = Path("README.md")
GITIGNORE_FILE = Path(".gitignore")
REPORT_FILE = Path("output") / "vitals_report.txt"
FOLLOWUP_FILE = Path("output") / "followup_list.txt"
REPORT_LABELS = (
    "usable encounters",
    "skipped rows",
    "patients seen",
    "mean systolic",
    "highest systolic",
    "lowest systolic",
)

FIELD_COUNT = 3
SYSTOLIC_MIN = 60
SYSTOLIC_MAX = 250
CUTOFF_MIN = 120
CUTOFF_MAX = 180

MEAN_TOLERANCE = 0.1
COUNT_TOLERANCE = 0.5
REASON_MIN_LENGTH = 20
REASON_MAX_LENGTH = 300
DESCRIPTION_MIN_LENGTH = 30
DESCRIPTION_MAX_LENGTH = 300

# A Python command (`python`, `python3`, or the Windows `py` launcher, any version)
# that runs a `.py` script, wherever it sits in the `## Run` section: alone on a
# line, inside a sentence (a full stop may follow), in a bullet, or in a code
# fence. `.pyc` and `.py.bak` are not scripts.
RUN_COMMAND = re.compile(
    r"(?<![\w.-])(?:python(?:3(?:\.\d+)?)?|py(?:\s+-3(?:\.\d+)?)?)\s+[\w./\\-]*\.py(?!\w|\.\w)",
    re.IGNORECASE,
)
# NumPy prints a scalar as `np.float64(129.44)`, which is a correct answer
# written by a correct program. A number has to start a token, so the digits
# inside `P018` or `float64` are never read as the answer.
NUMPY_SCALAR = re.compile(r"^(?:np|numpy)\.\w+\((.*)\)$")
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
BARE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# The bytecode cache, written the standard ways: the directory Python creates,
# or a glob for the files it puts there, including the `*.py[codz]` line in
# GitHub's own Python template.
CACHE_DIRECTORY_PATTERN = re.compile(r"^/?(?:\*\*/)?__pycache__/?(?:\*\*?/?)?$")
CACHE_FILE_PATTERN = re.compile(
    r"^/?(?:\*\*/)?(?:__pycache__/)?\*(?:\.py(?:[cod]|\[[codz]+\])|\$py\.class)$",
    re.IGNORECASE,
)
LABEL_LINES = ("cutoff", "reason")


# A saved line is shown in feedback up to this many characters, and a list of patient IDs up to this many IDs.
SHOWN_LENGTH = 80
SHOWN_IDS = 6
# How closely a line's label must resemble a missing label to be shown as the likely attempt at it.
SIMILAR_LABEL = 0.6
# What each report line holds, as Task 2.3 words it.
REPORT_MEANINGS = {
    "usable encounters": "how many data rows were usable",
    "skipped rows": "how many data rows were skipped",
    "patients seen": "how many different patient IDs appear among the usable encounters",
    "mean systolic": "the mean of every usable reading",
    "highest systolic": "the largest usable reading",
    "lowest systolic": "the smallest usable reading",
}
REPORT_FIX = "Fix vitals_tools.py or clinic_report.py, rerun clinic_report.py, and commit output/vitals_report.txt."
LIST_FIX = "Then rerun clinic_report.py and commit output/followup_list.txt."


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]
    # The file the check reads and the check's name within it, for the report's `Left to fix` line.
    artifact: str = ""
    label: str = ""


@dataclass(frozen=True)
class SuppliedEncounters:
    """The supplied encounters, summarized the way the assignment defines them."""

    usable: tuple[tuple[str, int], ...]
    data_rows: int
    all_ids: frozenset[str]

    @property
    def skipped_rows(self) -> int:
        return self.data_rows - len(self.usable)

    @property
    def readings(self) -> tuple[int, ...]:
        return tuple(systolic for _, systolic in self.usable)

    @property
    def patients(self) -> set[str]:
        return {patient_id.casefold() for patient_id, _ in self.usable}

    def patients_at_or_above(self, cutoff: float) -> set[str]:
        return {patient_id.casefold() for patient_id, systolic in self.usable if systolic >= cutoff}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _join(items, limit: int | None = None) -> str:
    """`a`, `a and b`, or `a, b and c`, naming at most `limit` items before `and K more`."""
    items = list(items)
    if limit is not None and len(items) > limit + 1:
        items = items[:limit] + [f"{len(items) - limit} more"]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _shown(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SHOWN_LENGTH else text[: SHOWN_LENGTH - 3] + "..."


def _file_state(path: Path) -> str:
    """What stands where a file belongs, in plain words."""
    if path.is_symlink():
        return "is a link, not a file"
    if path.is_dir():
        return "is a folder, not a file"
    return "is not a regular file" if path.exists() else "is missing"


def _read_artifact(root: Path, relative: Path, save: str) -> str:
    """Read a saved artifact as text, tolerating the BOM Notepad and Excel add.

    `save` says how the artifact is made, as a sentence that the messages end with.
    """
    path, name = root / relative, relative.as_posix()
    if not path.is_file() or path.is_symlink():
        start = "Delete it, then" if path.exists() or path.is_symlink() else ""
        state = _file_state(path)
        raise AssertionError(f"{name} {state}. " + (f"{start} {save[0].lower()}{save[1:]}" if start else save))
    try:
        # utf-8-sig drops a leading byte-order mark and is otherwise plain UTF-8.
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise AssertionError(
            f"{name} is not saved as UTF-8 text, so it cannot be read. Save it again as UTF-8 (in Python, "
            'open the file with encoding="utf-8"), then commit it.'
        ) from None
    except OSError as error:
        raise AssertionError(f"{name} cannot be opened ({error.strerror or 'unreadable'}). {save}") from None


def _fingerprint(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _data_rows(text: str) -> list[str]:
    """Every line after the header, as `splitlines()` reads it.

    The fingerprint forgives a blank line appended to the export, but the student's
    own loop counts it as one more skipped row, so it is counted here too.
    """
    return text.splitlines()[1:]


DATA_RESTORE = (
    f"Restore it with `git checkout {DATA_FILE.as_posix()}`, rerun clinic_report.py on it, "
    "and commit the new output files."
)


def load_supplied_encounters(root: Path) -> SuppliedEncounters:
    """Apply the assignment's usable-row rule to the supplied encounter file."""
    path, name = root / DATA_FILE, DATA_FILE.as_posix()
    _assert(
        path.is_file() and not path.is_symlink(),
        f"{name} {_file_state(path)}, and the answers are recomputed from it. {DATA_RESTORE}",
    )
    try:
        # utf-8-sig drops a leading byte-order mark and is otherwise plain UTF-8.
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = None  # not the supplied file, which is UTF-8
    except OSError as error:
        raise AssertionError(f"{name} cannot be opened ({error.strerror or 'unreadable'}). {DATA_RESTORE}") from None
    _assert(
        text is not None and _fingerprint(text) == DATA_FINGERPRINT,
        f"{name} is not the file the assignment supplies, and the answers are recomputed from it. {DATA_RESTORE}",
    )

    usable: list[tuple[str, int]] = []
    all_ids: set[str] = set()
    rows = _data_rows(text)
    for row in rows:
        fields = row.strip().split(",")
        if fields[0].strip():
            all_ids.add(fields[0].strip().casefold())
        if len(fields) != FIELD_COUNT:
            continue
        try:
            systolic = int(fields[2])
        except ValueError:
            continue
        if SYSTOLIC_MIN <= systolic <= SYSTOLIC_MAX:
            usable.append((fields[0].strip(), systolic))
    return SuppliedEncounters(usable=tuple(usable), data_rows=len(rows), all_ids=frozenset(all_ids))


def _labelled_values(text: str) -> dict[str, str]:
    """Map each `Label: value` line to its value, keeping the first of a repeat."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        values.setdefault(_label(label), value.strip())
    return values


def _label(text: str) -> str:
    """Normalize a label: case, inner spacing, and any bullet or emphasis marks."""
    return " ".join(text.split()).strip("-*#>• ").casefold()


def _closest(text: str, label: str, known: tuple[str, ...]) -> str | None:
    """The line that most resembles a `label:` line, or None.

    A line already read as one of the `known` labels belongs to that label and is never shown.
    """
    best, best_ratio = None, SIMILAR_LABEL
    for line in text.splitlines():
        if not line.strip() or (":" in line and _label(line.partition(":")[0]) in known):
            continue
        head = line.partition(":")[0] if ":" in line else re.split(r"[\d=]", line, maxsplit=1)[0]
        candidate = _label(re.sub(r"[^\w\s]", " ", head))
        ratio = difflib.SequenceMatcher(None, label, candidate).ratio() if candidate else 0
        if ratio >= best_ratio:
            best, best_ratio = line, ratio
    return best


def _closest_line(text: str, label: str, known: tuple[str, ...]) -> str:
    """`; the closest line reads ...` naming the line that most resembles one for `label`, or nothing."""
    best = _closest(text, label, known)
    return f"; the closest line reads `{_shown(best)}`" if best is not None else ""


def _parse_number(raw: str) -> tuple[float, int] | None:
    """Find the number on a labelled line and how many decimals it was written to.

    Words around the number are ignored, so `129.4 mm Hg`, `129.4 mmHg (mean)`
    and `np.float64(129.44)` all read as the number the student computed.
    """
    text = raw.strip()
    wrapped = NUMPY_SCALAR.match(text)
    if wrapped is not None:
        text = wrapped.group(1).strip()
    found = NUMBER.search(text)
    if found is None:
        return None
    written = found.group(0)
    _, point, fraction = written.partition(".")
    decimals = len(fraction) if point and "e" not in fraction.lower() else 0
    return float(written), decimals


def _report_text(root: Path) -> str:
    return _read_artifact(
        root,
        REPORT_FILE,
        "Run clinic_report.py so it writes the six summary lines there (Task 2.3), then commit it.",
    )


def _report_values(root: Path) -> dict[str, str]:
    return _labelled_values(_report_text(root))


def _report_number(root: Path, label: str) -> tuple[float, int]:
    text = _report_text(root)
    values = _labelled_values(text)
    shown = label.capitalize()
    _assert(
        label in values,
        f"{REPORT_FILE.as_posix()} has no `{shown}:` line{_closest_line(text, label, REPORT_LABELS)}. "
        f"Write it as Task 2.3 shows: `{shown}: <number>`, the label, a colon, then {REPORT_MEANINGS[label]}. "
        + REPORT_FIX,
    )
    parsed = _parse_number(values[label])
    _assert(
        parsed is not None,
        f"{REPORT_FILE.as_posix()} has `{shown}: {_shown(values[label])}`, with no number after the colon. "
        f"Task 2.3 asks for {REPORT_MEANINGS[label]} there, as a number. " + REPORT_FIX,
    )
    return parsed


def _wrong(root: Path, label: str, expected: str, hint: str) -> str:
    written = _report_values(root)[label]
    return (
        f"{REPORT_FILE.as_posix()} has `{label.capitalize()}: {_shown(written)}`, but {expected}. {hint} "
        + REPORT_FIX
    )


def _check_count(root: Path, label: str, expected: int, found: str, hint: str) -> None:
    reported, _ = _report_number(root, label)
    if abs(reported - expected) < COUNT_TOLERANCE:
        return
    raise AssertionError(_wrong(root, label, found, hint))


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _readme_section(root: Path, heading: str) -> str | None:
    readme = _read_artifact(
        root,
        README_FILE,
        "Restore it with `git checkout README.md`, then replace its two TODO lines (Tasks 1.1 and 1.2).",
    )
    found = re.search(
        rf"^## {heading}\s*$\n(.*?)(?=\n## |\Z)",
        readme,
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    return None if found is None else found.group(1)


def check_readme_description(root: Path) -> None:
    section = _readme_section(root, "Project description")
    _assert(
        section is not None,
        "README.md has no `## Project description` heading. Put the heading back at the top of README.md and "
        f"write {DESCRIPTION_MIN_LENGTH}-{DESCRIPTION_MAX_LENGTH} characters of your own under it (Task 1.1).",
    )
    written = " ".join(section.split())
    _assert(
        "TODO" not in written,
        "README.md still has the TODO line under `## Project description`. Replace it with "
        f"{DESCRIPTION_MIN_LENGTH}-{DESCRIPTION_MAX_LENGTH} characters of your own saying what this project "
        "reads and what it produces (Task 1.1).",
    )
    if DESCRIPTION_MIN_LENGTH <= len(written) <= DESCRIPTION_MAX_LENGTH:
        return
    change = (
        "Say more about what this project reads and what it produces."
        if len(written) < DESCRIPTION_MIN_LENGTH
        else f"Shorten it to {DESCRIPTION_MAX_LENGTH} characters or fewer."
    )
    raise AssertionError(
        f"README.md has {len(written)} characters under `## Project description`, and Task 1.1 asks for "
        f"{DESCRIPTION_MIN_LENGTH}-{DESCRIPTION_MAX_LENGTH}. {change}"
    )


def check_readme_run_command(root: Path) -> None:
    section = _readme_section(root, "Run")
    _assert(
        section is not None,
        "README.md has no `## Run` heading. Put the heading back and write the command that runs your report "
        "script under it (Task 1.2).",
    )
    if RUN_COMMAND.search(section) is not None:
        return
    lines = [line.strip() for line in section.splitlines() if line.strip() and not line.strip().startswith("```")]
    if "TODO" in section:
        found = "README.md still has the TODO line under `## Run`."
    elif not lines:
        found = "README.md has nothing under `## Run`."
    else:
        found = f"Under `## Run`, README.md reads `{_shown(lines[0])}`, which is not a command that runs a Python script."
    raise AssertionError(
        f"{found} Replace it with the command that runs your report script (Task 1.2): `python3`, `python`, or "
        "`py`, a space, and the script's full file name, as in `python3 clinic_report.py` or "
        "`py -3.13 clinic_report.py`. A sentence, a bullet, or a code fence around the command is fine."
    )


def check_gitignore_cache(root: Path) -> None:
    text = _read_artifact(
        root,
        GITIGNORE_FILE,
        "Restore it with `git checkout .gitignore`, then replace its two TODO lines (Task 1.3).",
    )
    patterns = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if any(CACHE_DIRECTORY_PATTERN.match(pattern) for pattern in patterns) or any(
        CACHE_FILE_PATTERN.match(pattern) for pattern in patterns
    ):
        return
    if "TODO" in text:
        found = "it still has its TODO lines"
    elif patterns:
        found = "it lists only " + _join((f"`{_shown(pattern)}`" for pattern in patterns), limit=4)
    else:
        found = "it lists no patterns"
    raise AssertionError(
        f".gitignore has no pattern for Python's bytecode cache: {found}. Replace the two TODO lines with "
        "`__pycache__/` for the cache folder and `*.pyc` (or `*.py[cod]`) for the compiled files (Task 1.3)."
    )


def check_report_format(root: Path) -> None:
    text = _report_text(root)
    values = _labelled_values(text)
    missing = [label for label in REPORT_LABELS if label not in values]
    if not missing:
        return
    closest = list(dict.fromkeys(
        f"`{_shown(line)}`" for line in (_closest(text, label, REPORT_LABELS) for label in missing) if line
    ))
    found = ""
    if closest:
        found = (f"; the closest line reads {closest[0]}" if len(closest) == 1
                 else f"; the closest lines read {_join(closest, limit=2)}")
    raise AssertionError(
        f"{REPORT_FILE.as_posix()} has no "
        + _join(f"`{label.capitalize()}:`" for label in missing)
        + f" line{'s' if len(missing) > 1 else ''}{found}. Task 2.3 asks for six lines, each the label as "
        "shown, a colon, and the number, as in `Patients seen: <number>`. " + REPORT_FIX
    )


def check_usable_encounters(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "usable encounters",
        len(encounters.usable),
        f"{DATA_FILE.as_posix()} has {len(encounters.usable)} usable data rows",
        "A usable row (Task 2.1) has exactly three comma-separated fields, a systolic value `int()` can read, "
        f"and a reading from {SYSTOLIC_MIN} to {SYSTOLIC_MAX} mmHg.",
    )


def check_skipped_rows(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "skipped rows",
        encounters.skipped_rows,
        f"{DATA_FILE.as_posix()} has {encounters.skipped_rows} data rows to skip",
        "Every data row that is not usable is skipped (Task 2.1): the header is not a data row, but the blank "
        "line inside the export is one, the way Demo 3 reports it.",
    )


def check_distinct_patients(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "patients seen",
        len(encounters.patients),
        f"the usable encounters name {len(encounters.patients)} different patients",
        "Count each patient ID once across the usable encounters only (Task 2.3): some patients visited twice, "
        "and a patient whose only row was skipped was not seen.",
    )


def check_mean_systolic(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    reported, decimals = _report_number(root, "mean systolic")
    expected = sum(encounters.readings) / len(encounters.readings)
    # Compared at the precision the student wrote, so rounding never decides a grade.
    tolerance = max(MEAN_TOLERANCE, 0.5 * 10**-decimals)
    if abs(reported - expected) <= tolerance:
        return
    raise AssertionError(
        _wrong(
            root,
            "mean systolic",
            f"the usable readings average {expected:.2f} mmHg",
            "Average every usable reading, including a patient's second visit (Task 2.3). Any value within "
            f"{MEAN_TOLERANCE} mmHg passes, and so does the mean rounded to the decimals you wrote.",
        )
    )


def check_highest_systolic(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "highest systolic",
        max(encounters.readings),
        f"the largest usable reading is {max(encounters.readings)} mmHg",
        "Take the largest usable reading, not the largest number in the file (Task 2.3).",
    )


def check_lowest_systolic(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _check_count(
        root,
        "lowest systolic",
        min(encounters.readings),
        f"the smallest usable reading is {min(encounters.readings)} mmHg",
        f"Take the smallest usable reading, so nothing below {SYSTOLIC_MIN} mmHg counts (Task 2.3).",
    )


def _followup_text(root: Path) -> str:
    return _read_artifact(
        root,
        FOLLOWUP_FILE,
        "Run clinic_report.py so it writes your `Cutoff:` line, `Reason:` line, and patient list there "
        "(Task 3.1), then commit it.",
    )


def _listed_patients(root: Path, known: frozenset[str]) -> set[str]:
    """The patient IDs named in the follow-up list, ignoring anything else on the page.

    A line counts only when the whole line is one patient ID, allowing a bullet
    marker and trailing punctuation. That is what the README promises, so a
    heading or a note that happens to mention a patient cannot change the
    graded answer.
    """
    listed: set[str] = set()
    for line in _followup_text(root).splitlines():
        entry = line.strip()
        label, separator, _ = entry.partition(":")
        if separator:
            if _label(label) in LABEL_LINES:
                continue
            entry = label.strip()
        entry = entry.lstrip("-*• \t").strip().rstrip(",;")
        if entry.casefold() in known:
            listed.add(entry.casefold())
    return listed


def _declared_cutoff(root: Path) -> float:
    text = _followup_text(root)
    values = _labelled_values(text)
    name = FOLLOWUP_FILE.as_posix()
    _assert(
        "cutoff" in values,
        f"{name} has no `Cutoff:` line{_closest_line(text, 'cutoff', LABEL_LINES)}. Start the file with "
        f"`Cutoff: <number> mmHg`, using a cutoff from {CUTOFF_MIN} to {CUTOFF_MAX} mmHg you choose (Task 3.1). "
        + LIST_FIX,
    )
    parsed = _parse_number(values["cutoff"])
    _assert(
        parsed is not None,
        f"{name} has `Cutoff: {_shown(values['cutoff'])}`, with no number after the colon. Write the cutoff you "
        f"chose as a number from {CUTOFF_MIN} to {CUTOFF_MAX}, as in `Cutoff: <number> mmHg` (Task 3.1). "
        + LIST_FIX,
    )
    cutoff = parsed[0]
    _assert(
        CUTOFF_MIN <= cutoff <= CUTOFF_MAX,
        f"{name} declares a cutoff of {cutoff:g} mmHg, and Task 3.1 asks for one from {CUTOFF_MIN} to "
        f"{CUTOFF_MAX} mmHg. Choose a cutoff in that range, rerun clinic_report.py, and commit the new list.",
    )
    return cutoff


def check_followup_cutoff(root: Path) -> None:
    _declared_cutoff(root)


def check_followup_reason(root: Path) -> None:
    text = _followup_text(root)
    values = _labelled_values(text)
    name = FOLLOWUP_FILE.as_posix()
    _assert(
        "reason" in values,
        f"{name} has no `Reason:` line{_closest_line(text, 'reason', LABEL_LINES)}. Add one line, "
        f"`Reason: <why you chose your cutoff>`, of {REASON_MIN_LENGTH}-{REASON_MAX_LENGTH} characters (Task 3.1). "
        + LIST_FIX,
    )
    reason = " ".join(values["reason"].split())
    if REASON_MIN_LENGTH <= len(reason) <= REASON_MAX_LENGTH:
        return
    change = (
        "Say more about why you chose your cutoff."
        if len(reason) < REASON_MIN_LENGTH
        else f"Shorten it to {REASON_MAX_LENGTH} characters or fewer."
    )
    raise AssertionError(
        f"{name} has a `Reason:` of {len(reason)} characters, and Task 3.1 asks for "
        f"{REASON_MIN_LENGTH}-{REASON_MAX_LENGTH} on one line. {change} {LIST_FIX}"
    )


def check_followup_patients(root: Path) -> None:
    encounters = load_supplied_encounters(root)
    _followup_text(root)  # a missing list gives the same advice as the checks before it
    try:
        cutoff = _declared_cutoff(root)
    except AssertionError as error:
        # The list is read against the cutoff, so that line has to be right first.
        raise AssertionError(
            f"{error} The patient list is checked against your cutoff, so it scores once the `Cutoff:` line passes."
        ) from None
    listed = _listed_patients(root, encounters.all_ids)
    expected = encounters.patients_at_or_above(cutoff)

    missing = sorted(patient.upper() for patient in expected - listed)
    extra = sorted(patient.upper() for patient in listed - expected)
    problems = []
    if missing:
        problems.append(f"it leaves out {_join(missing, limit=SHOWN_IDS)}")
    if extra:
        verb = "has" if len(extra) == 1 else "have"
        problems.append(
            f"it also lists {_join(extra, limit=SHOWN_IDS)}, which {verb} no usable reading at or above "
            f"{cutoff:g} mmHg"
        )
    _assert(
        not problems,
        f"{FOLLOWUP_FILE.as_posix()} should list the {len(expected)} patients with a usable reading at or above "
        f"your cutoff of {cutoff:g} mmHg, but " + "; ".join(problems) + ". List one patient ID per line "
        "(Task 3.1), rerun clinic_report.py, and commit the new list.",
    )


_REPORT = REPORT_FILE.as_posix()
_FOLLOWUP = FOLLOWUP_FILE.as_posix()
CHECKS = (
    Check("README project description", check_readme_description, "README.md", "project description"),
    Check("README run command", check_readme_run_command, "README.md", "run command"),
    Check(".gitignore bytecode cache", check_gitignore_cache, ".gitignore", "bytecode cache"),
    Check("vitals report format", check_report_format, _REPORT, "format"),
    Check("usable encounters", check_usable_encounters, _REPORT, "usable encounters"),
    Check("skipped rows", check_skipped_rows, _REPORT, "skipped rows"),
    Check("patients seen", check_distinct_patients, _REPORT, "patients seen"),
    Check("mean systolic", check_mean_systolic, _REPORT, "mean systolic"),
    Check("highest systolic", check_highest_systolic, _REPORT, "highest systolic"),
    Check("lowest systolic", check_lowest_systolic, _REPORT, "lowest systolic"),
    Check("follow-up cutoff", check_followup_cutoff, _FOLLOWUP, "cutoff"),
    Check("follow-up reason", check_followup_reason, _FOLLOWUP, "reason"),
    Check("follow-up patient list", check_followup_patients, _FOLLOWUP, "patient list"),
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
