# Lectures 01–03 content and assessment alignment

> Historical review snapshot. Any Classroom50 or GitHub Classroom language
> below records a superseded intermediate delivery plan, not current course
> policy. The release design is one repository per assignment with repo-local
> pytest/GitHub Actions; no Classroom service is used.

## Review status and fixed decisions

This is an evidence-backed design artifact for the terminal-foundations range. It reviews the current lecture, bonus, demo, assignment, and test files but does not modify student-facing material.

The following decisions are fixed for this review:

- Python work remains script-and-terminal based through Lecture 03. The current files already state that boundary in Lectures 01, 02, and 03 (`01/README.md:14-18`, `02/README.md:14-20`, `03/README.md:14-19`).
- Required Git work is GUI-first through VS Code Source Control or GitHub Desktop. Command-line Git is bonus. The existing Git bonus already labels its command-line material as a power-user track (`02/bonus/advanced_git.md:1-7`), but the main Lecture 02 and Assignment 02 currently contradict it (`02/README.md:158-199`, `02/assignment/README.md:27-53`).
- Lecture 01 still receives a complete content/demo/assignment review even though its current assignment package is absent.
- Content order is settled before demos and assignments are rebuilt. The proposed artifacts below are contracts, not implementation instructions.
- Each lecture converges on two or three required demos. No lecture-calendar or duration plan is included.

## Range-level dependency contract

```text
supported terminal + project folder
  -> working directory + relative path + safe file operations
  -> create/edit/run a top-to-bottom Python script
  -> scalar values + lists + conditions + simple loops + tracebacks
  -> GitHub account and supported GUI Git workflow
  -> function interfaces + dictionaries + import-safe local modules
  -> interpreter/package/dependency/environment vocabulary
  -> recreate an environment and run a dependency-backed script
  -> ndarray shape/dtype/indexing/masks/vectorized operations/axis
  -> Jupyter cell/kernel/runtime/state and pandas structures in Lecture 04
```

### Exact incoming capabilities

Lecture 01 has no programming prerequisite; the current source explicitly says so (`01/README.md:14-18`). It does assume access to a supported computer, permission to install or use course tools, a browser, and a supported terminal. Those operational assumptions should be stated as onboarding requirements rather than hidden inside the learning objectives.

### Exact outgoing capabilities to Lecture 04

By the end of Lecture 03, a student should independently be able to:

1. identify the current working directory, resolve a relative path, and run a script from the intended project directory;
2. open a project in VS Code, edit `.py` files, and use the integrated terminal;
3. inspect, stage, commit, synchronize, branch, and merge through a supported Git GUI;
4. define and call a function, distinguish parameter from argument and return value, and import a side-effect-free local module guarded by `if __name__ == "__main__":`;
5. explain interpreter, package, direct dependency, environment, activation, and requirements file, then recreate a NumPy-capable environment;
6. create an ndarray and reason about its `shape`, `ndim`, `size`, `dtype`, indexing, masks, vectorized arithmetic, reductions, and `axis`;
7. run a terminal-only NumPy analysis script from a clean environment.

Lecture 04 can then introduce notebook, cell, kernel/runtime, state, execution order, restart, and run-all before expecting notebook debugging. Its current opening already contrasts top-to-bottom scripts with out-of-order notebook execution (`04/README.md:10-14`) and defines cells and kernel later (`04/README.md:23-40`); the revised handoff should make that contrast explicit rather than reteach paths, imports, environments, or NumPy.

## Cross-range blockers

### P0: the supported shell is not defined consistently

Lecture 01 presents WSL, native PowerShell, and Codespaces as alternatives (`01/README.md:46-64`). Lecture 03 later requires POSIX utilities and syntax such as `cut`, `sort`, `uniq`, pipes, backslash continuation, `awk`, and shell redirection (`03/README.md:514-590`; `03/assignment/README.md:47-97`). Native PowerShell does not provide that exact interface. Before the content rewrite, the course must either:

- declare macOS/Linux/WSL/Codespaces Bash as the supported shell for required early work and make PowerShell a setup bridge only; or
- provide and assess equivalent PowerShell instructions.

The first option is the smaller and more coherent match for the current curriculum.

### P0: Git platform evidence may not survive Classroom 50 grading

Assignment 02 currently requests multiple branches and at least three commits on each (`02/assignment/README.md:31-45`), while its test checks only for `.git` and at least one commit (`02/assignment/.github/test/test_assignment.py:19-41`, `02/assignment/.github/test/test_assignment.py:170-184`). The Classroom 50 pilot must determine whether student repositories, branch refs, and adequate history are present in the grading checkout. If they are not, GUI Git competence needs a small manual checkpoint or a submitted evidence file; public tests should not pretend to certify branch/merge behavior.

### P0: Assignment 01 is absent

Lecture 01 describes assignments generically (`01/README.md:28-34`) and its setup guide says assignment repositories will be used (`01/demo/01_github_vscode_setup_guide.md:38-48`), but there is no current `01/assignment/` package. The absence creates no assessment evidence for any Lecture 01 objective and requires a redesign rather than an unchanged historical restore.

---

## Lecture 01: terminal orientation and first Python scripts

### Intended role

Give a novice enough command-line, editor, and Python fluency to create, run, inspect, and debug one small script without introducing functions, file formats, automation, or Git mastery.

### Proposed measurable objectives

By the end of Lecture 01, a student can:

1. **O1.1 — Navigate and manage a scratch project:** use `pwd`, `ls`, `cd`, `mkdir`, `touch`, `cp`, and `mv`, and remove only a named scratch file after confirming its path.
2. **O1.2 — Create and run a script:** open a project folder in VS Code, edit a `.py` file, and run it from the integrated terminal with the course-standard Python command.
3. **O1.3 — Work with values and a list:** assign scalar values, convert a numeric string, create/index a list, perform a calculation, and format deterministic output.
4. **O1.4 — Control simple execution:** write one `if`/`elif`/`else` decision and one `for` loop over a list or `range`.
5. **O1.5 — Diagnose a beginner error:** use the final line and referenced source line in a traceback to correct a `NameError`, `TypeError`, `ValueError`, `IndexError`, or `IndentationError`.

GitHub account creation, private-email configuration, tool installation, and a Classroom 50 repository-access check remain required onboarding tasks, but they are readiness checks rather than programming objectives. The current objective list incorrectly combines setup with functions and broad “collections” (`01/README.md:7-12`).

### Incoming prerequisites

- No prior programming or terminal knowledge (`01/README.md:14-18`).
- Access to one declared supported shell, VS Code, Python 3.12, Git, a browser, and a GitHub account or the ability to create one.
- Permission to work only in an instructor-designated scratch directory during destructive-command practice.

### Outgoing prerequisites supplied to Lecture 02

- Locate the working directory and explain a relative path.
- Create, rename, copy, inspect, and deliberately remove a file.
- Open a folder in VS Code and run a `.py` file in its integrated terminal.
- Use strings, numbers, booleans, a list, zero-based indexing, comparisons, conditions, and a simple loop.
- Read a basic traceback and make one correction.
- GitHub account, private/noreply email selection, and Git installed; no independent staging, commit, branch, merge, push, or pull knowledge is assumed.

### Term first-definition and first-independent-use ledger

| Term or skill | Current first definition/guided evidence | Current first independent or demanding use | Finding and target |
|---|---|---|---|
| command line / shell | The command line is described as typed interaction at `01/README.md:186-192`. | The navigation sequence begins immediately at `01/README.md:196-215`; Demo 02 executes a shell script (`01/demo/02_cli_navigation_demo.sh:1-16`). | **Revise:** name the shell separately from a command and state the supported Bash environment before commands are used. |
| working directory | `pwd` is glossed as “where am I?” at `01/README.md:196-205`. | Demo 02 changes directories and embeds `$(pwd)` (`01/demo/02_cli_navigation_demo.sh:13-16`). | **Revise:** define working directory as the base used to resolve relative paths before the demo. |
| path / relative path | Paths appear in `cd [path]` and project examples (`01/README.md:200-205`, `01/README.md:229-250`) without a plain-language absolute/relative distinction. | Demo 02 deliberately fails because `grades.csv` is resolved from `scripts/` (`01/demo/02_cli_navigation_demo.sh:31-42`). | **P0 revise:** define path, absolute path, relative path, and resolution from the working directory before the failure exercise. |
| variable / value / type | A variable is described as a labeled box at `01/README.md:396-399`, followed by numeric/string/boolean examples (`01/README.md:400-441`). | Demo 03 expects type conversion and `isinstance` (`01/demo/03_python_basics_demo.py:47-66`). | **Revise:** distinguish variable name, value, and type; keep `type()` core and defer `isinstance` unless taught. |
| list / element / index | The lecture uses a list in membership testing at `01/README.md:522-525` and loops at `01/README.md:560-589` without a list-definition section. | Demo 03 creates, indexes, and appends to lists (`01/demo/03_python_basics_demo.py:147-179`). | **P0 move earlier:** define list, element, length, and zero-based index before comparisons or loops over a list. |
| condition / boolean expression | Comparisons are listed at `01/README.md:505-525`; branching follows at `01/README.md:527-558`. | Demo 04 asks students to reason through compound branches (`01/demo/04_control_structures_demo.py:75-139`). | **Keep/revise:** explicitly define a condition as an expression evaluated to `True` or `False`. |
| loop / iterable / `range` | For-loop syntax is shown at `01/README.md:560-589`; “iterable” is not defined. | Demo 03 already contains indexed loops (`01/demo/03_python_basics_demo.py:200-215`) before the dedicated control-flow demo. | **Reorder:** define loop target and iterable, then use direct list iteration before index-based iteration. |
| traceback / exception type | Error type and message are described, but the displayed examples omit a real traceback and source location (`01/README.md:660-717`). | Demo 03 prints simulated error text rather than producing and reading a traceback (`01/demo/03_python_basics_demo.py:268-301`). | **Revise:** show one real, harmless failing script, read it bottom-up, fix it, then run successfully. Do not teach `try`/`except` yet. |
| function / parameter / return | “Functions” appears in objective 4 (`01/README.md:10`) but has no Lecture 01 definition. | Demo 04 defines and calls functions at `01/demo/04_control_structures_demo.py:82-123`; Demo 05 defines recursion and a main guard later (`01/demo/05_integration_workflow_demo.py:370-435`). | **P0 move to Lecture 02:** Lecture 01 may call built-ins such as `print` and `len`, but students do not define functions independently. |
| repository / commit | The setup guide previews initializing a repository (`01/demo/01_github_vscode_setup_guide.md:116-121`) and makes broad permanence claims (`01/demo/01_github_vscode_setup_guide.md:44-48`). | No current Assignment 01 can establish an independent use. | **Preview only:** account/privacy/access setup now; repository, commit, and staging definitions belong to Lecture 02. |

### Objective-to-artifact alignment matrix

| Objective | Current lecture section | Current guided demo | Current assignment evidence | Current test/rubric evidence | Current status |
|---|---|---|---|---|---|
| O1.1 Navigate/manage scratch project | Navigation, operations, and viewing at `01/README.md:186-297` | Demo 02 at `01/demo/DEMO_GUIDE.md:53-81` | No current Assignment 01 | None | **practiced but unassessed**; the demo's intentional failure also conflicts with the quick-run success expectation (`01/demo/DEMO_GUIDE.md:3-15`). |
| O1.2 Create/run script in VS Code terminal | Editor and run modes at `01/README.md:104-125`, `01/README.md:329-363` | Setup guide creates `hello.py` at `01/demo/01_github_vscode_setup_guide.md:78-97` | None | None | **practiced but unassessed**. |
| O1.3 Values/list/calculation/output | Values and operations at `01/README.md:396-501`; the list model is missing | Demo 03 at `01/demo/03_python_basics_demo.py:22-179` | None | None | **partially aligned** because lists are demanded before being defined and the demo adds `isinstance`, float precision, and advanced formatting. |
| O1.4 Condition and simple loop | `01/README.md:505-589` | Demo 04 starts aligned, then expands to functions, while/nested loops, comprehensions, exceptions, and dictionaries (`01/demo/DEMO_GUIDE.md:118-149`) | None | None | **practiced but unassessed** with substantial out-of-scope demo content. |
| O1.5 Read and fix traceback | Error discussion at `01/README.md:660-750` | Simulated errors at `01/demo/03_python_basics_demo.py:268-333` | None | None | **partially aligned**; no real traceback-reading evidence. |

### Recommended required demos

1. **Demo 1 — Project folder, paths, and first script.** In one scratch directory: open VS Code, confirm `pwd`, create a folder/file, run `hello.py`, deliberately run it from the wrong directory, explain the relative-path error, correct location/path, and rerun. This consolidates the current separate setup and navigation demos without a generated here-document or untaught file I/O.
2. **Demo 2 — Values, lists, decisions, and loops.** Build a short list-based script incrementally: scalars and `type`, numeric conversion, list/index/`len`, one calculation, one condition, one direct loop, and formatted output. Keep all data in the script.
3. **Demo 3 — Read, fix, rerun.** Execute small prepared variants that produce a real `NameError`, `TypeError`, `IndexError`, and `IndentationError`; identify exception type and source line, fix one issue at a time, and finish with a clean top-to-bottom run.

The current five-demo set is too broad: Demo 04 introduces functions and advanced control flow (`01/demo/DEMO_GUIDE.md:118-149`), while Demo 05 adds `pathlib`, CSV/JSON, file I/O, exceptions, dictionaries, comprehensions, lambdas, sorting, recursion, and a main guard (`01/demo/05_integration_workflow_demo.py:14-19`, `01/demo/05_integration_workflow_demo.py:57-199`, `01/demo/05_integration_workflow_demo.py:278-435`).

### Assignment 01 redesign contract

**Purpose:** certify tool readiness and beginner script competence without treating Git as already learned.

**Delivery:** a Classroom 50 repository is provisioned for every student, but repository access and the first GUI sync are fully guided platform steps, not an assessed Git objective. Do not require command-line Git.

**Three incremental questions:**

1. Complete `readiness.py` so it prints the Python version family, a supplied project label, and the current script filename. Run it from the terminal.
2. Complete a list-based calculation using supplied values, one condition, and one loop; write deterministic labeled output to stdout.
3. Fix three prepared beginner errors and produce `output/readiness.txt` by redirecting the final clean script run or by a supplied wrapper. Do not require `open()`, functions, dictionaries, exception handling, CSV/JSON, imports beyond supplied machinery, or notebooks.

**Public-test contract:**

- execute the script from both the repository root and the script's documented directory if path independence is intended;
- assert exact labels and numeric values rather than only file existence;
- include a different input list to ensure the calculation is general;
- include known-correct, one-error, partial, and hard-coded-wrong fixtures in grader validation;
- report GUI repository access/sync as a separate checklist item because a test runner cannot prove that the student operated the GUI.

**Success evidence:** one clean terminal run, correct behavior on alternate inputs, a readable output artifact, and successful Classroom 50 submission. No Git history rubric in Assignment 01.

### Core/bonus/drop disposition

| Current material | Disposition | Evidence and rationale |
|---|---|---|
| Tool installation, account, email privacy, and repository access | **Revise; move to onboarding/pre-class checklist** | Setup dominates the opening (`01/README.md:36-174`) but is operational readiness rather than lecture concepts. Remove instructor identity examples and privacy absolutes at `01/README.md:133-163`. |
| Basic `pwd`, `ls`, `cd`, `mkdir`, `touch`, `cp`, `mv`, `cat`, `head`, `tail`, and `Ctrl+C` | **Keep/revise** | These support the terminal-to-script workflow (`01/README.md:196-323`). Add path definitions and practice only in a safe scratch directory. |
| `rm -r`, brace expansion, globbing, command substitution, `find -exec`, dated backup patterns | **Move to bonus** | The “common patterns” jump from first navigation to multiple shell expansions and `find -exec` (`01/README.md:254-269`). |
| REPL, script, notebook preview | **Keep, script-first** | The three modes are distinguished at `01/README.md:333-347`; notebooks remain preview-only. |
| Variables, scalar types, conversion, lists, comparisons, conditions, simple `for` | **Keep/reorder** | Core foundations are present (`01/README.md:396-589`) but lists need their own definition before unrestricted use. |
| Advanced f-string alignment, date formatting, and interactive `input()` | **Move to bonus** | These expand beyond the assignment spine (`01/README.md:613-655`). Keep only simple decimal formatting in core. |
| Functions, while/nested loops, `break`/`continue`, comprehensions, exceptions | **Move later** | They appear in Demo 04 before Lecture 02 defines functions (`01/demo/DEMO_GUIDE.md:118-149`). Functions move to Lecture 02; other items move to later/bonus. |
| CSV/JSON pipelines, `pathlib`, logging, recursion, lambdas, sorting | **Drop from required Lecture 01 demo; relocate selectively** | Demo 05 is an entire data pipeline rather than novice integration (`01/demo/05_integration_workflow_demo.py:14-19`, `01/demo/05_integration_workflow_demo.py:278-435`). |
| Git command configuration and repository initialization | **Guided setup only; CLI commands to Lecture 02 bonus** | Current setup uses CLI config (`01/README.md:150-174`) and initializes a repo (`01/demo/01_github_vscode_setup_guide.md:116-121`) before the Git mental model. GUI identity configuration/access may remain onboarding. |
| “Suggested class plan” and broad professional claims | **Drop** | Timing was not requested (`01/README.md:20-26`); claims such as a universal 80/20 split are unsupported and distract from the learning contract (`01/README.md:827-831`). |

### Ordered content outline

1. Learning contract and supported environment; no prior knowledge; scripts/terminal only.
2. Terminal, shell, command, prompt, working directory, path, absolute versus relative path.
3. Safe navigation and file inspection in a supplied scratch directory.
4. Open a project folder in VS Code and use the integrated terminal.
5. Create and run a first top-to-bottom `.py` script.
6. Statements, comments, indentation, variable name, value, scalar type, and conversion.
7. List, element, length, zero-based index, and direct iteration.
8. Arithmetic, comparisons, boolean expressions, one `if`/`elif`/`else` pattern.
9. One `for` loop over a list and one `range` example.
10. Simple formatted output.
11. Real traceback anatomy and fix-rerun workflow.
12. Integrated mini-script and readiness checklist; preview that Git concepts come next.

---

## Lecture 02: GUI Git, functions, and local modules

### Intended role

Teach one supported visual Git workflow and transform a top-to-bottom beginner script into small reusable, import-safe functions and modules.

### Proposed measurable objectives

By the end of Lecture 02, a student can:

1. **O2.1 — Explain the Git state model:** distinguish repository, working tree, change/diff, staging area, commit, local branch, and remote.
2. **O2.2 — Complete the daily GUI workflow:** inspect a diff, stage selected changes, write a focused commit message, commit, and synchronize/push/pull through VS Code Source Control or GitHub Desktop.
3. **O2.3 — Use a branch through the GUI:** create/switch a feature branch, commit one focused change, merge it into `main`, and resolve a prepared simple text conflict without discarding the other intended change.
4. **O2.4 — Define a clear function interface:** write and call functions using parameters, arguments, return values, local variables, a short docstring, and an empty-input guard.
5. **O2.5 — Build an import-safe two-file program:** move reusable functions into a local module, import them from a driver script, and use `main()` plus a correct main guard so importing the module has no analysis or file-writing side effects.

### Incoming prerequisites from Lecture 01

- Navigate to a project directory and resolve relative paths.
- Create/edit/run a `.py` script from VS Code's terminal.
- Use scalar values, lists, indices, comparisons, conditions, and simple loops.
- Read a basic traceback.
- GitHub account and Git installation are ready, but repository/staging/commit/branch knowledge is not assumed.

The current prerequisite block is close but says only “use Git for a small project” as an outgoing Lecture 03 capability (`03/README.md:14-18`); the revised wording should specify the supported GUI actions above.

### Outgoing prerequisites supplied to Lecture 03

- Use a GUI to inspect, stage, commit, synchronize, create a branch, and merge.
- Explain repository, working tree, staging area, commit, branch, and remote.
- Use lists and a minimal dictionary for named records/results.
- Define/call a function; distinguish parameter, argument, return value, and local variable.
- Read/write one small text file with a path already resolved; complex CSV parsing is not assumed.
- Import a local module without causing top-level analysis or file writes.
- Put orchestration in `main()` under `if __name__ == "__main__":`.

### Term first-definition and first-independent-use ledger

| Term or skill | Current first definition/guided evidence | Current first independent or demanding use | Finding and target |
|---|---|---|---|
| repository | Defined as a Git-tracked project folder at `02/README.md:117-123`. | Assignment asks students to create a separate repository at `02/assignment/README.md:27-45`. | **Keep/revise:** explain that the Classroom 50 submission repository is already a repository; use one coherent repository rather than a second ungraded location. |
| working tree | Named in objective 1 (`02/README.md:7`) and later called working directory in a three-stage workflow (`02/README.md:158-160`), but not plainly defined. | Tests assert `.git` exists without validating state understanding (`02/assignment/.github/test/test_assignment.py:19-41`). | **P0 define first:** current checked-out files a student can edit; distinguish it from “working directory” as a shell location. |
| staging area / stage | Introduced operationally in command lists and GUI clicks (`02/README.md:164-180`, `02/README.md:221-250`). | Assignment requires commit history but never asks students to reason about selected staged changes (`02/assignment/README.md:201-215`). | **Revise:** define staging as the proposed content of the next commit, then assess selective staging via GUI checkpoint. |
| commit | Defined at `02/README.md:125-129`. | Assignment requires at least three commits per branch (`02/assignment/README.md:41-45`). | **Revise:** retain focused commits but reduce arbitrary commit-count gaming; one meaningful commit per coherent task is enough for competence. |
| remote / sync / push / pull | Remote is defined at `02/README.md:131-135`; GUI sync is procedural at `02/README.md:221-251`. | Assignment tells students to commit and push (`02/assignment/README.md:246-258`) but is otherwise CLI-centered. | **Reorder:** show local state first, then remote synchronization; keep command spellings in bonus. |
| branch / merge / conflict | Branch is defined at `02/README.md:137-143`, immediately followed by “branches come later,” then taught in the same lecture (`02/README.md:253-292`). | Assignment requires two feature branches and merges (`02/assignment/README.md:31-45`), but tests inspect neither branches nor merges. | **P0 resolve contradiction:** keep one GUI feature branch and one prepared simple conflict; remove arbitrary multi-branch history requirements. |
| dictionary / key / value | Dictionaries are defined at `02/README.md:660-687`. | Assignment independently requires dictionaries for detailed results (`02/assignment/README.md:187-192`). | **Keep/reorder:** teach a minimal dictionary before functions that accept or return named results; sets are not required. |
| function / call | First defined after more than 700 lines at `02/README.md:719-743`. | Assignment requires eleven named functions across two scripts (`02/assignment/README.md:155-185`). | **P0 move earlier/reduce:** define functions immediately after the Git workflow and require a much smaller coherent interface. |
| parameter / argument / return value / local variable | Parameters and return are listed at `02/README.md:723-728`; argument and local-variable meanings are not explained. | Assignment prescribes signatures at `02/assignment/README.md:155-185`. | **Define before use:** use one annotated call diagram that maps argument values to parameters and return value to caller; then practice. |
| docstring | The example contains a docstring (`02/README.md:732-738`) without defining its purpose or location. | Demos and assignment demand many documented functions (`02/demo/python_functions_demo.py:103-126`, `02/assignment/README.md:155-185`). | **Revise:** define a one-sentence docstring as function documentation placed first in the function body. |
| module / import | Modules are named in objective 6 (`02/README.md:12`) but have no valid main-lecture definition/example. | Demo 03 manipulates `sys.path` and imports eight functions (`02/demo/03_module_usage_demo.py:8-28`); Assignment 02 calls duplicate scripts “modular” without requiring one to import the other (`02/assignment/README.md:170-185`). | **P0 teach correctly:** a `.py` file used via `import`; keep both files in one directory so no `sys.path` manipulation is needed. |
| `__name__`, `main()`, main guard | Current markup is malformed and outside a Python code fence (`02/README.md:746-752`). | The assignment requires `main()` but does not explicitly test an import-safe main guard (`02/assignment/.github/test/test_assignment.py:94-116`). | **P0 revise:** define script execution versus import, show exact syntax, and test no stdout/file writes on import. |
| command-line Git | Main content teaches commands at `02/README.md:158-199` and throughout branching. | Assignment gives a CLI repository/branch contract (`02/assignment/README.md:27-53`). | **Move to bonus:** the bonus already declares a command-line power-user track (`02/bonus/advanced_git.md:1-7`). |

### Objective-to-artifact alignment matrix

| Objective | Current lecture section | Current guided demo | Current assignment evidence | Current test/rubric evidence | Current status |
|---|---|---|---|---|---|
| O2.1 Git state model | Definitions at `02/README.md:117-160` | Demo guide repeats three interfaces rather than one model (`02/demo/DEMO_GUIDE.md:18-57`) | Separate-repo Git task at `02/assignment/README.md:27-53` | `.git`, README, requirements existence only (`02/assignment/.github/test/test_assignment.py:19-41`) | **partially aligned**; working tree/staging meanings and GUI evidence are weak. |
| O2.2 Daily GUI workflow | VS Code sequence at `02/README.md:221-251` | Every step is repeated CLI/VS Code/GitHub at `02/demo/DEMO_GUIDE.md:20-199` | Assignment is CLI-oriented and asks for arbitrary history (`02/assignment/README.md:27-53`) | Only one commit is required in practice (`02/assignment/.github/test/test_assignment.py:170-184`) | **taught but unpracticed** in the required interface. |
| O2.3 GUI branch/merge/conflict | Branch and conflict section at `02/README.md:253-292` | Demo offers all three interfaces and later destructive reset material (`02/demo/DEMO_GUIDE.md:200-384`) | Two branches/merges requested at `02/assignment/README.md:31-45` | No branch, merge, or conflict assertion | **partially aligned**; required interface and grader evidence disagree. |
| O2.4 Function interface | Function section at `02/README.md:719-743` | Refactor demo at `02/demo/python_functions_demo.py:102-188` | Eleven required functions at `02/assignment/README.md:155-185` | Source-string checks for a subset of `def` names (`02/assignment/.github/test/test_assignment.py:94-116`) | **partially aligned**; assignment scope is excessive and tests do not validate behavior or edge cases. |
| O2.5 Import-safe two-file program | Objective only plus malformed main guard at `02/README.md:12`, `02/README.md:746-752` | Import demo at `02/demo/03_module_usage_demo.py:8-28`, but importing `python_functions_demo` runs extensive top-level code and writes files (`02/demo/python_functions_demo.py:8-100`, `02/demo/python_functions_demo.py:191-269`) | “Modular” second script duplicates rather than imports (`02/assignment/README.md:170-185`) | Tests only search function names and run scripts (`02/assignment/.github/test/test_assignment.py:94-140`) | **practiced but unassessed**, with the import-safety lesson contradicted by the demo module. |
| Orphan: shell automation | Shell scripting is a late review section (`02/README.md:866-920`) rather than a stated objective | Required `02_cli_advanced_demo.sh` (`02/demo/DEMO_GUIDE.md:3-12`) | Six-point required scaffold with heredocs/chmod (`02/assignment/README.md:97-131`) | Tests source-text spellings and executable bit (`02/assignment/.github/test/test_assignment.py:43-90`) | **orphaned**; move shell scripting to bonus and remove it from Assignment 02. |

### Recommended required demos

1. **Demo 1 — One GUI Git state cycle.** Starting with an instructor-provided disposable repository, edit one line, inspect the diff, stage only that file, commit with a focused message, synchronize, create a feature branch through the supported GUI, merge, and inspect history. Use a prepared conflict for one guided resolution. CLI Git spellings appear only in bonus/reference, not on the projected required path.
2. **Demo 2 — From duplication to functions.** Begin with a short Lecture 01-style script containing duplicated list calculations. Extract two functions while explicitly mapping argument to parameter, tracing a local variable, returning a value, adding one empty-input guard, and writing a one-sentence docstring.
3. **Demo 3 — From functions to an import-safe module.** Move the two functions into `analysis_utils.py`; import them into `main.py`; show that `import analysis_utils` produces no report or output; add `main()` and the main guard; run `python main.py` from the terminal.

Remove the required advanced shell demo. The current demo guide is also far too long and repeats operations in three interfaces (`02/demo/DEMO_GUIDE.md:18-384`), includes destructive `git reset --hard` and force-push material (`02/demo/DEMO_GUIDE.md:343-384`), and later embeds full solution-scale Python programs instead of concise direct demo steps (`02/demo/DEMO_GUIDE.md:802-1216`).

### Assignment 02 redesign contract

**Purpose:** demonstrate one GUI Git workflow and one coherent function-to-module refactor.

**Repository contract:** students work in the Classroom 50 assignment repository itself. Remove the instruction to create a separate repository, which currently makes the grader inspect a different location from the one described (`02/assignment/README.md:31-32`; `02/assignment/.github/test/test_assignment.py:22-24`).

**Three incremental questions:**

1. Use the supported GUI to inspect a starter change, stage it, commit it, create a feature branch, and commit a small README or code correction. Merge through the GUI. A supplied `GIT_CHECKPOINT.md` records commit IDs/branch name only if Classroom 50 cannot expose reliable history to the grader.
2. Refactor duplicated analysis in `main.py` into three functions in `analysis_utils.py`: one loader or supplied-data adapter, one calculation, and one report formatter. Each has a clear signature, return value, and one meaningful edge case.
3. Import the functions into `main.py`, orchestrate in `main()`, protect it with the main guard, run from the terminal, and create one deterministic report.

**Explicit non-requirements:** command-line Git, shell scripts, heredocs, executable permissions, a second repository, two parallel analysis scripts, generic extension dispatch, advanced grade distributions, and arbitrary commits-per-branch counts.

**Public-test contract:**

- import each function and test returned values with at least two fixtures, including empty input;
- import `analysis_utils` in a clean temporary directory and assert no stdout and no generated file;
- run `main.py` and validate exact report invariants rather than source text;
- check that required functions are actually used by `main.py` through behavior/mocking where practical, not merely that `def` strings exist;
- validate the GUI Git checkpoint separately from Python tests if history is unavailable;
- do not make private/hidden tests necessary for basic success; Classroom 50 public tests should state the observable contract.

### Core/bonus/drop disposition

| Current material | Disposition | Evidence and rationale |
|---|---|---|
| Themes, icon schemes, sidebars, breadcrumbs, Zen mode, extension shopping | **Drop or onboarding bonus** | These precede the central Git/function content (`02/README.md:31-90`) and are not course outcomes. |
| Repository, working tree, staging, commit, diff, remote | **Keep/revise as GUI-first core** | The mental model exists (`02/README.md:117-160`) but working tree/staging need explicit definitions and a single visual workflow. |
| Blob, tree, reference, HEAD internals | **Move to bonus** | Internal object vocabulary at `02/README.md:145-154` is not required for daily GUI Git competence. |
| CLI `git init/add/commit/push/pull/branch/merge` | **Move to `advanced_git.md`** | Main CLI lists and examples (`02/README.md:158-199`, `02/README.md:253-283`) conflict with the fixed GUI-first policy and duplicate existing bonus material. |
| One GUI feature branch and simple conflict | **Keep/revise** | Core branch creation via GUI is practical; current “branches come later” sentence contradicts the same-lecture branch section (`02/README.md:137-143`, `02/README.md:253-292`). |
| Pull requests, issues, web editing, broad GitHub interface | **Move to bonus or later collaboration material** | The web-interface survey at `02/README.md:294-347` is not necessary for the daily local workflow. |
| Minimal `.gitignore` and README/Markdown | **Keep as supporting practice, not separate survey** | `.gitignore` protects data/secrets (`02/README.md:308-341`); the broad Markdown section (`02/README.md:349-385`) can be reduced to the exact syntax used in assignments. |
| Repeated Python scalar/control/list recap | **Consolidate to Lecture 01** | The Python survey restarts at `02/README.md:387` and delays functions until `02/README.md:719`. Retain only a short prerequisite check. |
| Dictionaries | **Keep, reduce** | Named records/results are useful (`02/README.md:660-687`); sets and broad sequence-function surveys are not prerequisites. |
| Functions, parameters, returns, docstrings, empty-input guard | **Keep/reorder earlier** | Core function material at `02/README.md:719-743` is the lecture's main Python contribution and needs more precise definitions. |
| Modules, imports, `main()`, main guard | **Add/repair as core** | Objective 6 promises modules (`02/README.md:12`), but current content lacks a valid module explanation and the guard is malformed (`02/README.md:746-752`). |
| File I/O | **Reduce to one text read/write example** | Current file section is broad (`02/README.md:560-593`); only enough to support one deterministic report is necessary. CSV parsing can be supplied or postponed. |
| Shell scripting, advanced chaining, command shortcuts | **Move to bonus** | The late command-line review and scripting sections (`02/README.md:754-920`) duplicate Lecture 01 and create an unrelated Assignment 02 strand. |
| Reset-hard, force push, advanced history rewriting | **Bonus only, with recovery framing** | Current required guide includes destructive rollback (`02/demo/DEMO_GUIDE.md:343-384`); this is inappropriate for beginner core demonstrations. |
| “Suggested class plan” | **Drop** | Timing is outside the requested review scope (`02/README.md:22-28`). |

### Ordered content outline

1. Learning contract, Lecture 01 prerequisite check, and GUI-first Git policy.
2. Why version control: one evolving file, not a survey of tools.
3. Repository, working tree, change/diff, staging area, commit, local branch, remote.
4. One VS Code/GitHub Desktop edit → inspect → stage → commit → synchronize cycle.
5. Focused commit messages and minimal `.gitignore`/README practice.
6. Feature branch creation/switch/merge in the GUI; one prepared conflict and safe resolution.
7. Motivation for reusable code through visible duplication.
8. Minimal dictionary as a named record/result when needed.
9. Function definition/call; parameter versus argument; return versus print; local variable; docstring; empty input.
10. Module and import; script execution versus import; `main()` and exact main guard.
11. Two-file terminal workflow and import-safety check.
12. Bonus links: CLI Git, deeper Git internals, pull requests, advanced function features, and shell scripting.

---

## Lecture 03: reproducible environments and NumPy arrays

### Intended role

Teach one reproducible local environment workflow and the ndarray mental model strongly enough that students can execute a terminal-based NumPy analysis and enter Lecture 04 ready to compare arrays with labeled pandas objects.

### Proposed measurable objectives

By the end of Lecture 03, a student can:

1. **O3.1 — Recreate an environment:** distinguish interpreter, package, direct dependency, transitive dependency, environment, activation, and requirements file; create/activate a Python 3.12 environment with uv, install direct requirements, verify the selected interpreter and NumPy version, and recreate the environment from the recorded file. Standard-library `venv` is a supported fallback, not a second independently assessed path.
2. **O3.2 — Build a bounded terminal pipeline:** use `head`/`tail`, `cut`, `sort`, `uniq`, `wc`, a pipe, and output redirection to preview, select, count, and save a result from a simple CSV in the declared Bash environment.
3. **O3.3 — Explain and select array data:** create 1D/2D ndarrays, inspect `shape`, `ndim`, `size`, and `dtype`, select elements/rows/columns/slices, and distinguish a slice view from an explicit copy.
4. **O3.4 — Compute without element loops:** create a boolean mask, perform vectorized scalar/element-wise arithmetic, and calculate reductions over the whole array and along a named axis.
5. **O3.5 — Change and predict shape:** reshape and transpose an array and predict the shape/result of one simple scalar or compatible 1D broadcasting operation.

Array concatenation/stacking is not required for the Lecture 04 handoff and moves to bonus. If it remains a course objective, add one core example and assessment; the current objective says “combine” while all combination material is bonus (`03/README.md:11-12`; `03/BONUS.md:38-55`).

### Incoming prerequisites from Lecture 02

- Navigate directories and run scripts from the terminal.
- Use the supported GUI Git workflow for the assignment repository.
- Define/call functions and import an import-safe local module.
- Use lists, a minimal dictionary, conditions, and loops.
- Resolve paths for one small text input/output.

### Outgoing prerequisites supplied to Lecture 04

- Explain environment isolation and identify the active interpreter.
- Install from and recreate an environment using a direct-dependency file.
- Run a NumPy-backed `.py` script from a clean terminal environment.
- Explain 1D versus 2D, element, dimension, shape, dtype, and axis.
- Use array indexing/slicing, boolean masks, vectorized arithmetic, reductions, reshape, transpose, and one simple broadcast.
- Recognize that a slice may share data with its source.
- No notebook, cell, kernel, runtime, pandas index/label, DataFrame, or Colab knowledge is assumed; Lecture 04 defines them.

### Term first-definition and first-independent-use ledger

| Term or skill | Current first definition/guided evidence | Current first independent or demanding use | Finding and target |
|---|---|---|---|
| interpreter | The current environment section discusses Python versions and environments but never plainly distinguishes the interpreter executable from its packages (`03/README.md:37-63`). | Assignment asks students to create/activate a venv and run Python (`03/assignment/README.md:99-138`). | **P0 define first:** the program that executes Python code; verify its path before package installation. |
| package / dependency | `pip install` and “dependencies” are used operationally (`03/README.md:47-63`); direct dependency is named only in objective 4 (`03/README.md:10`). | Assignment installs NumPy from requirements (`03/assignment/README.md:99-138`). | **P0 define:** package is installable code; dependency is a package the project needs; distinguish direct from transitive before recording. |
| environment / isolation / activation | The project-version problem and environment solution are introduced at `03/README.md:37-46`; commands follow at `03/README.md:47-87`. | Assignment independently uses `venv`, not the lecture's uv default (`03/assignment/README.md:99-130`). | **Revise/align:** uv is the required path; venv is a clearly labeled fallback with equivalent verification steps. |
| requirements file / reproducibility | The lecture uses `uv pip freeze > requirements.txt` (`03/README.md:53-63`), which records transitive packages rather than just the direct dependencies promised by objective 4. | Assignment supplies requirements and treats successful import as proof of environment management (`03/assignment/README.md:99-138`). | **P0 revise:** teach direct requirements deliberately, then demonstrate recreation in a clean environment; a successful import alone does not prove isolation. |
| pipeline / pipe / redirection | Pipeline is defined at `03/README.md:514-536`; commands follow at `03/README.md:538-590`. | Assignment independently requires four pipelines and `awk` (`03/assignment/README.md:47-97`). | **Keep/reduce:** core uses a bounded, portable Bash subset; `awk`, `sed`, `tr`, sparklines, and gnuplot move to bonus. |
| ndarray / element / dimension | `ndarray` is described at `03/README.md:195-200`; 1D/2D creation follows at `03/README.md:206-222`. | Demo 03 uses 2D grade arrays and functions immediately (`03/demo/demo3_student_analysis.py:9-34`). | **Keep/revise:** explicitly map list → 1D array and nested lists → 2D array, then define element and dimension. |
| shape / `ndim` / size / dtype | Properties are demonstrated at `03/README.md:224-253`. | Assignment works with a structured dtype supplied in the loader (`03/assignment/README.md:148-170`). | **P0 align:** assess ordinary homogeneous numeric arrays; structured dtype is advanced and currently defined only in bonus (`03/BONUS.md:219-242`). |
| index / slice / view / copy | Basic/multidimensional indexing and slicing appear at `03/README.md:255-312`; view/copy is defined at `03/README.md:332-351`. | Demo 03 uses rows/columns and later `grades.flat`, `reshape`, and transpose (`03/demo/demo3_student_analysis.py:25-34`, `03/demo/demo3_student_analysis.py:80-96`). | **Keep/reorder:** basic selection → slice → demonstrate shared mutation → `.copy()` before more complex transformations. Fancy indexing moves to bonus. |
| boolean mask / boolean indexing | Defined and exemplified at `03/README.md:293-312`. | Assignment independently filters structured fields at `03/assignment/README.md:193-210`. | **Keep, change data model:** test masks on ordinary 1D/2D numeric arrays, not structured fields. |
| vectorized operation | Defined at `03/README.md:195-200` and demonstrated at `03/README.md:353-374`. | Demo compares list and NumPy timing (`03/demo/demo3_numpy_performance.py:10-60`). | **Keep/revise:** emphasize element-wise meaning and readable code; report measured results without promising a universal speed factor. |
| reduction / axis | Statistical methods and `axis=0/1` examples appear at `03/README.md:376-394`. | Demo asks for row/column means and later uses `argmin`/`argsort` before those are core (`03/demo/demo3_student_analysis.py:35-55`, `03/demo/demo3_student_analysis.py:98-125`). | **Keep/reduce:** define axis as the dimension collapsed by a reduction; use labeled rows/columns and one prediction exercise. Move ranking/sorting to bonus. |
| reshape / transpose / broadcasting | Reshape/transpose are core at `03/README.md:396-411`; broadcasting is only named at `03/README.md:195-200` and defined in bonus (`03/BONUS.md:22-36`). | Demo independently reshapes and transposes (`03/demo/demo3_student_analysis.py:80-96`); no assignment checks either. | **P0 align:** retain reshape/transpose plus one simple core broadcast, or remove them from required objectives. |
| structured array | Defined only in bonus (`03/BONUS.md:219-242`). | Assignment loader and all analysis use structured fields (`03/assignment/README.md:148-210`). | **P0 untaught but assessed:** replace the assignment data representation or promote and practice structured arrays. Replacement is preferable before pandas. |

### Objective-to-artifact alignment matrix

| Objective | Current lecture section | Current guided demo | Current assignment evidence | Current test/rubric evidence | Current status |
|---|---|---|---|---|---|
| O3.1 Recreate environment | uv default and venv fallback at `03/README.md:37-113` | Quick run uses uv (`03/demo/DEMO_GUIDE.md:3-20`), but detailed guide calls uv optional (`03/demo/DEMO_GUIDE.md:641-680`) | Assignment requires venv (`03/assignment/README.md:99-138`) | Tests explicitly do not test the environment (`03/assignment/.github/test/test_assignment.py:18-20`) | **partially aligned** with a three-way policy contradiction and no recreation evidence. |
| O3.2 Bounded terminal pipeline | Broad command-line processing at `03/README.md:514-590` | Demo 4 is manual and claims a 1,500-row file (`03/demo/DEMO_GUIDE.md:780-805`) | Four required shell outputs including `awk` at `03/assignment/README.md:47-97` | Tests only check broad output ranges/formats (`03/assignment/.github/test/test_assignment.py:23-92`) | **partially aligned**; required tool scope is too wide and tests can accept plausible hard-coded values. |
| O3.3 Array model/selection/view-copy | Array properties and selection at `03/README.md:206-351` | Student analysis covers properties/indexing but not a visible view-mutation lesson (`03/demo/demo3_student_analysis.py:9-34`) | Assignment uses supplied structured arrays (`03/assignment/README.md:148-210`) | Report-level tests do not exercise indexing or view/copy (`03/assignment/.github/test/test_assignment.py:95-165`) | **partially aligned**; ordinary array reasoning is not assessed. |
| O3.4 Masks/vectorized reductions/axis | `03/README.md:293-394` | Demo covers masks and axes at `03/demo/demo3_student_analysis.py:35-78` | Current scaffold asks means and masks on fields (`03/assignment/analyze_health_data.py:32-68`) | Tests only search report text and plausible numbers (`03/assignment/.github/test/test_assignment.py:115-165`) | **partially aligned**; implementation behavior is not tested and the representation is untaught. |
| O3.5 Reshape/transpose/simple broadcast | Reshape/transpose at `03/README.md:396-411`; broadcast only bonus at `03/BONUS.md:22-36` | Reshape/transpose demo at `03/demo/demo3_student_analysis.py:80-96` | No required task | No test | **practiced but unassessed**; broadcasting is **taught only in bonus** despite the current objective. |
| Orphan: structured arrays | Bonus only at `03/BONUS.md:219-242` | No focused required demo | Central assignment representation at `03/assignment/README.md:148-210` | Indirect report tests only | **untaught but assessed**. |

### Recommended required demos

1. **Demo 1 — Reproduce one environment.** Starting from a small supplied project: identify the current interpreter, create `.venv` with uv/Python 3.12, activate it, verify interpreter path, install direct requirements, run a NumPy version check and a tiny script, then recreate the same project environment in a separate disposable directory from `requirements.txt`. A short venv fallback box mirrors the same outcomes without becoming a second demo.
2. **Demo 2 — Build the ndarray mental model.** Use tiny literal arrays whose values fit on screen. Progress through list → ndarray, 1D/2D, shape/ndim/size/dtype, element/row/column/slice, visible view mutation and `.copy()`, mask, vectorized arithmetic, whole-array and axis reductions, reshape/transpose, and one shape-compatible broadcast.
3. **Demo 3 — One terminal analysis.** On a small deterministic CSV, use a short Bash pipeline to preview/select/count and save an output, then run a provided-loader Python script that returns a plain 2D numeric ndarray. Students fill or predict masks and axis reductions and run the completed `.py` script from the activated environment.

The current guide runs five scripts (`03/demo/DEMO_GUIDE.md:15-20`), devotes its first 634 lines to an Assignment 02 solution (`03/demo/DEMO_GUIDE.md:24-234`, `03/demo/DEMO_GUIDE.md:236-633`), and includes a redundant “potpourri” demo of Lecture 01/02 material (`03/demo/demo2_python_potpourri.py:9-128`). The performance script can be a brief optional hook; its universal “10-100x” conclusion is not justified by one measurement (`03/demo/demo3_numpy_performance.py:46-60`).

### Assignment 03 redesign contract

**Purpose:** demonstrate environment recreation, a bounded CLI pipeline, and core NumPy reasoning using homogeneous numeric arrays.

**Three incremental questions:**

1. Run a supplied `environment_check.py` from an activated uv environment created from `requirements.txt`; save deterministic interpreter-major/minor and NumPy-version evidence to `output/environment_check.txt`. The environment directory is never committed. The instructions include a clearly separated venv fallback.
2. Complete functions in `array_analysis.py` over a plain 2D numeric ndarray returned by supplied loader machinery: inspect metadata, select a column/slice, make an explicit copy before mutation, create a mask, calculate whole/row/column reductions, and reshape one supplied vector. Function signatures and return values are the assessed interface.
3. Complete a short `cli_checks.sh` or equivalent declared-Bash command file using only `head`/`tail`, `cut`, `sort`, `uniq`, `wc`, pipes, and redirection; then run `analysis.py` to combine the returned NumPy results into one deterministic report. No notebook is introduced.

**Data contract:** use a small, pinned, deterministic CSV and a provided loader that returns a homogeneous numeric ndarray. Avoid medical “abnormal” labels unless domain thresholds and synthetic-data caveats are explicit. The current generator's `patient_num` parameter is ignored (`03/assignment/generate_health_data.py:26-35`) and it samples IDs randomly while claiming exactly 10,000 unique patients (`03/assignment/generate_health_data.py:81-94`); fixtures must instead guarantee their documented invariants.

**Explicit non-requirements:** structured arrays, `np.genfromtxt` mastery, random-data debugging, 50,000 rows, `awk`/`sed`/`tr`, terminal plotting, fancy indexing, sorting/ranking, concatenation, Jupyter, or Colab.

**Public-test contract:**

- import and test each required NumPy function directly on at least two arrays with different shapes and values;
- assert exact shapes, dtypes where relevant, returned arrays/statistics, mask behavior, and axis semantics;
- test that copy-required work does not mutate the original;
- execute the CLI artifact against a tiny alternate fixture and compare exact output, preventing hard-coded plausible counts;
- execute `analysis.py` from a temporary working directory if path independence is promised;
- validate exact report invariants, not merely the presence of words such as “Average” or five arbitrary numbers (the current tests do the latter at `03/assignment/.github/test/test_assignment.py:120-165`);
- validate the grader with known-correct, partially correct, loop-based where prohibited, wrong-axis, hard-coded, and broken-path fixtures.

### Core/bonus/drop disposition

| Current material | Disposition | Evidence and rationale |
|---|---|---|
| One uv/Python 3.12 environment workflow | **Keep/revise as required core** | The current default is explicit (`03/README.md:47-63`), but detailed demos and assignment must stop presenting different defaults. |
| Standard-library venv | **Keep as supported fallback** | The fallback is useful (`03/README.md:65-87`) but should not be a second objective or the only assignment path. |
| Additional uv commands and Conda | **Consolidate / move Conda to bonus** | Repeated environment creation at `03/README.md:89-113` adds little; Conda at `03/README.md:115-136` is a third path. |
| Interpreter/package/direct/transitive dependency/requirements/recreation | **Add/revise as core** | Objective 4 promises direct dependencies (`03/README.md:10`) but current commands use undifferentiated freeze output (`03/README.md:53-63`). |
| Python potpourri | **Drop from required Lecture 03** | Type checking and f-strings repeat earlier material and even use NumPy before its introduction (`03/README.md:138-172`). |
| ndarray creation/properties/dtype | **Keep** | These establish the mental model (`03/README.md:195-253`). |
| Basic indexing/slicing, masks, views/copies | **Keep/reorder** | Core daily operations are present (`03/README.md:255-351`) and directly prepare pandas selection. |
| Fancy indexing | **Move to bonus** | It is useful but not required for the Lecture 04 bridge (`03/README.md:314-330`). |
| Vectorized arithmetic and reductions by axis | **Keep** | Central NumPy concepts at `03/README.md:353-394`; examples need clearer axis language and direct tests. |
| Reshape, transpose, one simple broadcast | **Keep/revise** | Reshape/transpose are already core (`03/README.md:396-411`); add one minimal broadcasting example if it remains objective. |
| Concatenation/stacking and advanced broadcasting rules | **Keep in bonus** | Existing bonus location is appropriate (`03/BONUS.md:22-55`). |
| Ufunc survey, `np.where`, boolean methods, sorting, RNG | **Move to bonus** | These appear under an admitted “Semi-Advanced” section (`03/README.md:413-502`) and are not needed by the proposed assignment. |
| Bounded CLI pipeline | **Keep/reduce as core** | CLI processing is an explicit early-course thread (`03/README.md:514-590`), but only a small declared-Bash subset should be required. |
| `awk`, `sed`, `tr`, sparklines, gnuplot | **Move to bonus** | Advanced processing and terminal visualization at `03/README.md:562-622` expand platform dependencies and distract from NumPy. |
| Structured arrays | **Keep in bonus; remove from assignment** | They are correctly located in bonus (`03/BONUS.md:219-242`) but currently assessed (`03/assignment/README.md:148-210`). |
| Large random health generator | **Replace** | Its documented unique-patient invariant is false because random sampling ignores `patient_num` (`03/assignment/generate_health_data.py:26-35`, `03/assignment/generate_health_data.py:81-94`). Small deterministic fixtures support better reasoning and tests. |
| “Suggested class plan” | **Drop** | Timing is outside the requested review scope (`03/README.md:21-27`). |

### Ordered content outline

1. Learning contract, exact Lecture 02 prerequisites, final pre-Jupyter boundary, and supported Bash environment.
2. Reproducibility problem: interpreter, package, dependency, direct/transitive dependency, environment, activation, requirements file.
3. One uv workflow: create, activate, verify interpreter, install direct requirements, run, record, recreate; venv fallback box.
4. Bounded terminal pipeline: stdout, pipe, redirection, preview/select/count/save using the supported command subset.
5. Why arrays: homogeneous numeric data and vectorized operations, with measured—not promised—performance.
6. ndarray, element, dimension, shape, `ndim`, size, dtype; 1D versus 2D.
7. Element/row/column indexing and slices; view versus explicit copy.
8. Boolean masks and filtered selection.
9. Scalar and element-wise vectorized arithmetic.
10. Whole-array reductions; axis as the dimension collapsed; row/column predictions.
11. Reshape and transpose; one simple compatible broadcast and predicted output shape.
12. Integrated terminal script, environment recreation check, and explicit bridge to Lecture 04 Series/DataFrame labels and notebook runtime/state.

---

## Range-wide content disposition summary

| Canonical home | Required core | Bonus/later boundary |
|---|---|---|
| Lecture 01 | terminal orientation, paths, safe file work, VS Code terminal, scalar values, lists, conditions, simple loops, traceback reading | shell expansion/find, advanced formatting/input, functions, file formats, exceptions, automation |
| Lecture 02 | GUI Git model/workflow/branch, minimal dictionary, functions, text I/O, modules/import/main guard | CLI Git, Git internals/history rewriting, shell scripting, advanced functions/comprehensions, broad GitHub collaboration |
| Lecture 03 | uv environment/recreation, dependency vocabulary, bounded Bash pipeline, ndarray model/indexing/masks/views/vectorization/reductions/axis/reshape/simple broadcast | Conda, advanced shell utilities/terminal plots, fancy indexing, structured arrays, concatenation, RNG/sorting/advanced ufuncs |

## Ordered implementation sequence after approval

1. Declare the supported required shell and document the Classroom 50 Git-history limitation or capability.
2. Approve the proposed objectives, prerequisites, term homes, and ordered outlines for Lectures 01–03.
3. Rewrite Lecture 01 core and bonus boundaries; then rebuild its three demos and new Assignment 01.
4. Rewrite Lecture 02 around GUI Git and functions/modules; then rebuild its three demos and Assignment 02.
5. Rewrite Lecture 03 around one environment workflow, bounded CLI, and ordinary ndarrays; then rebuild its three demos and Assignment 03.
6. Independently verify Lecture 01→02, 02→03, and 03→04 boundaries.
7. Run all scripts from clean directories/environments and validate known-correct, partial, hard-coded, wrong-axis, import-side-effect, and broken-path assignment fixtures.
8. Package all three assignments for Classroom 50 only after their content/test contracts pass. No Colab certification is needed for Lectures 01–03.

## Remaining instructor decisions

1. **Resolved in the course map:** require the shared POSIX command subset through Bash on Linux/WSL/supported cloud or default zsh on macOS; native PowerShell is onboarding only.
2. **GUI branch evidence:** confirm whether Classroom 50 preserves sufficient history and branch refs; otherwise approve a small manual checkpoint.
3. **Assignment 01 submission:** approve the first guided GUI sync as unassessed platform onboarding so Lecture 01 does not assess Git before Lecture 02.
4. **Resolved in the course map:** deliberate direct requirements belong in `requirements.txt`; exact transitive versions belong in the release constraints/lock artifact. Do not call an unreviewed `freeze` result a direct-dependency list.
5. **Resolved in the course map:** retain one simple compatible-shape broadcasting example in core; concatenation and advanced broadcasting rules remain bonus.
