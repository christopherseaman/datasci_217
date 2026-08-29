# Advanced Python CLI Topics

*Optional reference for students interested in command-line data workflows.*

This page owns shell and CLI-specific extensions. Python function design and
object-model extensions live in `bonus_python_concepts.md`; the core lecture
already introduces ordinary functions, lambdas, and the main guard.

## Shell scripts with arguments

Shell scripts can turn a repeatable pipeline into a small command-line tool.
Quote paths and validate inputs before processing them.

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s FILE\n' "$0" >&2
    exit 2
fi

input=$1
[[ -f "$input" ]] || { printf 'not a file: %s\n' "$input" >&2; exit 1; }

# grep status 1 means "no matches," which is a valid clean result. Capture the
# matches so statuses above 1 can still stop the script as real failures.
if matches=$(grep -i 'error' "$input"); then
    :
else
    status=$?
    if (( status != 1 )); then
        exit "$status"
    fi
fi
printf '%s' "$matches" | sort | tee errors.txt
```

Useful shell variables include `$1` (the first argument), `$@` (all
arguments), `$#` (argument count), and `$?` (the previous command's status).

## Pipelines, redirection, and process substitution

Pipes connect stdout to stdin. `&&` continues only after success, `||` handles
failure, and explicit redirections make output destinations clear.

```bash
grep -i 'error' logfile.txt | wc -l > error-count.txt
backup_script.sh > backup.log 2>&1
diff <(sort file1.txt) <(sort file2.txt)
```

For larger batches, `find` can safely pass null-delimited paths to a loop:

```bash
find data -name '*.csv' -print0 |
while IFS= read -r -d '' file; do
    printf 'processing %s\n' "$file"
done
```

## Command-line data processing

Small Unix tools are useful for inspection before a Python program takes over:

```bash
cut -d',' -f1,3 data.csv |        # select fields
  tr '[:lower:]' '[:upper:]' |   # normalize case
  sort -t',' -k2,2n |            # order by field 2
  head -n 10 > results.csv
```

`grep`, `cut`, `sort`, `uniq`, `tr`, `sed`, and `awk` each do one focused
transformation. Check quoting and delimiters for the actual input format.

## Calling Python from a shell

The shell is often the orchestrator while Python owns domain logic. Pass input
through arguments or standard input rather than relying on hidden state:

```bash
python3 summarize.py data.csv --output summary.json
python3 -c 'import sys; print(sum(map(float, sys.stdin)))' < values.txt
```

## Further directions

Explore `xargs`, `tee`, process substitution, and shell completion in a
disposable practice directory. For substantial transformations, prefer a
tested Python script so parsing, errors, and edge cases are explicit.
