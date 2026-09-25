#!/usr/bin/env bash
# Publish one assignment's student-facing folder to its own GitHub repository.
#
#   scripts/publish_assignment.sh 02 UCSF-DataSci/ds217-26f-02 [--dry-run] [--replace-history]
#
# The repository gets exactly what is in NN/assignment/: the README, scaffolds, data and
# .github/ workflow a student forks. NN/assignment_checks/ is course-side and never goes.
# Files the target has and this folder does not are deleted, so the repository ends up
# matching the source folder rather than accumulating leftovers.
#
# By default this makes an ordinary commit on top of the existing history, which is enough
# to replace the visible content and keeps anything already forked working. --replace-history
# discards the target's history and force-pushes a single fresh commit; it is destructive and
# breaks existing forks and clones, so it is opt-in.
set -euo pipefail

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

[ $# -ge 2 ] || usage
NUMBER="$1"; TARGET="$2"; shift 2
DRY_RUN=no; REPLACE_HISTORY=no
for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=yes ;;
    --replace-history) REPLACE_HISTORY=yes ;;
    *) echo "unknown option: $argument" >&2; usage ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/$NUMBER/assignment"
[ -d "$SOURCE" ] || { echo "no such assignment folder: $SOURCE" >&2; exit 1; }
case "$TARGET" in
  */datasci_217|*/datasci_217.git) echo "refusing to publish into the course repository" >&2; exit 1 ;;
  */*) ;;
  *) echo "target must be owner/repo, a git URL, or a local path" >&2; exit 1 ;;
esac
# owner/repo is expanded; a URL or an on-disk path (handy for testing) is used as given.
case "$TARGET" in
  *://*|git@*|/*|./*|../*) URL="$TARGET" ;;
  *) URL="https://github.com/$TARGET.git" ;;
esac

# The checker must actually run before anyone grades with it.
python3 -c "
import json, pathlib, subprocess, sys, tempfile
source = pathlib.Path('$SOURCE')
if not (source / 'check_assignment.py').exists():
    sys.exit(0)
with tempfile.TemporaryDirectory() as empty:
    run = subprocess.run([sys.executable, '-B', str(source / 'check_assignment.py'), empty, '--json'],
                         cwd=source, capture_output=True, text=True)
report = json.loads(run.stdout or '{}')
assert report.get('tests'), 'checker reported no tests'
assert report.get('max-score'), 'checker reported no points'
print(f\"  checker: {len(report['tests'])} checks, {report['max-score']} points, scores an empty submission {report['score']}\")
" || { echo "the assignment's own checker does not run; fix that before publishing" >&2; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
echo "  cloning $URL"
git clone --quiet "$URL" "$WORK/repo" 2>/dev/null || { echo "cannot clone $URL (does it exist, and do you have access?)" >&2; exit 1; }

# Mirror the folder into the checkout, keeping .git and dropping build clutter.
rsync -a --delete --exclude '.git/' \
      --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
      --exclude '_grader_selftest/' \
      "$SOURCE/" "$WORK/repo/"
find "$WORK/repo" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

cd "$WORK/repo"
git add -A
# Matching content still needs a push when the point is to discard the old history.
if git diff --cached --quiet && [ "$REPLACE_HISTORY" = no ]; then
  echo "  no change: $TARGET already matches $NUMBER/assignment"; exit 0
fi

if git diff --cached --quiet; then
  echo "  content already matches; replacing history anyway"
else
  echo "  changes to publish:"
  git diff --cached --stat | sed 's/^/    /'
fi

if [ "$DRY_RUN" = yes ]; then echo "  dry run: nothing pushed"; exit 0; fi

MESSAGE="Publish assignment $NUMBER from the course repository

Source: $(git -C "$ROOT" rev-parse --short HEAD) on $(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"

if [ "$REPLACE_HISTORY" = yes ]; then
  echo "  replacing history (this breaks existing forks and clones)"
  BRANCH="$(git symbolic-ref --short HEAD)"
  git checkout --quiet --orphan published
  git add -A
  git commit --quiet -m "$MESSAGE"
  git branch -M published "$BRANCH"
  git push --force --quiet origin "$BRANCH"
else
  git commit --quiet -m "$MESSAGE"
  git push --quiet origin HEAD
fi
echo "  published $NUMBER/assignment to $TARGET"
