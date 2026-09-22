"""Size metrics per lecture: prose words, code lines, and structure."""
import re, sys
from pathlib import Path

def metrics(path):
    lines = Path(path).read_text().splitlines()
    fence = False
    prose_words = code_lines = topics = subs = cards = snips = images = demos = tables = 0
    for l in lines:
        if l.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            code_lines += 1
            continue
        if l.strip() == "# LIVE DEMO!" or l.strip() == "# LIVE DEMO":
            demos += 1
            continue
        m = re.match(r"^(#{1,4}) (.*)", l)
        if m:
            lvl, title = len(m.group(1)), m.group(2)
            if lvl == 1: topics += 1
            elif lvl == 2: subs += 1
            if title.startswith("Reference Card"): cards += 1
            if title.startswith("Code Snippet"): snips += 1
            continue
        if l.strip().startswith("!["): images += 1; continue
        if l.strip().startswith("|"): tables += 1; continue
        prose_words += len(l.split())
    return dict(lines=len(lines), prose_words=prose_words, code=code_lines, topics=topics,
                subs=subs, cards=cards, snippets=snips, images=images, table_rows=tables, demos=demos)

print(f"{'lec':<5}{'lines':>7}{'prose':>8}{'code':>7}{'topics':>8}{'subs':>6}{'cards':>7}{'snips':>7}{'imgs':>6}{'tbl':>6}{'demos':>7}")
tot = {}
for nn in [f"{i:02d}" for i in range(1, 12)]:
    m = metrics(f"{nn}/README.md")
    print(f"{nn:<5}{m['lines']:>7}{m['prose_words']:>8}{m['code']:>7}{m['topics']:>8}{m['subs']:>6}{m['cards']:>7}{m['snippets']:>7}{m['images']:>6}{m['table_rows']:>6}{m['demos']:>7}")
    for k, v in m.items(): tot[k] = tot.get(k, 0) + v
print(f"{'sum':<5}{tot['lines']:>7}{tot['prose_words']:>8}{tot['code']:>7}{tot['topics']:>8}{tot['subs']:>6}{tot['cards']:>7}{tot['snippets']:>7}{tot['images']:>6}{tot['table_rows']:>6}{tot['demos']:>7}")
