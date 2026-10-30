#!/usr/bin/env python3
"""Replace a published page's placeholder image, video, PDF, and bookmark blocks with real ones.

`ntn pages edit` does not resolve the `file-upload://` sources that
scripts/notion_publish.py emits: it stores each one as an external block with an
empty URL, which renders broken. This walks a published page, matches every such
block to its local file by type and caption, uploads the file, inserts a proper
block of the same type in its place, and deletes the placeholder. Each bookmark
arrives as a paragraph holding only its URL as a link (scripts/notion_publish.py),
and is replaced the same way by a bookmark block.

    python3 scripts/notion_attach_media.py <page-id> <source.md> [--dry-run]

Run it immediately after publishing that page. An upload that is never attached
expires about an hour after it is created.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notion_media import IMAGE_SUFFIXES, MEDIA_TAG, bookmark_urls, run_ntn, upload  # noqa: E402

MEDIA_TYPES = ("image", "video", "pdf")
LINK = re.compile(r"!\[([^]]*)\]\(([^)]+)\)")
ENV = {**os.environ, "NOTION_KEYRING": "0"}


def api(path: str, *arguments: str, body: dict | None = None) -> dict:
    command = ["ntn", "api", path, *arguments]
    if body is not None:
        command += ["-d", json.dumps(body)]
    result = run_ntn(command)
    if result.returncode != 0:
        raise RuntimeError(f"{path}: {result.stderr.strip()[:300]}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def page_blocks(page: str) -> list[dict]:
    """Every block on the page, including those inside columns, callouts, and lists."""
    cursor, found = None, []
    while True:
        query = f"/v1/blocks/{page}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
        data = api(query)
        found += data.get("results", [])
        if not data.get("has_more"):
            break
        cursor = data["next_cursor"]
    for block in list(found):
        if block.get("has_children") and block["type"] not in ("child_page", "child_database"):
            found += page_blocks(block["id"])
    return found


def parent_of(block: dict) -> str:
    parent = block.get("parent", {})
    return parent.get("block_id") or parent.get("page_id")


def captioned_media(source: Path) -> list[tuple[str, str, Path]]:
    """Every local image, video, and PDF the document embeds, as (block type, caption, path), in order."""
    found, fenced = [], False
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        images = [("image", caption, target) for caption, target in LINK.findall(line)
                  if Path(target.split("#", 1)[0]).suffix.lower() in IMAGE_SUFFIXES]
        for kind, caption, target in images + [(k, c, t) for k, t, c in MEDIA_TAG.findall(line)]:
            if target.startswith(("http://", "https://", "data:", "file-upload://")):
                continue
            local = (source.parent / target.split("#", 1)[0]).resolve()
            if local.is_file():
                found.append((kind, caption.strip(), local))
    return found


def caption_key(caption: str) -> str:
    """A caption as Notion stores it: plain text, so Markdown code and emphasis marks are gone."""
    return re.sub(r"[`*_]", "", caption).strip()


def placeholder(block: dict) -> bool:
    """True when the media block has no usable source, however it is typed."""
    media = block.get(block["type"], {})
    return not (media.get(media.get("type")) or {}).get("url")


def caption_of(block: dict) -> str:
    return "".join(run.get("plain_text", "") for run in block[block["type"]].get("caption", [])).strip()


def link_only(block: dict) -> str | None:
    """The URL of a paragraph whose whole text is that URL linked to itself, as a bookmark placeholder is."""
    if block["type"] != "paragraph":
        return None
    runs = block["paragraph"].get("rich_text", [])
    text = "".join(run.get("plain_text", "") for run in runs)
    return text if runs and all(run.get("href") == text for run in runs) else None


def attach_bookmarks(every: list[dict], after: dict, urls: list[str], page: str, dry_run: bool) -> int:
    """Swap each bookmark placeholder paragraph for a bookmark block in the same place."""
    count = 0
    for block in every:
        url = link_only(block)
        if url not in urls:
            continue
        following = after.get(block["id"])
        if following and following["type"] == "bookmark" and following["bookmark"].get("url") == url:
            if not dry_run:
                api(f"/v1/blocks/{block['id']}", "-X", "DELETE")
            print(f"bookmark already present for {url}; removed its placeholder")
        elif dry_run:
            print(f"would bookmark {url}")
        else:
            parent = block.get("parent", {}).get("block_id") or page
            api(f"/v1/blocks/{parent}/children", "-X", "PATCH", body={
                "position": {"type": "after_block", "after_block": {"id": block["id"]}},
                "children": [{"object": "block", "type": "bookmark", "bookmark": {"url": url}}],
            })
            api(f"/v1/blocks/{block['id']}", "-X", "DELETE")
            print(f"bookmarked {url}")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page")
    parser.add_argument("source", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    media = captioned_media(args.source)
    wanted = {(kind, caption_key(caption)): path for kind, caption, path in media if caption}
    # Uncaptioned media have no key, so the nth uncaptioned block of a type on the page gets the nth one in the file.
    # ponytail: nested blocks come after top-level ones in page_blocks, so uncaptioned media inside columns may pair out of order.
    every = page_blocks(args.page)
    blocks = [b for b in every if b["type"] in MEDIA_TYPES]
    by_order = {}
    for kind in MEDIA_TYPES:
        uncaptioned = [path for k, caption, path in media if k == kind and not caption]
        by_order.update({b["id"]: path for b, path in zip(
            [b for b in blocks if b["type"] == kind and not caption_of(b)], uncaptioned)})
    broken = [b for b in blocks if placeholder(b)]
    # The block right after each block in its own parent: a rerun after a failed delete finds the copy already there.
    # page_blocks lists each parent's children together and in page order.
    after = {}
    for parent in {parent_of(b) for b in every}:
        siblings = [b for b in every if parent_of(b) == parent]
        after.update({a["id"]: b for a, b in zip(siblings, siblings[1:])})
    marked = attach_bookmarks(every, after, bookmark_urls(args.source), args.page, args.dry_run)
    if not broken:
        print(f"{marked} bookmark(s), no placeholder media on this page")
        return 0

    repaired, skipped = 0, []
    for block in broken:
        caption = caption_of(block)
        kind = block["type"]
        local = wanted.get((kind, caption_key(caption))) if caption else by_order.get(block["id"])
        if local is None:
            skipped.append(caption[:70] or "(no caption)")
            continue
        following = after.get(block["id"])
        if (following and following["type"] == kind and not placeholder(following)
                and caption_of(following) == caption):
            if not args.dry_run:
                api(f"/v1/blocks/{block['id']}", "-X", "DELETE")
            print(f"already attached {local.name}; removed its placeholder")
            repaired += 1
            continue
        if args.dry_run:
            print(f"would attach {local.name} for {caption[:60]!r}")
            repaired += 1
            continue
        upload_id = upload(local)
        # A nested block (inside a column, say) is replaced within its own parent block.
        parent = block.get("parent", {}).get("block_id") or args.page
        api(f"/v1/blocks/{parent}/children", "-X", "PATCH", body={
            "position": {"type": "after_block", "after_block": {"id": block["id"]}},
            "children": [{
                "object": "block", "type": kind,
                kind: {
                    "type": "file_upload",
                    "file_upload": {"id": upload_id},
                    "caption": block[kind].get("caption", []),
                },
            }],
        })
        api(f"/v1/blocks/{block['id']}", "-X", "DELETE")
        print(f"attached {local.name}")
        repaired += 1

    print(f"{marked} bookmark(s), {repaired} repaired, {len(skipped)} unmatched")
    for caption in skipped:
        print(f"  no local media captioned {caption!r}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
