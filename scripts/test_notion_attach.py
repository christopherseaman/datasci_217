import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import notion_attach_media
import notion_media


class RunNtnTest(unittest.TestCase):
    def test_retries_while_rate_limited_then_returns_success(self):
        with tempfile.TemporaryDirectory(dir="scratch") as tmp:
            count = Path(tmp) / "count"
            fake = Path(tmp) / "ntn"
            # Fails with a 429 twice, then succeeds.
            fake.write_text(
                f"#!/bin/sh\nn=$(cat {count} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {count}\n"
                'if [ $n -lt 3 ]; then echo "429 Too Many Requests" >&2; exit 1; fi\necho \'{"ok": true}\'\n'
            )
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            path = f"{tmp}{os.pathsep}{os.environ['PATH']}"
            with mock.patch.object(notion_media, "RETRY_DELAYS", (0, 0, 0, 0)), \
                    mock.patch.dict(os.environ, {"PATH": path}):
                result = notion_media.run_ntn(["ntn", "api", "/v1/x"])
            self.assertEqual(result.returncode, 0)
            self.assertEqual(count.read_text().strip(), "3")

    def test_other_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory(dir="scratch") as tmp:
            count = Path(tmp) / "count"
            fake = Path(tmp) / "ntn"
            fake.write_text(f"#!/bin/sh\necho x >> {count}\necho 'not found' >&2\nexit 1\n")
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            path = f"{tmp}{os.pathsep}{os.environ['PATH']}"
            with mock.patch.object(notion_media, "RETRY_DELAYS", (0, 0, 0)), \
                    mock.patch.dict(os.environ, {"PATH": path}):
                result = notion_media.run_ntn(["ntn", "api", "/v1/x"])
            self.assertEqual(result.returncode, 1)
            self.assertEqual(len(count.read_text().split()), 1)


def image(block_id, caption, url=""):
    return {"id": block_id, "type": "image", "parent": {"page_id": "page"},
            "image": {"type": "external", "external": {"url": url},
                      "caption": [{"plain_text": caption}]}}


class AttachTest(unittest.TestCase):
    def test_rerun_removes_placeholder_without_attaching_a_second_copy(self):
        with tempfile.TemporaryDirectory(dir="scratch") as tmp:
            (Path(tmp) / "a.png").write_bytes(b"png")
            source = Path(tmp) / "page.md"
            source.write_text("![Chart A](a.png)\n", encoding="utf-8")
            # A failed delete left the placeholder with its attached copy right after it.
            blocks = [image("old", "Chart A"), image("new", "Chart A", url="https://files/a.png")]
            calls = []
            with mock.patch.object(notion_attach_media, "page_blocks", return_value=blocks), \
                    mock.patch.object(notion_attach_media, "upload", side_effect=AssertionError("uploaded again")), \
                    mock.patch.object(notion_attach_media, "api", side_effect=lambda *a, **k: calls.append(a) or {}), \
                    mock.patch.object(sys, "argv", ["attach", "page", str(source)]):
                self.assertEqual(notion_attach_media.main(), 0)
            self.assertEqual(calls, [("/v1/blocks/old", "-X", "DELETE")])


    def test_uncaptioned_video_and_pdf_placeholders_get_uploads_of_their_type(self):
        with tempfile.TemporaryDirectory(dir="scratch") as tmp:
            for name in ("clip.mp4", "a.pdf", "b.pdf"):
                (Path(tmp) / name).write_bytes(b"x")
            source = Path(tmp) / "page.md"
            source.write_text('<video src="clip.mp4"></video>\n\n<pdf src="a.pdf"></pdf>\n'
                              '<pdf src="b.pdf"></pdf>\n', encoding="utf-8")

            def media(block_id, kind):
                return {"id": block_id, "type": kind, "parent": {"page_id": "page"},
                        kind: {"type": "external", "external": {"url": ""}, "caption": []}}
            blocks = [media("v", "video"), media("p1", "pdf"), media("p2", "pdf")]
            calls = []
            with mock.patch.object(notion_attach_media, "page_blocks", return_value=blocks), \
                    mock.patch.object(notion_attach_media, "upload", side_effect=lambda path: f"up-{path.name}"), \
                    mock.patch.object(notion_attach_media, "api", side_effect=lambda *a, **k: calls.append((a, k)) or {}), \
                    mock.patch.object(sys, "argv", ["attach", "page", str(source)]):
                self.assertEqual(notion_attach_media.main(), 0)
            inserted = [(k["body"]["position"]["after_block"]["id"], k["body"]["children"][0]) for _, k in calls if k]
            self.assertEqual([(after, child["type"], child[child["type"]]["file_upload"]["id"]) for after, child in inserted],
                             [("v", "video", "up-clip.mp4"), ("p1", "pdf", "up-a.pdf"), ("p2", "pdf", "up-b.pdf")])
            self.assertEqual([a for a, k in calls if not k],
                             [(f"/v1/blocks/{i}", "-X", "DELETE") for i in ("v", "p1", "p2")])

    def test_bookmark_placeholder_is_replaced_in_place_and_a_rerun_only_deletes(self):
        url = "https://qz.com/a-story"
        with tempfile.TemporaryDirectory(dir="scratch") as tmp:
            source = Path(tmp) / "page.md"
            source.write_text(f'<bookmark url="{url}"></bookmark>\n\n[{url}]({url}) in a sentence\n', encoding="utf-8")

            def paragraph(block_id, text, href):
                return {"id": block_id, "type": "paragraph", "parent": {"page_id": "page"},
                        "paragraph": {"rich_text": [{"plain_text": text, "href": href}]}}
            placeholder = paragraph("ph", url, url)
            prose = paragraph("prose", "see the story", url)
            mark = {"id": "bm", "type": "bookmark", "parent": {"page_id": "page"}, "bookmark": {"url": url}}

            def attach(blocks):
                calls = []
                with mock.patch.object(notion_attach_media, "page_blocks", return_value=blocks), \
                        mock.patch.object(notion_attach_media, "api", side_effect=lambda *a, **k: calls.append((a, k)) or {}), \
                        mock.patch.object(sys, "argv", ["attach", "page", str(source)]):
                    self.assertEqual(notion_attach_media.main(), 0)
                return calls

            calls = attach([prose, placeholder])
            self.assertEqual(calls, [
                (("/v1/blocks/page/children", "-X", "PATCH"), {"body": {
                    "position": {"type": "after_block", "after_block": {"id": "ph"}},
                    "children": [{"object": "block", "type": "bookmark", "bookmark": {"url": url}}]}}),
                (("/v1/blocks/ph", "-X", "DELETE"), {}),
            ])
            # A failed delete left the placeholder before its bookmark: the rerun deletes it and adds nothing.
            self.assertEqual(attach([placeholder, mark]), [(("/v1/blocks/ph", "-X", "DELETE"), {})])


if __name__ == "__main__":
    unittest.main()
