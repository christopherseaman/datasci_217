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


if __name__ == "__main__":
    unittest.main()
