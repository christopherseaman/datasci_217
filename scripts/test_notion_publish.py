import tempfile
import unittest
from pathlib import Path

import notion_media
import notion_publish
from notion_publish import prepare, _replace_links, table


class PrepareTest(unittest.TestCase):
    def test_child_pages_stay_beside_their_descriptions(self):
        with tempfile.TemporaryDirectory(dir="scratch") as tmp:
            root = Path(tmp)
            target = root / "lecture.md"
            target.write_text("---\nnotion:\n  url: https://app.notion.com/p/lecture\n---\n", encoding="utf-8")
            source = root / "index.md"
            source.write_text("# Resources\n\nSome references.\n\n# Lectures\n\n[Lecture](lecture.md)\n\n> Lecture description.\n", encoding="utf-8")
            child = '<page url="https://app.notion.com/p/lecture">Lecture</page>'
            current = root / "current.md"
            current.write_text(child, encoding="utf-8")
            result = prepare(source, current)
            self.assertEqual(result.count(child), 1)
            self.assertLess(result.index("# Lectures"), result.index(child))
            self.assertLess(result.index(child), result.index("> Lecture description."))

    def test_media_uses_notion_uploads_when_a_manifest_is_loaded(self):
        source = Path("01/README.md")
        self.assertIn(
            "raw.githubusercontent.com/christopherseaman/datasci_217/main/01/media/duck_typing.jpg",
            _replace_links("![Git](media/duck_typing.jpg)", source),
        )
        notion_publish.MEDIA.update({"01/media/duck_typing.jpg": "file-upload://abc123"})
        try:
            self.assertEqual(
                _replace_links("![Git](media/duck_typing.jpg)", source),
                "![Git](file-upload://abc123)",
            )
            with self.assertRaises(ValueError):
                _replace_links("![Other](media/not_uploaded.png)", source)
        finally:
            notion_publish.MEDIA.clear()

    def test_video_and_pdf_tags_use_notion_uploads(self):
        source = Path("08/README.md")
        video = '<video src="media/swedish-delight.mp4"></video>'
        self.assertEqual(
            _replace_links(video, source),
            '<video src="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/08/media/swedish-delight.mp4"></video>',
        )
        notion_publish.MEDIA.update({"08/media/swedish-delight.mp4": "file-upload://vid1",
                                     "08/media/RedSoloCupTracking.pdf": "file-upload://pdf1"})
        try:
            self.assertEqual(_replace_links(video, source), '<video src="file-upload://vid1"></video>')
            self.assertEqual(_replace_links('<pdf src="media/RedSoloCupTracking.pdf">Cups</pdf>', source),
                             '<pdf src="file-upload://pdf1">Cups</pdf>')
            self.assertEqual(_replace_links('`<pdf src="media/x.pdf"></pdf>`', source),
                             '`<pdf src="media/x.pdf"></pdf>`')
            with self.assertRaises(ValueError):
                _replace_links('<pdf src="media/missing.pdf"></pdf>', source)
        finally:
            notion_publish.MEDIA.clear()
        self.assertEqual(
            [p.name for p in notion_media.media_targets(source)][:3],
            ["swedish-delight.mp4", "RedSoloCupTracking.pdf", "Regression-Abuse-SIM-44-0.pdf"],
        )

    def test_bookmark_becomes_a_url_paragraph_outside_fences(self):
        with tempfile.TemporaryDirectory(dir="scratch") as tmp:
            root = Path(tmp)
            source = root / "README.md"
            url = "https://qz.com/a-story"
            source.write_text(f'# A\n\n<bookmark url="{url}"></bookmark>\n\n```html\n<bookmark url="{url}"></bookmark>\n```\n',
                              encoding="utf-8")
            current = root / "current.md"
            current.write_text("", encoding="utf-8")
            output = prepare(source, current)
            self.assertIn(f"\n[{url}]({url})\n", output)
            self.assertEqual(output.count(f'<bookmark url="{url}"></bookmark>'), 1)
            self.assertEqual(notion_media.bookmark_urls(source), [url])
        self.assertEqual(notion_media.bookmark_urls(Path("08/README.md")),
                         ["https://qz.com/anthropic-mithril-medicine-acquisition-drug-programs-102826"])

    def test_inline_code_and_tables(self):
        source = Path("01/README.md")
        self.assertEqual(_replace_links("`[example](file.md)`", source), "`[example](file.md)`")
        self.assertIn("/blob/main/01/example.py", _replace_links("[`example.py`](example.py)", source))
        self.assertEqual(_replace_links("[Here](#section)", source), "[Here](#section)")
        self.assertEqual(
            _replace_links("[site](/references/)", source),
            "[site](https://github.com/christopherseaman/datasci_217/tree/main/references/)",
        )
        self.assertIn("<td>`x`</td>", table(["| Name |\n", "| --- |\n", "| `x` |\n"]))
        self.assertEqual(table(["| ordinary text\n"]), "| ordinary text\n")

    def test_preserves_fence_children_and_resolves_links(self):
        with tempfile.TemporaryDirectory(dir="scratch") as tmp:
            root = Path(tmp)
            (root / "media").mkdir()
            source = root / "README.md"
            target = root / "child.md"
            source.write_text(
                "---\nnotion:\n  url: https://app.notion.com/p/source\n  title_line: Lesson title\n---\n"
                "Lesson title\n\n# Section\n\n[Child](child.md) ![plot](media/p.png)\n\n````markdown\n```python\n# [keep](child.md)\n```\n````\n",
                encoding="utf-8",
            )
            target.write_text("---\nnotion:\n  url: https://app.notion.com/p/child\n---\nChild\n", encoding="utf-8")
            current = root / "current.md"
            current.write_text('<page title="Child" />\n---\nLesson title\n', encoding="utf-8")
            output = prepare(source, current)
            self.assertTrue(output.startswith('<page title="Child" />\n\n# Section'))
            self.assertIn("https://app.notion.com/p/child", output)
            self.assertIn("/media/p.png", output)
            self.assertIn("# [keep](child.md)", output)


if __name__ == "__main__":
    unittest.main()
