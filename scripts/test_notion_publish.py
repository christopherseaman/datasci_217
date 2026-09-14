import tempfile
import unittest
from pathlib import Path

from notion_publish import prepare, _replace_links, table


class PrepareTest(unittest.TestCase):
    def test_inline_code_and_tables(self):
        source = Path("01/README.md")
        self.assertEqual(_replace_links("`[example](file.md)`", source), "`[example](file.md)`")
        self.assertIn("/blob/main/01/example.py", _replace_links("[`example.py`](example.py)", source))
        self.assertEqual(_replace_links("[Here](#section)", source), "[Here](#section)")
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
