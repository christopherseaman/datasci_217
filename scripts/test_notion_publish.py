import tempfile
import unittest
from pathlib import Path

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
