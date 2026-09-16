"""Regression checks for Assignment 01 saved artifacts only."""

from pathlib import Path
import sys
import tempfile


ASSIGNMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT))
from _assignment_checks import EXPECTED_READINESS, ROSTER_HASHES
from grading import grade_submission
from process_email import process_email


def run() -> None:
    import hashlib

    expected_identity = hashlib.sha256(b"alicesmith").hexdigest()
    for address in ("Alice.Smith@ucsf.edu", " alice-smith@ucsf.edu ", "\tALICE.SMITH@UCSF.EDU\n", "alice_smith@ucsf.edu"):
        assert process_email(address)["hash"] == expected_identity
    assert len(ROSTER_HASHES) == 40
    for address in ("alice.smith@example.com", "alice.smith@ucsf.edu.example.com", "alice.smith@notucsf.edu", "alice.smith@sub.ucsf.edu", "alice.smith", "alice@@ucsf.edu", "@ucsf.edu", "...@ucsf.edu", "alice smith@ucsf.edu", "alice@ ucsf.edu"):
        try:
            process_email(address)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid roster email was accepted")
    valid_identity = sorted(ROSTER_HASHES)[0]
    with tempfile.TemporaryDirectory(dir=ASSIGNMENT.parents[1] / "scratch", prefix="a01-artifacts-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()
        assert grade_submission(root)["score"] == 0
        practice = root / "terminal-practice"
        practice.mkdir()
        for name in ("source.txt", "path-check.txt"):
            (practice / name).write_text("", encoding="utf-8")
        output = root / "output"
        output.mkdir()
        (output / "readiness.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
        assert grade_submission(root)["score"] == 20
        for identity_hash in ROSTER_HASHES:
            for text in (identity_hash, identity_hash + "\n", "\t " + identity_hash.upper() + " \n"):
                (output / "student_identity.txt").write_text(text, encoding="utf-8")
                assert grade_submission(root)["score"] == 100
        (output / "student_identity.txt").write_text("0" * 64 + "\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 20
        (output / "student_identity.txt").write_text(valid_identity + "\n", encoding="utf-8")
        (root / "notes.txt").write_text("extra files are allowed\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 100
        (output / "readiness.txt").write_text("wrong\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 20
        (output / "readiness.txt").unlink()
        assert grade_submission(root)["score"] == 20
        (output / "readiness.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
        (output / "student_identity.txt").write_text("not-a-hash\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 20
        (output / "student_identity.txt").write_text(valid_identity + "\n" + valid_identity, encoding="utf-8")
        assert grade_submission(root)["score"] == 20
        (output / "student_identity.txt").write_text(valid_identity + "\n", encoding="utf-8")
        (practice / "source.txt").unlink()
        assert grade_submission(root)["score"] == 80
    print("Assignment 01: all 40 roster hashes, non-roster hashes, starter, extra-file, missing, and wrong-artifact checks passed.")


if __name__ == "__main__":
    run()
