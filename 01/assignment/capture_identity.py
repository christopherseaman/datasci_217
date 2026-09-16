"""Supplied private identity capture helper; stores only the username hash."""

from pathlib import Path

from process_email import process_email


email_address = input("Email address (not saved): ")
try:
    identity_hash = process_email(email_address)["hash"]
except ValueError as error:
    raise SystemExit(str(error)) from None
output_path = Path(__file__).resolve().parent / "output" / "student_identity.txt"
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(identity_hash + "\n", encoding="utf-8")
print("Wrote output/student_identity.txt.")
