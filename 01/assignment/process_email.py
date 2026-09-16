#!/usr/bin/env python3
"""Create the Assignment 01 identity hash without storing the email address."""

import hashlib
import re
import sys


def process_email(email_address):
    """Return the historical Assignment 01 username and SHA-256 hash."""
    email_address = email_address.strip().lower()
    if email_address.count("@") != 1:
        raise ValueError("Enter your course roster email address ending in @ucsf.edu.")
    username, domain = email_address.split("@", 1)
    if domain != "ucsf.edu" or not username or any(char.isspace() for char in username):
        raise ValueError("Enter your course roster email address ending in @ucsf.edu.")
    username_clean = re.sub(r"[^a-z0-9]", "", username.lower().strip())
    if not username_clean:
        raise ValueError("The email username must contain letters or numbers.")
    username_hash = hashlib.sha256(username_clean.encode("utf-8")).hexdigest()
    return {"username": username_clean, "hash": username_hash}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_email.py your_email@ucsf.edu")
        raise SystemExit(1)
    try:
        print(process_email(sys.argv[1])["hash"])
    except ValueError as error:
        raise SystemExit(str(error)) from None
