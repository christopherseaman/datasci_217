#!/usr/bin/env python3
"""
Test suite for Assignment 01 - Setup and Email Processing
Tests email validation and reflection completion
"""

import os
import hashlib
import re


def test_processed_email_exists():
    """Test that processed_email.txt exists"""
    assert os.path.exists("processed_email.txt"), "processed_email.txt file not found. Did you run process_email.py?"


def test_email_hash_valid():
    """Test that the email hash is in the pre-approved list"""
    # Pre-hashed list of valid student emails
    # These are SHA256 hashes of cleaned usernames (lowercase, alphanumeric only)
    valid_hashes = [
        "042de7d63dd818a2cdbb4e29d095b3ae055e28d0da90d53de50eb77e13375647",
        "04edd70d3fde2adf5054fb19b3309b7b4fa7aafc468152e7524d70d54524fcbd",
        "07c719b4532d3d8acbd7fe3ee88eee18d1a6917df627e05aadb75555863f63ea",
        "124c44fca353a751ce8439391f18b83e1b6182d17a9ad152205f916057c71ed0",
        "138215b37c8bda19e69635d84ca7427922deb2c4565edc4161e69a56592de54c",
        "1422882332977f7e7acc311c57b22ea9e642ec4ed326f853794cfdeaad0d6554",
        "1a9863ce354e6ca4d926de9ff1a0b9f3da45a360e7d2ebdca8327189eeae0d2b",
        "1c9ed27ecddc296ac90a21112d3393c860a1b1f6aa474087e6426f95707a2937",
        "1ce5b2fc796f7e29b9bb030ce2a51430bd2447bbabc928382cf411db2de9139c",
        "21160b1690002b9d8caba2749d00616dbca84d6b640967fbbe307ac77174dcb5",
        "268dffd00de602897c13142ee88e17267656df26f5052ec669bd26a44bc2c55e",
        "299d5a0a4f2a53250bb795f6c4eaf3530bd415add9e2e045878244ce161ae21c",
        "2a2871901079e9f77bb99f7f662260986f1c0a034f0865a87282356e59c505eb",
        "364078b38c138a41f4a583706aa5cb7c970586cc3b8bd12bf0189e0c310443db",
        "39f01b347aef82d56a2922dcac2b84c198f539703459e0237dd9b4c8ee50819d",
        "46b3239056bc6f2b3e725f46ac226166f4fa9612fb7bcdadf06eb946b93aab0a",
        "4f86617cd650b3fb956fe5127a33b39e485e4224d0ec2aacc87acf9cd73a8f08",
        "5548c153730c80c3bab6e4ca26c39212c5199c6bb8b8fdd3bb18c1c73ededecb",
        "6371b85722ea50fe82f37c87ae7d4fd4b2959dec5994c2025990eb9f0cfc3492",
        "7a707c8a88e997c5b9d0498bd3a740813f98ac32dd81067060ab5fcbd0d24f8f",
        "7d3d6e37aef24f67a9349001e3f11505e58cf7d5f4cdc3591e00be9e94ef76d4",
        "7d94b55d0c91acf0ffc9b84bec1131452326bbcf5c177d97dd97205d90a6a2d5",
        "94afe77f864c2365b4274b33fc4f44e0b0c451262baab67558035e72a6d6fa70",
        "9ee3d9c36d454e5a095b5f12bc3c02f830b39310a4f5882774b62a954a782a5c",
        "9f784ddc20bcedd9ff14892db31db7c34cf380a21854ba989c01184e11137688",
        "ac948541bcac67153ad5f6dd44dbe85f318a0f4a3b0294632f4ddc3f9fcedebc",
        "ada09398f9068fbb9768120c3b2cf542000413f98949ffe48123cb43f0650b30",
        "b107815729cc3f17b5fff4a86b1dce722c161b1184251c55e16564bd9312b110",
        "b3b0dcdab039ff1ecd38097c190887688ca0e48b40a59d6c4709f21a58c64b32",
        "b3e75663ad8f65fc82430672fcf39bb084dc76f5b7e2023b5e26a79524575225",
        "ba59958684beac750f85a10c5993eb2afc8a24c583797c90bcc656c37b5dca04",
        "cb56fd9de33b1adcfadc51175bf8c922c837fc1cb4e4dab524bd03e7ce47b9dd",
        "ce0a2529a63fbf1fc5846d32859aee1169840d0dfb56e4e7498250b4adef9654",
        "d2e1b7428d83c0b939415777b2d8cd49006893532ca6320e08145f550e42f2bb",
        "d793158628ad3cc8624061252a418a71843a2f784e39c9fbbbe743818b1f90f2",
        "d7bb88aae579282c0c326e81793730b37f0ec33342b28284f90b815a42f5a9c4",
        "e8f65ef495b91c4ed789032d102a9697c0f6ee0ffa5a3354478cf06cf435543e",
        "e96ed33350e276d858264495dfb7e2df84fa7d057c09c848f34f988b3a272e76",
        "e9d7ea3eccdf8d6ecf9169096bc5e6c9b66c048e67c2552a9ef89ee118b3300f",
        "ed1a8c3015d5a5d5b4d7be2f7390c4bf59cfd0fcfe82a5d5143d1a02b245465c",
        "ed82cca0f5f496a8c821530c9f67d5955a331faf17a3cb916fe3ab39d3a094d0",
        "f6df5aee62ade4205c4836135b0c55b71a6bfbff71a9b07a388ec45d2f9b70c3",
        "fccbe9a00585237b09ae788f96272b9e9d8d3e952e5abfef63a55acd8fb92dc3",
    ]

    # Read the submitted hash
    assert os.path.exists("processed_email.txt"), "processed_email.txt not found"

    with open("processed_email.txt", "r") as f:
        submitted_hash = f.read().strip()

    # Verify hash format (64 hex characters)
    assert len(submitted_hash) == 64, f"Invalid hash length: expected 64 characters, got {len(submitted_hash)}"
    assert all(c in '0123456789abcdef' for c in submitted_hash.lower()), "Invalid hash format: not a valid hex string"

    # Check if hash is in valid list
    assert submitted_hash in valid_hashes, "Email hash not found in student roster. Please use your UCSF email."


def test_reflection_exists():
    """Test that reflection.md exists and has been modified"""
    assert os.path.exists("reflection.md"), "reflection.md file not found"

    with open("reflection.md", "r") as f:
        content = f.read()

    # Check that file has content
    assert len(content) > 100, "reflection.md appears to be empty or too short"


def test_reflection_modified():
    """Test that reflection has been meaningfully modified from template"""
    with open("reflection.md", "r") as f:
        content = f.read()

    # Original template is ~650 characters
    # With answers it should be significantly longer
    original_template_size = 650

    # Should be at least 50% longer than template (conservative)
    assert len(content) > original_template_size * 1.5, \
        "reflection.md doesn't appear to have been answered. Please add your responses."

    # Check that there aren't too many placeholder brackets
    # Students should replace [bracketed text] with their answers
    bracket_count = content.count('[')

    # Original has 4 bracketed placeholders, should have fewer after answering
    # But be flexible - students might use brackets in their answers
    assert bracket_count < 8, \
        "Too many placeholder brackets found. Please replace the [bracketed text] with your actual answers."


def test_reflection_has_url():
    """Test that reflection contains at least one URL (for question 4)"""
    with open("reflection.md", "r") as f:
        content = f.read()

    # Look for URL patterns
    url_patterns = [
        r'https?://[^\s]+',  # HTTP/HTTPS URLs
        r'www\.[^\s]+',      # www URLs
        r'[a-zA-Z0-9]+\.(com|org|edu|io|dev|net|gov)[^\s]*'  # Common domains
    ]

    has_url = any(re.search(pattern, content) for pattern in url_patterns)

    assert has_url, \
        "No URL found in reflection. Please include a link to something you enjoy in question 4."