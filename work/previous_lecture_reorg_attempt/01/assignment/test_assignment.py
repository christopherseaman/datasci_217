#!/usr/bin/env python3
"""
Assignment 01 - Automated Tests

This file contains pytest tests for automated grading via GitHub Classroom.
Students should not modify this file.

Tests verify:
1. Output file exists
2. Output file has correct format  
3. Content appears valid
4. Basic workflow completion
"""

import pytest
import os
import re
import hashlib
import subprocess
import sys

class TestAssignment01:
    
    def test_output_file_exists(self):
        """Test that the processed_email.txt file was created"""
        assert os.path.exists('processed_email.txt'), "processed_email.txt file not found. Make sure you ran the script successfully."
    
    def test_output_file_not_empty(self):
        """Test that the output file contains content"""
        assert os.path.exists('processed_email.txt'), "Output file does not exist"
        
        with open('processed_email.txt', 'r') as f:
            content = f.read().strip()
        
        assert len(content) > 0, "Output file is empty. Make sure the script ran successfully."
    
    def test_output_format(self):
        """Test that output file has the expected CSV format"""
        assert os.path.exists('processed_email.txt'), "Output file does not exist"
        
        with open('processed_email.txt', 'r') as f:
            line = f.read().strip()
        
        # Should be: username,hash,status
        parts = line.split(',')
        assert len(parts) == 3, f"Expected 3 comma-separated values, got {len(parts)}. Format should be: username,hash,status"
        
        username, hash_value, status = parts
        
        # Basic format checks
        assert len(username) > 0, "Username part is empty"
        assert len(hash_value) == 64, f"Hash should be 64 characters (SHA256), got {len(hash_value)}"
        assert status == 'processed', f"Status should be 'processed', got '{status}'"
    
    def test_username_format(self):
        """Test that username appears to be valid UCSF format"""
        assert os.path.exists('processed_email.txt'), "Output file does not exist"
        
        with open('processed_email.txt', 'r') as f:
            line = f.read().strip()
        
        username = line.split(',')[0]
        
        # Username should be lowercase and contain only valid characters
        assert username.islower(), "Username should be lowercase"
        assert re.match(r'^[a-z0-9._%+-]+$', username), "Username contains invalid characters"
        assert len(username) >= 2, "Username seems too short"
    
    def test_hash_validity(self):
        """Test that hash appears to be a valid SHA256 hash"""
        assert os.path.exists('processed_email.txt'), "Output file does not exist"
        
        with open('processed_email.txt', 'r') as f:
            line = f.read().strip()
        
        hash_value = line.split(',')[1]
        
        # SHA256 hash should be 64 hex characters
        assert len(hash_value) == 64, f"Hash length should be 64, got {len(hash_value)}"
        assert re.match(r'^[a-f0-9]{64}$', hash_value), "Hash should contain only lowercase hex characters"
    
    def test_hash_consistency(self):
        """Test that hash matches the username (verifies script logic)"""
        assert os.path.exists('processed_email.txt'), "Output file does not exist"
        
        with open('processed_email.txt', 'r') as f:
            line = f.read().strip()
        
        username, provided_hash, status = line.split(',')
        
        # Recreate hash to verify consistency
        expected_hash = hashlib.sha256(username.encode()).hexdigest()
        
        assert provided_hash == expected_hash, "Hash doesn't match username - script may not have run correctly"
    
    def test_script_executable(self):
        """Test that the processing script exists and is readable"""
        assert os.path.exists('process_email.py'), "process_email.py script not found"
        
        # Try to read the script
        with open('process_email.py', 'r') as f:
            content = f.read()
        
        assert 'def process_email' in content, "process_email function not found in script"
        assert 'hashlib' in content, "Script should use hashlib for hashing"

    def test_reflection_exists(self):
        """Test that reflection file exists (if provided)"""
        if os.path.exists('reflection.md'):
            with open('reflection.md', 'r') as f:
                content = f.read().strip()
            
            assert len(content) > 50, "Reflection appears too short - please answer all questions thoughtfully"

# Additional helper functions for manual grading support

def get_submission_info():
    """Helper function to extract submission information for manual review"""
    info = {}
    
    if os.path.exists('processed_email.txt'):
        with open('processed_email.txt', 'r') as f:
            line = f.read().strip()
        
        if ',' in line:
            parts = line.split(',')
            info['username'] = parts[0] if len(parts) > 0 else 'unknown'
            info['hash'] = parts[1] if len(parts) > 1 else 'unknown'
            info['status'] = parts[2] if len(parts) > 2 else 'unknown'
    
    if os.path.exists('reflection.md'):
        with open('reflection.md', 'r') as f:
            info['reflection_length'] = len(f.read().strip())
    
    return info

if __name__ == "__main__":
    # Run tests when executed directly (for local testing)
    pytest.main([__file__, '-v'])
    
    # Show submission info
    print("\n" + "="*50)
    print("SUBMISSION INFORMATION")  
    print("="*50)
    info = get_submission_info()
    for key, value in info.items():
        print(f"{key}: {value}")