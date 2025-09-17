#!/usr/bin/env python3
"""
Assignment 01 - Email Processing Script

This script processes a UCSF email address and creates a hash for verification.
Students run this script with their email to demonstrate basic CLI and Python skills.

Usage: python process_email.py your_email@ucsf.edu
"""

import sys
import hashlib
import re

def process_email(email_address):
    """
    Process email address through the following steps:
    1. Validate UCSF email format
    2. Extract username (part before @)
    3. Convert to lowercase
    4. Create hash for verification
    5. Return processed results
    """
    
    print(f"Processing email: {email_address}")
    
    # Extract username
    username = email_address.split('@')[0]
    print(f"Extracting username: {username}")
    
    # Convert to lowercase and strip whitespace
    username_clean = username.lower().strip()
    print(f"Converting to lowercase: {username_clean}")
    # Remove any non-alphanumeric characters
    username_clean = re.sub(r'[^a-z0-9]', '', username_clean)
    print(f"Cleaning username: {username_clean}")
    
    # Create hash for verification (SHA256)
    hash_object = hashlib.sha256(username_clean.encode())
    username_hash = hash_object.hexdigest()
    print(f"Creating verification hash: {username_hash[:16]}...")
    
    # Prepare results
    results = {
        'original_email': email_address,
        'username': username_clean,
        'hash': username_hash,
    }
    
    return results

def save_results(results, output_file='processed_email.txt'):
    """Save processing results to output file"""
    
    if results is None:
        return False
        
    try:
        with open(output_file, 'w') as f:
            f.write(f"{results['hash']}\n")
        print(f"Results saved to {output_file}")
        return True
    
    except IOError as e:
        print(f"Error saving results: {e}")
        return False

def main():
    """Main function - process command line arguments and run email processing"""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python process_email.py your_email@ucsf.edu")
        print("Example: python process_email.py alice.smith@ucsf.edu")
        sys.exit(1)
    
    email_address = sys.argv[1]
    
    # Process the email
    results = process_email(email_address)
    
    if results is None:
        print("Processing failed. Please check your email format and try again.")
        sys.exit(1)
    
    # Save results
    if save_results(results):
        print("\n✓ Assignment completed successfully!")
        print("Next steps:")
        print("1. Check that 'processed_email.txt' was created")
        print("2. Add the file to Git: git add processed_email.txt") 
        print("3. Commit: git commit -m 'Add processed email results'")
        print("4. Push: git push")
    else:
        print("Failed to save results. Please check file permissions and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()