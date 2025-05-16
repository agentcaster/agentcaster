#!/usr/bin/env python

"""
Script to automatically fix Antialiasing references in the SHARPpy codebase.

This script searches for instances of QtGui.QPainter.RenderHint.Antialiasing and QtGui.QPainter.RenderHint.TextAntialiasing
and replaces them with QtGui.QPainter.RenderHint.Antialiasing and 
QtGui.QPainter.RenderHint.TextAntialiasing respectively.
"""

import os
import re
import sys

def find_py_files(directory):
    """Find all Python files in the given directory and its subdirectories."""
    py_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def fix_antialiasing_in_file(file_path):
    """Fix Antialiasing references in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the file contains any Antialiasing references
    if 'Antialiasing' not in content:
        return False, 0
    
    # Replace QtGui.QPainter.RenderHint.Antialiasing with QtGui.QPainter.RenderHint.Antialiasing
    pattern1 = r'qp\.Antialiasing'
    replacement1 = r'QtGui.QPainter.RenderHint.Antialiasing'
    new_content = re.sub(pattern1, replacement1, content)
    
    # Replace QtGui.QPainter.RenderHint.TextAntialiasing with QtGui.QPainter.RenderHint.TextAntialiasing
    pattern2 = r'qp\.TextAntialiasing'
    replacement2 = r'QtGui.QPainter.RenderHint.TextAntialiasing'
    new_content = re.sub(pattern2, replacement2, new_content)
    
    # Count the number of replacements
    count1 = len(re.findall(pattern1, content))
    count2 = len(re.findall(pattern2, content))
    total_count = count1 + count2
    
    # Only write to the file if changes were made
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True, total_count
    
    return False, 0

def main():
    # Use the current directory as the base directory
    base_dir = os.getcwd()
    
    # Find all Python files
    py_files = find_py_files(base_dir)
    print(f"Found {len(py_files)} Python files to check.")
    
    # Process each file
    total_files_modified = 0
    total_replacements = 0
    
    for file_path in py_files:
        modified, count = fix_antialiasing_in_file(file_path)
        if modified:
            total_files_modified += 1
            total_replacements += count
            print(f"Modified {file_path} - {count} replacements")
    
    print(f"\nSummary:")
    print(f"Total files modified: {total_files_modified}")
    print(f"Total replacements made: {total_replacements}")

if __name__ == "__main__":
    main()
