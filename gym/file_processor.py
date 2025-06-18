"""
File processing utility with I/O bugs for testing the SWE agent.
"""

import json
import os
from typing import Any


class FileProcessor:
    def __init__(self, base_directory: str = "."):
        self.base_directory = base_directory
        self.processed_files = []

    def read_file(self, filename: str) -> str:
        """Read a text file."""
        # Bug: No error handling for file not found or permission errors
        filepath = os.path.join(self.base_directory, filename)
        file = open(filepath)
        content = file.read()
        # Bug: File is never closed
        return content

    def write_file(self, filename: str, content: str) -> bool:
        """Write content to a file."""
        filepath = os.path.join(self.base_directory, filename)
        # Bug: No error handling for permission errors or disk full
        with open(filepath, "w") as file:
            file.write(content)
        return True

    def read_json(self, filename: str) -> dict[str, Any]:
        """Read a JSON file."""
        filepath = os.path.join(self.base_directory, filename)
        # Bug: No error handling for invalid JSON or file not found
        with open(filepath) as file:
            return json.load(file)

    def write_json(self, filename: str, data: dict[str, Any]) -> bool:
        """Write data to a JSON file."""
        filepath = os.path.join(self.base_directory, filename)
        # Bug: No error handling for serialization errors
        with open(filepath, "w") as file:
            json.dump(data, file)
        return True

    def process_directory(self, directory: str) -> list[str]:
        """Process all text files in a directory."""
        processed = []
        full_dir = os.path.join(self.base_directory, directory)

        # Bug: No check if directory exists
        for filename in os.listdir(full_dir):
            filepath = os.path.join(full_dir, filename)

            # Bug: No check if it's actually a file
            if filename.endswith(".txt"):
                content = self.read_file(filepath)
                # Bug: Processing the full path instead of just the filename
                processed_content = self.process_text(content)
                self.write_file(f"processed_{filename}", processed_content)
                processed.append(filename)

        return processed

    def process_text(self, text: str) -> str:
        """Process text by removing extra whitespace and converting to lowercase."""
        # Bug: This will strip all whitespace, including spaces between words
        return text.strip().replace(" ", "").lower()

    def backup_file(self, filename: str) -> bool:
        """Create a backup of a file."""
        source = os.path.join(self.base_directory, filename)
        backup = os.path.join(self.base_directory, f"{filename}.backup")

        # Bug: No error handling for file operations
        content = self.read_file(filename)
        self.write_file(f"{filename}.backup", content)
        return True

    def get_file_stats(self, filename: str) -> dict[str, Any]:
        """Get file statistics."""
        filepath = os.path.join(self.base_directory, filename)

        # Bug: No error handling for file not found
        stats = os.stat(filepath)

        return {
            "size": stats.st_size,
            "modified": stats.st_mtime,
            "created": stats.st_ctime,
            # Bug: Accessing non-existent attribute
            "permissions": stats.st_perm,
        }

    def merge_files(self, file_list: list[str], output_file: str) -> bool:
        """Merge multiple files into one."""
        merged_content = ""

        for filename in file_list:
            # Bug: No error handling if any file in the list doesn't exist
            content = self.read_file(filename)
            merged_content += content
            # Bug: No separator between files

        self.write_file(output_file, merged_content)
        return True


def main():
    """Demo function with file processing bugs."""
    processor = FileProcessor()

    # These operations will demonstrate the bugs
    try:
        # Bug: Will fail if sample.txt doesn't exist
        content = processor.read_file("sample.txt")
        print(f"File content: {content}")

        # Bug: Will fail if directory doesn't exist
        processed = processor.process_directory("data")
        print(f"Processed files: {processed}")

        # Bug: Will fail if file doesn't exist
        stats = processor.get_file_stats("nonexistent.txt")
        print(f"File stats: {stats}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
