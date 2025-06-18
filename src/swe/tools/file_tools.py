"""File manipulation tools for the SWE agent."""

from pathlib import Path

import aiofiles


async def read_file(file_path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """
    Read a file and return its contents.

    Args:
        file_path: Path to the file to read
        start_line: Optional starting line number (1-indexed)
        end_line: Optional ending line number (1-indexed)

    Returns:
        File contents as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        if start_line is None and end_line is None:
            content = await f.read()
            return str(content)

        lines = await f.readlines()

        if start_line is not None:
            start_idx = max(0, start_line - 1)  # Convert to 0-indexed
        else:
            start_idx = 0

        if end_line is not None:
            end_idx = min(len(lines), end_line)  # Convert to 0-indexed + 1
        else:
            end_idx = len(lines)

        return "".join(lines[start_idx:end_idx])


async def edit_file(file_path: str, old_content: str, new_content: str) -> bool:
    """
    Edit a file by replacing old_content with new_content.

    Args:
        file_path: Path to the file to edit
        old_content: Content to be replaced
        new_content: Content to replace with

    Returns:
        True if the edit was successful

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If old_content is not found in the file
        PermissionError: If the file can't be written
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Read the current content
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        current_content = await f.read()

    # Check if old_content exists in the file
    if old_content not in current_content:
        raise ValueError(f"Content not found in file: {old_content[:100]}...")

    # Replace the content
    updated_content = current_content.replace(old_content, new_content)

    # Write the updated content back
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(updated_content)

    return True


async def create_file(file_path: str, content: str) -> bool:
    """
    Create a new file with the given content.

    Args:
        file_path: Path where the file should be created
        content: Content to write to the file

    Returns:
        True if the file was created successfully

    Raises:
        FileExistsError: If the file already exists
        PermissionError: If the file can't be created
    """
    path = Path(file_path)

    if path.exists():
        raise FileExistsError(f"File already exists: {file_path}")

    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)

    return True
