"""Search tools for the SWE agent."""

import asyncio
import subprocess


async def search_files(
    query: str,
    directory: str = ".",
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    use_regex: bool = False,
    max_results: int = 50,
) -> list[tuple[str, int, str]]:
    """
    Search for text within files using ripgrep (rg) or grep as fallback.

    Args:
        query: Text or pattern to search for
        directory: Directory to search in (default: current directory)
        file_pattern: Optional file pattern to filter (e.g., "*.py", "*.js")
        case_sensitive: Whether the search should be case sensitive
        use_regex: Whether to treat query as a regular expression
        max_results: Maximum number of results to return

    Returns:
        List of tuples containing (file_path, line_number, line_content)
    """
    results = []

    # Try to use ripgrep first (faster and more feature-rich)
    try:
        await _search_with_rg(
            query,
            directory,
            file_pattern,
            case_sensitive,
            use_regex,
            max_results,
            results,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fall back to grep if ripgrep is not available
        try:
            await _search_with_grep(
                query,
                directory,
                file_pattern,
                case_sensitive,
                use_regex,
                max_results,
                results,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"Neither ripgrep nor grep is available: {e}")

    return results[:max_results]


async def _search_with_rg(
    query: str,
    directory: str,
    file_pattern: str | None,
    case_sensitive: bool,
    use_regex: bool,
    max_results: int,
    results: list[tuple[str, int, str]],
) -> None:
    """Search using ripgrep (rg)."""
    cmd = ["rg", "--line-number", "--no-heading", "--color=never"]

    if not case_sensitive:
        cmd.append("--ignore-case")

    if not use_regex:
        cmd.append("--fixed-strings")

    if file_pattern:
        cmd.extend(["--glob", file_pattern])

    cmd.extend(["--max-count", str(max_results)])
    cmd.append(query)
    cmd.append(directory)

    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await process.communicate()

    if process.returncode != 0 and process.returncode != 1:  # 1 means no matches found
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

    if stdout:
        for line in stdout.decode().strip().split("\n"):
            if line:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path, line_num, content = parts[0], parts[1], parts[2]
                    try:
                        results.append((file_path, int(line_num), content))
                    except ValueError:
                        continue


async def _search_with_grep(
    query: str,
    directory: str,
    file_pattern: str | None,
    case_sensitive: bool,
    use_regex: bool,
    max_results: int,
    results: list[tuple[str, int, str]],
) -> None:
    """Search using grep as fallback."""
    cmd = ["grep", "-rn", "--color=never"]

    if not case_sensitive:
        cmd.append("-i")

    if not use_regex:
        cmd.append("-F")

    if file_pattern:
        cmd.extend(["--include", file_pattern])

    cmd.append(query)
    cmd.append(directory)

    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await process.communicate()

    if process.returncode != 0 and process.returncode != 1:  # 1 means no matches found
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

    if stdout:
        count = 0
        for line in stdout.decode().strip().split("\n"):
            if line and count < max_results:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path, line_num, content = parts[0], parts[1], parts[2]
                    try:
                        results.append((file_path, int(line_num), content))
                        count += 1
                    except ValueError:
                        continue


async def find_files(name_pattern: str, directory: str = ".", max_results: int = 50) -> list[str]:
    """
    Find files by name pattern using find command.

    Args:
        name_pattern: File name pattern to search for (e.g., "*.py", "test_*")
        directory: Directory to search in (default: current directory)
        max_results: Maximum number of results to return

    Returns:
        List of file paths matching the pattern
    """
    cmd = ["find", directory, "-name", name_pattern, "-type", "f"]

    try:
        process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

        if stdout:
            files = stdout.decode().strip().split("\n")
            return [f for f in files if f][:max_results]

        return []

    except FileNotFoundError:
        raise RuntimeError("find command is not available")
