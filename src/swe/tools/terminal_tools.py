"""Terminal command execution tools for the SWE agent."""

import asyncio
import subprocess
from pathlib import Path


async def execute_command(command: str, working_directory: str | None = None, confirm: bool = True, timeout: int = 30) -> str:
    """
    Execute a terminal command with optional confirmation.

    Args:
        command: The command to execute
        working_directory: Directory to run the command in (default: current directory)
        confirm: Whether to prompt for confirmation (default: True)
        timeout: Command timeout in seconds (default: 30)

    Returns:
        Combined stdout and stderr output from the command

    Raises:
        RuntimeError: If user cancels the execution or command fails
        TimeoutError: If command times out
    """
    if confirm:
        # Show command details and prompt for confirmation
        print("\n🔧 TERMINAL COMMAND REQUEST:")
        print(f"   Command: {command}")
        print(f"   Directory: {working_directory or 'current directory'}")
        print(f"   Timeout: {timeout} seconds")

        # Show some example use cases
        print("\n💡 Terminal commands can be useful for:")
        print("   • Running tests: pytest, npm test, cargo test")
        print("   • Text processing: grep, awk, sed, sort")
        print("   • Network diagnostics: ping, curl, wget")
        print("   • System info: ps, df, top, uname")
        print("   • Build tools: make, npm build, cargo build")
        print("   • Git operations: git status, git log, git diff")

        print("\n⚠️  WARNING: This will execute a terminal command on your system!")
        print("   Only approve commands you trust and understand.")

        print("\n❓ Do you want to execute this command? (type 'yes' to confirm): ", end="")

        # Get user input in a way that works with asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, input)

        if response.lower().strip() != "yes":
            raise RuntimeError("Command execution cancelled by user")

        print("✅ Command execution approved.")

    # Validate working directory if provided
    if working_directory:
        work_dir = Path(working_directory)
        if not work_dir.exists():
            raise RuntimeError(f"Working directory does not exist: {working_directory}")
        if not work_dir.is_dir():
            raise RuntimeError(f"Working directory is not a directory: {working_directory}")

    try:
        # Execute the command
        print(f"\n🚀 Executing: {command}")

        process = await asyncio.wait_for(
            asyncio.create_subprocess_shell(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_directory), timeout=timeout
        )

        stdout, stderr = await process.communicate()

        # Decode bytes to strings
        stdout_str = stdout.decode("utf-8") if stdout else ""
        stderr_str = stderr.decode("utf-8") if stderr else ""

        # Combine output
        output_parts = []
        if stdout_str.strip():
            output_parts.append(f"STDOUT:\n{stdout_str}")
        if stderr_str.strip():
            output_parts.append(f"STDERR:\n{stderr_str}")

        combined_output = "\n".join(output_parts) if output_parts else "(no output)"

        # Check return code
        if process.returncode != 0:
            error_msg = f"Command failed with exit code {process.returncode}\n{combined_output}"
            print(f"❌ {error_msg}")
            return f"ERROR: {error_msg}"

        print(f"✅ Command completed successfully (exit code: {process.returncode})")
        return combined_output

    except asyncio.TimeoutError:
        error_msg = f"Command timed out after {timeout} seconds"
        print(f"⏰ {error_msg}")
        raise TimeoutError(error_msg)
    except Exception as e:
        error_msg = f"Failed to execute command: {e}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)


async def check_command_safety(command: str) -> tuple[bool, str]:
    """
    Basic safety check for terminal commands.

    Args:
        command: The command to check

    Returns:
        Tuple of (is_safe, warning_message)
    """
    # List of potentially dangerous commands/patterns
    dangerous_patterns = [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf *",
        "sudo rm",
        "> /dev/",
        "dd if=",
        "mkfs",
        "fdisk",
        "format",
        "del /f /s /q",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        ":(){ :|:& };:",  # fork bomb
    ]

    command_lower = command.lower().strip()

    for pattern in dangerous_patterns:
        if pattern.lower() in command_lower:
            return False, f"Potentially dangerous command detected: contains '{pattern}'"

    # Check for some risky patterns
    if command_lower.startswith("sudo"):
        return False, "Command requires sudo privileges - potentially risky"

    if "rm " in command_lower and ("-r" in command_lower or "-f" in command_lower):
        return False, "Recursive or forced file deletion detected - potentially risky"

    return True, "Command appears safe"
