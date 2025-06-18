"""SWE Agent - Main agent implementation."""

import json
import re
from collections.abc import Callable
from pathlib import Path

from swe.llm import ChatModel, Message, Role, llm_chat_conversation
from swe.tools import delete_file, edit_file, execute_command, read_file, search_files
from swe.tools.search_tools import find_files


class SWEAgent:
    """Software Engineering Agent that can perform various coding tasks."""

    def __init__(
        self,
        model: ChatModel = ChatModel.GEMINI_2_5_FLASH,
        working_directory: Path = Path("."),
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        self.model = model
        self.working_directory = working_directory
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Default system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Available tools
        self.tools = {
            "read_file": self._tool_read_file,
            "edit_file": self._tool_edit_file,
            "search_files": self._tool_search_files,
            "find_files": self._tool_find_files,
            "list_directory": self._tool_list_directory,
            "create_file": self._tool_create_file,
            "delete_file": self._tool_delete_file,
            "execute_command": self._tool_execute_command,
        }

        # Agent loop configuration
        self.max_iterations = 10
        self.conversation_history = []
        self.stream_callback = None

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent."""
        return f"""You are SWE, a helpful Software Engineering Agent. You can help with:

- Reading and analyzing code files
- Editing and modifying files
- Searching for code patterns and text
- Finding files and understanding project structure
- Debugging and fixing code issues
- Writing new code and features
- Explaining code and providing documentation

You have access to the following tools:
- read_file(file_path, start_line?, end_line?) - Read a file or specific lines
- edit_file(file_path, old_content, new_content) - Edit a file by replacing content
- search_files(query, file_pattern?, case_sensitive?, use_regex?) - Search for text in files
- find_files(name_pattern) - Find files by name pattern
- list_directory(directory_path?) - List contents of a directory
- create_file(file_path, content) - Create a new file
- delete_file(file_path, confirm?) - Delete a file (requires confirmation by default)
- execute_command(command, working_directory?, confirm?, timeout?) - Execute terminal commands (requires confirmation by default)

The execute_command tool is useful for:
• Running tests: pytest, npm test, cargo test, make test
• Text processing: grep, awk, sed, sort, uniq, cut
• Network diagnostics: ping, curl, wget, netstat
• System information: ps, df, top, uname, which
• Build tools: make, npm build, cargo build, mvn compile
• Git operations: git status, git log, git diff, git branch
• Package management: npm install, pip install, apt list

Current working directory: {self.working_directory}

When using tools, format your tool calls as JSON in this format:
```json
{{"tool": "tool_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}
```

IMPORTANT INSTRUCTIONS:
1. Work iteratively towards completing the task
2. Use tools as needed to gather information and make changes
3. Always explain what you're doing and why
4. When you believe the task is complete, end your response with: TASK COMPLETE
5. If you need more information or encounter issues, continue working through them
6. Be thorough and systematic in your approach
"""

    async def process_message(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Process a user message and return a response using an iterative agent loop."""

        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []

        # Store stream callback for use during processing
        self.stream_callback = stream_callback

        # Add user message to conversation history
        conversation_history.append({"role": "user", "content": user_message})

        # Stream initial message
        if self.stream_callback:
            self.stream_callback(f"[bold green]Starting task:[/bold green] {user_message}\n\n")

        # Agent loop - continue until task is complete or max iterations reached
        iteration = 0
        full_response_parts = []

        while iteration < self.max_iterations:
            iteration += 1

            # Stream iteration header
            if iteration > 1 and self.stream_callback:
                iteration_header = f"\n[bold blue]--- Iteration {iteration} ---[/bold blue]\n"
                self.stream_callback(iteration_header)
                full_response_parts.append(iteration_header)

            # Build messages for the LLM
            messages = [Message(role=Role.SYSTEM, content=self.system_prompt)]

            # Add conversation history
            for msg in conversation_history:
                role = Role.USER if msg["role"] == "user" else Role.ASSISTANT
                messages.append(Message(role=role, content=msg["content"]))

            # Stream thinking message
            if self.stream_callback:
                self.stream_callback(f"🤔 [bold yellow]Thinking...[/bold yellow] " f"(Iteration {iteration})\n" f"[bold cyan]Agent reasoning:[/bold cyan] ")

            # Get response from LLM (this will stream directly if callback is set)
            response = await self._get_llm_response(messages)
            assistant_response = response.content

            # Process any tool calls in the response
            processed_response = await self._process_tool_calls_streaming(assistant_response)

            # Add assistant response to conversation history
            conversation_history.append({"role": "assistant", "content": processed_response})

            # Add to full response
            full_response_parts.append(processed_response)

            # Check if task is complete
            if self._is_task_complete(processed_response):
                if self.stream_callback:
                    self.stream_callback("\n✅ [bold green]Task completed![/bold green]\n")
                break

            # Check if agent wants to continue (look for tool calls or continuation indicators)
            if not self._should_continue(processed_response):
                # If no tools were called and no clear continuation, ask if more work is needed
                follow_up = (
                    "Are there any other aspects of this task that need to be " "addressed? If the task is complete, please respond with " "TASK COMPLETE."
                )
                conversation_history.append({"role": "user", "content": follow_up})
                if self.stream_callback:
                    self.stream_callback("\n❓ [bold yellow]Checking if task is complete..." "[/bold yellow]\n")
                continue

        # Store conversation history for potential future use
        self.conversation_history = conversation_history

        return "\n".join(full_response_parts)

    async def _get_llm_response(self, messages: list[Message]):
        """Get response from LLM using the conversation messages with streaming."""
        # Use streaming conversation-based chat
        if self.stream_callback:
            # Stream the response in real-time
            full_content = ""
            stream_iterator = await llm_chat_conversation(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )

            # Stream chunks in real-time
            async for chunk in stream_iterator:
                if chunk.content:
                    full_content += chunk.content
                    # Stream each chunk immediately
                    self.stream_callback(chunk.content)

            # Create a mock response object with the full content
            class MockResponse:
                def __init__(self, content):
                    self.content = content

            self.stream_callback("\n\n")  # Add spacing after streaming
            return MockResponse(full_content)
        else:
            # Non-streaming fallback
            response = await llm_chat_conversation(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False,
            )
            return response

    def _is_task_complete(self, response: str) -> bool:
        """Check if the agent considers the task complete."""
        completion_indicators = [
            "TASK COMPLETE",
            "task is complete",
            "task has been completed",
            "finished",
            "done with",
        ]

        response_lower = response.lower()
        return any(indicator.lower() in response_lower for indicator in completion_indicators)

    def _should_continue(self, response: str) -> bool:
        """Determine if the agent should continue working."""
        # Check if tools were called (indication of ongoing work)
        if "Tool Result" in response:
            return True

        # Check for continuation phrases
        continuation_indicators = [
            "let me",
            "i'll",
            "next",
            "now i",
            "i need to",
            "i should",
            "let's",
            "i'm going to",
            "continuing",
            "next step",
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in continuation_indicators)

    def _clean_tool_call_display(self, response: str) -> str:
        """Replace JSON tool call blocks with nicely formatted tool calls."""
        # Find JSON blocks that might contain tool calls - more flexible pattern
        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",  # Standard json blocks
            r"\*\*Tool Call:\*\*\s*\n```json\s*(\{.*?\})\s*```",  # With "Tool Call:" prefix
        ]

        def replace_tool_call(match):
            # Extract JSON from the match - handle different capture groups
            json_str = None
            for i in range(1, len(match.groups()) + 1):
                if match.group(i):
                    json_str = match.group(i)
                    break

            if not json_str:
                json_str = match.group(0)

            try:
                tool_call = json.loads(json_str)

                if "tool" in tool_call and "args" in tool_call:
                    tool_name = tool_call["tool"]
                    tool_args = tool_call["args"]

                    # Format arguments nicely
                    if tool_args:
                        args_list = []
                        for key, value in tool_args.items():
                            if isinstance(value, str) and len(value) > 50:
                                # Truncate long string values
                                display_value = f'"{value[:47]}..."'
                            else:
                                display_value = repr(value)
                            args_list.append(f"  [yellow]{key}[/yellow]=[cyan]{display_value}[/cyan]")
                        args_display = "\n".join(args_list)

                        return f"[dim]📋[/dim] [bold magenta]Calling tool:[/bold magenta] " f"[bold white]{tool_name}[/bold white]\n{args_display}"
                    else:
                        return f"[dim]📋[/dim] [bold magenta]Calling tool:[/bold magenta] " f"[bold white]{tool_name}[/bold white]"
                else:
                    # Not a tool call, return original
                    return match.group(0)
            except (json.JSONDecodeError, KeyError):
                # Not a valid tool call, return original
                return match.group(0)

        # Replace all tool call JSON blocks with all patterns
        cleaned = response
        for pattern in json_patterns:
            cleaned = re.sub(pattern, replace_tool_call, cleaned, flags=re.DOTALL)

        return cleaned

    async def _process_tool_calls(self, response: str) -> str:
        """Process any tool calls found in the response (non-streaming version)."""
        return await self._process_tool_calls_streaming(response)

    async def _process_tool_calls_streaming(self, response: str) -> str:
        """Process any tool calls found in the response with streaming support."""
        # Find JSON blocks that might contain tool calls
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        json_matches = re.findall(json_pattern, response, re.DOTALL)

        if not json_matches:
            return response

        # Process each tool call
        updated_response = response

        for json_str in json_matches:
            try:
                tool_call = json.loads(json_str)

                if "tool" in tool_call and "args" in tool_call:
                    tool_name = tool_call["tool"]
                    tool_args = tool_call["args"]

                    if tool_name in self.tools:
                        # Stream tool execution
                        if self.stream_callback:
                            args_display = ", ".join([f"{k}={repr(v)}" for k, v in tool_args.items()])
                            self.stream_callback(
                                f"🔧 [bold magenta]Executing tool:[/bold magenta] " f"[cyan]{tool_name}[/cyan]([yellow]{args_display}[/yellow])\n"
                            )

                        # Execute the tool
                        tool_result = await self.tools[tool_name](**tool_args)

                        # Stream tool result
                        if self.stream_callback:
                            self.stream_callback(f"[bold green]Tool Result ({tool_name}):[/bold green]\n" f"[dim]```[/dim]\n{tool_result}\n[dim]```[/dim]\n\n")

                        # Replace the JSON block with the tool result
                        tool_result_text = f"\n**Tool Result ({tool_name}):**\n```\n{tool_result}\n```\n"
                        updated_response = updated_response.replace(f"```json\n{json_str}\n```", tool_result_text)
            except (json.JSONDecodeError, KeyError, TypeError):
                # If we can't parse the JSON or it's not a valid tool call, leave it as is
                continue

        return updated_response

    # Tool implementations

    async def _tool_read_file(self, file_path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read a file tool."""
        try:
            # Make path relative to working directory
            full_path = self.working_directory / file_path
            content = await read_file(str(full_path), start_line, end_line)
            return f"Content of {file_path}:\n{content}"
        except Exception as e:
            return f"Error reading {file_path}: {e}"

    async def _tool_edit_file(self, file_path: str, old_content: str, new_content: str) -> str:
        """Edit a file tool."""
        try:
            # Make path relative to working directory
            full_path = self.working_directory / file_path
            await edit_file(str(full_path), old_content, new_content)
            return f"Successfully edited {file_path}"
        except Exception as e:
            return f"Error editing {file_path}: {e}"

    async def _tool_search_files(
        self,
        query: str,
        file_pattern: str | None = None,
        case_sensitive: bool = False,
        use_regex: bool = False,
        max_results: int = 20,
    ) -> str:
        """Search files tool."""
        try:
            results = await search_files(
                query=query,
                directory=str(self.working_directory),
                file_pattern=file_pattern,
                case_sensitive=case_sensitive,
                use_regex=use_regex,
                max_results=max_results,
            )

            if not results:
                return f"No results found for '{query}'"

            result_text = f"Found {len(results)} results for '{query}':\n"
            for file_path, line_num, line_content in results:
                # Make path relative to working directory
                rel_path = Path(file_path).relative_to(self.working_directory)
                result_text += f"{rel_path}:{line_num}: {line_content.strip()}\n"

            return result_text
        except Exception as e:
            return f"Error searching files: {e}"

    async def _tool_find_files(self, name_pattern: str, max_results: int = 20) -> str:
        """Find files tool."""
        try:
            results = await find_files(
                name_pattern=name_pattern,
                directory=str(self.working_directory),
                max_results=max_results,
            )

            if not results:
                return f"No files found matching pattern '{name_pattern}'"

            result_text = f"Found {len(results)} files matching '{name_pattern}':\n"
            for file_path in results:
                # Make path relative to working directory
                rel_path = Path(file_path).relative_to(self.working_directory)
                result_text += f"{rel_path}\n"

            return result_text
        except Exception as e:
            return f"Error finding files: {e}"

    async def _tool_list_directory(self, directory_path: str | None = None) -> str:
        """List directory contents tool."""
        try:
            if directory_path is None:
                target_dir = self.working_directory
            else:
                target_dir = self.working_directory / directory_path

            if not target_dir.exists():
                return f"Directory does not exist: {directory_path or '.'}"

            if not target_dir.is_dir():
                return f"Path is not a directory: {directory_path or '.'}"

            items = []
            for item in sorted(target_dir.iterdir()):
                if item.is_dir():
                    items.append(f"{item.name}/")
                else:
                    items.append(item.name)

            if not items:
                return f"Directory is empty: {directory_path or '.'}"

            return f"Contents of {directory_path or '.'}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing directory: {e}"

    async def _tool_create_file(self, file_path: str, content: str) -> str:
        """Create a new file tool."""
        try:
            from swe.tools.file_tools import create_file

            # Make path relative to working directory
            full_path = self.working_directory / file_path
            await create_file(str(full_path), content)
            return f"Successfully created {file_path}"
        except Exception as e:
            return f"Error creating {file_path}: {e}"

    async def _tool_delete_file(self, file_path: str, confirm: bool = True) -> str:
        """Delete a file tool with confirmation."""
        try:
            # Make path relative to working directory
            full_path = self.working_directory / file_path
            await delete_file(str(full_path), confirm=confirm)
            return f"Successfully deleted {file_path}"
        except RuntimeError as e:
            if "cancelled" in str(e).lower():
                return f"File deletion cancelled: {e}"
            return f"Error deleting {file_path}: {e}"
        except Exception as e:
            return f"Error deleting {file_path}: {e}"

    async def _tool_execute_command(self, command: str, working_directory: str | None = None, confirm: bool = True, timeout: int = 30) -> str:
        """Execute a terminal command with confirmation."""
        try:
            # Use provided working directory or default to agent's working directory
            work_dir = working_directory or str(self.working_directory)

            # Import safety check function
            from swe.tools.terminal_tools import check_command_safety

            # Perform basic safety check
            is_safe, warning = await check_command_safety(command)
            if not is_safe:
                return f"SAFETY WARNING: {warning}\nCommand execution blocked for safety reasons."

            # Execute the command
            result = await execute_command(command=command, working_directory=work_dir, confirm=confirm, timeout=timeout)
            return result

        except RuntimeError as e:
            if "cancelled" in str(e).lower():
                return f"Command execution cancelled: {e}"
            return f"Command execution failed: {e}"
        except TimeoutError as e:
            return f"Command timed out: {e}"
        except Exception as e:
            return f"Error executing command: {e}"
