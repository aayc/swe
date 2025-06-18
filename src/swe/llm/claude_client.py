from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic.types import Message as AnthropicMessage

from .types import (
    APIError,
    AuthenticationError,
    ChatRequest,
    ChatResponse,
    ConfigError,
    LLMProvider,
    Message,
    RateLimitError,
    Role,
    StreamingChunk,
    Usage,
)


class ClaudeClient(LLMProvider):
    """Anthropic Claude API client implementation using official anthropic library."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 30.0,
        anthropic_version: str = "2023-06-01",
    ):
        if not api_key:
            raise ConfigError("Anthropic API key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.anthropic_version = anthropic_version

        # Initialize the official Anthropic client
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def get_provider_name(self) -> str:
        return "anthropic"

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Convert internal message format to Claude format."""
        system_message = ""
        claude_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_message = msg.content
            else:
                claude_messages.append({"role": msg.role.value, "content": msg.content})

        return system_message, claude_messages

    def _handle_anthropic_error(self, error: Exception) -> None:
        """Convert Anthropic library exceptions to our custom exceptions."""
        if isinstance(error, anthropic.AuthenticationError):
            raise AuthenticationError("Invalid API key")
        elif isinstance(error, anthropic.RateLimitError):
            raise RateLimitError("Rate limit exceeded")
        elif isinstance(error, anthropic.APIError):
            raise APIError(f"Claude API error: {error}")
        else:
            raise APIError(f"Claude client error: {error}")

    def _extract_usage(self, message: AnthropicMessage) -> Usage | None:
        """Extract usage information from Anthropic response."""
        if not message.usage:
            return None

        return Usage(
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            total_tokens=message.usage.input_tokens + message.usage.output_tokens,
        )

    def _extract_content(self, message: AnthropicMessage) -> str:
        """Extract text content from Anthropic message."""
        content = ""
        for block in message.content:
            if block.type == "text":
                content += block.text
        return content

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request and return the response."""
        if request.stream:
            raise ValueError("Use chat_stream() for streaming requests")

        try:
            system_message, messages = self._convert_messages(request.messages)

            # Create request parameters
            params: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or 4096,  # Claude requires max_tokens
            }

            if system_message:
                params["system"] = system_message
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p

            # Make the API call
            response = await self.client.messages.create(**params)

            # Extract response data
            content = self._extract_content(response)
            usage = self._extract_usage(response)

            return ChatResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.stop_reason,
            )

        except Exception as e:
            self._handle_anthropic_error(e)
            raise  # This should never be reached due to _handle_anthropic_error raising

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamingChunk]:
        """Send a chat request and return streaming response."""
        try:
            system_message, messages = self._convert_messages(request.messages)

            # Create request parameters
            params: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or 4096,  # Claude requires max_tokens
                "stream": True,
            }

            if system_message:
                params["system"] = system_message
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p

            # Make the streaming API call
            stream = await self.client.messages.create(**params)

            async for event in stream:
                if event.type == "content_block_delta":
                    content = event.delta.text if hasattr(event.delta, "text") else ""

                    yield StreamingChunk(
                        content=content,
                        model=request.model,
                        finish_reason=None,
                        usage=None,
                    )

                elif event.type == "message_delta":
                    finish_reason = event.delta.stop_reason if hasattr(event.delta, "stop_reason") else None
                    usage = None

                    if hasattr(event, "usage") and event.usage:
                        usage = Usage(
                            prompt_tokens=event.usage.input_tokens,
                            completion_tokens=event.usage.output_tokens,
                            total_tokens=event.usage.input_tokens + event.usage.output_tokens,
                        )

                    yield StreamingChunk(
                        content="",
                        model=request.model,
                        finish_reason=finish_reason,
                        usage=usage,
                    )

        except Exception as e:
            self._handle_anthropic_error(e)
            raise  # This should never be reached due to _handle_anthropic_error raising
