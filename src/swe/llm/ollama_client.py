from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from .types import (
    APIError,
    ChatRequest,
    ChatResponse,
    LLMProvider,
    Message,
    StreamingChunk,
    Usage,
)


class OllamaClient(LLMProvider):
    """Ollama API client implementation using OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Initialize OpenAI client pointing to Ollama's OpenAI-compatible endpoint
        # Ollama serves OpenAI-compatible API at /v1
        self.client = AsyncOpenAI(
            api_key="ollama",  # Ollama doesn't require a real API key
            base_url=f"{base_url}/v1",
            timeout=timeout,
        )

    def get_provider_name(self) -> str:
        return "ollama"

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal message format to OpenAI format (which Ollama supports)."""
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    def _handle_ollama_error(self, error: Exception) -> None:
        """Convert Ollama errors to our custom exceptions."""
        error_str = str(error).lower()

        if "model not found" in error_str or "404" in error_str:
            raise APIError("Model not found. Make sure the model is installed in Ollama.")
        else:
            raise APIError(f"Ollama client error: {error}")

    def _extract_usage(self, completion: ChatCompletion) -> Usage | None:
        """Extract usage information from Ollama response."""
        if not completion.usage:
            return None

        return Usage(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request and return the response."""
        if request.stream:
            raise ValueError("Use chat_stream() for streaming requests")

        try:
            # Create request parameters
            params: dict[str, Any] = {
                "model": request.model,
                "messages": self._convert_messages(request.messages),
                "stream": False,
            }

            # Ollama uses different parameter names for some options
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p

            # Make the API call
            completion = await self.client.chat.completions.create(**params)

            # Extract response data
            choice = completion.choices[0]
            usage = self._extract_usage(completion)

            return ChatResponse(
                content=choice.message.content or "",
                model=completion.model,
                usage=usage,
                finish_reason=choice.finish_reason,
            )

        except Exception as e:
            self._handle_ollama_error(e)
            raise  # This should never be reached due to _handle_ollama_error raising

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamingChunk]:
        """Send a chat request and return streaming response."""
        try:
            # Create request parameters
            params: dict[str, Any] = {
                "model": request.model,
                "messages": self._convert_messages(request.messages),
                "stream": True,
            }

            # Ollama uses different parameter names for some options
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p

            # Make the streaming API call
            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                content = delta.content or ""
                finish_reason = choice.finish_reason

                # Handle usage information (typically only in the last chunk)
                usage = None
                if chunk.usage:
                    usage = Usage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )

                yield StreamingChunk(
                    content=content,
                    model=chunk.model or request.model,
                    finish_reason=finish_reason,
                    usage=usage,
                )

        except Exception as e:
            self._handle_ollama_error(e)
            raise  # This should never be reached due to _handle_ollama_error raising
