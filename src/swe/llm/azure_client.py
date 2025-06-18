from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

from .types import (
    APIError,
    AuthenticationError,
    ChatRequest,
    ChatResponse,
    ConfigError,
    LLMProvider,
    Message,
    RateLimitError,
    StreamingChunk,
    Usage,
)


class AzureOpenAIClient(LLMProvider):
    """Azure OpenAI API client implementation using official openai library."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str = "2024-02-01",
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ConfigError("Azure OpenAI API key is required")
        if not endpoint:
            raise ConfigError("Azure OpenAI endpoint is required")

        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        self.timeout = timeout

        # Initialize the official Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            timeout=timeout,
        )

    def get_provider_name(self) -> str:
        return "azure_openai"

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal message format to Azure OpenAI format."""
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    def _handle_azure_error(self, error: Exception) -> None:
        """Convert Azure OpenAI library exceptions to our custom exceptions."""
        from openai import APIError as OpenAIAPIError
        from openai import AuthenticationError as OpenAIAuthError
        from openai import RateLimitError as OpenAIRateLimitError

        if isinstance(error, OpenAIAuthError):
            raise AuthenticationError("Invalid API key")
        elif isinstance(error, OpenAIRateLimitError):
            raise RateLimitError("Rate limit exceeded")
        elif isinstance(error, OpenAIAPIError):
            raise APIError(f"Azure OpenAI API error: {error}")
        else:
            raise APIError(f"Azure OpenAI client error: {error}")

    def _extract_usage(self, completion: ChatCompletion) -> Usage | None:
        """Extract usage information from Azure OpenAI response."""
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
                "model": request.model,  # This is the deployment name
                "messages": self._convert_messages(request.messages),
                "stream": False,
            }

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
            self._handle_azure_error(e)
            raise  # This should never be reached due to _handle_azure_error raising

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamingChunk]:
        """Send a chat request and return streaming response."""
        try:
            # Create request parameters
            params: dict[str, Any] = {
                "model": request.model,  # This is the deployment name
                "messages": self._convert_messages(request.messages),
                "stream": True,
            }

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
            self._handle_azure_error(e)
            raise  # This should never be reached due to _handle_azure_error raising
