from collections.abc import AsyncIterator
from typing import Any

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

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


class GeminiClient(LLMProvider):
    """Google Gemini API client implementation using official google-generativeai library."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ConfigError("Google API key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Configure the official Google Generative AI library
        genai.configure(api_key=api_key)  # type: ignore

    def get_provider_name(self) -> str:
        return "gemini"

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Convert internal message format to Gemini format."""
        system_instruction = ""
        gemini_contents = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
            else:
                # Skip empty or whitespace-only messages
                if not msg.content or not msg.content.strip():
                    continue

                # Gemini uses "user" and "model" instead of "assistant"
                role = "model" if msg.role == Role.ASSISTANT else "user"
                gemini_contents.append({"role": role, "parts": [{"text": msg.content}]})

        return system_instruction, gemini_contents

    def _handle_gemini_error(self, error: Exception) -> None:
        """Convert Google Generative AI library exceptions to our custom exceptions."""
        # Google's client library uses different exception types
        error_str = str(error).lower()

        if "api key" in error_str or "permission" in error_str or "auth" in error_str:
            raise AuthenticationError("Invalid API key")
        elif "quota" in error_str or "rate limit" in error_str:
            raise RateLimitError("Rate limit exceeded")
        elif "400" in error_str or "invalid" in error_str:
            raise APIError(f"Gemini API error: {error}")
        else:
            raise APIError(f"Gemini client error: {error}")

    def _extract_usage(self, response: GenerateContentResponse) -> Usage | None:
        """Extract usage information from Gemini response."""
        if not hasattr(response, "usage_metadata") or not response.usage_metadata:
            return None

        return Usage(
            prompt_tokens=response.usage_metadata.prompt_token_count,
            completion_tokens=response.usage_metadata.candidates_token_count,
            total_tokens=response.usage_metadata.total_token_count,
        )

    def _extract_content(self, response: GenerateContentResponse) -> str:
        """Extract text content from Gemini response."""
        if not response.candidates:
            return ""

        content = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text"):
                content += part.text

        return content

    def _extract_finish_reason(self, response: GenerateContentResponse) -> str | None:
        """Extract finish reason from Gemini response."""
        if not response.candidates:
            return None

        finish_reason = response.candidates[0].finish_reason
        if finish_reason:
            return str(finish_reason.name.lower())
        return None

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request and return the response."""
        if request.stream:
            raise ValueError("Use chat_stream() for streaming requests")

        system_instruction, contents = self._convert_messages(request.messages)

        # Validate that we have at least one valid user message
        valid_user_messages = [content for content in contents if content["role"] == "user" and content["parts"][0]["text"].strip()]

        if not valid_user_messages:
            raise ValueError("No valid user message found in the conversation")

        try:
            # Create the model instance
            if system_instruction:
                model = genai.GenerativeModel(request.model, system_instruction=system_instruction)  # type: ignore
            else:
                model = genai.GenerativeModel(request.model)  # type: ignore

            # Create generation config
            generation_config: dict[str, Any] = {}
            if request.max_tokens is not None:
                generation_config["max_output_tokens"] = request.max_tokens
            if request.temperature is not None:
                generation_config["temperature"] = float(request.temperature)
            if request.top_p is not None:
                generation_config["top_p"] = float(request.top_p)

            # Handle different content formats based on conversation type
            if len(contents) == 1 and contents[0]["role"] == "user":
                # Single user prompt - pass as simple string
                content_input = contents[0]["parts"][0]["text"]
                response = await model.generate_content_async(  # type: ignore[arg-type]
                    content_input,
                    generation_config=generation_config if generation_config else None,
                )
            else:
                # Multi-turn conversation - pass as structured content
                # Gemini API accepts a list of Content objects for multi-turn conversations
                response = await model.generate_content_async(  # type: ignore[arg-type]
                    contents,
                    generation_config=generation_config if generation_config else None,
                )

            # Extract response data
            content = self._extract_content(response)  # type: ignore
            usage = self._extract_usage(response)  # type: ignore
            finish_reason = self._extract_finish_reason(response)  # type: ignore

            return ChatResponse(
                content=content,
                model=request.model,
                usage=usage,
                finish_reason=finish_reason,
            )

        except Exception as e:
            self._handle_gemini_error(e)
            raise  # This should never be reached due to _handle_gemini_error raising

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamingChunk]:
        """Send a chat request and return streaming response."""
        system_instruction, contents = self._convert_messages(request.messages)

        # Validate that we have at least one valid user message
        valid_user_messages = [content for content in contents if content["role"] == "user" and content["parts"][0]["text"].strip()]

        if not valid_user_messages:
            raise ValueError("No valid user message found in the conversation")

        try:
            # Create the model instance
            if system_instruction:
                model = genai.GenerativeModel(request.model, system_instruction=system_instruction)  # type: ignore
            else:
                model = genai.GenerativeModel(request.model)  # type: ignore

            # Create generation config
            generation_config: dict[str, Any] = {}
            if request.max_tokens is not None:
                generation_config["max_output_tokens"] = request.max_tokens
            if request.temperature is not None:
                generation_config["temperature"] = float(request.temperature)
            if request.top_p is not None:
                generation_config["top_p"] = float(request.top_p)

            # Handle different content formats based on conversation type
            if len(contents) == 1 and contents[0]["role"] == "user":
                # Single user prompt - pass as simple string
                content_input = contents[0]["parts"][0]["text"]
                response_stream = await model.generate_content_async(  # type: ignore[arg-type]
                    content_input,
                    generation_config=generation_config if generation_config else None,
                    stream=True,
                )
            else:
                # Multi-turn conversation - pass as structured content
                # Gemini API accepts a list of Content objects for multi-turn conversations
                response_stream = await model.generate_content_async(  # type: ignore[arg-type]
                    contents,
                    generation_config=generation_config if generation_config else None,
                    stream=True,
                )

            async for chunk in response_stream:
                content = ""
                finish_reason = None
                usage = None

                # Extract content from the chunk
                if chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, "text"):
                            content += part.text

                    # Check for finish reason
                    if chunk.candidates[0].finish_reason:
                        finish_reason = chunk.candidates[0].finish_reason.name.lower()

                # Extract usage information (typically only in final chunk)
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = Usage(
                        prompt_tokens=chunk.usage_metadata.prompt_token_count,
                        completion_tokens=chunk.usage_metadata.candidates_token_count,
                        total_tokens=chunk.usage_metadata.total_token_count,
                    )

                yield StreamingChunk(
                    content=content,
                    model=request.model,
                    finish_reason=finish_reason,
                    usage=usage,
                )

        except Exception as e:
            self._handle_gemini_error(e)
            raise  # This should never be reached due to _handle_gemini_error raising
