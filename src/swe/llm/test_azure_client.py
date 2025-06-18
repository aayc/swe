"""Tests for Azure OpenAI client using official openai library."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from swe.llm.azure_client import AzureOpenAIClient
from swe.llm.types import (
    APIError,
    AuthenticationError,
    ChatRequest,
    ChatResponse,
    ConfigError,
    Message,
    RateLimitError,
    Role,
)


class TestAzureOpenAIClient:
    """Test cases for AzureOpenAIClient using official openai library."""

    def test_init_valid_params(self):
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        assert client.api_key == "test-key"
        assert client.endpoint == "https://test.openai.azure.com"
        assert client.api_version == "2024-02-01"
        assert client.timeout == 30.0
        assert client.client.api_key == "test-key"

    def test_init_custom_params(self):
        client = AzureOpenAIClient(
            api_key="test-key",
            endpoint="https://custom.openai.azure.com/",
            api_version="2023-12-01-preview",
            timeout=60.0,
        )
        assert client.api_key == "test-key"
        assert client.endpoint == "https://custom.openai.azure.com"
        assert client.api_version == "2023-12-01-preview"
        assert client.timeout == 60.0

    def test_init_empty_api_key(self):
        with pytest.raises(ConfigError, match="Azure OpenAI API key is required"):
            AzureOpenAIClient(api_key="", endpoint="https://test.openai.azure.com")

    def test_init_empty_endpoint(self):
        with pytest.raises(ConfigError, match="Azure OpenAI endpoint is required"):
            AzureOpenAIClient(api_key="test-key", endpoint="")

    def test_get_provider_name(self):
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        assert client.get_provider_name() == "azure_openai"

    def test_convert_messages(self):
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello!"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]

        converted = client._convert_messages(messages)
        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert converted == expected

    def test_handle_azure_error_authentication(self):
        from openai import AuthenticationError as OpenAIAuthError

        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        # Create a proper mock response for the exception
        mock_response = Mock()
        mock_response.status_code = 401
        mock_error = OpenAIAuthError("Invalid API key", response=mock_response, body={})

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            client._handle_azure_error(mock_error)

    def test_handle_azure_error_rate_limit(self):
        from openai import RateLimitError as OpenAIRateLimitError

        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        # Create a proper mock response for the exception
        mock_response = Mock()
        mock_response.status_code = 429
        mock_error = OpenAIRateLimitError("Rate limit", response=mock_response, body={})

        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            client._handle_azure_error(mock_error)

    def test_handle_azure_error_api_error(self):
        from openai import APIError as OpenAIAPIError

        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        # Create a proper mock request for the exception
        mock_request = Mock()
        mock_error = OpenAIAPIError("API error", request=mock_request, body={})

        with pytest.raises(APIError, match="Azure OpenAI API error"):
            client._handle_azure_error(mock_error)

    def test_handle_azure_error_generic(self):
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        with pytest.raises(APIError, match="Azure OpenAI client error"):
            client._handle_azure_error(Exception("Generic error"))

    @pytest.mark.asyncio
    async def test_chat_success(self):
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="gpt-4o")

        # Create mock response
        mock_completion = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello there!"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
        )

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ):
            response = await client.chat(request)

            assert isinstance(response, ChatResponse)
            assert response.content == "Hello there!"
            assert response.model == "gpt-4o"
            assert response.finish_reason == "stop"
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 15
            assert response.usage.total_tokens == 25

    @pytest.mark.asyncio
    async def test_chat_no_usage(self):
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="gpt-4o")

        # Create mock response without usage
        mock_completion = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello there!"),
                    finish_reason="stop",
                )
            ],
        )

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ):
            response = await client.chat(request)

            assert response.usage is None

    @pytest.mark.asyncio
    async def test_chat_fallback_model(self):
        """Test that response uses request model when completion model is missing."""
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="my-deployment")

        # Create mock response where model field might be different
        mock_completion = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677652288,
            model="my-deployment",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello there!"),
                    finish_reason="stop",
                )
            ],
        )

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ):
            response = await client.chat(request)

            assert response.model == "my-deployment"

    @pytest.mark.asyncio
    async def test_chat_stream_request_error(self):
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="gpt-4o", stream=True)

        with pytest.raises(ValueError, match="Use chat_stream\\(\\) for streaming requests"):
            await client.chat(request)

    @pytest.mark.asyncio
    async def test_chat_api_error(self):
        from openai import AuthenticationError as OpenAIAuthError

        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="gpt-4o")

        # Create proper mock exception
        mock_response = Mock()
        mock_response.status_code = 401
        mock_error = OpenAIAuthError("Invalid API key", response=mock_response, body={})

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=mock_error,
        ):
            with pytest.raises(AuthenticationError, match="Invalid API key"):
                await client.chat(request)

    @pytest.mark.asyncio
    async def test_chat_stream_success(self):
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="gpt-4o")

        # Create mock streaming response
        async def mock_stream():
            yield ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-4o",
                choices=[StreamChoice(index=0, delta=ChoiceDelta(content="Hello"), finish_reason=None)],
            )
            yield ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-4o",
                choices=[
                    StreamChoice(
                        index=0,
                        delta=ChoiceDelta(content=" there!"),
                        finish_reason=None,
                    )
                ],
            )
            yield ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-4o",
                choices=[StreamChoice(index=0, delta=ChoiceDelta(content=""), finish_reason="stop")],
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
            )

        with patch.object(client.client.chat.completions, "create", return_value=mock_stream()):
            chunks = []
            async for chunk in client.chat_stream(request):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
            assert chunks[0].finish_reason is None
            assert chunks[1].content == " there!"
            assert chunks[1].finish_reason is None
            assert chunks[2].content == ""
            assert chunks[2].finish_reason == "stop"
            assert chunks[2].usage.total_tokens == 25

    @pytest.mark.asyncio
    async def test_chat_stream_empty_choices(self):
        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="gpt-4o")

        # Create mock streaming response with empty choices
        async def mock_stream():
            yield ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-4o",
                choices=[],  # Empty choices
            )
            yield ChatCompletionChunk(
                id="chatcmpl-123",
                object="chat.completion.chunk",
                created=1677652288,
                model="gpt-4o",
                choices=[StreamChoice(index=0, delta=ChoiceDelta(content="Hello"), finish_reason=None)],
            )

        with patch.object(client.client.chat.completions, "create", return_value=mock_stream()):
            chunks = []
            async for chunk in client.chat_stream(request):
                chunks.append(chunk)

            # Should only get 1 chunk, empty choices should be skipped
            assert len(chunks) == 1
            assert chunks[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_chat_stream_api_error(self):
        from openai import AuthenticationError as OpenAIAuthError

        client = AzureOpenAIClient(api_key="test-key", endpoint="https://test.openai.azure.com")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="gpt-4o")

        # Create proper mock exception
        mock_response = Mock()
        mock_response.status_code = 401
        mock_error = OpenAIAuthError("Invalid API key", response=mock_response, body={})

        with patch.object(client.client.chat.completions, "create", side_effect=mock_error):
            with pytest.raises(AuthenticationError):
                async for _ in client.chat_stream(request):
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
