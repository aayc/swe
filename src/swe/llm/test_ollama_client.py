"""Tests for Ollama client."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from swe.llm.ollama_client import OllamaClient
from swe.llm.types import (
    APIError,
    ChatRequest,
    ChatResponse,
    Message,
    Role,
)


class TestOllamaClient:
    """Test cases for OllamaClient."""

    def test_init_default(self):
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 30.0
        assert client.client.api_key == "ollama"
        assert str(client.client.base_url) == "http://localhost:11434/v1/"

    def test_init_custom_params(self):
        client = OllamaClient(
            base_url="http://custom.ollama.server:8080/",
            timeout=60.0,
        )
        assert client.base_url == "http://custom.ollama.server:8080"
        assert client.timeout == 60.0
        # The implementation passes base_url directly to OpenAI client, so it includes the trailing slash
        assert str(client.client.base_url) == "http://custom.ollama.server:8080//v1/"

    def test_get_provider_name(self):
        client = OllamaClient()
        assert client.get_provider_name() == "ollama"

    def test_convert_messages(self):
        client = OllamaClient()
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

    def test_handle_ollama_error_model_not_found(self):
        client = OllamaClient()

        # Mock model not found error
        mock_error = Exception("Model not found")

        with pytest.raises(
            APIError,
            match="Model not found. Make sure the model is installed in Ollama.",
        ):
            client._handle_ollama_error(mock_error)

    def test_handle_ollama_error_404(self):
        client = OllamaClient()

        # Mock 404 error
        mock_error = Exception("404 error")

        with pytest.raises(
            APIError,
            match="Model not found. Make sure the model is installed in Ollama.",
        ):
            client._handle_ollama_error(mock_error)

    def test_handle_ollama_error_generic(self):
        client = OllamaClient()

        # Mock generic error
        mock_error = Exception("Some other error")

        with pytest.raises(APIError, match="Ollama client error"):
            client._handle_ollama_error(mock_error)

    @pytest.mark.asyncio
    async def test_chat_success(self):
        client = OllamaClient()
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="llama3.2:3b")

        # Mock OpenAI ChatCompletion response
        mock_message = Mock()
        mock_message.content = "Hello there!"

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 15
        mock_usage.total_tokens = 25

        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        mock_completion.model = "llama3.2:3b"
        mock_completion.usage = mock_usage

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ):
            response = await client.chat(request)

            assert isinstance(response, ChatResponse)
            assert response.content == "Hello there!"
            assert response.model == "llama3.2:3b"
            assert response.finish_reason == "stop"
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 15
            assert response.usage.total_tokens == 25

    @pytest.mark.asyncio
    async def test_chat_no_usage(self):
        client = OllamaClient()
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="llama3.2:3b")

        # Mock OpenAI ChatCompletion response without usage
        mock_message = Mock()
        mock_message.content = "Hello there!"

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        mock_completion.model = "llama3.2:3b"
        mock_completion.usage = None

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ):
            response = await client.chat(request)

            assert response.usage is None

    @pytest.mark.asyncio
    async def test_chat_no_content(self):
        client = OllamaClient()
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="llama3.2:3b")

        # Mock OpenAI ChatCompletion response with no content
        mock_message = Mock()
        mock_message.content = None

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        mock_completion.model = "llama3.2:3b"
        mock_completion.usage = None

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ):
            response = await client.chat(request)

            assert response.content == ""

    @pytest.mark.asyncio
    async def test_chat_stream_request_error(self):
        client = OllamaClient()
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="llama3.2:3b", stream=True)

        with pytest.raises(ValueError, match="Use chat_stream\\(\\) for streaming requests"):
            await client.chat(request)

    @pytest.mark.asyncio
    async def test_chat_api_error(self):
        client = OllamaClient()
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="llama3.2:3b")

        # Mock OpenAI client to raise an exception
        test_exception = Exception("Test error")

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=test_exception,
        ):
            with pytest.raises(APIError, match="Ollama client error"):
                await client.chat(request)

    @pytest.mark.asyncio
    async def test_chat_stream_success(self):
        client = OllamaClient()
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="llama3.2:3b")

        # Mock streaming chunks
        mock_delta1 = Mock()
        mock_delta1.content = "Hello"

        mock_choice1 = Mock()
        mock_choice1.delta = mock_delta1
        mock_choice1.finish_reason = None

        mock_chunk1 = Mock()
        mock_chunk1.choices = [mock_choice1]
        mock_chunk1.model = "llama3.2:3b"
        mock_chunk1.usage = None

        mock_delta2 = Mock()
        mock_delta2.content = " there!"

        mock_choice2 = Mock()
        mock_choice2.delta = mock_delta2
        mock_choice2.finish_reason = None

        mock_chunk2 = Mock()
        mock_chunk2.choices = [mock_choice2]
        mock_chunk2.model = "llama3.2:3b"
        mock_chunk2.usage = None

        # Final chunk with finish reason and usage
        mock_delta3 = Mock()
        mock_delta3.content = ""

        mock_choice3 = Mock()
        mock_choice3.delta = mock_delta3
        mock_choice3.finish_reason = "stop"

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 15
        mock_usage.total_tokens = 25

        mock_chunk3 = Mock()
        mock_chunk3.choices = [mock_choice3]
        mock_chunk3.model = "llama3.2:3b"
        mock_chunk3.usage = mock_usage

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2
            yield mock_chunk3

        with patch.object(client.client.chat.completions, "create", return_value=mock_stream()):
            chunks = []
            async for chunk in client.chat_stream(request):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
            assert chunks[0].finish_reason is None
            assert chunks[0].usage is None
            assert chunks[1].content == " there!"
            assert chunks[1].finish_reason is None
            assert chunks[1].usage is None
            assert chunks[2].content == ""
            assert chunks[2].finish_reason == "stop"
            assert chunks[2].usage.total_tokens == 25

    @pytest.mark.asyncio
    async def test_chat_stream_api_error(self):
        client = OllamaClient()
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="llama3.2:3b")

        # Mock OpenAI client to raise an exception during streaming
        test_exception = Exception("Test error")

        with patch.object(client.client.chat.completions, "create", side_effect=test_exception):
            with pytest.raises(APIError, match="Ollama client error"):
                async for _ in client.chat_stream(request):
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
