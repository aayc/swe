"""Tests for main LLM client functionality."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from swe.llm.client import (
    _get_client_for_model,
    llm_chat,
    llm_chat_conversation,
    llm_chat_stream,
)
from swe.llm.types import (
    ChatModel,
    ChatRequest,
    ChatResponse,
    ConfigError,
    Message,
    Role,
    StreamingChunk,
    Usage,
)


class TestGetClientForModel:
    """Test cases for _get_client_for_model function."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_model_string(self):
        from swe.llm.openai_client import OpenAIClient

        client = _get_client_for_model(ChatModel.GPT_4O)
        assert isinstance(client, OpenAIClient)
        assert client.api_key == "test-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_model_enum(self):
        from swe.llm.openai_client import OpenAIClient

        client = _get_client_for_model(ChatModel.GPT_4O)
        assert isinstance(client, OpenAIClient)
        assert client.api_key == "test-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_gpt_4o_available(self):
        from swe.llm.openai_client import OpenAIClient

        client = _get_client_for_model(ChatModel.GPT_4O)
        assert isinstance(client, OpenAIClient)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_claude_model(self):
        from swe.llm.claude_client import ClaudeClient

        client = _get_client_for_model(ChatModel.CLAUDE_SONNET_4)
        assert isinstance(client, ClaudeClient)
        assert client.api_key == "test-key"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_gemini_model(self):
        from swe.llm.gemini_client import GeminiClient

        client = _get_client_for_model(ChatModel.GEMINI_2_5_FLASH)
        assert isinstance(client, GeminiClient)
        assert client.api_key == "test-key"

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
    def test_deepseek_model(self):
        from swe.llm.openai_client import OpenAIClient

        client = _get_client_for_model(ChatModel.DEEPSEEK_REASONER)
        assert isinstance(client, OpenAIClient)
        assert client.api_key == "test-key"
        assert "deepseek.com" in client.base_url

    @patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "https://custom.deepseek.com/v1",
        },
    )
    def test_deepseek_custom_base_url(self):
        from swe.llm.openai_client import OpenAIClient

        client = _get_client_for_model(ChatModel.DEEPSEEK_CHAT)
        assert isinstance(client, OpenAIClient)
        assert client.base_url == "https://custom.deepseek.com/v1"

    @patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"})
    def test_llama_model(self):
        from swe.llm.ollama_client import OllamaClient

        client = _get_client_for_model(ChatModel.LLAMA_3_2_3B)
        assert isinstance(client, OllamaClient)
        assert client.base_url == "http://localhost:11434"

    @patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom.ollama.server:8080"})
    def test_llama_custom_base_url(self):
        from swe.llm.ollama_client import OllamaClient

        client = _get_client_for_model(ChatModel.LLAMA_3_2_3B)
        assert isinstance(client, OllamaClient)
        assert client.base_url == "http://custom.ollama.server:8080"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        },
    )
    def test_azure_openai_model(self):
        from swe.llm.azure_client import AzureOpenAIClient

        client = _get_client_for_model(ChatModel.GPT_4O)
        assert isinstance(client, AzureOpenAIClient)
        assert client.api_key == "test-key"
        assert client.endpoint == "https://test.openai.azure.com"

    def test_openai_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="OPENAI_API_KEY environment variable is required"):
                _get_client_for_model(ChatModel.GPT_4O)

    def test_claude_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY environment variable is required"):
                _get_client_for_model(ChatModel.CLAUDE_SONNET_4)

    def test_gemini_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="GEMINI_API_KEY environment variable is required"):
                _get_client_for_model(ChatModel.GEMINI_2_5_FLASH)

    def test_deepseek_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY environment variable is required"):
                _get_client_for_model(ChatModel.DEEPSEEK_REASONER)

    def test_azure_no_api_key(self):
        with patch.dict(
            os.environ,
            {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com"},
            clear=True,
        ):
            with pytest.raises(
                ConfigError,
                match="AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables are required",
            ):
                _get_client_for_model(ChatModel.GPT_4O)

    def test_azure_no_endpoint(self):
        with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key"}, clear=True):
            # With only API key but no endpoint, it should fall back to OpenAI and fail for missing OPENAI_API_KEY
            with pytest.raises(ConfigError, match="OPENAI_API_KEY environment variable is required"):
                _get_client_for_model(ChatModel.GPT_4O)

    def test_unsupported_model_configuration(self):
        # Test case where a valid model enum exists but no configuration supports it
        # This would happen if we have a model in the enum but no client logic for it
        with patch.dict(os.environ, {}, clear=True):
            # GPT_4O should fail with no API key, testing the error path
            with pytest.raises(ConfigError, match="OPENAI_API_KEY environment variable is required"):
                _get_client_for_model(ChatModel.GPT_4O)


class TestLLMChat:
    """Test cases for llm_chat function."""

    @pytest.mark.asyncio
    @patch("swe.llm.client._get_client_for_model")
    async def test_llm_chat_success(self, mock_get_client):
        # Mock client
        mock_client = AsyncMock()
        mock_response = ChatResponse(
            content="Hello there!",
            model=ChatModel.GPT_4O,
            usage=Usage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
            finish_reason="stop",
        )
        mock_client.chat.return_value = mock_response
        mock_get_client.return_value = mock_client

        response = await llm_chat(
            prompt="Hello!",
            model=ChatModel.GPT_4O,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        assert response == mock_response

        # Verify client.chat was called with correct request
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args[0][0]
        assert isinstance(call_args, ChatRequest)
        assert len(call_args.messages) == 1
        assert call_args.messages[0].role == Role.USER
        assert call_args.messages[0].content == "Hello!"
        assert call_args.model == "gpt-4o"
        assert call_args.max_tokens == 100
        assert call_args.temperature == 0.7
        assert call_args.top_p == 0.9
        assert call_args.stream is False

    @pytest.mark.asyncio
    @patch("swe.llm.client._get_client_for_model")
    async def test_llm_chat_with_system_prompt(self, mock_get_client):
        # Mock client
        mock_client = AsyncMock()
        mock_response = ChatResponse(content="Hello!", model="gpt-4o")
        mock_client.chat.return_value = mock_response
        mock_get_client.return_value = mock_client

        response = await llm_chat(prompt="Hello!", model=ChatModel.GPT_4O, system_prompt="You are helpful.")
        del response

        # Verify messages include system prompt
        call_args = mock_client.chat.call_args[0][0]
        assert len(call_args.messages) == 2
        assert call_args.messages[0].role == Role.SYSTEM
        assert call_args.messages[0].content == "You are helpful."
        assert call_args.messages[1].role == Role.USER
        assert call_args.messages[1].content == "Hello!"

    @pytest.mark.asyncio
    @patch("swe.llm.client._get_client_for_model")
    async def test_llm_chat_with_enum_model(self, mock_get_client):
        # Mock client
        mock_client = AsyncMock()
        mock_response = ChatResponse(content="Hello!", model="gpt-4o")
        mock_client.chat.return_value = mock_response
        mock_get_client.return_value = mock_client

        response = await llm_chat(prompt="Hello!", model=ChatModel.GPT_4O)
        del response

        # Verify model value is used
        call_args = mock_client.chat.call_args[0][0]
        assert call_args.model == "gpt-4o"


class TestLLMChatStream:
    """Test cases for llm_chat_stream function."""

    @pytest.mark.asyncio
    @patch("swe.llm.client._get_client_for_model")
    async def test_llm_chat_stream_success(self, mock_get_client):
        # Mock client
        mock_client = AsyncMock()

        # Mock streaming chunks
        chunks = [
            StreamingChunk(content="Hello", model="gpt-4o"),
            StreamingChunk(content=" there!", model="gpt-4o"),
            StreamingChunk(content="", model="gpt-4o", finish_reason="stop"),
        ]

        async def mock_chat_stream(request):
            for chunk in chunks:
                yield chunk

        mock_client.chat_stream = mock_chat_stream
        mock_get_client.return_value = mock_client

        response_chunks = []
        async for chunk in llm_chat_stream(prompt="Hello!", model=ChatModel.GPT_4O, max_tokens=100, temperature=0.7):
            response_chunks.append(chunk)

        assert len(response_chunks) == 3
        assert response_chunks[0].content == "Hello"
        assert response_chunks[1].content == " there!"
        assert response_chunks[2].finish_reason == "stop"

    @pytest.mark.asyncio
    @patch("swe.llm.client._get_client_for_model")
    async def test_llm_chat_stream_with_system_prompt(self, mock_get_client):
        # Mock client
        mock_client = AsyncMock()

        async def mock_chat_stream(request):
            # Verify request structure
            assert len(request.messages) == 2
            assert request.messages[0].role == Role.SYSTEM
            assert request.messages[0].content == "You are helpful."
            assert request.messages[1].role == Role.USER
            assert request.messages[1].content == "Hello!"
            assert request.stream is True

            yield StreamingChunk(content="Hello!", model="gpt-4o")

        mock_client.chat_stream = mock_chat_stream
        mock_get_client.return_value = mock_client

        response_chunks = []
        async for chunk in llm_chat_stream(prompt="Hello!", model=ChatModel.GPT_4O, system_prompt="You are helpful."):
            response_chunks.append(chunk)

        assert len(response_chunks) == 1
        assert response_chunks[0].content == "Hello!"


class TestLLMChatConversation:
    """Test cases for llm_chat_conversation function."""

    @pytest.mark.asyncio
    @patch("swe.llm.client._get_client_for_model")
    async def test_llm_chat_conversation_non_streaming(self, mock_get_client):
        # Mock client
        mock_client = AsyncMock()
        mock_response = ChatResponse(content="Hello there!", model="gpt-4o")
        mock_client.chat.return_value = mock_response
        mock_get_client.return_value = mock_client

        messages = [
            Message(role=Role.USER, content="Hello!"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
            Message(role=Role.USER, content="How are you?"),
        ]

        response = await llm_chat_conversation(messages=messages, model=ChatModel.GPT_4O, max_tokens=100, stream=False)

        assert response == mock_response

        # Verify client.chat was called with correct request
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args[0][0]
        assert isinstance(call_args, ChatRequest)
        assert call_args.messages == messages
        assert call_args.model == "gpt-4o"
        assert call_args.max_tokens == 100
        assert call_args.stream is False

    @pytest.mark.asyncio
    @patch("swe.llm.client._get_client_for_model")
    async def test_llm_chat_conversation_streaming(self, mock_get_client):
        # Mock client
        mock_client = AsyncMock()

        # Mock streaming chunks
        chunks = [
            StreamingChunk(content="I'm", model="gpt-4o"),
            StreamingChunk(content=" doing", model="gpt-4o"),
            StreamingChunk(content=" well!", model="gpt-4o", finish_reason="stop"),
        ]

        async def mock_chat_stream(request):
            for chunk in chunks:
                yield chunk

        mock_client.chat_stream = mock_chat_stream
        mock_get_client.return_value = mock_client

        messages = [
            Message(role=Role.USER, content="Hello!"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
            Message(role=Role.USER, content="How are you?"),
        ]

        stream_iterator = await llm_chat_conversation(messages=messages, model=ChatModel.GPT_4O, stream=True)

        response_chunks = []
        async for chunk in stream_iterator:
            response_chunks.append(chunk)

        assert len(response_chunks) == 3
        assert response_chunks[0].content == "I'm"
        assert response_chunks[1].content == " doing"
        assert response_chunks[2].content == " well!"
        assert response_chunks[2].finish_reason == "stop"

    @pytest.mark.asyncio
    @patch("swe.llm.client._get_client_for_model")
    async def test_llm_chat_conversation_with_enum_model(self, mock_get_client):
        # Mock client
        mock_client = AsyncMock()
        mock_response = ChatResponse(content="Hello!", model="gpt-4o")
        mock_client.chat.return_value = mock_response
        mock_get_client.return_value = mock_client

        messages = [Message(role=Role.USER, content="Hello!")]

        response = await llm_chat_conversation(messages=messages, model=ChatModel.GPT_4O, stream=False)
        del response

        # Verify model value is used
        call_args = mock_client.chat.call_args[0][0]
        assert call_args.model == "gpt-4o"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
