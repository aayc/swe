"""Tests for LLM types and enums."""

import pytest

from swe.llm.types import (
    APIError,
    AuthenticationError,
    ChatModel,
    ChatRequest,
    ChatResponse,
    ConfigError,
    EmbeddingModel,
    LLMError,
    Message,
    RateLimitError,
    Role,
    StreamingChunk,
    Usage,
)


class TestRole:
    """Test cases for Role enum."""

    def test_role_values(self):
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.SYSTEM.value == "system"


class TestMessage:
    """Test cases for Message dataclass."""

    def test_message_creation(self):
        msg = Message(role=Role.USER, content="Hello, world!")
        assert msg.role == Role.USER
        assert msg.content == "Hello, world!"

    def test_message_equality(self):
        msg1 = Message(role=Role.USER, content="Hello")
        msg2 = Message(role=Role.USER, content="Hello")
        msg3 = Message(role=Role.ASSISTANT, content="Hello")

        assert msg1 == msg2
        assert msg1 != msg3


class TestChatRequest:
    """Test cases for ChatRequest dataclass."""

    def test_chat_request_creation(self):
        messages = [Message(role=Role.USER, content="Hello")]
        request = ChatRequest(
            messages=messages,
            model="gpt-4o",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stream=False,
        )

        assert request.messages == messages
        assert request.model == "gpt-4o"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stream is False

    def test_chat_request_defaults(self):
        messages = [Message(role=Role.USER, content="Hello")]
        request = ChatRequest(messages=messages, model="gpt-4o")

        assert request.messages == messages
        assert request.model == "gpt-4o"
        assert request.max_tokens is None
        assert request.temperature is None
        assert request.top_p is None
        assert request.stream is False


class TestUsage:
    """Test cases for Usage dataclass."""

    def test_usage_creation(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestChatResponse:
    """Test cases for ChatResponse dataclass."""

    def test_chat_response_creation(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = ChatResponse(
            content="Hello there!",
            model="gpt-4o",
            usage=usage,
            finish_reason="stop",
        )

        assert response.content == "Hello there!"
        assert response.model == "gpt-4o"
        assert response.usage == usage
        assert response.finish_reason == "stop"

    def test_chat_response_defaults(self):
        response = ChatResponse(content="Hello", model="gpt-4o")

        assert response.content == "Hello"
        assert response.model == "gpt-4o"
        assert response.usage is None
        assert response.finish_reason is None


class TestStreamingChunk:
    """Test cases for StreamingChunk dataclass."""

    def test_streaming_chunk_creation(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        chunk = StreamingChunk(
            content="Hello",
            model="gpt-4o",
            finish_reason="stop",
            usage=usage,
        )

        assert chunk.content == "Hello"
        assert chunk.model == "gpt-4o"
        assert chunk.finish_reason == "stop"
        assert chunk.usage == usage

    def test_streaming_chunk_defaults(self):
        chunk = StreamingChunk(content="Hello", model="gpt-4o")

        assert chunk.content == "Hello"
        assert chunk.model == "gpt-4o"
        assert chunk.finish_reason is None
        assert chunk.usage is None


class TestExceptions:
    """Test cases for LLM exceptions."""

    def test_llm_error(self):
        error = LLMError("Base error")
        assert str(error) == "Base error"
        assert isinstance(error, Exception)

    def test_rate_limit_error(self):
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, LLMError)

    def test_authentication_error(self):
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, LLMError)

    def test_api_error(self):
        error = APIError("API error")
        assert str(error) == "API error"
        assert isinstance(error, LLMError)

    def test_config_error(self):
        error = ConfigError("Configuration error")
        assert str(error) == "Configuration error"
        assert isinstance(error, LLMError)


class TestChatModel:
    """Test cases for ChatModel enum."""

    def test_openai_models(self):
        assert ChatModel.GPT_4O.value == "gpt-4o"
        assert ChatModel.GPT_4_1.value == "gpt-4.1"

    def test_anthropic_models(self):
        assert ChatModel.CLAUDE_SONNET_4.value == "claude-sonnet-4-20250514"
        assert ChatModel.CLAUDE_OPUS_4.value == "claude-opus-4-20250514"
        assert ChatModel.CLAUDE_HAIKU_4.value == "claude-3-haiku-20240307"

    def test_google_models(self):
        assert ChatModel.GEMINI_2_5_FLASH.value == "gemini-2.5-flash"
        assert ChatModel.GEMINI_2_5_PRO.value == "gemini-2.5-pro"

    def test_deepseek_models(self):
        assert ChatModel.DEEPSEEK_REASONER.value == "deepseek-reasoner"
        assert ChatModel.DEEPSEEK_CHAT.value == "deepseek-chat"

    def test_llama_models(self):
        assert ChatModel.LLAMA_3_2_3B.value == "llama3.2:3b"
        assert ChatModel.LLAMA_3_2_1B.value == "llama3.2:1b"


class TestEmbeddingModel:
    """Test cases for EmbeddingModel enum."""

    def test_openai_embeddings(self):
        assert EmbeddingModel.TEXT_EMBEDDING_3_LARGE.value == "text-embedding-3-large"
        assert EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value == "text-embedding-3-small"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
