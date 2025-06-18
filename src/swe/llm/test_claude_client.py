"""Tests for Claude client."""

from unittest.mock import AsyncMock, Mock, patch

import anthropic
import pytest

from swe.llm.claude_client import ClaudeClient
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


class TestClaudeClient:
    """Test cases for ClaudeClient."""

    def test_init_valid_api_key(self):
        client = ClaudeClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.anthropic.com"
        assert client.timeout == 30.0
        assert client.anthropic_version == "2023-06-01"
        assert client.client.api_key == "test-key"

    def test_init_custom_params(self):
        client = ClaudeClient(
            api_key="test-key",
            base_url="https://custom.api.com/",
            timeout=60.0,
            anthropic_version="2024-01-01",
        )
        assert client.api_key == "test-key"
        assert client.base_url == "https://custom.api.com"
        assert client.timeout == 60.0
        assert client.anthropic_version == "2024-01-01"

    def test_init_empty_api_key(self):
        with pytest.raises(ConfigError, match="Anthropic API key is required"):
            ClaudeClient(api_key="")

    def test_get_provider_name(self):
        client = ClaudeClient(api_key="test-key")
        assert client.get_provider_name() == "anthropic"

    def test_convert_messages_with_system(self):
        client = ClaudeClient(api_key="test-key")
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello!"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]

        system_message, claude_messages = client._convert_messages(messages)

        assert system_message == "You are a helpful assistant."
        expected_messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert claude_messages == expected_messages

    def test_convert_messages_no_system(self):
        client = ClaudeClient(api_key="test-key")
        messages = [
            Message(role=Role.USER, content="Hello!"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]

        system_message, claude_messages = client._convert_messages(messages)

        assert system_message == ""
        expected_messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert claude_messages == expected_messages

    def test_handle_anthropic_error_authentication(self):
        client = ClaudeClient(api_key="test-key")

        # Create a mock exception that's an instance of the expected type
        mock_auth_error = Mock(spec=anthropic.AuthenticationError)

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            client._handle_anthropic_error(mock_auth_error)

    def test_handle_anthropic_error_rate_limit(self):
        client = ClaudeClient(api_key="test-key")

        # Create a mock exception that's an instance of the expected type
        mock_rate_error = Mock(spec=anthropic.RateLimitError)

        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            client._handle_anthropic_error(mock_rate_error)

    def test_handle_anthropic_error_api_error(self):
        client = ClaudeClient(api_key="test-key")

        # Create a mock exception that's an instance of the expected type
        mock_api_error = Mock(spec=anthropic.APIError)

        with pytest.raises(APIError, match="Claude API error"):
            client._handle_anthropic_error(mock_api_error)

    def test_handle_anthropic_error_generic(self):
        client = ClaudeClient(api_key="test-key")

        with pytest.raises(APIError, match="Claude client error"):
            client._handle_anthropic_error(Exception("Generic error"))

    @pytest.mark.asyncio
    async def test_chat_success(self):
        client = ClaudeClient(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="claude-sonnet-4-20250514")

        # Mock the Anthropic response
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello there!"

        mock_usage = Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 15

        mock_response = Mock()
        mock_response.content = [mock_text_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = mock_usage

        with patch.object(
            client.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await client.chat(request)

            assert isinstance(response, ChatResponse)
            assert response.content == "Hello there!"
            assert response.model == "claude-sonnet-4-20250514"
            assert response.finish_reason == "end_turn"
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 15
            assert response.usage.total_tokens == 25

    @pytest.mark.asyncio
    async def test_chat_multiple_text_blocks(self):
        client = ClaudeClient(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="claude-sonnet-4-20250514")

        # Mock multiple text blocks
        mock_text_block1 = Mock()
        mock_text_block1.type = "text"
        mock_text_block1.text = "Hello "

        mock_text_block2 = Mock()
        mock_text_block2.type = "text"
        mock_text_block2.text = "there!"

        mock_response = Mock()
        mock_response.content = [mock_text_block1, mock_text_block2]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = None

        with patch.object(
            client.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await client.chat(request)

            assert response.content == "Hello there!"

    @pytest.mark.asyncio
    async def test_chat_no_usage(self):
        client = ClaudeClient(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="claude-sonnet-4-20250514")

        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello there!"

        mock_response = Mock()
        mock_response.content = [mock_text_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = None

        with patch.object(
            client.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await client.chat(request)

            assert response.usage is None

    @pytest.mark.asyncio
    async def test_chat_stream_request_error(self):
        client = ClaudeClient(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="claude-sonnet-4-20250514", stream=True)

        with pytest.raises(ValueError, match="Use chat_stream\\(\\) for streaming requests"):
            await client.chat(request)

    @pytest.mark.asyncio
    async def test_chat_stream_success(self):
        client = ClaudeClient(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="claude-sonnet-4-20250514")

        # Mock streaming events
        mock_event1 = Mock()
        mock_event1.type = "content_block_delta"
        mock_event1.delta = Mock()
        mock_event1.delta.text = "Hello"

        mock_event2 = Mock()
        mock_event2.type = "content_block_delta"
        mock_event2.delta = Mock()
        mock_event2.delta.text = " there!"

        mock_event3 = Mock()
        mock_event3.type = "message_delta"
        mock_event3.delta = Mock()
        mock_event3.delta.stop_reason = "end_turn"
        mock_event3.usage = Mock()
        mock_event3.usage.input_tokens = 10
        mock_event3.usage.output_tokens = 15

        async def mock_stream():
            yield mock_event1
            yield mock_event2
            yield mock_event3

        with patch.object(client.client.messages, "create", return_value=mock_stream()):
            chunks = []
            async for chunk in client.chat_stream(request):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
            assert chunks[0].finish_reason is None
            assert chunks[1].content == " there!"
            assert chunks[1].finish_reason is None
            assert chunks[2].content == ""
            assert chunks[2].finish_reason == "end_turn"
            assert chunks[2].usage.total_tokens == 25

    @pytest.mark.asyncio
    async def test_chat_stream_unknown_event_type(self):
        client = ClaudeClient(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="claude-sonnet-4-20250514")

        # Mock streaming events with unknown type
        mock_event1 = Mock()
        mock_event1.type = "unknown_event"

        mock_event2 = Mock()
        mock_event2.type = "content_block_delta"
        mock_event2.delta = Mock()
        mock_event2.delta.text = "Hello"

        async def mock_stream():
            yield mock_event1
            yield mock_event2

        with patch.object(client.client.messages, "create", return_value=mock_stream()):
            chunks = []
            async for chunk in client.chat_stream(request):
                chunks.append(chunk)

            # Should only get the content_block_delta chunk
            assert len(chunks) == 1
            assert chunks[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_chat_stream_invalid_json(self):
        client = ClaudeClient(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="claude-sonnet-4-20250514")

        # Test with proper mock setup - delta without text attribute
        mock_event1 = Mock()
        mock_event1.type = "content_block_delta"
        mock_event1.delta = Mock(spec=[])  # Delta without text attribute

        mock_event2 = Mock()
        mock_event2.type = "content_block_delta"
        mock_event2.delta = Mock()
        mock_event2.delta.text = "Hello"

        async def mock_stream():
            yield mock_event1
            yield mock_event2

        with patch.object(client.client.messages, "create", return_value=mock_stream()):
            chunks = []
            async for chunk in client.chat_stream(request):
                chunks.append(chunk)

            # Should get both chunks, first with empty content due to missing text
            assert len(chunks) == 2
            assert chunks[0].content == ""
            assert chunks[1].content == "Hello"

    @pytest.mark.asyncio
    async def test_chat_stream_api_error(self):
        client = ClaudeClient(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello!")]
        request = ChatRequest(messages=messages, model="claude-sonnet-4-20250514")

        # Test with a generic exception (will be handled as APIError)
        test_exception = Exception("Test error")

        with patch.object(client.client.messages, "create", side_effect=test_exception):
            with pytest.raises(APIError, match="Claude client error"):
                async for _ in client.chat_stream(request):
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
