"""Tests for Gemini client."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from swe.llm.gemini_client import GeminiClient
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


class TestGeminiClient:
    """Test cases for GeminiClient."""

    def test_init_valid_api_key(self):
        with patch("google.generativeai.configure") as mock_configure:
            client = GeminiClient(api_key="test-key")
            assert client.api_key == "test-key"
            assert client.base_url == "https://generativelanguage.googleapis.com"
            assert client.timeout == 30.0
            mock_configure.assert_called_once_with(api_key="test-key")

    def test_init_custom_params(self):
        with patch("google.generativeai.configure") as mock_configure:
            client = GeminiClient(
                api_key="test-key",
                base_url="https://custom.gemini.api.com/",
                timeout=60.0,
            )
            assert client.api_key == "test-key"
            assert client.base_url == "https://custom.gemini.api.com"
            assert client.timeout == 60.0
            mock_configure.assert_called_once_with(api_key="test-key")

    def test_init_empty_api_key(self):
        with pytest.raises(ConfigError, match="Google API key is required"):
            GeminiClient(api_key="")

    def test_get_provider_name(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            assert client.get_provider_name() == "gemini"

    def test_convert_messages_with_system(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [
                Message(role=Role.SYSTEM, content="You are a helpful assistant."),
                Message(role=Role.USER, content="Hello!"),
                Message(role=Role.ASSISTANT, content="Hi there!"),
            ]

            system_instruction, gemini_contents = client._convert_messages(messages)

            assert system_instruction == "You are a helpful assistant."
            expected_contents = [
                {"role": "user", "parts": [{"text": "Hello!"}]},
                {"role": "model", "parts": [{"text": "Hi there!"}]},
            ]
            assert gemini_contents == expected_contents

    def test_convert_messages_no_system(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [
                Message(role=Role.USER, content="Hello!"),
                Message(role=Role.ASSISTANT, content="Hi there!"),
            ]

            system_instruction, gemini_contents = client._convert_messages(messages)

            assert system_instruction == ""
            expected_contents = [
                {"role": "user", "parts": [{"text": "Hello!"}]},
                {"role": "model", "parts": [{"text": "Hi there!"}]},
            ]
            assert gemini_contents == expected_contents

    def test_handle_gemini_error_authentication(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")

            # Mock authentication error
            mock_error = Exception("Invalid API key")

            with pytest.raises(AuthenticationError, match="Invalid API key"):
                client._handle_gemini_error(mock_error)

    def test_handle_gemini_error_rate_limit(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")

            # Mock rate limit error
            mock_error = Exception("Rate limit exceeded")

            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                client._handle_gemini_error(mock_error)

    def test_handle_gemini_error_api_error(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")

            # Mock API error
            mock_error = Exception("Invalid request")

            with pytest.raises(APIError, match="Gemini API error"):
                client._handle_gemini_error(mock_error)

    def test_handle_gemini_error_generic(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")

            # Mock generic error
            mock_error = Exception("Some other error")

            with pytest.raises(APIError, match="Gemini client error"):
                client._handle_gemini_error(mock_error)

    @pytest.mark.asyncio
    async def test_chat_success(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [Message(role=Role.USER, content="Hello!")]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro")

            # Mock the response parts
            mock_part = Mock()
            mock_part.text = "Hello there!"

            # Mock the candidate
            mock_candidate = Mock()
            mock_candidate.content = Mock()
            mock_candidate.content.parts = [mock_part]
            mock_candidate.finish_reason = Mock()
            mock_candidate.finish_reason.name = "STOP"

            # Mock the usage metadata
            mock_usage = Mock()
            mock_usage.prompt_token_count = 10
            mock_usage.candidates_token_count = 15
            mock_usage.total_token_count = 25

            # Mock the response
            mock_response = Mock()
            mock_response.candidates = [mock_candidate]
            mock_response.usage_metadata = mock_usage

            # Mock the model and its generate_content_async method
            mock_model = AsyncMock()
            mock_model.generate_content_async.return_value = mock_response

            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                response = await client.chat(request)

                assert isinstance(response, ChatResponse)
                assert response.content == "Hello there!"
                assert response.model == "gemini-1.5-pro"
                assert response.finish_reason == "stop"
                assert response.usage.prompt_tokens == 10
                assert response.usage.completion_tokens == 15
                assert response.usage.total_tokens == 25

    @pytest.mark.asyncio
    async def test_chat_multiple_parts(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [Message(role=Role.USER, content="Hello!")]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro")

            # Mock multiple parts
            mock_part1 = Mock()
            mock_part1.text = "Hello "
            mock_part2 = Mock()
            mock_part2.text = "there!"

            # Mock the candidate
            mock_candidate = Mock()
            mock_candidate.content = Mock()
            mock_candidate.content.parts = [mock_part1, mock_part2]
            mock_candidate.finish_reason = Mock()
            mock_candidate.finish_reason.name = "STOP"

            # Mock the response
            mock_response = Mock()
            mock_response.candidates = [mock_candidate]
            mock_response.usage_metadata = None

            # Mock the model
            mock_model = AsyncMock()
            mock_model.generate_content_async.return_value = mock_response

            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                response = await client.chat(request)

                assert response.content == "Hello there!"

    @pytest.mark.asyncio
    async def test_chat_no_candidates(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [Message(role=Role.USER, content="Hello!")]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro")

            # Mock response with no candidates
            mock_response = Mock()
            mock_response.candidates = []
            mock_response.usage_metadata = None

            # Mock the model
            mock_model = AsyncMock()
            mock_model.generate_content_async.return_value = mock_response

            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                response = await client.chat(request)

                assert response.content == ""

    @pytest.mark.asyncio
    async def test_chat_no_usage(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [Message(role=Role.USER, content="Hello!")]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro")

            # Mock the response parts
            mock_part = Mock()
            mock_part.text = "Hello there!"

            # Mock the candidate
            mock_candidate = Mock()
            mock_candidate.content = Mock()
            mock_candidate.content.parts = [mock_part]
            mock_candidate.finish_reason = Mock()
            mock_candidate.finish_reason.name = "STOP"

            # Mock the response with no usage metadata
            mock_response = Mock()
            mock_response.candidates = [mock_candidate]
            mock_response.usage_metadata = None

            # Mock the model
            mock_model = AsyncMock()
            mock_model.generate_content_async.return_value = mock_response

            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                response = await client.chat(request)

                assert response.usage is None

    @pytest.mark.asyncio
    async def test_chat_stream_request_error(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [Message(role=Role.USER, content="Hello!")]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro", stream=True)

            with pytest.raises(ValueError, match="Use chat_stream\\(\\) for streaming requests"):
                await client.chat(request)

    @pytest.mark.asyncio
    async def test_chat_stream_success(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [Message(role=Role.USER, content="Hello!")]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro")

            # Mock streaming chunks
            mock_part1 = Mock()
            mock_part1.text = "Hello"

            mock_candidate1 = Mock()
            mock_candidate1.content = Mock()
            mock_candidate1.content.parts = [mock_part1]
            mock_candidate1.finish_reason = None

            mock_chunk1 = Mock()
            mock_chunk1.candidates = [mock_candidate1]
            mock_chunk1.usage_metadata = None

            mock_part2 = Mock()
            mock_part2.text = " there!"

            mock_candidate2 = Mock()
            mock_candidate2.content = Mock()
            mock_candidate2.content.parts = [mock_part2]
            mock_candidate2.finish_reason = None

            mock_chunk2 = Mock()
            mock_chunk2.candidates = [mock_candidate2]
            mock_chunk2.usage_metadata = None

            # Final chunk with finish reason and usage
            mock_part3 = Mock()
            mock_part3.text = ""

            mock_candidate3 = Mock()
            mock_candidate3.content = Mock()
            mock_candidate3.content.parts = [mock_part3]
            mock_candidate3.finish_reason = Mock()
            mock_candidate3.finish_reason.name = "STOP"

            mock_usage = Mock()
            mock_usage.prompt_token_count = 10
            mock_usage.candidates_token_count = 15
            mock_usage.total_token_count = 25

            mock_chunk3 = Mock()
            mock_chunk3.candidates = [mock_candidate3]
            mock_chunk3.usage_metadata = mock_usage

            async def mock_stream():
                yield mock_chunk1
                yield mock_chunk2
                yield mock_chunk3

            # Mock the model
            mock_model = AsyncMock()
            mock_model.generate_content_async.return_value = mock_stream()

            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
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
    async def test_chat_stream_api_error(self):
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [Message(role=Role.USER, content="Hello!")]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro")

            # Mock the model to raise an exception
            mock_model = AsyncMock()
            mock_model.generate_content_async.side_effect = Exception("Test error")

            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                with pytest.raises(APIError, match="Gemini client error"):
                    async for _ in client.chat_stream(request):
                        pass

    @pytest.mark.asyncio
    async def test_chat_empty_messages_filtered(self):
        """Test that empty messages are filtered out during conversion."""
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hi!"),
                Message(role=Role.USER, content="   "),  # Empty message
                Message(role=Role.USER, content="How are you?"),
            ]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro")

            # Mock response
            mock_part = Mock()
            mock_part.text = "I'm doing well!"

            mock_candidate = Mock()
            mock_candidate.content = Mock()
            mock_candidate.content.parts = [mock_part]
            mock_candidate.finish_reason = Mock()
            mock_candidate.finish_reason.name = "STOP"

            mock_response = Mock()
            mock_response.candidates = [mock_candidate]
            mock_response.usage_metadata = None

            mock_model = AsyncMock()
            mock_model.generate_content_async.return_value = mock_response

            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                response = await client.chat(request)

                # Verify the response
                assert response.content == "I'm doing well!"

                # Verify that empty messages were filtered out
                call_args = mock_model.generate_content_async.call_args
                passed_contents = call_args[0][0]

                # Should have 3 messages (Hello, Hi!, How are you?) but not the empty one
                assert len(passed_contents) == 3
                assert passed_contents[0]["parts"][0]["text"] == "Hello"
                assert passed_contents[1]["parts"][0]["text"] == "Hi!"
                assert passed_contents[2]["parts"][0]["text"] == "How are you?"

    @pytest.mark.asyncio
    async def test_chat_all_empty_user_messages(self):
        """Test that having all empty user messages raises an error."""
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [
                Message(role=Role.SYSTEM, content="You are helpful"),
                Message(role=Role.USER, content="   "),  # Empty
                Message(role=Role.ASSISTANT, content="Hi!"),
                Message(role=Role.USER, content=""),  # Empty
            ]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro")

            with pytest.raises(ValueError, match="No valid user message found"):
                await client.chat(request)

    @pytest.mark.asyncio
    async def test_chat_multi_turn_conversation(self):
        """Test that multi-turn conversations preserve full context."""
        with patch("google.generativeai.configure"):
            client = GeminiClient(api_key="test-key")
            messages = [
                Message(role=Role.SYSTEM, content="You are a math tutor"),
                Message(role=Role.USER, content="What is 2+2?"),
                Message(role=Role.ASSISTANT, content="2+2 = 4"),
                Message(role=Role.USER, content="What about 3+3?"),
            ]
            request = ChatRequest(messages=messages, model="gemini-1.5-pro")

            # Mock response
            mock_part = Mock()
            mock_part.text = "3+3 = 6"

            mock_candidate = Mock()
            mock_candidate.content = Mock()
            mock_candidate.content.parts = [mock_part]
            mock_candidate.finish_reason = Mock()
            mock_candidate.finish_reason.name = "STOP"

            mock_response = Mock()
            mock_response.candidates = [mock_candidate]
            mock_response.usage_metadata = None

            mock_model = AsyncMock()
            mock_model.generate_content_async.return_value = mock_response

            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                response = await client.chat(request)

                assert response.content == "3+3 = 6"

                # Verify that the full conversation context was passed
                call_args = mock_model.generate_content_async.call_args
                passed_contents = call_args[0][0]

                # Should have the full conversation (3 non-system messages)
                assert len(passed_contents) == 3
                assert passed_contents[0]["role"] == "user"
                assert passed_contents[0]["parts"][0]["text"] == "What is 2+2?"
                assert passed_contents[1]["role"] == "model"
                assert passed_contents[1]["parts"][0]["text"] == "2+2 = 4"
                assert passed_contents[2]["role"] == "user"
                assert passed_contents[2]["parts"][0]["text"] == "What about 3+3?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
