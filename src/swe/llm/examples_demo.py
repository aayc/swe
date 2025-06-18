#!/usr/bin/env python3
"""
Demo examples for the LLM module using mock responses.

This script demonstrates the LLM module functionality without requiring real API keys.
It shows the structure and usage patterns that would work with real API calls.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from swe.llm.client import llm_chat, llm_chat_conversation, llm_chat_stream
from swe.llm.types import ChatModel, Message, Role, Usage


async def demo_simple_chat():
    """Demo: Simple chat request with mocked response."""
    print("🤖 Demo: Simple Chat Request")
    print("-" * 40)

    # Mock the Gemini SDK response
    mock_candidate = Mock()
    mock_candidate.content.parts = [Mock(text="The capital of France is Paris.")]

    mock_usage = Mock()
    mock_usage.prompt_token_count = 8
    mock_usage.candidates_token_count = 7
    mock_usage.total_token_count = 15

    mock_response = Mock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = mock_usage

    # Mock the Gemini model and its methods
    mock_model = AsyncMock()
    mock_model.generate_content_async.return_value = mock_response

    with (
        patch("google.generativeai.GenerativeModel", return_value=mock_model),
        patch("google.generativeai.configure"),
        patch.dict("os.environ", {"GEMINI_API_KEY": "mock-key"}),
    ):
        response = await llm_chat(
            prompt="What is the capital of France?",
            model=ChatModel.GEMINI_2_5_FLASH,
            max_tokens=100,
        )

        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        print()


async def demo_streaming_chat():
    """Demo: Streaming chat response with mocked data."""
    print("🌊 Demo: Streaming Chat")
    print("-" * 40)

    # Mock streaming chunks
    haiku_parts = [
        "AI systems learn",
        " patterns from data,",
        "\nTransforming information\nInto insight and action.",
    ]

    async def mock_stream():
        for text in haiku_parts:
            chunk = Mock()
            chunk.candidates = [Mock()]
            chunk.candidates[0].content.parts = [Mock(text=text)]
            yield chunk

    # Mock the Gemini model
    mock_model = AsyncMock()
    mock_model.generate_content_async.return_value = mock_stream()

    with (
        patch("google.generativeai.GenerativeModel", return_value=mock_model),
        patch("google.generativeai.configure"),
        patch.dict("os.environ", {"GEMINI_API_KEY": "mock-key"}),
    ):
        print("Streaming haiku about AI:")
        print("Response: ", end="", flush=True)

        async for chunk in llm_chat_stream(
            prompt="Write a haiku about artificial intelligence",
            model=ChatModel.GEMINI_2_5_FLASH,
            temperature=0.7,
        ):
            if chunk.content:
                print(chunk.content, end="", flush=True)

            if chunk.finish_reason and chunk.usage:
                print(f"\n\nTokens used: {chunk.usage.total_tokens}")

        print("\n")  # Better formatting


async def demo_conversation():
    """Demo: Multi-turn conversation with mocked response."""
    print("💬 Demo: Multi-turn Conversation")
    print("-" * 40)

    # Mock the Gemini SDK response
    mock_candidate = Mock()
    mock_candidate.content.parts = [
        Mock(
            text=(
                "Certainly! Let me break down 15 × 23 step by step:\n\n1. Break down 23 into 20 + 3\n2. Multiply 15 × 20 = 300\n3. Multiply 15 × 3 = 45\n"
                "4. Add the results: 300 + 45 = 345\n\nThis method uses the distributive property: 15 × (20 + 3) = (15 × 20) + (15 × 3)"
            )
        )
    ]

    mock_usage = Mock()
    mock_usage.prompt_token_count = 45
    mock_usage.candidates_token_count = 62
    mock_usage.total_token_count = 107

    mock_response = Mock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = mock_usage

    # Mock the Gemini model
    mock_model = AsyncMock()
    mock_model.generate_content_async.return_value = mock_response

    with (
        patch("google.generativeai.GenerativeModel", return_value=mock_model),
        patch("google.generativeai.configure"),
        patch.dict("os.environ", {"GEMINI_API_KEY": "mock-key"}),
    ):
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful math tutor."),
            Message(role=Role.USER, content="What is 15 × 23?"),
            Message(role=Role.ASSISTANT, content="15 × 23 = 345"),
            Message(
                role=Role.USER,
                content="Can you show me how to solve this step by step?",
            ),
        ]

        response = await llm_chat_conversation(
            messages=messages,
            model=ChatModel.GEMINI_2_5_FLASH,
            max_tokens=200,
            temperature=0.3,
        )

        print("Conversation context:")
        for msg in messages:
            role_emoji = {"system": "⚙️", "user": "👤", "assistant": "🤖"}
            emoji = role_emoji.get(msg.role.value, "❓")
            print(f"{emoji} {msg.role.value}: {msg.content}")

        print(f"\n🤖 Assistant: {response.content}")
        print()


async def demo_error_handling():
    """Demo: Error handling scenarios."""
    print("⚠️  Demo: Error Handling")
    print("-" * 40)

    from llm.types import ConfigError

    # Demo 1: Missing API key
    try:
        response = await llm_chat(
            prompt="This will fail",
            model=ChatModel.GPT_4O,  # No OPENAI_API_KEY in environment
        )
        print(f"Unexpected success: {response.content}")

    except ConfigError as e:
        print(f"✅ Caught expected error: {e}")

    # Demo 2: Using factory with ChatModel enum
    with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://localhost:11434"}):
        try:
            # This would work if Ollama was running with the model
            response = await llm_chat(
                prompt="Hello",
                model=ChatModel.LLAMA_3_2_3B,  # ChatModel enum
                max_tokens=50,
            )
        except Exception as e:
            print(f"✅ Ollama not available: {type(e).__name__}")

    print()


async def demo_model_factory():
    """Demo: How the model factory selects different clients."""
    print("🏭 Demo: Model Factory Selection")
    print("-" * 40)

    from llm.client import _get_client_for_model

    # Show which client gets selected for different models
    model_examples = [
        (ChatModel.GPT_4O, "OpenAI"),
        (ChatModel.CLAUDE_SONNET_4, "Anthropic"),
        (ChatModel.GEMINI_2_5_FLASH, "Google"),
        (ChatModel.LLAMA_3_2_3B, "Ollama"),
        (ChatModel.DEEPSEEK_REASONER, "DeepSeek (OpenAI-compatible)"),
    ]

    for model, expected_provider in model_examples:
        try:
            with patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "mock-openai",
                    "ANTHROPIC_API_KEY": "mock-anthropic",
                    "GOOGLE_API_KEY": "mock-google",
                    "DEEPSEEK_API_KEY": "mock-deepseek",
                    "OLLAMA_BASE_URL": "http://localhost:11434",
                },
            ):
                client = _get_client_for_model(model)
                provider_name = client.get_provider_name()

                model_str = model.value if hasattr(model, "value") else model
                print(f"✅ {model_str} → {provider_name} client ({expected_provider})")

        except Exception as e:
            print(f"❌ {model}: {e}")

    print()


def demo_types_and_enums():
    """Demo: Working with types and enums."""
    print("📝 Demo: Types and Enums")
    print("-" * 40)

    # Create messages using the type system
    messages = [
        Message(role=Role.SYSTEM, content="You are helpful."),
        Message(role=Role.USER, content="Hello!"),
    ]

    print("Message types:")
    for msg in messages:
        print(f"  Role: {msg.role} (value: '{msg.role.value}')")
        print(f"  Content: {msg.content}")

    print("\nAvailable ChatModels:")
    openai_models = [ChatModel.GPT_4O]
    anthropic_models = [ChatModel.CLAUDE_SONNET_4, ChatModel.CLAUDE_OPUS_4]
    google_models = [ChatModel.GEMINI_2_5_FLASH, ChatModel.GEMINI_2_5_PRO]

    print(f"  OpenAI: {[m.value for m in openai_models]}")
    print(f"  Anthropic: {[m.value for m in anthropic_models]}")
    print(f"  Google: {[m.value for m in google_models]}")

    # Show usage object
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    print(f"\nUsage tracking: {usage}")

    print()


async def main():
    """Run all demos."""
    print("🎭 LLM Module Demo Examples")
    print("=" * 50)
    print("These examples use mocked responses to demonstrate functionality")
    print("without requiring real API keys.\n")

    # Run demos
    await demo_simple_chat()
    await demo_streaming_chat()
    await demo_conversation()
    await demo_error_handling()
    await demo_model_factory()
    demo_types_and_enums()

    print("🎉 Demos completed!")
    print()
    print("💡 Next steps:")
    print("- Set real API keys to try with actual LLM services")
    print("- Use 'python -m llm.examples' for real API examples")
    print("- Check out the test files for more usage patterns")


if __name__ == "__main__":
    asyncio.run(main())
