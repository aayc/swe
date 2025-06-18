#!/usr/bin/env python3
"""
Examples demonstrating how to use the LLM module.

This script shows how to:
1. Use the factory function to get clients for different models
2. Make simple chat requests
3. Use streaming responses
4. Handle different message formats and parameters

To run these examples, you'll need to set the appropriate API keys:
- GEMINI_API_KEY for Gemini models
- OPENAI_API_KEY for OpenAI models
- ANTHROPIC_API_KEY for Claude models
- OLLAMA_BASE_URL for Ollama (defaults to http://localhost:11434)
"""

import asyncio
import os

from swe.llm.client import llm_chat, llm_chat_conversation, llm_chat_stream
from swe.llm.types import ChatModel, Message, Role


async def example_simple_chat():
    """Example: Simple chat request using the factory function."""
    print("🤖 Example: Simple Chat Request")
    print("-" * 40)

    try:
        # Use the factory function - it automatically selects the right client
        response = await llm_chat(
            prompt="What is the capital of France?",
            model=ChatModel.GEMINI_2_5_FLASH,  # Using Gemini model
            max_tokens=100,
        )

        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        if response.usage:
            print(f"Tokens used: {response.usage.total_tokens}")
        print()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have GEMINI_API_KEY set in your environment")
        print()


async def example_streaming_chat():
    """Example: Streaming chat response."""
    print("🌊 Example: Streaming Chat")
    print("-" * 40)

    try:
        print("Streaming response from Gemini:")
        print("Response: ", end="", flush=True)

        # Stream the response token by token
        async for chunk in llm_chat_stream(
            prompt="Write a haiku about artificial intelligence",
            model=ChatModel.GEMINI_2_5_FLASH,
            temperature=0.7,
        ):
            if chunk.content:
                print(chunk.content, end="", flush=True)

            # Print usage info when done
            if chunk.finish_reason and chunk.usage:
                print(f"\n\nTokens used: {chunk.usage.total_tokens}")

        print("\n")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have GEMINI_API_KEY set in your environment")
        print()


async def example_conversation():
    """Example: Multi-turn conversation."""
    print("💬 Example: Multi-turn Conversation")
    print("-" * 40)

    try:
        # Build a conversation with multiple messages
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
            print(f"{msg.role.value}: {msg.content}")

        print(f"\nAssistant: {response.content}")
        print()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have GEMINI_API_KEY set in your environment")
        print()


async def example_different_models():
    """Example: Using different LLM providers."""
    print("🔄 Example: Different LLM Providers")
    print("-" * 40)

    # List of models to try (with their required environment variables)
    models_to_try = [
        (ChatModel.GEMINI_2_5_FLASH, "GEMINI_API_KEY"),
        (ChatModel.GPT_4O, "OPENAI_API_KEY"),
        (ChatModel.CLAUDE_HAIKU_4, "ANTHROPIC_API_KEY"),
        (ChatModel.LLAMA_3_2_3B, "OLLAMA_BASE_URL"),  # Ollama model
    ]

    prompt = "What is machine learning in one sentence?"

    for model, env_var in models_to_try:
        if env_var == "OLLAMA_BASE_URL" or os.getenv(env_var):
            try:
                print(f"Trying {model}...")
                response = await llm_chat(prompt=prompt, model=model, max_tokens=50, temperature=0.5)
                print(f"✅ {response.model}: {response.content[:100]}...")

            except Exception as e:
                print(f"❌ {model}: {e}")
        else:
            print(f"⏭️  Skipping {model} (no {env_var} found)")

    print()


async def example_streaming_conversation():
    """Example: Streaming a multi-turn conversation."""
    print("🌊💬 Example: Streaming Conversation")
    print("-" * 40)

    try:
        # Create a conversation
        messages = [
            Message(role=Role.SYSTEM, content="You are a creative storyteller."),
            Message(
                role=Role.USER,
                content="Tell me a short story about a robot learning to paint.",
            ),
        ]

        print("Streaming story from Gemini:")
        print("Story: ", end="", flush=True)

        # Stream the conversation
        stream_generator = await llm_chat_conversation(
            messages=messages,
            model=ChatModel.GEMINI_2_5_FLASH,
            max_tokens=300,
            temperature=0.8,
            stream=True,
        )

        async for chunk in stream_generator:
            if chunk.content:
                print(chunk.content, end="", flush=True)

            if chunk.finish_reason:
                print(f"\n\nStory completed! Reason: {chunk.finish_reason}")
                if chunk.usage:
                    print(f"Tokens used: {chunk.usage.total_tokens}")

        print()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have GEMINI_API_KEY set in your environment")
        print()


async def example_error_handling():
    """Example: Proper error handling."""
    print("⚠️  Example: Error Handling")
    print("-" * 40)

    from swe.llm.types import APIError, AuthenticationError, ConfigError, RateLimitError

    try:
        # This will fail if no API key is set
        response = await llm_chat(
            prompt="This will fail",
            model=ChatModel.GPT_4O,  # Requires OPENAI_API_KEY
        )
        print(f"Unexpected success: {response.content}")

    except ConfigError as e:
        print(f"Configuration error: {e}")
        print("This is expected if OPENAI_API_KEY is not set")
    except AuthenticationError as e:
        print(f"Authentication error: {e}")
        print("Check your API key")
    except RateLimitError as e:
        print(f"Rate limit error: {e}")
        print("You're making requests too quickly")
    except APIError as e:
        print(f"API error: {e}")
        print("There was an issue with the API request")
    except Exception as e:
        print(f"Unexpected error: {e}")

    print()


async def main():
    """Run all examples."""
    print("🚀 LLM Module Examples")
    print("=" * 50)
    print()

    # Check for available API keys
    available_keys = []
    for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        if os.getenv(key):
            available_keys.append(key)

    if available_keys:
        print(f"✅ Found API keys: {', '.join(available_keys)}")
    else:
        print("⚠️  No API keys found in environment variables")
        print("   Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY to try the examples")

    if os.getenv("OLLAMA_BASE_URL") or os.path.exists("/usr/local/bin/ollama"):
        print("✅ Ollama detected")

    print()

    # Run examples
    await example_simple_chat()
    await example_streaming_chat()
    await example_conversation()
    await example_different_models()
    await example_streaming_conversation()
    await example_error_handling()

    print("🎉 Examples completed!")
    print()
    print("💡 Tips:")
    print("- Set environment variables for the APIs you want to use")
    print("- Use ChatModel enum for type safety")
    print("- Handle errors appropriately in production code")
    print("- Use streaming for better user experience with long responses")


if __name__ == "__main__":
    asyncio.run(main())
