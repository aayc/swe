# LLM Module

A unified interface for multiple Large Language Model providers including OpenAI, Anthropic Claude, Google Gemini, Ollama, and Azure OpenAI.

## Features

- **Unified API** - Single interface for all providers
- **Factory Pattern** - Automatic client selection based on model
- **Streaming Support** - Real-time response streaming
- **Type Safety** - Full TypeScript-style type hints
- **Comprehensive Testing** - 160+ tests with zero API costs
- **Error Handling** - Proper error types and handling

## Quick Start

```python
import asyncio
from llm.client import llm_chat
from llm.types import ChatModel

async def main():
    # Simple chat - factory automatically selects the right client
    response = await llm_chat(
        prompt="What is the capital of France?",
        model=ChatModel.GEMINI_2_5_FLASH,
        max_tokens=100
    )

    print(f"Response: {response.content}")
    print(f"Tokens used: {response.usage.total_tokens}")

asyncio.run(main())
```

## Environment Variables

Set the appropriate API keys for the services you want to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-google-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export OLLAMA_BASE_URL="http://localhost:11434"  # Optional, defaults to localhost

# For Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

## Examples

### 1. Run Interactive Demo (No API Keys Required)
```bash
python -m llm.examples_demo
```

### 2. Run Real Examples (Requires API Keys)
```bash
python -m llm.examples
```

### 3. Streaming Chat
```python
async for chunk in llm_chat_stream(
    prompt="Write a haiku about AI",
    model=ChatModel.GEMINI_2_5_FLASH,
    temperature=0.7
):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 4. Multi-turn Conversation
```python
from llm.types import Message, Role

messages = [
    Message(role=Role.SYSTEM, content="You are a helpful assistant."),
    Message(role=Role.USER, content="Hello!"),
    Message(role=Role.ASSISTANT, content="Hi there!"),
    Message(role=Role.USER, content="How are you?"),
]

response = await llm_chat_conversation(
    messages=messages,
    model=ChatModel.GPT_4O,
    max_tokens=100
)
```

## Supported Models

### OpenAI
- `gpt-4o`, `gpt-4.1`

### Anthropic Claude
- `claude-sonnet-4-20250514`, `claude-opus-4-20250514`, `claude-3-haiku-20240307`

### Google Gemini
- `gemini-2.5-flash`, `gemini-2.5-pro`

### Ollama (Local)
- `llama3.2:3b`, `llama3.2:1b`, etc.

### DeepSeek
- `deepseek-reasoner`, `deepseek-chat`

## Error Handling

```python
from llm.types import ConfigError, APIError, AuthenticationError, RateLimitError

try:
    response = await llm_chat(prompt="Hello", model=ChatModel.GPT_4O)
except ConfigError as e:
    print(f"Configuration issue: {e}")
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
except RateLimitError as e:
    print(f"Rate limited: {e}")
except APIError as e:
    print(f"API error: {e}")
```

## Testing

Run the comprehensive test suite:

```bash
# All tests (160 tests, ~0.2s)
python -m pytest llm/ -v

# Specific test files
python -m pytest llm/test_types.py -v
python -m pytest llm/test_openai_client.py -v
python -m pytest llm/test_client.py -v
```

All tests use mocked responses, so no API keys are required and no costs are incurred.

## Architecture

- `client.py` - Main factory functions and high-level API
- `types.py` - Data types, enums, and exceptions
- `*_client.py` - Individual provider implementations
- `test_*.py` - Comprehensive test suite with mocking
- `examples.py` - Real examples (requires API keys)
- `examples_demo.py` - Demo with mocked responses

The module uses a factory pattern where `llm_chat()` automatically selects the appropriate client based on the model name, making it easy to switch between providers without changing your code.
