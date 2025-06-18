import os
from collections.abc import AsyncIterator

from .azure_client import AzureOpenAIClient
from .claude_client import ClaudeClient
from .gemini_client import GeminiClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .types import (
    ChatModel,
    ChatRequest,
    ChatResponse,
    ConfigError,
    LLMProvider,
    Message,
    Role,
    StreamingChunk,
)


def _get_client_for_model(model: ChatModel) -> LLMProvider:
    """Get the appropriate client for the given model."""
    model_value = model.value

    # Azure OpenAI models (check first for OpenAI-compatible models)
    if model_value in ["gpt-4o", "gpt-4.1"] and os.getenv("AZURE_OPENAI_ENDPOINT"):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise ConfigError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables are required for Azure OpenAI")
        return AzureOpenAIClient(api_key=api_key, endpoint=endpoint)

    # OpenAI models
    elif model_value in ["gpt-4o", "gpt-4.1"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError("OPENAI_API_KEY environment variable is required for OpenAI models")
        return OpenAIClient(api_key=api_key)

    # Anthropic models
    elif model_value.startswith("claude-") or model_value.startswith("claude-sonnet-") or model_value.startswith("claude-opus-"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError("ANTHROPIC_API_KEY environment variable is required for Claude models")
        return ClaudeClient(api_key=api_key)

    # Google models
    elif model_value.startswith("gemini-"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigError("GEMINI_API_KEY environment variable is required for Gemini models")
        return GeminiClient(api_key=api_key)

    # DeepSeek models (using OpenAI-compatible API)
    elif model_value in ["deepseek-chat", "deepseek-reasoner"]:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        if not api_key:
            raise ConfigError("DEEPSEEK_API_KEY environment variable is required for DeepSeek models")
        return OpenAIClient(api_key=api_key, base_url=base_url)

    # Llama models (Ollama)
    elif model_value.startswith("llama"):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaClient(base_url=base_url)

    else:
        raise ConfigError(f"No client configured for model: {model_value}")


async def llm_chat(
    prompt: str,
    model: ChatModel,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> ChatResponse:
    """
    Send a chat request to the appropriate LLM provider.

    Args:
        prompt: The user prompt/message
        model: The model to use (ChatModel enum)
        system_prompt: Optional system prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Top-p sampling parameter

    Returns:
        ChatResponse with the model's response

    Raises:
        ConfigError: If required API keys or configuration is missing
        APIError: If the API request fails
    """
    messages = []

    if system_prompt:
        messages.append(Message(role=Role.SYSTEM, content=system_prompt))

    messages.append(Message(role=Role.USER, content=prompt))

    request = ChatRequest(
        messages=messages,
        model=model.value,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )

    client = _get_client_for_model(model)
    return await client.chat(request)


async def llm_chat_stream(
    prompt: str,
    model: ChatModel,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> AsyncIterator[StreamingChunk]:
    """
    Send a streaming chat request to the appropriate LLM provider.

    Args:
        prompt: The user prompt/message
        model: The model to use (ChatModel enum)
        system_prompt: Optional system prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Top-p sampling parameter

    Yields:
        StreamingChunk objects with incremental response content

    Raises:
        ConfigError: If required API keys or configuration is missing
        APIError: If the API request fails
    """
    messages = []

    if system_prompt:
        messages.append(Message(role=Role.SYSTEM, content=system_prompt))

    messages.append(Message(role=Role.USER, content=prompt))

    request = ChatRequest(
        messages=messages,
        model=model.value,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    client = _get_client_for_model(model)
    async for chunk in client.chat_stream(request):
        yield chunk


async def llm_chat_conversation(
    messages: list[Message],
    model: ChatModel,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool = False,
) -> ChatResponse | AsyncIterator[StreamingChunk]:
    """
    Send a conversation with multiple messages to the appropriate LLM provider.

    Args:
        messages: List of Message objects representing the conversation
        model: The model to use (ChatModel enum)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Top-p sampling parameter
        stream: Whether to stream the response

    Returns:
        ChatResponse if stream=False, AsyncIterator[StreamingChunk] if stream=True

    Raises:
        ConfigError: If required API keys or configuration is missing
        APIError: If the API request fails
    """
    request = ChatRequest(
        messages=messages,
        model=model.value,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
    )

    client = _get_client_for_model(model)

    if stream:
        return client.chat_stream(request)
    else:
        return await client.chat(request)
