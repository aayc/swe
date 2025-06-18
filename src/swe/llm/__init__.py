"""
LLM Remote - A unified interface for multiple LLM providers.

This module provides a consistent interface for interacting with various LLM providers
including OpenAI, Anthropic Claude, Google Gemini, DeepSeek, and Ollama.
"""

# Individual clients for advanced usage
from .azure_client import AzureOpenAIClient
from .claude_client import ClaudeClient
from .client import llm_chat, llm_chat_conversation, llm_chat_stream
from .gemini_client import GeminiClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .types import (
    APIError,
    AuthenticationError,
    ChatModel,
    ChatRequest,
    ChatResponse,
    ConfigError,
    EmbeddingModel,
    LLMError,
    LLMProvider,
    Message,
    RateLimitError,
    Role,
    StreamingChunk,
    Usage,
)

__all__ = [
    # Main interface functions
    "llm_chat",
    "llm_chat_stream",
    "llm_chat_conversation",
    # Types and enums
    "ChatModel",
    "EmbeddingModel",
    "Message",
    "Role",
    "ChatRequest",
    "ChatResponse",
    "StreamingChunk",
    "Usage",
    "LLMProvider",
    # Exceptions
    "LLMError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "ConfigError",
    # Individual clients
    "OpenAIClient",
    "ClaudeClient",
    "GeminiClient",
    "OllamaClient",
    "AzureOpenAIClient",
]
