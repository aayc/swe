from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class ChatRequest:
    messages: list[Message]
    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    content: str
    model: str
    usage: Usage | None = None
    finish_reason: str | None = None


@dataclass
class StreamingChunk:
    content: str
    model: str
    finish_reason: str | None = None
    usage: Usage | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request and return the response."""
        pass

    @abstractmethod
    def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamingChunk]:
        """Send a chat request and return streaming response."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        pass


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class RateLimitError(LLMError):
    """Raised when API rate limit is exceeded."""

    pass


class AuthenticationError(LLMError):
    """Raised when API authentication fails."""

    pass


class APIError(LLMError):
    """Raised when API returns an error."""

    pass


class ConfigError(LLMError):
    """Raised when configuration is invalid."""

    pass


class ChatModel(Enum):
    """Enumeration of available chat models."""

    # OpenAI Models
    GPT_4O = "gpt-4o"
    GPT_4_1 = "gpt-4.1"

    # Anthropic Models
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_HAIKU_4 = "claude-3-haiku-20240307"  # Claude 4 Haiku not yet released

    # Google Models
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"

    # DeepSeek Models
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"

    # Meta Models
    LLAMA_3_2_3B = "llama3.2:3b"
    LLAMA_3_2_1B = "llama3.2:1b"


class EmbeddingModel(Enum):
    """Enumeration of available embedding models."""

    # OpenAI Embeddings
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
