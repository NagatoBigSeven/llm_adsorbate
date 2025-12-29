"""
Abstract base class for LLM backends.

This module defines the interface that all LLM backends must implement,
following the same pattern as src/calculators/base.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass
class LLMConfig:
    """
    Configuration for LLM initialization.

    This is a unified configuration object that can be passed to any backend.
    Backend-specific options should be placed in `extra_options`.

    Attributes:
        backend: Backend identifier ("google", "openrouter", "ollama", "huggingface")
        api_key: API key for cloud backends (None for local backends)
        model: Model identifier (backend-specific)
        temperature: Sampling temperature (0.0 for deterministic output)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        extra_options: Additional backend-specific options
    """

    backend: str
    api_key: Optional[str] = None
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120
    extra_options: Dict[str, Any] = field(default_factory=dict)


class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    All LLM backends (Google, OpenRouter, Ollama, HuggingFace) must inherit
    from this class and implement the required abstract methods.

    The interface is designed to work with LangChain, so all models returned
    must be LangChain-compatible BaseChatModel instances.
    """

    @abstractmethod
    def get_chat_model(self, config: LLMConfig) -> BaseChatModel:
        """
        Return a LangChain-compatible chat model instance.

        Args:
            config: LLM configuration

        Returns:
            A BaseChatModel that can be used with LangChain

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If model initialization fails
        """
        pass

    @abstractmethod
    def get_default_config(self, api_key: Optional[str] = None) -> LLMConfig:
        """
        Return default configuration for this backend.

        Args:
            api_key: Optional API key (required for cloud backends)

        Returns:
            Default LLMConfig for this backend
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'google', 'ollama')."""
        pass

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Whether this backend requires an API key."""
        pass

    @property
    def is_available(self) -> bool:
        """
        Check if this backend is available (dependencies installed).

        Override this in subclasses to perform actual availability checks.
        """
        return True
