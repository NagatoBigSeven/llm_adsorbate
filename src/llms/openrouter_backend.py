"""
OpenRouter backend implementation.

This backend provides access to multiple LLM providers through OpenRouter's
unified API, including Gemini, GPT-4, Claude, and more.

Benefits:
- Access to many models via single API
- Fallback options if one provider is down
- Usage-based billing across providers
"""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.llms.base import BaseLLMBackend, LLMConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model (Gemini through OpenRouter)
DEFAULT_MODEL = "google/gemini-2.5-pro"


class OpenRouterBackend(BaseLLMBackend):
    """
    OpenRouter LLM backend.

    Provides access to multiple LLM providers through a unified API.
    This was the original default backend, preserved for backward compatibility.
    """

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def requires_api_key(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        """Check if langchain-openai is installed."""
        try:
            from langchain_openai import ChatOpenAI

            return True
        except ImportError:
            return False

    def get_chat_model(self, config: LLMConfig) -> BaseChatModel:
        """
        Return a ChatOpenAI instance configured for OpenRouter.

        Args:
            config: LLM configuration

        Returns:
            ChatOpenAI instance pointing to OpenRouter API
        """
        from langchain_openai import ChatOpenAI

        if not config.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable "
                "or provide it in the configuration."
            )

        logger.info(
            f"Initializing OpenRouter backend (model: {config.model}, "
            f"temperature: {config.temperature})"
        )

        return ChatOpenAI(
            openai_api_base=OPENROUTER_BASE_URL,
            openai_api_key=config.api_key,
            model=config.model,
            streaming=False,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            seed=42,  # For reproducibility
        )

    def get_default_config(self, api_key: Optional[str] = None) -> LLMConfig:
        """
        Return default configuration for OpenRouter.

        Args:
            api_key: OpenRouter API key

        Returns:
            Default LLMConfig for OpenRouter
        """
        return LLMConfig(
            backend=self.name,
            api_key=api_key,
            model=DEFAULT_MODEL,
            temperature=0.0,
            max_tokens=4096,
            timeout=120,
        )
