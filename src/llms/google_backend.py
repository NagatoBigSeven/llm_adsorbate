"""
Google AI backend implementation.

This is the default LLM backend for AdsKRK, providing direct access to
Google's Gemini models via the Google AI Studio API.

Benefits over OpenRouter:
- Lower latency (no proxy)
- Direct access to latest Gemini features
"""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.llms.base import BaseLLMBackend, LLMConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default model for scientific reasoning
DEFAULT_MODEL = "gemini-2.5-pro"


class GoogleBackend(BaseLLMBackend):
    """
    Google AI (Gemini) LLM backend.

    This is the default and recommended backend for AdsKRK, using Google's
    Gemini models for chemical reasoning and planning.
    """

    @property
    def name(self) -> str:
        return "google"

    @property
    def requires_api_key(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        """Check if langchain-google-genai is installed."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return True
        except ImportError:
            return False

    def get_chat_model(self, config: LLMConfig) -> BaseChatModel:
        """
        Return a ChatGoogleGenerativeAI instance.

        Args:
            config: LLM configuration

        Returns:
            ChatGoogleGenerativeAI instance
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not config.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or provide it in the configuration."
            )

        logger.info(
            f"Initializing Google AI backend (model: {config.model}, "
            f"temperature: {config.temperature})"
        )

        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )

    def get_default_config(self, api_key: Optional[str] = None) -> LLMConfig:
        """
        Return default configuration for Google AI.

        Args:
            api_key: Google API key

        Returns:
            Default LLMConfig for Google AI
        """
        return LLMConfig(
            backend=self.name,
            api_key=api_key,
            model=DEFAULT_MODEL,
            temperature=0.0,
            max_tokens=4096,
            timeout=120,
        )
