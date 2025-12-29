"""
Ollama backend implementation.

This backend provides access to locally running LLM models via Ollama,
enabling privacy-focused and cost-free inference.

Benefits:
- No API costs
- Data stays local (privacy)
- No internet required after model download

Setup:
    1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
    2. Pull a model: ollama pull qwen3:8b
    3. Set backend: ADSKRK_LLM_BACKEND=ollama
"""

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.llms.base import BaseLLMBackend, LLMConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default Ollama configuration
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


class OllamaBackend(BaseLLMBackend):
    """
    Ollama LLM backend.

    Provides access to locally running LLM models for privacy-focused
    and cost-free inference. Ideal for sensitive chemical data.
    """

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def requires_api_key(self) -> bool:
        return False

    @property
    def is_available(self) -> bool:
        """Check if langchain-ollama is installed."""
        try:
            from langchain_ollama import ChatOllama

            return True
        except ImportError:
            return False

    def get_chat_model(self, config: LLMConfig) -> BaseChatModel:
        """
        Return a ChatOllama instance.

        Args:
            config: LLM configuration

        Returns:
            ChatOllama instance
        """
        from langchain_ollama import ChatOllama

        # Get host from config or environment
        host = config.extra_options.get(
            "host", os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
        )

        logger.info(
            f"Initializing Ollama backend (host: {host}, model: {config.model}, "
            f"temperature: {config.temperature})"
        )

        return ChatOllama(
            base_url=host,
            model=config.model,
            temperature=config.temperature,
            num_predict=config.max_tokens,
        )

    def get_default_config(self, api_key: Optional[str] = None) -> LLMConfig:
        """
        Return default configuration for Ollama.

        Args:
            api_key: Ignored (Ollama doesn't need API key)

        Returns:
            Default LLMConfig for Ollama
        """
        return LLMConfig(
            backend=self.name,
            api_key=None,  # No API key needed
            model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
            temperature=0.0,
            max_tokens=4096,
            timeout=300,  # Longer timeout for local inference
            extra_options={
                "host": os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
            },
        )
