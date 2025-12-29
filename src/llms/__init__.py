"""
LLM Backend Module for AdsKRK.

This module provides a factory pattern for LLM backends, allowing easy switching
between different LLM providers (cloud and local).

Supported backends:
- google: Google AI (Gemini) - Default, direct access
- openrouter: OpenRouter API - Access to multiple models via unified API
- ollama: Ollama local service - Privacy-focused, no API cost
- huggingface: HuggingFace Transformers - Offline, customizable

Usage:
    from src.llms import get_llm_backend, get_available_llm_backends

    # Get a specific backend
    backend = get_llm_backend("google")
    config = backend.get_default_config(api_key="your-key")
    llm = backend.get_chat_model(config)

    # List available backends
    available = get_available_llm_backends()
"""

from src.llms.factory import get_llm_backend, get_available_llm_backends

__all__ = ["get_llm_backend", "get_available_llm_backends"]
