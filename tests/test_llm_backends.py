"""
Unit tests for LLM backend factory.

Tests the multi-backend LLM support including factory pattern,
configuration, and backend availability checks.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestLLMFactory:
    """Test the LLM backend factory."""
    
    def test_factory_import(self):
        """Test that factory can be imported."""
        from src.llms import get_llm_backend, get_available_llm_backends
        assert callable(get_llm_backend)
        assert callable(get_available_llm_backends)
    
    def test_get_google_backend(self):
        """Test getting Google backend."""
        from src.llms import get_llm_backend
        backend = get_llm_backend("google")
        assert backend.name == "google"
        assert backend.requires_api_key is True
    
    def test_get_openrouter_backend(self):
        """Test getting OpenRouter backend."""
        from src.llms import get_llm_backend
        backend = get_llm_backend("openrouter")
        assert backend.name == "openrouter"
        assert backend.requires_api_key is True
    
    def test_get_ollama_backend(self):
        """Test getting Ollama backend."""
        from src.llms import get_llm_backend
        backend = get_llm_backend("ollama")
        assert backend.name == "ollama"
        assert backend.requires_api_key is False
    
    def test_get_huggingface_backend(self):
        """Test getting HuggingFace backend."""
        from src.llms import get_llm_backend
        backend = get_llm_backend("huggingface")
        assert backend.name == "huggingface"
        assert backend.requires_api_key is False
    
    def test_unknown_backend_raises_error(self):
        """Test that unknown backend raises ValueError."""
        from src.llms import get_llm_backend
        with pytest.raises(ValueError) as exc_info:
            get_llm_backend("unknown_backend")
        assert "Unknown LLM backend" in str(exc_info.value)
    
    def test_available_backends(self):
        """Test listing available backends."""
        from src.llms import get_available_llm_backends
        available = get_available_llm_backends()
        # At minimum, google and openrouter should be available
        assert "google" in available or "openrouter" in available


class TestLLMConfig:
    """Test LLM configuration dataclass."""
    
    def test_config_creation(self):
        """Test creating LLMConfig."""
        from src.llms.base import LLMConfig
        config = LLMConfig(
            backend="google",
            api_key="test-key",
            model="gemini-2.5-pro"
        )
        assert config.backend == "google"
        assert config.api_key == "test-key"
        assert config.model == "gemini-2.5-pro"
        assert config.temperature == 0.0  # Default
    
    def test_config_defaults(self):
        """Test LLMConfig default values."""
        from src.llms.base import LLMConfig
        config = LLMConfig(backend="test")
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.timeout == 120
        assert config.extra_options == {}


class TestGoogleBackend:
    """Test Google AI backend."""
    
    def test_default_config(self):
        """Test default configuration."""
        from src.llms import get_llm_backend
        backend = get_llm_backend("google")
        config = backend.get_default_config(api_key="test-key")
        
        assert config.backend == "google"
        assert config.api_key == "test-key"
        assert config.model == "gemini-2.5-pro"
        assert config.temperature == 0.0
    
    def test_is_available(self):
        """Test availability check."""
        from src.llms import get_llm_backend
        backend = get_llm_backend("google")
        # Should be True if langchain_google_genai is installed
        assert isinstance(backend.is_available, bool)


class TestOpenRouterBackend:
    """Test OpenRouter backend."""
    
    def test_default_config(self):
        """Test default configuration."""
        from src.llms import get_llm_backend
        backend = get_llm_backend("openrouter")
        config = backend.get_default_config(api_key="test-key")
        
        assert config.backend == "openrouter"
        assert config.api_key == "test-key"
        assert config.model == "google/gemini-2.5-pro"


class TestOllamaBackend:
    """Test Ollama backend."""
    
    def test_default_config(self):
        """Test default configuration."""
        from src.llms import get_llm_backend
        backend = get_llm_backend("ollama")
        config = backend.get_default_config()
        
        assert config.backend == "ollama"
        assert config.api_key is None
        assert config.model == "qwen3:8b"
        assert "host" in config.extra_options
    
    def test_custom_host_from_env(self):
        """Test custom host from environment variable."""
        from src.llms import get_llm_backend
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://custom:11434"}):
            backend = get_llm_backend("ollama")
            config = backend.get_default_config()
            assert config.extra_options["host"] == "http://custom:11434"


class TestHuggingFaceBackend:
    """Test HuggingFace backend."""
    
    def test_default_config(self):
        """Test default configuration."""
        from src.llms import get_llm_backend
        backend = get_llm_backend("huggingface")
        config = backend.get_default_config()
        
        assert config.backend == "huggingface"
        assert config.api_key is None
        assert config.model == "Qwen/Qwen3-8B"
        assert "device" in config.extra_options
        assert "quantize" in config.extra_options


class TestConfigModule:
    """Test config module LLM functions."""
    
    def test_get_llm_backend_name_default(self):
        """Test default backend name."""
        from src.utils.config import get_llm_backend_name, DEFAULT_LLM_BACKEND
        # Clear any environment override
        with patch.dict(os.environ, {}, clear=True):
            backend = get_llm_backend_name()
            assert backend == DEFAULT_LLM_BACKEND
    
    def test_is_cloud_backend(self):
        """Test cloud backend detection."""
        from src.utils.config import is_cloud_backend
        assert is_cloud_backend("google") is True
        assert is_cloud_backend("openrouter") is True
        assert is_cloud_backend("ollama") is False
        assert is_cloud_backend("huggingface") is False


class TestAgentIntegration:
    """Test agent.py integration with LLM backends."""
    
    def test_get_llm_function_signature(self):
        """Test get_llm function signature."""
        from src.agent.agent import get_llm
        import inspect
        sig = inspect.signature(get_llm)
        params = list(sig.parameters.keys())
        assert "api_key" in params
        assert "backend_name" in params
        assert "llm_config" in params
    
    def test_agent_state_has_llm_fields(self):
        """Test AgentState has LLM configuration fields."""
        from src.agent.agent import AgentState
        # TypedDict annotations
        annotations = AgentState.__annotations__
        assert "llm_backend" in annotations
        assert "llm_config" in annotations
    
    def test_prepare_initial_state_signature(self):
        """Test _prepare_initial_state function signature."""
        from src.agent.agent import _prepare_initial_state
        import inspect
        sig = inspect.signature(_prepare_initial_state)
        params = list(sig.parameters.keys())
        assert "llm_backend" in params
        assert "llm_config" in params
