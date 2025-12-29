"""
Configuration management utilities for AdsKRK.

Handles loading and saving API keys from/to a local JSON configuration file.
Supports auto-detection of API keys from multiple sources with priority:
environment variable > config file > user input.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Literal

# Configuration file path (stored in user's home directory)
CONFIG_DIR = Path.home() / ".adskrk"
CONFIG_FILE_PATH = CONFIG_DIR / "config.json"

# Type alias for API key source
ApiKeySource = Literal["env", "config", None]


def load_config() -> dict:
    """
    Load configuration from the JSON config file.
    
    Returns:
        dict: Configuration dictionary, or empty dict if file doesn't exist.
    """
    if not CONFIG_FILE_PATH.exists():
        return {}
    
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_config(config: dict) -> bool:
    """
    Save configuration to the JSON config file.
    
    Args:
        config: Configuration dictionary to save.
        
    Returns:
        bool: True if save was successful, False otherwise.
    """
    try:
        # Create config directory if it doesn't exist
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except IOError:
        return False


def get_api_key() -> Tuple[Optional[str], ApiKeySource]:
    """
    Get the OpenRouter API key from available sources.
    
    Priority order:
    1. Environment variable (OPENROUTER_API_KEY)
    2. Config file (~/.adskrk/config.json)
    3. None (user must input)
    
    Returns:
        Tuple of (api_key, source) where source is "env", "config", or None.
    """
    # Priority 1: Environment variable
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return (env_key, "env")
    
    # Priority 2: Config file
    config = load_config()
    config_key = config.get("openrouter_api_key")
    if config_key:
        return (config_key, "config")
    
    # No key found
    return (None, None)


def save_api_key(api_key: str) -> bool:
    """
    Save the OpenRouter API key to the config file.
    
    Args:
        api_key: The API key to save.
        
    Returns:
        bool: True if save was successful, False otherwise.
    """
    config = load_config()
    config["openrouter_api_key"] = api_key
    return save_config(config)


def is_env_key_set() -> bool:
    """
    Check if the API key is set via environment variable.
    
    Returns:
        bool: True if OPENROUTER_API_KEY environment variable is set.
    """
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def get_calculator_backend() -> str:
    """
    Get the calculator backend name from environment variable.
    
    The backend can be configured via the ADSKRK_BACKEND environment variable.
    Defaults to "mace" if not set.
    
    Available backends:
    - "mace": MACE-MP foundation model (default, recommended)
    - "openmd": OpenMD (not yet implemented)
    
    Returns:
        str: Backend name (e.g., "mace", "openmd")
    """
    return os.environ.get("ADSKRK_BACKEND", "mace")


# ============================================================
# LLM Backend Configuration
# ============================================================

# Default LLM backend
DEFAULT_LLM_BACKEND = "google"

# Environment variable to API key mapping
LLM_API_KEY_ENV_VARS = {
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Config file key mapping
LLM_API_KEY_CONFIG_KEYS = {
    "google": "google_api_key",
    "openrouter": "openrouter_api_key",
}


def get_llm_backend_name() -> str:
    """
    Get the LLM backend name from environment variable or config.
    
    Priority order:
    1. Environment variable (ADSKRK_LLM_BACKEND)
    2. Config file (~/.adskrk/config.json -> llm_backend)
    3. Default ("google")
    
    Available backends:
    - "google": Google AI (Gemini) - Default
    - "openrouter": OpenRouter API
    - "ollama": Local Ollama service
    - "huggingface": Local HuggingFace Transformers
    
    Returns:
        str: LLM backend name
    """
    # Environment variable takes priority
    env_backend = os.environ.get("ADSKRK_LLM_BACKEND")
    if env_backend:
        return env_backend
    
    # Check config file
    config = load_config()
    config_backend = config.get("llm_backend")
    if config_backend:
        return config_backend
    
    return DEFAULT_LLM_BACKEND


def get_api_key_for_backend(backend: str) -> Tuple[Optional[str], ApiKeySource]:
    """
    Get the API key for a specific LLM backend.
    
    Priority order:
    1. Environment variable (backend-specific, e.g., GOOGLE_API_KEY)
    2. Config file (~/.adskrk/config.json)
    3. None (user must input)
    
    Args:
        backend: LLM backend name ("google", "openrouter", etc.)
        
    Returns:
        Tuple of (api_key, source) where source is "env", "config", or None.
    """
    # Get the environment variable name for this backend
    env_var = LLM_API_KEY_ENV_VARS.get(backend)
    if env_var:
        env_key = os.environ.get(env_var)
        if env_key:
            return (env_key, "env")
    
    # Check config file
    config_key_name = LLM_API_KEY_CONFIG_KEYS.get(backend)
    if config_key_name:
        config = load_config()
        config_key = config.get(config_key_name)
        if config_key:
            return (config_key, "config")
    
    # No key found (not required for local backends)
    return (None, None)


def save_api_key_for_backend(backend: str, api_key: str) -> bool:
    """
    Save the API key for a specific LLM backend to the config file.
    
    Args:
        backend: LLM backend name ("google", "openrouter", etc.)
        api_key: The API key to save.
        
    Returns:
        bool: True if save was successful, False otherwise.
    """
    config_key_name = LLM_API_KEY_CONFIG_KEYS.get(backend)
    if not config_key_name:
        return False
    
    config = load_config()
    config[config_key_name] = api_key
    return save_config(config)


def save_llm_backend(backend: str) -> bool:
    """
    Save the preferred LLM backend to the config file.
    
    Args:
        backend: LLM backend name to save as default.
        
    Returns:
        bool: True if save was successful, False otherwise.
    """
    config = load_config()
    config["llm_backend"] = backend
    return save_config(config)


def is_cloud_backend(backend: str) -> bool:
    """
    Check if a backend is a cloud backend (requires API key).
    
    Args:
        backend: LLM backend name
        
    Returns:
        bool: True if the backend requires an API key
    """
    return backend in ("google", "openrouter")

