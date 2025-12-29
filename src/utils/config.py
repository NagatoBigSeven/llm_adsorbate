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
