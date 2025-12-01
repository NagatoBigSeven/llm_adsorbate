"""
Centralized logging configuration.

This module provides a consistent logging interface across all components,
replacing scattered print statements with structured, level-based logging.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a configured logger instance.
    
    Args:
        name: Logger name, typically __name__ from calling module
        log_file: Optional path to log file. If None, only logs to console
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_global_log_level(level: int):
    """
    Set logging level for all existing loggers.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logging.root.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)
