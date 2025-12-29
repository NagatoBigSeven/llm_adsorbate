"""
Calculator backend abstraction module.

This module provides a unified interface for different atomistic simulation calculators
(MACE, OpenMD, DeePMD-kit, etc.) while preserving ASE compatibility.

Usage:
    from src.calculators import get_backend, CalculatorConfig

    backend = get_backend("mace")  # or from env: os.getenv("ADSKRK_BACKEND", "mace")
    config = CalculatorConfig(device="cpu", model="small")
    calc = backend.get_calculator(config)
"""

from src.calculators.base import BaseBackend, CalculatorConfig
from src.calculators.factory import get_backend, get_available_backends

__all__ = [
    "BaseBackend",
    "CalculatorConfig",
    "get_backend",
    "get_available_backends",
]
