"""
Calculator backend factory.

This module provides the factory function to get calculator backends by name.
"""

from typing import Dict, List, Type

from src.calculators.base import BaseBackend


def _get_backend_registry() -> Dict[str, Type[BaseBackend]]:
    """
    Lazily build the backend registry.

    Using lazy imports to avoid loading heavy dependencies (like MACE)
    until actually needed.
    """
    from src.calculators.mace_backend import MACEBackend
    from src.calculators.openmd_backend import OpenMDBackend

    return {
        "mace": MACEBackend,
        "openmd": OpenMDBackend,
    }


def get_backend(name: str) -> BaseBackend:
    """
    Get a calculator backend by name.

    Args:
        name: Backend name ("mace", "openmd", etc.)

    Returns:
        An instance of the requested backend

    Raises:
        ValueError: If the backend name is unknown

    Example:
        >>> backend = get_backend("mace")
        >>> config = backend.get_default_config(has_gpu=False)
        >>> calc = backend.get_calculator(config)
    """
    registry = _get_backend_registry()

    if name not in registry:
        available = list(registry.keys())
        raise ValueError(
            f"Unknown calculator backend: '{name}'. "
            f"Available backends: {available}"
        )

    backend_class = registry[name]
    return backend_class()


def get_available_backends() -> List[str]:
    """
    Get a list of all available backend names.

    Returns:
        List of backend names that are installed and available
    """
    registry = _get_backend_registry()
    available = []

    for name, backend_class in registry.items():
        try:
            backend = backend_class()
            if backend.is_available:
                available.append(name)
        except Exception:
            pass

    return available
