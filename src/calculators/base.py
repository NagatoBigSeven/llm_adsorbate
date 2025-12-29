"""
Abstract base class for calculator backends.

This module defines the interface that all calculator backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ase.calculators.calculator import Calculator


@dataclass
class CalculatorConfig:
    """
    Configuration for calculator initialization.

    This is a unified configuration object that can be passed to any backend.
    Backend-specific options should be placed in `extra_options`.

    Attributes:
        device: Compute device ("cpu", "cuda", "mps")
        model: Model identifier (backend-specific, e.g., "small", "medium", "large" for MACE)
        precision: Floating point precision ("float32", "float64")
        use_dispersion: Whether to enable dispersion corrections (D3/D4)
        extra_options: Additional backend-specific options
    """

    device: str = "cpu"
    model: str = "small"
    precision: str = "float32"
    use_dispersion: bool = False
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def cache_key(self) -> tuple:
        """Generate a hashable key for caching calculators."""
        return (self.device, self.model, self.precision, self.use_dispersion)


@dataclass
class RelaxationParams:
    """
    Parameters for structure relaxation.

    Attributes:
        fmax: Maximum force tolerance for optimization (eV/Å)
        steps: Maximum optimization steps
        md_steps: Number of MD warmup steps (0 to disable)
        md_temp: MD temperature in Kelvin
    """

    fmax: float = 0.05
    steps: int = 500
    md_steps: int = 20
    md_temp: float = 150.0


class BaseBackend(ABC):
    """
    Abstract base class for calculator backends.

    All calculator backends (MACE, OpenMD, DeePMD, etc.) must inherit from this
    class and implement the required abstract methods.

    The interface is designed to work with ASE (Atomic Simulation Environment),
    so all calculators returned must be ASE-compatible.
    """

    @abstractmethod
    def get_calculator(self, config: CalculatorConfig) -> Calculator:
        """
        Return an ASE-compatible calculator instance.

        Args:
            config: Calculator configuration

        Returns:
            An ASE Calculator object that implements get_potential_energy(),
            get_forces(), and optionally get_stress().

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If calculator initialization fails
        """
        pass

    @abstractmethod
    def get_default_config(self, has_gpu: bool) -> CalculatorConfig:
        """
        Return default configuration for this backend.

        This method encapsulates platform-specific defaults (e.g., float32 for
        macOS, float64 for Linux).

        Args:
            has_gpu: Whether CUDA GPU is available

        Returns:
            Default CalculatorConfig for this backend
        """
        pass

    @abstractmethod
    def get_default_relaxation_params(self, has_gpu: bool) -> RelaxationParams:
        """
        Return default MD/optimization parameters for this backend.

        Args:
            has_gpu: Whether CUDA GPU is available

        Returns:
            Default RelaxationParams for this backend
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'mace', 'openmd')."""
        pass

    @property
    def is_available(self) -> bool:
        """
        Check if this backend is available (dependencies installed).

        Override this in subclasses to perform actual availability checks.
        """
        return True
