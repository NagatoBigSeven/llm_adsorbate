"""
MACE backend implementation.

This module provides the MACEBackend class that wraps MACE-MP (Materials Project)
foundation model for atomistic simulations.

MACE is the default and primary calculator for AdsKRK. This implementation
preserves all platform-specific logic:
- Apple Silicon (Darwin): float32 precision (MPS doesn't support FP64)
- Linux CPU: float64 precision
- CUDA GPU: float64 precision + larger model + dispersion corrections
"""

import platform
from typing import Optional

from ase.calculators.calculator import Calculator

from src.calculators.base import BaseBackend, CalculatorConfig, RelaxationParams
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MACEBackend(BaseBackend):
    """
    MACE-MP calculator backend.

    This is the primary calculator backend for AdsKRK, using the MACE-MP
    foundation model for energy and force calculations.

    The backend implements calculator caching to avoid reloading the model
    on every call, which significantly improves performance.
    """

    # Class-level cache for calculator reuse
    _cached_calculator: Optional[Calculator] = None
    _cached_config_key: Optional[tuple] = None

    @property
    def name(self) -> str:
        return "mace"

    @property
    def is_available(self) -> bool:
        """Check if MACE is installed."""
        try:
            from mace.calculators import mace_mp

            return True
        except ImportError:
            return False

    def get_calculator(self, config: CalculatorConfig) -> Calculator:
        """
        Return a MACE-MP calculator instance.

        Uses caching to avoid reloading the model when the same configuration
        is requested multiple times.

        Args:
            config: Calculator configuration

        Returns:
            MACE-MP ASE calculator
        """
        from mace.calculators import mace_mp

        current_key = config.cache_key()

        # Check if we can reuse cached calculator
        if (
            MACEBackend._cached_calculator is not None
            and MACEBackend._cached_config_key == current_key
        ):
            logger.info("Reusing cached MACE calculator")
            return MACEBackend._cached_calculator

        # Create new calculator
        logger.info(
            f"Initializing MACE Calculator (Model: {config.model}, "
            f"Device: {config.device}, Precision: {config.precision})"
        )

        calculator = mace_mp(
            model=config.model,
            device=config.device,
            default_dtype=config.precision,
            dispersion=config.use_dispersion,
        )

        # Cache for reuse
        MACEBackend._cached_calculator = calculator
        MACEBackend._cached_config_key = current_key

        return calculator

    def get_default_config(self, has_gpu: bool) -> CalculatorConfig:
        """
        Return platform-specific default configuration for MACE.

        CRITICAL: This method preserves the exact platform-specific logic
        from the original implementation:
        - macOS (Darwin): float32 (Apple Silicon MPS doesn't support FP64)
        - Linux CPU: float64
        - CUDA GPU: float64 + large model + dispersion

        Args:
            has_gpu: Whether CUDA GPU is available

        Returns:
            Default CalculatorConfig for MACE
        """
        if has_gpu:
            # CUDA GPU: Production quality settings
            return CalculatorConfig(
                device="cuda",
                model="large",
                precision="float64",
                use_dispersion=True,
            )
        else:
            # CPU mode: Determine precision based on platform
            # Apple Silicon MPS does NOT support FP64 - using float64 will crash!
            is_macos = platform.system() == "Darwin"
            precision = "float32" if is_macos else "float64"

            return CalculatorConfig(
                device="cpu",
                model="small",
                precision=precision,
                use_dispersion=False,
            )

    def get_default_relaxation_params(self, has_gpu: bool) -> RelaxationParams:
        """
        Return default relaxation parameters for MACE.

        GPU mode uses tighter tolerances and more steps for production quality.
        CPU mode uses looser tolerances for faster prototyping.

        Args:
            has_gpu: Whether CUDA GPU is available

        Returns:
            Default RelaxationParams for MACE
        """
        if has_gpu:
            # GPU: Tighter tolerances, more steps, MD warmup
            return RelaxationParams(
                fmax=0.05,
                steps=500,
                md_steps=20,
                md_temp=150.0,
            )
        else:
            # CPU: Faster settings for prototyping
            return RelaxationParams(
                fmax=0.10,
                steps=200,
                md_steps=0,  # No MD warmup on CPU
                md_temp=150.0,
            )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached calculator (useful for testing)."""
        cls._cached_calculator = None
        cls._cached_config_key = None
