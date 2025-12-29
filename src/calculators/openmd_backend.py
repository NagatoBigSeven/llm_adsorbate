"""
OpenMD backend placeholder.

This module provides a placeholder for OpenMD integration. OpenMD does not have
a native ASE calculator interface, so full implementation would require either:
1. A subprocess-based wrapper that calls OpenMD CLI
2. Python bindings (if available)

For now, this serves as a template for future backend implementations.
"""

from ase.calculators.calculator import Calculator

from src.calculators.base import BaseBackend, CalculatorConfig, RelaxationParams


class OpenMDBackend(BaseBackend):
    """
    Placeholder backend for OpenMD integration.

    This backend is not yet implemented. It serves as an interface placeholder
    for future extensibility.
    """

    @property
    def name(self) -> str:
        return "openmd"

    @property
    def is_available(self) -> bool:
        """OpenMD is not yet implemented."""
        return False

    def get_calculator(self, config: CalculatorConfig) -> Calculator:
        """
        Return an OpenMD calculator.

        Raises:
            NotImplementedError: OpenMD backend is not yet implemented
        """
        raise NotImplementedError(
            "OpenMD backend is not yet implemented. "
            "Contributions welcome! See: https://github.com/schwallergroup/llm_adsorbate"
        )

    def get_default_config(self, has_gpu: bool) -> CalculatorConfig:
        """Return default configuration (placeholder)."""
        return CalculatorConfig(
            device="cpu",
            model="default",
            precision="float64",
            use_dispersion=False,
        )

    def get_default_relaxation_params(self, has_gpu: bool) -> RelaxationParams:
        """Return default relaxation parameters (placeholder)."""
        return RelaxationParams(
            fmax=0.05,
            steps=500,
            md_steps=0,
            md_temp=300.0,
        )
