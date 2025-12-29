# Calculator Backend System

AdsKRK supports pluggable calculator backends for atomistic simulations. This allows flexibility in choosing different computational engines while maintaining a unified interface.

## Quick Start

The default backend is **MACE** (MACE-MP foundation model). No configuration is needed for basic usage.

```python
# The agent automatically uses MACE with platform-optimized settings
streamlit run src/app/app.py
```

## Backend Selection

### Via Environment Variable

```bash
# Use MACE (default)
export ADSKRK_BACKEND=mace

# Future: Use OpenMD (not yet implemented)
export ADSKRK_BACKEND=openmd
```

### Programmatic Usage

```python
from src.calculators import get_backend, CalculatorConfig

# Get backend instance
backend = get_backend("mace")

# Get platform-specific configuration
import torch
config = backend.get_default_config(has_gpu=torch.cuda.is_available())

# Get calculator
calc = backend.get_calculator(config)
```

## Platform-Specific Behavior

MACE automatically adapts to your platform:

| Platform | Device | Precision | Model | Notes |
|----------|--------|-----------|-------|-------|
| macOS (Apple Silicon) | cpu | float32 | small | MPS doesn't support FP64 |
| Linux (CPU) | cpu | float64 | small | Standard CPU mode |
| Linux (CUDA GPU) | cuda | float64 | large | Production quality |

## Adding New Backends

To add a new backend (e.g., DeePMD-kit):

1. Create `src/calculators/deepmd_backend.py`:

```python
from src.calculators.base import BaseBackend, CalculatorConfig, RelaxationParams

class DeePMDBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "deepmd"
    
    def get_calculator(self, config: CalculatorConfig):
        from deepmd.calculator import DP
        return DP(model=config.model)
    
    # ... implement other abstract methods
```

1. Register in `src/calculators/factory.py`:

```python
from src.calculators.deepmd_backend import DeePMDBackend

_BACKENDS = {
    "mace": MACEBackend,
    "openmd": OpenMDBackend,
    "deepmd": DeePMDBackend,  # Add this
}
```

## API Reference

### CalculatorConfig

Configuration dataclass for calculator initialization:

- `device`: Compute device ("cpu", "cuda", "mps")
- `model`: Model identifier (backend-specific)
- `precision`: Floating point precision ("float32", "float64")
- `use_dispersion`: Enable dispersion corrections

### RelaxationParams

Parameters for structure relaxation:

- `fmax`: Max force tolerance (eV/Å)
- `steps`: Max optimization steps
- `md_steps`: MD warmup steps (0 to disable)
- `md_temp`: MD temperature (K)
