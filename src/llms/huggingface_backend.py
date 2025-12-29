"""
HuggingFace backend implementation.

This backend provides access to local HuggingFace Transformers models,
enabling fully offline and customizable LLM inference.

Benefits:
- Fully offline (air-gapped environments)
- Customizable models
- Direct GPU utilization

Setup:
    1. Install dependencies: pip install transformers accelerate
    2. (Optional) For quantization: pip install bitsandbytes
    3. Set backend: ADSKRK_LLM_BACKEND=huggingface

Environment variables:
    - HF_MODEL: Model name/path (default: Qwen/Qwen3-8B)
    - HF_DEVICE: Device selection (cuda, cpu, auto)
    - HF_QUANTIZE: Quantization mode (4bit, 8bit, none)
"""

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.llms.base import BaseLLMBackend, LLMConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default HuggingFace configuration
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_DEVICE = "auto"
DEFAULT_QUANTIZE = "none"

# Cache for loaded model (lazy loading)
_cached_pipeline = None
_cached_model_name = None


class HuggingFaceBackend(BaseLLMBackend):
    """
    HuggingFace Transformers LLM backend.

    Provides access to locally loaded HuggingFace models for fully offline
    inference. Supports quantization for memory-constrained environments.
    """

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def requires_api_key(self) -> bool:
        return False

    @property
    def is_available(self) -> bool:
        """Check if transformers and langchain-huggingface are installed."""
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            import transformers

            return True
        except ImportError:
            return False

    def get_chat_model(self, config: LLMConfig) -> BaseChatModel:
        """
        Return a ChatHuggingFace instance.

        Uses lazy loading and caching to avoid reloading the model
        on every call.

        Args:
            config: LLM configuration

        Returns:
            ChatHuggingFace instance
        """
        global _cached_pipeline, _cached_model_name

        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        model_name = config.model
        device = config.extra_options.get("device", DEFAULT_DEVICE)
        quantize = config.extra_options.get("quantize", DEFAULT_QUANTIZE)

        # Check if we can reuse cached pipeline
        if _cached_pipeline is not None and _cached_model_name == model_name:
            logger.info(f"Reusing cached HuggingFace pipeline: {model_name}")
            return ChatHuggingFace(llm=_cached_pipeline, model_id=model_name)

        # Load new model
        logger.info(
            f"Loading HuggingFace model: {model_name} "
            f"(device: {device}, quantize: {quantize})"
        )

        pipeline = self._create_pipeline(model_name, device, quantize, config)

        # Cache for reuse
        _cached_pipeline = pipeline
        _cached_model_name = model_name

        return ChatHuggingFace(llm=pipeline, model_id=model_name)

    def _create_pipeline(
        self, model_name: str, device: str, quantize: str, config: LLMConfig
    ) -> "HuggingFacePipeline":
        """
        Create a HuggingFacePipeline with optional quantization.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use (cuda, cpu, auto)
            quantize: Quantization mode (4bit, 8bit, none)
            config: LLM configuration

        Returns:
            HuggingFacePipeline instance
        """
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
        }

        # Handle quantization
        if quantize == "4bit":
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = "auto"
                logger.info("Using 4-bit quantization")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed, falling back to no quantization. "
                    "Install with: pip install bitsandbytes"
                )
        elif quantize == "8bit":
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_kwargs["device_map"] = "auto"
                logger.info("Using 8-bit quantization")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed, falling back to no quantization. "
                    "Install with: pip install bitsandbytes"
                )
        else:
            # No quantization
            if device == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.float16

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Create text-generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature if config.temperature > 0 else None,
            do_sample=config.temperature > 0,
            return_full_text=False,
        )

        return HuggingFacePipeline(pipeline=pipe)

    def get_default_config(self, api_key: Optional[str] = None) -> LLMConfig:
        """
        Return default configuration for HuggingFace.

        Args:
            api_key: Ignored (HuggingFace models are local)

        Returns:
            Default LLMConfig for HuggingFace
        """
        return LLMConfig(
            backend=self.name,
            api_key=None,  # No API key needed
            model=os.environ.get("HF_MODEL", DEFAULT_MODEL),
            temperature=0.0,
            max_tokens=4096,
            timeout=600,  # Longer timeout for local inference
            extra_options={
                "device": os.environ.get("HF_DEVICE", DEFAULT_DEVICE),
                "quantize": os.environ.get("HF_QUANTIZE", DEFAULT_QUANTIZE),
            },
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached pipeline (useful for testing)."""
        global _cached_pipeline, _cached_model_name
        _cached_pipeline = None
        _cached_model_name = None
