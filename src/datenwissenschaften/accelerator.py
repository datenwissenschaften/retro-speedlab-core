from __future__ import annotations

import os
from functools import lru_cache

from loguru import logger

CUDA_ALLOCATOR_SETTINGS = "expandable_segments:True"


@lru_cache(maxsize=1)
def configure_accelerator() -> str:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOCATOR_SETTINGS
    import torch

    if torch.cuda.is_available():
        # "high" permits TensorFloat-32 matrix multiplications on supported
        # NVIDIA GPUs while retaining float32 inputs and outputs.
        torch.set_float32_matmul_precision("high")

        # Retro observations have fixed dimensions, so cuDNN can benchmark
        # kernels once and reuse the fastest implementation.
        torch.backends.cudnn.benchmark = True
        device = "cuda"
        logger.info(f"Configured CUDA accelerator: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple MPS accelerator detected")
    else:
        device = "cpu"
        logger.info("No supported GPU accelerator found; using CPU")

    return device
