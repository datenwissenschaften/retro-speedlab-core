import torch

LARGE_GPU_BYTES = int(7.5 * 1024**3)
LARGE_MINIBATCH = 16
SMALL_MINIBATCH = 8


def autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.float16
    return torch.bfloat16


def minibatch_size(device: torch.device) -> int:
    if device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory >= LARGE_GPU_BYTES:
        return LARGE_MINIBATCH
    return SMALL_MINIBATCH
