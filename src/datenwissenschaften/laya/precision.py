import torch


def autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.float16
    return torch.bfloat16
