import torch


def gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This model trains on a GPU, and torch sees none: run the app with --gpus all")
    return torch.device("cuda")
