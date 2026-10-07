import torch

from datenwissenschaften.laya import precision
from datenwissenschaften.laya.precision import autocast_dtype

GIB = 1024**3
CUDA = torch.device("cuda")


def test_gpus_without_native_bfloat16_train_in_float16(monkeypatch):
    requests = []

    def supported(including_emulation: bool) -> bool:
        requests.append(including_emulation)
        return False

    monkeypatch.setattr(precision.torch.cuda, "is_bf16_supported", supported)

    assert autocast_dtype(CUDA) == torch.float16
    assert requests == [False]


def test_modern_gpus_and_the_cpu_train_in_bfloat16(monkeypatch):
    monkeypatch.setattr(precision.torch.cuda, "is_bf16_supported", lambda including_emulation: True)

    assert autocast_dtype(CUDA) == torch.bfloat16
    assert autocast_dtype(torch.device("cpu")) == torch.bfloat16
