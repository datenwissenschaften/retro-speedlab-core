import torch

from datenwissenschaften.accelerator import configure_accelerator


def test_selects_cuda_when_available(monkeypatch):
    configure_accelerator.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    set_precision_calls = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", set_precision_calls.append)

    device = configure_accelerator()

    assert device == "cuda"
    assert set_precision_calls == ["high"]
    assert torch.backends.cudnn.benchmark is True
    configure_accelerator.cache_clear()


def test_selects_mps_when_cuda_unavailable_but_mps_present(monkeypatch):
    configure_accelerator.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device = configure_accelerator()

    assert device == "mps"
    configure_accelerator.cache_clear()


def test_falls_back_to_cpu_when_no_accelerator_present(monkeypatch):
    configure_accelerator.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    device = configure_accelerator()

    assert device == "cpu"
    configure_accelerator.cache_clear()


def test_falls_back_to_cpu_when_mps_backend_is_absent(monkeypatch):
    configure_accelerator.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", None)

    device = configure_accelerator()

    assert device == "cpu"
    configure_accelerator.cache_clear()


def test_result_is_cached_between_calls(monkeypatch):
    configure_accelerator.cache_clear()
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (calls.append(1), False)[1])
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    configure_accelerator()
    configure_accelerator()

    assert len(calls) == 1
    configure_accelerator.cache_clear()
