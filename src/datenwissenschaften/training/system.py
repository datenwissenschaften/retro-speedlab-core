import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch

CPU_INFO = Path("/proc/cpuinfo")
NVIDIA_SMI_TIMEOUT_SECONDS = 2
GIB = 1024**3


def system_metadata() -> dict[str, object]:
    total_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu": {"name": _cpu_name(), "processor": platform.processor(), "logical_cores": os.cpu_count()},
        "memory": {"total_bytes": total_memory, "total_gib": round(total_memory / GIB, 2)},
        "gpu": _gpu_metadata(),
    }


def _cpu_name() -> str:
    if CPU_INFO.is_file():
        for line in CPU_INFO.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def _gpu_metadata() -> dict[str, object]:
    devices = [torch.cuda.get_device_properties(index) for index in range(torch.cuda.device_count())]
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": len(devices),
        "torch_version": torch.__version__,
        "devices": [
            {"index": index, "name": device.name, "total_memory_bytes": device.total_memory}
            for index, device in enumerate(devices)
        ],
        "nvidia_smi": _nvidia_smi(),
    }


def _nvidia_smi() -> list[dict[str, object]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    query = [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    output = subprocess.run(query, check=True, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_SECONDS)
    gpus = []
    for line in output.stdout.splitlines():
        name, memory_total_mb, driver_version = (part.strip() for part in line.split(",", 2))
        gpus.append({"name": name, "memory_total_mb": int(memory_total_mb), "driver_version": driver_version})
    return gpus
