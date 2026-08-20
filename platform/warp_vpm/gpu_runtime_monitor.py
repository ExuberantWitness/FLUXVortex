"""Non-scientific CUDA runtime instrumentation for validation runners."""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GpuRuntimeEvidence:
    gpu_device: str
    gpu_uuid: str
    gpu_utilization_observed: bool
    gpu_utilization_peak_percent: int
    gpu_memory_peak_mib: float
    sample_count: int


class GpuRuntimeMonitor:
    """Poll NVIDIA telemetry while CUDA science kernels run.

    Polling is instrumentation only: values never enter a model state, force,
    coefficient, gate, or score.
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.25) -> None:
        self.device_index = int(device_index)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[int, float, str, str]] = []

    def __enter__(self) -> "GpuRuntimeMonitor":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime monitor requires an NVIDIA GPU")
        torch_device = torch.device(f"cuda:{self.device_index}")
        torch.empty(1, device=torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
        self._sample()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 4.0 * self.interval_s))
        self._sample()

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        command = [
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=utilization.gpu,memory.used,name,uuid",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        fields = [field.strip() for field in completed.stdout.strip().split(",")]
        if len(fields) != 4:
            raise RuntimeError("unexpected nvidia-smi telemetry schema")
        self._samples.append((int(fields[0]), float(fields[1]), fields[2], fields[3]))

    def evidence(self) -> GpuRuntimeEvidence:
        if not self._samples:
            raise RuntimeError("GPU runtime monitor recorded no samples")
        utilization = max(sample[0] for sample in self._samples)
        nvidia_memory = max(sample[1] for sample in self._samples)
        torch_memory = torch.cuda.max_memory_allocated(
            torch.device(f"cuda:{self.device_index}")
        ) / (1024**2)
        names = {sample[2] for sample in self._samples}
        uuids = {sample[3] for sample in self._samples}
        if len(names) != 1 or len(uuids) != 1:
            raise RuntimeError("GPU telemetry identity changed during the run")
        return GpuRuntimeEvidence(
            gpu_device=next(iter(names)),
            gpu_uuid=next(iter(uuids)),
            gpu_utilization_observed=utilization > 0,
            gpu_utilization_peak_percent=utilization,
            gpu_memory_peak_mib=max(nvidia_memory, torch_memory),
            sample_count=len(self._samples),
        )


__all__ = ["GpuRuntimeEvidence", "GpuRuntimeMonitor"]
