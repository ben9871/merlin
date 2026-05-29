"""Shared utilities for memristor TBPTT benchmark scripts."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class CudaMemorySnapshot:
    """CUDA memory counters captured around one measured benchmark run."""

    cuda_memory_start_allocated_bytes: int | None
    cuda_memory_end_allocated_bytes: int | None
    cuda_memory_peak_allocated_bytes: int | None
    cuda_memory_start_reserved_bytes: int | None
    cuda_memory_end_reserved_bytes: int | None
    cuda_memory_peak_reserved_bytes: int | None


def resolve_benchmark_device(requested_device: str) -> torch.device:
    """Resolve a CLI device argument into a concrete torch device.

    Parameters
    ----------
    requested_device : str
        Device argument from the command line. ``"auto"`` selects CUDA when it
        is available and CPU otherwise.

    Returns
    -------
    torch.device
        Resolved torch device.

    Raises
    ------
    RuntimeError
        If a CUDA device is requested but CUDA is not available.
    """
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested_device!r} was requested, but CUDA is not available."
        )
    return device


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA work before or after a timed region."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_cuda_memory_stats(device: torch.device) -> None:
    """Reset CUDA peak-memory counters for a benchmark run."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def capture_cuda_memory(device: torch.device) -> CudaMemorySnapshot:
    """Capture CUDA memory counters for the selected device.

    Parameters
    ----------
    device : torch.device
        Device being benchmarked.

    Returns
    -------
    CudaMemorySnapshot
        CUDA memory counters. Values are ``None`` for non-CUDA devices.
    """
    if device.type != "cuda":
        return CudaMemorySnapshot(
            cuda_memory_start_allocated_bytes=None,
            cuda_memory_end_allocated_bytes=None,
            cuda_memory_peak_allocated_bytes=None,
            cuda_memory_start_reserved_bytes=None,
            cuda_memory_end_reserved_bytes=None,
            cuda_memory_peak_reserved_bytes=None,
        )

    return CudaMemorySnapshot(
        cuda_memory_start_allocated_bytes=torch.cuda.memory_allocated(device),
        cuda_memory_end_allocated_bytes=None,
        cuda_memory_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_memory_start_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_memory_end_reserved_bytes=None,
        cuda_memory_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def finish_cuda_memory_snapshot(
    device: torch.device,
    start_snapshot: CudaMemorySnapshot,
) -> CudaMemorySnapshot:
    """Complete a CUDA memory snapshot at the end of a run."""
    if device.type != "cuda":
        return start_snapshot

    return CudaMemorySnapshot(
        cuda_memory_start_allocated_bytes=start_snapshot.cuda_memory_start_allocated_bytes,
        cuda_memory_end_allocated_bytes=torch.cuda.memory_allocated(device),
        cuda_memory_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_memory_start_reserved_bytes=start_snapshot.cuda_memory_start_reserved_bytes,
        cuda_memory_end_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_memory_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def cuda_memory_fields(snapshot: CudaMemorySnapshot) -> dict[str, int | None]:
    """Return a serializable mapping for a CUDA memory snapshot."""
    return asdict(snapshot)


def move_batches_to_device(
    batches: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    """Move pre-built benchmark batches to the selected device."""
    return [batch.to(device=device) for batch in batches]


def benchmark_environment(device: torch.device) -> dict[str, Any]:
    """Return environment metadata stored with benchmark outputs."""
    environment: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "requested_device": str(device),
    }
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        environment.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_index": device_index,
                "cuda_runtime_version": torch.version.cuda,
                "cuda_device_capability": torch.cuda.get_device_capability(device_index),
                "cuda_total_memory_bytes": torch.cuda.get_device_properties(
                    device_index
                ).total_memory,
            }
        )
    return environment


def format_megabytes(value: int | None) -> str:
    """Format an optional byte count as megabytes."""
    return "n/a" if value is None else f"{value / 1_000_000:.1f}"
