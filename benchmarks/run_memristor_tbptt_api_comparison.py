"""Compare blackbox memristive TBPTT with user-managed TBPTT.

This benchmark is intentionally a standalone script. It compares two APIs on
the same memristive QuantumLayer implementation:

* blackbox: ``num_backprop_steps=k`` lets QuantumLayer maintain its own
  truncated memristive window;
* manual: ``num_backprop_steps=None`` keeps the full recurrent graph until the
  user calls ``detach_memristive_state()`` at chunk boundaries.

The benchmark measures wall time, approximate memory, measured layer-output
calls, and calls into the underlying computation process. It does not update
model parameters; it exercises forward and backward graph construction.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import merlin as ML
from benchmarks.memristor_benchmark_utils import (
    benchmark_environment,
    capture_cuda_memory,
    cuda_memory_fields,
    finish_cuda_memory_snapshot,
    format_megabytes,
    reset_cuda_memory_stats,
    resolve_benchmark_device,
    synchronize_if_cuda,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional benchmark dependency
    psutil = None


PROCESS_METHODS = (
    "compute",
    "compute_superposition_state",
    "compute_ebs_simultaneously",
)


@dataclass(frozen=True)
class WorkloadConfig:
    """Configuration for one benchmark workload."""

    n_modes: int
    n_photons: int
    input_size: int
    memristor_count: int
    entangling_layers: int
    batch_size: int
    timesteps: int
    k: int
    chunk_len: int
    seed: int
    device: str


@dataclass
class RunResult:
    """Measured result for one API and workload."""

    api: str
    config: dict[str, int]
    seconds: float
    layer_output_calls: int
    process_calls: dict[str, int]
    process_calls_total: int
    rss_start_bytes: int | None
    rss_peak_bytes: int | None
    rss_end_bytes: int | None
    rss_peak_delta_bytes: int | None
    rss_end_delta_bytes: int | None
    tracemalloc_peak_bytes: int
    cuda_memory_start_allocated_bytes: int | None
    cuda_memory_end_allocated_bytes: int | None
    cuda_memory_peak_allocated_bytes: int | None
    cuda_memory_start_reserved_bytes: int | None
    cuda_memory_end_reserved_bytes: int | None
    cuda_memory_peak_reserved_bytes: int | None
    retained_gradient_outputs: int
    history_lengths: list[int]
    output_checksum: float
    input_grad_norm_sum: float
    input_grad_norms: list[float]


def _update_from_first_probability(
    state: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Update the memristive state from the first measured probability."""
    return 0.85 * state + 0.15 * output[:, 0]


def _make_probability_update_rule(
    probability_index: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create a deterministic memristive update rule for a probability column."""

    def update_rule(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        column = min(probability_index, output.shape[1] - 1)
        return 0.85 * state + 0.15 * output[:, column]

    update_rule.__name__ = f"update_from_probability_{probability_index}"
    return update_rule


def _memristor_modes(n_modes: int, memristor_count: int) -> list[int]:
    """Return deterministic modes used by memristive phase shifters."""
    if memristor_count < 1:
        raise ValueError("memristor_count must be at least 1.")
    if memristor_count >= n_modes:
        raise ValueError(
            f"memristor_count={memristor_count} leaves no mode for inputs "
            f"when n_modes={n_modes}."
        )
    return list(range(memristor_count))


def _input_modes(
    n_modes: int, input_size: int, memristor_modes: list[int]
) -> list[int]:
    """Return deterministic angle-encoding modes excluding memristor modes."""
    modes = [mode for mode in range(n_modes) if mode not in set(memristor_modes)]
    if len(modes) < input_size:
        raise ValueError(
            f"Need at least {input_size} non-memristor modes, got {len(modes)}."
        )
    return modes[:input_size]


def build_memristive_layer(
    *,
    n_modes: int,
    n_photons: int,
    input_size: int,
    memristor_count: int,
    entangling_layers: int,
    num_backprop_steps: int | None,
    seed: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ML.QuantumLayer:
    """Build the benchmark memristive layer."""
    torch.manual_seed(seed)
    memristor_modes = _memristor_modes(n_modes, memristor_count)
    builder = ML.CircuitBuilder(n_modes=n_modes)
    if entangling_layers < 1:
        raise ValueError("entangling_layers must be at least 1.")

    builder.add_entangling_layer(trainable=True, name="U_pre")
    for index, mode in enumerate(memristor_modes):
        builder.add_memristive_ps(
            mode=mode,
            update_rule=_make_probability_update_rule(index),
            initial_state=0.25 + 0.05 * index,
            name=f"mem{index}_",
            num_backprop_steps=num_backprop_steps,
        )
    if entangling_layers >= 2:
        builder.add_entangling_layer(trainable=True, name="U_mid")
    builder.add_angle_encoding(
        modes=_input_modes(n_modes, input_size, memristor_modes),
        name="input",
    )
    if entangling_layers >= 3:
        builder.add_entangling_layer(trainable=True, name="U_post")

    input_state = [1 if mode < n_photons else 0 for mode in range(n_modes)]
    return ML.QuantumLayer(
        builder=builder,
        input_size=input_size,
        input_state=input_state,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        ),
        device=device,
        dtype=dtype,
    )


def make_inputs(config: WorkloadConfig, device: torch.device) -> list[torch.Tensor]:
    """Create one deterministic input sequence for a workload."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 1000)
    inputs = []
    for _ in range(config.timesteps):
        input_batch = torch.randn(
            config.batch_size,
            config.input_size,
            generator=generator,
        )
        inputs.append(input_batch.to(device=device).requires_grad_())
    return inputs


def rss_bytes() -> int | None:
    """Return current process RSS when psutil is available."""
    if psutil is None:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


class CallCounter:
    """Count calls into the layer and its computation process."""

    def __init__(self, layer: ML.QuantumLayer) -> None:
        self.layer = layer
        self.layer_output_calls = 0
        self.process_calls = dict.fromkeys(PROCESS_METHODS, 0)

    def install(self) -> None:
        """Wrap layer and process methods with counters."""
        if hasattr(self.layer, "_compute_layer_output"):
            original_layer_output = self.layer._compute_layer_output

            def counted_layer_output(*args: Any, **kwargs: Any) -> Any:
                self.layer_output_calls += 1
                return original_layer_output(*args, **kwargs)

            self.layer._compute_layer_output = counted_layer_output  # type: ignore[method-assign]

        process = self.layer.computation_process
        for method_name in PROCESS_METHODS:
            if not hasattr(process, method_name):
                continue
            original = getattr(process, method_name)

            def counted_process_method(
                *args: Any,
                _method_name: str = method_name,
                _original: Callable[..., Any] = original,
                **kwargs: Any,
            ) -> Any:
                self.process_calls[_method_name] += 1
                return _original(*args, **kwargs)

            setattr(process, method_name, counted_process_method)


def run_training_workload(api: str, config: WorkloadConfig) -> RunResult:
    """Run one measured forward/backward workload."""
    device = torch.device(config.device)
    num_backprop_steps = config.k if api == "blackbox" else None
    layer = build_memristive_layer(
        n_modes=config.n_modes,
        n_photons=config.n_photons,
        input_size=config.input_size,
        memristor_count=config.memristor_count,
        entangling_layers=config.entangling_layers,
        num_backprop_steps=num_backprop_steps,
        seed=config.seed,
        device=device,
    )
    layer = layer.to(device)
    layer.train()
    layer.reset(batch_size=config.batch_size)

    counter = CallCounter(layer)
    counter.install()

    inputs = make_inputs(config, device)
    rss_start = rss_bytes()
    rss_peak = rss_start

    def sample_rss() -> None:
        nonlocal rss_peak
        current = rss_bytes()
        if current is not None:
            rss_peak = current if rss_peak is None else max(rss_peak, current)

    synchronize_if_cuda(device)
    reset_cuda_memory_stats(device)
    cuda_memory_start = capture_cuda_memory(device)
    tracemalloc.start()
    start = time.perf_counter()
    output_checksum = 0.0
    loss_terms: list[torch.Tensor] = []

    for timestep, input_batch in enumerate(inputs):
        output = layer(input_batch)
        output_checksum += float(output.detach().sum().item())
        loss_terms.append(output[:, 0].mean())
        sample_rss()

        is_chunk_end = (timestep + 1) % config.chunk_len == 0
        is_last_step = timestep + 1 == config.timesteps
        if is_chunk_end or is_last_step:
            layer.zero_grad(set_to_none=True)
            loss = torch.stack(loss_terms).sum()
            loss.backward()
            sample_rss()
            layer.detach_memristive_state()
            loss_terms = []
            sample_rss()

    synchronize_if_cuda(device)
    seconds = time.perf_counter() - start
    cuda_memory = finish_cuda_memory_snapshot(device, cuda_memory_start)
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_end = rss_bytes()
    rss_peak_delta = (
        None if rss_start is None or rss_peak is None else rss_peak - rss_start
    )
    rss_end_delta = None if rss_start is None or rss_end is None else rss_end - rss_start

    grad_norms = [
        0.0 if input_batch.grad is None else float(input_batch.grad.abs().max().item())
        for input_batch in inputs
    ]
    retained_outputs = len(getattr(layer, "_memristive_gradient_outputs", []))
    history_lengths = [len(history) for history in layer.memristive_history]

    return RunResult(
        api=api,
        config=asdict(config),
        seconds=seconds,
        layer_output_calls=counter.layer_output_calls,
        process_calls=counter.process_calls,
        process_calls_total=sum(counter.process_calls.values()),
        rss_start_bytes=rss_start,
        rss_peak_bytes=rss_peak,
        rss_end_bytes=rss_end,
        rss_peak_delta_bytes=rss_peak_delta,
        rss_end_delta_bytes=rss_end_delta,
        tracemalloc_peak_bytes=tracemalloc_peak,
        **cuda_memory_fields(cuda_memory),
        retained_gradient_outputs=retained_outputs,
        history_lengths=history_lengths,
        output_checksum=output_checksum,
        input_grad_norm_sum=sum(grad_norms),
        input_grad_norms=grad_norms,
    )


def summarize(results: list[RunResult]) -> list[dict[str, Any]]:
    """Aggregate repeated measurements by API and workload."""
    grouped: dict[tuple[Any, ...], list[RunResult]] = {}
    for result in results:
        config = result.config
        key = (
            result.api,
            config["k"],
            config["chunk_len"],
            config["timesteps"],
            config["batch_size"],
            config["n_modes"],
            config["n_photons"],
            config["memristor_count"],
            config["entangling_layers"],
            config["device"],
        )
        grouped.setdefault(key, []).append(result)

    summary = []
    for key, group in grouped.items():
        (
            api,
            k,
            chunk_len,
            timesteps,
            batch_size,
            n_modes,
            n_photons,
            memristor_count,
            entangling_layers,
            device,
        ) = key
        summary.append({
            "api": api,
            "k": k,
            "chunk_len": chunk_len,
            "timesteps": timesteps,
            "batch_size": batch_size,
            "n_modes": n_modes,
            "n_photons": n_photons,
            "memristor_count": memristor_count,
            "entangling_layers": entangling_layers,
            "device": device,
            "seconds_mean": statistics.mean(result.seconds for result in group),
            "seconds_min": min(result.seconds for result in group),
            "layer_output_calls_mean": statistics.mean(
                result.layer_output_calls for result in group
            ),
            "process_calls_total_mean": statistics.mean(
                result.process_calls_total for result in group
            ),
            "rss_peak_bytes_max": max(
                (
                    result.rss_peak_bytes
                    for result in group
                    if result.rss_peak_bytes is not None
                ),
                default=None,
            ),
            "rss_peak_delta_bytes_max": max(
                (
                    result.rss_peak_delta_bytes
                    for result in group
                    if result.rss_peak_delta_bytes is not None
                ),
                default=None,
            ),
            "rss_end_delta_bytes_max": max(
                (
                    result.rss_end_delta_bytes
                    for result in group
                    if result.rss_end_delta_bytes is not None
                ),
                default=None,
            ),
            "tracemalloc_peak_bytes_max": max(
                result.tracemalloc_peak_bytes for result in group
            ),
            "cuda_memory_peak_allocated_bytes_max": max(
                (
                    result.cuda_memory_peak_allocated_bytes
                    for result in group
                    if result.cuda_memory_peak_allocated_bytes is not None
                ),
                default=None,
            ),
            "cuda_memory_peak_reserved_bytes_max": max(
                (
                    result.cuda_memory_peak_reserved_bytes
                    for result in group
                    if result.cuda_memory_peak_reserved_bytes is not None
                ),
                default=None,
            ),
            "retained_gradient_outputs_max": max(
                result.retained_gradient_outputs for result in group
            ),
            "output_checksum_mean": statistics.mean(
                result.output_checksum for result in group
            ),
            "input_grad_norm_sum_mean": statistics.mean(
                result.input_grad_norm_sum for result in group
            ),
        })

    return sorted(
        summary,
        key=lambda item: (
            item["k"],
            item["api"],
            item["timesteps"],
            item["batch_size"],
            item["n_modes"],
            item["n_photons"],
            item["memristor_count"],
            item["device"],
        ),
    )


def compare_apis(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare blackbox and manual results for matching workloads."""
    by_workload: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for item in summary:
        key = (
            item["k"],
            item["chunk_len"],
            item["timesteps"],
            item["batch_size"],
            item["n_modes"],
            item["n_photons"],
            item["memristor_count"],
            item["entangling_layers"],
            item["device"],
        )
        by_workload.setdefault(key, {})[item["api"]] = item

    comparisons = []
    for key, items in by_workload.items():
        if "blackbox" not in items or "manual" not in items:
            continue
        blackbox = items["blackbox"]
        manual = items["manual"]
        manual_seconds = manual["seconds_mean"]
        manual_layer_calls = manual["layer_output_calls_mean"]
        manual_process_calls = manual["process_calls_total_mean"]
        comparisons.append({
            "k": key[0],
            "chunk_len": key[1],
            "timesteps": key[2],
            "batch_size": key[3],
            "n_modes": key[4],
            "n_photons": key[5],
            "memristor_count": key[6],
            "entangling_layers": key[7],
            "device": key[8],
            "seconds_ratio_blackbox_over_manual": (
                blackbox["seconds_mean"] / manual_seconds
                if manual_seconds
                else None
            ),
            "layer_output_call_ratio_blackbox_over_manual": (
                blackbox["layer_output_calls_mean"] / manual_layer_calls
                if manual_layer_calls
                else None
            ),
            "process_call_ratio_blackbox_over_manual": (
                blackbox["process_calls_total_mean"] / manual_process_calls
                if manual_process_calls
                else None
            ),
            "output_checksum_abs_delta": abs(
                blackbox["output_checksum_mean"] - manual["output_checksum_mean"]
            ),
            "input_grad_norm_sum_abs_delta": abs(
                blackbox["input_grad_norm_sum_mean"]
                - manual["input_grad_norm_sum_mean"]
            ),
            "blackbox_rss_peak_bytes": blackbox["rss_peak_bytes_max"],
            "manual_rss_peak_bytes": manual["rss_peak_bytes_max"],
            "blackbox_rss_peak_delta_bytes": blackbox["rss_peak_delta_bytes_max"],
            "manual_rss_peak_delta_bytes": manual["rss_peak_delta_bytes_max"],
            "blackbox_tracemalloc_peak_bytes": blackbox[
                "tracemalloc_peak_bytes_max"
            ],
            "manual_tracemalloc_peak_bytes": manual["tracemalloc_peak_bytes_max"],
            "blackbox_cuda_memory_peak_allocated_bytes": blackbox[
                "cuda_memory_peak_allocated_bytes_max"
            ],
            "manual_cuda_memory_peak_allocated_bytes": manual[
                "cuda_memory_peak_allocated_bytes_max"
            ],
        })

    return sorted(
        comparisons,
        key=lambda item: (
            item["k"],
            item["timesteps"],
            item["batch_size"],
            item["n_modes"],
            item["n_photons"],
            item["memristor_count"],
            item["device"],
        ),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--timesteps", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--n-modes", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--n-photons", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--input-size", nargs="+", type=int, default=[2])
    parser.add_argument("--memristor-count", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--entangling-layers", nargs="+", type=int, default=[2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--chunk-rule",
        choices=("k", "k_plus_one"),
        default="k",
        help="Use chunk_len=k or chunk_len=k+1 for the manual TBPTT boundary.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for layers and benchmark tensors: cpu, cuda, cuda:0, or auto.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    return parser.parse_args()


def _format_megabytes(value: int | None) -> str:
    """Format an optional byte count as a right-aligned megabyte string."""
    return "       n/a" if value is None else f"{value / 1_000_000:10.1f}"


def print_summary(summary: list[dict[str, Any]]) -> None:
    """Print a compact human-readable summary table."""
    header = (
        "api      m  p  mem  k  chunk  T    batch  sec_mean  layer_calls  "
        "process_calls  rss_delta_mb  py_peak_mb  cuda_peak_mb"
    )
    print(header)
    print("-" * len(header))
    for item in summary:
        print(
            f"{item['api']:<8} "
            f"{item['n_modes']:>2} "
            f"{item['n_photons']:>2} "
            f"{item['memristor_count']:>4} "
            f"{item['k']:>2} "
            f"{item['chunk_len']:>6} "
            f"{item['timesteps']:>4} "
            f"{item['batch_size']:>6} "
            f"{item['seconds_mean']:>9.4f} "
            f"{item['layer_output_calls_mean']:>11.1f} "
            f"{item['process_calls_total_mean']:>13.1f} "
            f"{_format_megabytes(item['rss_peak_delta_bytes_max'])} "
            f"{item['tracemalloc_peak_bytes_max'] / 1_000_000:>10.1f} "
            f"{format_megabytes(item['cuda_memory_peak_allocated_bytes_max']):>12}"
        )


def print_comparisons(comparisons: list[dict[str, Any]]) -> None:
    """Print blackbox/manual ratios for matching workloads."""
    if not comparisons:
        return

    print()
    header = (
        "m  p  mem  k  chunk  T    batch  time_ratio  layer_call_ratio  "
        "process_call_ratio  output_delta  grad_delta"
    )
    print(header)
    print("-" * len(header))
    for item in comparisons:
        print(
            f"{item['n_modes']:>2} "
            f"{item['n_photons']:>2} "
            f"{item['memristor_count']:>4} "
            f"{item['k']:>2} "
            f"{item['chunk_len']:>6} "
            f"{item['timesteps']:>4} "
            f"{item['batch_size']:>6} "
            f"{item['seconds_ratio_blackbox_over_manual']:>10.3f} "
            f"{item['layer_output_call_ratio_blackbox_over_manual']:>17.3f} "
            f"{item['process_call_ratio_blackbox_over_manual']:>19.3f} "
            f"{item['output_checksum_abs_delta']:>12.3e} "
            f"{item['input_grad_norm_sum_abs_delta']:>10.3e}"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compact_group_label(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return a compact label for a scaling plot line."""
    labels = {
        "api": "api",
        "k": "k",
        "timesteps": "T",
        "batch_size": "B",
        "n_modes": "m",
        "n_photons": "p",
        "memristor_count": "mem",
        "entangling_layers": "L",
    }
    return "/".join(f"{labels.get(key, key)}{item[key]}" for key in keys)


def _plot_scaling_lines(
    *,
    plt: Any,
    plot_dir: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    group_keys: tuple[str, ...],
    filename: str,
    title: str,
    ylabel: str,
    baseline: float | None = None,
) -> None:
    """Plot one metric against a scaling variable for fixed conditions."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get(x_key) is None or row.get(y_key) is None:
            continue
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    series = [
        sorted(group, key=lambda item: item[x_key])
        for group in grouped.values()
        if len({item[x_key] for item in group}) > 1
    ]
    if not series:
        return

    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for group in series:
        x_values = [item[x_key] for item in group]
        y_values = [item[y_key] for item in group]
        axis.plot(
            x_values,
            y_values,
            marker="o",
            label=_compact_group_label(group[0], group_keys),
        )
    if baseline is not None:
        axis.axhline(baseline, color="black", linewidth=1)
    axis.set_xlabel(_compact_group_label({x_key: ""}, (x_key,)).rstrip())
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / filename, dpi=160)
    plt.close(fig)


def plot_figures(
    plot_dir: Path,
    summary: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    """Write benchmark figures when matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping figures.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        (
            f"m{item['n_modes']}/p{item['n_photons']}/mem{item['memristor_count']}"
            f"/k{item['k']}/T{item['timesteps']}/B{item['batch_size']}"
        )
        for item in comparisons
    ]

    def save_ratio_plot(metric: str, filename: str, ylabel: str) -> None:
        if not comparisons:
            return
        values = [item[metric] for item in comparisons]
        fig, axis = plt.subplots(figsize=(max(8, len(values) * 0.8), 4.5))
        axis.bar(range(len(values)), values)
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_title(ylabel)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    save_ratio_plot(
        "seconds_ratio_blackbox_over_manual",
        "time_ratio_blackbox_over_manual.png",
        "Blackbox / manual wall time",
    )
    save_ratio_plot(
        "process_call_ratio_blackbox_over_manual",
        "process_call_ratio_blackbox_over_manual.png",
        "Blackbox / manual process calls",
    )

    _plot_scaling_lines(
        plt=plt,
        plot_dir=plot_dir,
        rows=comparisons,
        x_key="k",
        y_key="seconds_ratio_blackbox_over_manual",
        group_keys=(
            "timesteps",
            "batch_size",
            "n_modes",
            "n_photons",
            "memristor_count",
        ),
        filename="scaling_time_ratio_vs_k.png",
        title="Wall-time ratio as TBPTT window grows",
        ylabel="Blackbox / manual wall time",
        baseline=1.0,
    )
    _plot_scaling_lines(
        plt=plt,
        plot_dir=plot_dir,
        rows=comparisons,
        x_key="k",
        y_key="process_call_ratio_blackbox_over_manual",
        group_keys=(
            "timesteps",
            "batch_size",
            "n_modes",
            "n_photons",
            "memristor_count",
        ),
        filename="scaling_process_call_ratio_vs_k.png",
        title="Process-call ratio as TBPTT window grows",
        ylabel="Blackbox / manual process calls",
        baseline=1.0,
    )
    _plot_scaling_lines(
        plt=plt,
        plot_dir=plot_dir,
        rows=comparisons,
        x_key="n_modes",
        y_key="seconds_ratio_blackbox_over_manual",
        group_keys=("k", "timesteps", "batch_size", "n_photons", "memristor_count"),
        filename="scaling_time_ratio_vs_modes.png",
        title="Wall-time ratio as mode count grows",
        ylabel="Blackbox / manual wall time",
        baseline=1.0,
    )
    _plot_scaling_lines(
        plt=plt,
        plot_dir=plot_dir,
        rows=comparisons,
        x_key="n_modes",
        y_key="process_call_ratio_blackbox_over_manual",
        group_keys=("k", "timesteps", "batch_size", "n_photons", "memristor_count"),
        filename="scaling_process_call_ratio_vs_modes.png",
        title="Process-call ratio as mode count grows",
        ylabel="Blackbox / manual process calls",
        baseline=1.0,
    )
    _plot_scaling_lines(
        plt=plt,
        plot_dir=plot_dir,
        rows=comparisons,
        x_key="n_photons",
        y_key="seconds_ratio_blackbox_over_manual",
        group_keys=("k", "timesteps", "batch_size", "n_modes", "memristor_count"),
        filename="scaling_time_ratio_vs_photons.png",
        title="Wall-time ratio as photon count grows",
        ylabel="Blackbox / manual wall time",
        baseline=1.0,
    )
    _plot_scaling_lines(
        plt=plt,
        plot_dir=plot_dir,
        rows=comparisons,
        x_key="n_photons",
        y_key="process_call_ratio_blackbox_over_manual",
        group_keys=("k", "timesteps", "batch_size", "n_modes", "memristor_count"),
        filename="scaling_process_call_ratio_vs_photons.png",
        title="Process-call ratio as photon count grows",
        ylabel="Blackbox / manual process calls",
        baseline=1.0,
    )

    if summary:
        summary_labels = [
            (
                f"{item['api']}:m{item['n_modes']}/p{item['n_photons']}"
                f"/mem{item['memristor_count']}/k{item['k']}"
                f"/T{item['timesteps']}/B{item['batch_size']}"
            )
            for item in summary
        ]
        seconds = [item["seconds_mean"] for item in summary]
        calls = [item["process_calls_total_mean"] for item in summary]
        rss_delta = [
            0.0
            if item["rss_peak_delta_bytes_max"] is None
            else item["rss_peak_delta_bytes_max"] / 1_000_000
            for item in summary
        ]
        py_peak = [
            item["tracemalloc_peak_bytes_max"] / 1_000_000 for item in summary
        ]

        for values, filename, ylabel in (
            (seconds, "wall_time_by_api.png", "Wall time (s)"),
            (calls, "process_calls_by_api.png", "Process calls"),
            (rss_delta, "rss_peak_delta_by_api.png", "Peak RSS delta (MB)"),
            (py_peak, "python_alloc_peak_by_api.png", "Python allocation peak (MB)"),
        ):
            fig, axis = plt.subplots(figsize=(max(10, len(values) * 0.5), 4.8))
            axis.bar(range(len(values)), values)
            axis.set_ylabel(ylabel)
            axis.set_xticks(range(len(values)))
            axis.set_xticklabels(summary_labels, rotation=45, ha="right")
            axis.set_title(ylabel)
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=160)
            plt.close(fig)

        _plot_scaling_lines(
            plt=plt,
            plot_dir=plot_dir,
            rows=summary,
            x_key="k",
            y_key="seconds_mean",
            group_keys=(
                "api",
                "timesteps",
                "batch_size",
                "n_modes",
                "n_photons",
                "memristor_count",
            ),
            filename="scaling_wall_time_vs_k_by_api.png",
            title="Absolute wall time as TBPTT window grows",
            ylabel="Wall time (s)",
        )


def iter_workload_configs(args: argparse.Namespace) -> list[WorkloadConfig]:
    """Build valid workload configurations from CLI sweep arguments."""
    configs = []
    for k in args.k:
        if k < 1:
            raise ValueError("This comparison expects k >= 1.")
        chunk_len = k if args.chunk_rule == "k" else k + 1
        for n_modes in args.n_modes:
            for n_photons in args.n_photons:
                if n_photons > n_modes:
                    print(f"Skipping invalid n_modes={n_modes}, n_photons={n_photons}.")
                    continue
                for input_size in args.input_size:
                    for memristor_count in args.memristor_count:
                        if input_size + memristor_count > n_modes:
                            print(
                                "Skipping invalid workload "
                                f"n_modes={n_modes}, input_size={input_size}, "
                                f"memristor_count={memristor_count}."
                            )
                            continue
                        for entangling_layers in args.entangling_layers:
                            for timesteps in args.timesteps:
                                for batch_size in args.batch_size:
                                    configs.append(
                                        WorkloadConfig(
                                            n_modes=n_modes,
                                            n_photons=n_photons,
                                            input_size=input_size,
                                            memristor_count=memristor_count,
                                            entangling_layers=entangling_layers,
                                            batch_size=batch_size,
                                            timesteps=timesteps,
                                            k=k,
                                            chunk_len=chunk_len,
                                            seed=args.seed,
                                            device=args.device,
                                        )
                                    )
    return configs


def main() -> None:
    """Run the benchmark comparison."""
    args = parse_args()
    device = resolve_benchmark_device(args.device)
    args.device = str(device)
    all_results: list[RunResult] = []

    configs = iter_workload_configs(args)
    for config in configs:
        for _ in range(args.warmups):
            run_training_workload("blackbox", config)
            run_training_workload("manual", config)
            gc.collect()
            synchronize_if_cuda(device)

        for repeat in range(args.repeats):
            repeat_config = WorkloadConfig(
                **{**asdict(config), "seed": args.seed + repeat}
            )
            all_results.append(run_training_workload("blackbox", repeat_config))
            gc.collect()
            synchronize_if_cuda(device)
            all_results.append(run_training_workload("manual", repeat_config))
            gc.collect()
            synchronize_if_cuda(device)

    summary = summarize(all_results)
    comparisons = compare_apis(summary)
    print_summary(summary)
    print_comparisons(comparisons)

    payload = {
        "summary": summary,
        "comparisons": comparisons,
        "runs": [asdict(result) for result in all_results],
        "notes": {
            "blackbox": "num_backprop_steps=k, automatic bounded memristive window",
            "manual": "num_backprop_steps=None, user calls detach_memristive_state() at chunk boundaries",
            "rss": "RSS requires optional psutil; tracemalloc captures Python allocations, not all PyTorch native allocations.",
        },
        "environment": benchmark_environment(device),
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.csv_output is not None:
        write_csv(args.csv_output.with_suffix(".summary.csv"), summary)
        write_csv(args.csv_output.with_suffix(".comparisons.csv"), comparisons)
        write_csv(
            args.csv_output.with_suffix(".runs.csv"),
            [asdict(result) for result in all_results],
        )
    if args.plot_dir is not None:
        plot_figures(args.plot_dir, summary, comparisons)


if __name__ == "__main__":
    main()
