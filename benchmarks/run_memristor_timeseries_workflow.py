"""Benchmark memristor TBPTT APIs on a time-series forecasting task.

The benchmark trains a small memristive quantum recurrent model to predict the
next value of a synthetic nonlinear delayed time-series. It compares practical
training behavior for:

* ``reservoir``: ``num_backprop_steps=0``;
* ``manual``: ``num_backprop_steps=None`` plus explicit TBPTT detaches;
* ``blackbox``: ``num_backprop_steps=k``.

The task is synthetic so the benchmark is reproducible and does not depend on a
network download, but it is application-shaped: train data, validation data,
optimizer steps, validation loss, raw seconds, call counts, and prediction
figures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from benchmarks.memristor_benchmark_utils import (
    benchmark_environment,
    capture_cuda_memory,
    cuda_memory_fields,
    finish_cuda_memory_snapshot,
    format_megabytes,
    move_batches_to_device,
    reset_cuda_memory_stats,
    resolve_benchmark_device,
    synchronize_if_cuda,
)
from benchmarks.run_memristor_tbptt_api_comparison import (
    CallCounter,
    build_memristive_layer,
    rss_bytes,
)


@dataclass(frozen=True)
class TimeSeriesConfig:
    """Configuration for one time-series benchmark workload."""

    n_modes: int
    n_photons: int
    input_size: int
    memristor_count: int
    entangling_layers: int
    batch_size: int
    timesteps: int
    n_batches: int
    val_batches: int
    epochs: int
    k: int
    chunk_len: int
    hidden_size: int
    learning_rate: float
    seed: int
    device: str


@dataclass
class TimeSeriesResult:
    """Measured result for one API and time-series workload."""

    api: str
    config: dict[str, Any]
    seconds: float
    train_seconds: float
    validation_seconds: float
    layer_output_calls: int
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
    optimizer_steps: int
    train_epoch_losses: list[float]
    validation_epoch_losses: list[float]
    final_train_loss: float
    final_validation_loss: float
    trainable_parameter_count: int


@dataclass
class PredictionSnapshot:
    """Prediction data for one trained model on the first validation sequence."""

    api: str
    config: dict[str, Any]
    target: list[float]
    prediction: list[float]


class TimeSeriesMemristorModel(nn.Module):
    """Small memristive model used for time-series forecasting."""

    def __init__(
        self, api: str, config: TimeSeriesConfig, initialization_seed: int
    ) -> None:
        super().__init__()
        if api == "reservoir":
            num_backprop_steps: int | None = 0
        elif api == "manual":
            num_backprop_steps = None
        elif api == "blackbox":
            num_backprop_steps = config.k
        else:
            raise ValueError(f"Unknown API {api!r}.")

        self.quantum = build_memristive_layer(
            n_modes=config.n_modes,
            n_photons=config.n_photons,
            input_size=config.input_size,
            memristor_count=config.memristor_count,
            entangling_layers=config.entangling_layers,
            num_backprop_steps=num_backprop_steps,
            seed=initialization_seed,
            device=torch.device(config.device),
        )
        torch.manual_seed(initialization_seed + 1)
        self.readout = nn.Sequential(
            nn.Linear(self.quantum.output_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1),
        )

    def reset_sequence(self, batch_size: int) -> None:
        """Reset the memristive state for a new batch of sequences."""
        self.quantum.reset(batch_size=batch_size)

    def detach_sequence_state(self) -> None:
        """Detach the recurrent memristive graph without changing its value."""
        self.quantum.detach_memristive_state()

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """Predict one next-step value from one sequence timestep."""
        return self.readout(self.quantum(x))


def _base_time_series(
    n_sequences: int, timesteps: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a nonlinear delayed time-series and an exogenous seasonal term."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    total_steps = timesteps + 8
    phase = 2.0 * math.pi * torch.rand(n_sequences, 1, generator=generator)
    values = 0.15 * torch.randn(n_sequences, total_steps, generator=generator)
    seasonal = torch.zeros(n_sequences, total_steps)

    for t in range(4, total_steps):
        seasonal_t = (
            0.45 * torch.sin(torch.tensor(0.37 * t) + phase[:, 0])
            + 0.20 * torch.sin(torch.tensor(0.11 * t) + 0.5 * phase[:, 0])
        )
        noise = 0.03 * torch.randn(n_sequences, generator=generator)
        previous = values[:, t - 1]
        delayed = values[:, t - 4]
        nonlinear_memory = 0.22 * delayed * (1.0 - previous.square())
        values[:, t] = 0.58 * previous + nonlinear_memory + seasonal_t + noise
        seasonal[:, t] = seasonal_t

    values = values[:, 4:]
    seasonal = seasonal[:, 4:]
    values = (values - values.mean()) / values.std().clamp_min(1e-6)
    seasonal = seasonal / seasonal.std().clamp_min(1e-6)
    return values, seasonal


def make_timeseries_batches(
    config: TimeSeriesConfig, *, n_batches: int, seed_offset: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Create fixed mini-batches for next-step forecasting."""
    n_sequences = n_batches * config.batch_size
    values, seasonal = _base_time_series(
        n_sequences=n_sequences,
        timesteps=config.timesteps + 1,
        seed=config.seed + seed_offset,
    )

    feature_columns = [values[:, : config.timesteps], seasonal[:, : config.timesteps]]
    if config.input_size >= 3:
        deltas = torch.zeros_like(values[:, : config.timesteps])
        deltas[:, 1:] = values[:, 1 : config.timesteps] - values[:, : config.timesteps - 1]
        feature_columns.append(deltas)
    if config.input_size >= 4:
        time = torch.linspace(-1.0, 1.0, config.timesteps).expand(n_sequences, -1)
        feature_columns.append(time)
    while len(feature_columns) < config.input_size:
        feature_columns.append(torch.zeros_like(feature_columns[0]))

    features = torch.stack(feature_columns[: config.input_size], dim=-1)
    targets = values[:, 1 : config.timesteps + 1].unsqueeze(-1)

    batches = []
    target_batches = []
    for batch_index in range(n_batches):
        start = batch_index * config.batch_size
        end = start + config.batch_size
        batches.append(features[start:end].contiguous())
        target_batches.append(targets[start:end].contiguous())
    return batches, target_batches


def _sample_rss(rss_peak: int | None) -> int | None:
    """Update a running RSS peak from the current process RSS."""
    current = rss_bytes()
    if current is None:
        return rss_peak
    return current if rss_peak is None else max(rss_peak, current)


def evaluate_model(
    model: TimeSeriesMemristorModel,
    batches: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> tuple[float, PredictionSnapshot | None]:
    """Evaluate validation loss and capture one prediction sequence."""
    criterion = nn.MSELoss()
    model.eval()
    losses = []
    snapshot: PredictionSnapshot | None = None

    with torch.no_grad():
        for batch, target in zip(batches, targets, strict=True):
            model.reset_sequence(batch_size=batch.shape[0])
            predictions = []
            for timestep in range(batch.shape[1]):
                prediction = model.forward_step(batch[:, timestep, :])
                predictions.append(prediction)
                losses.append(float(criterion(prediction, target[:, timestep, :]).item()))
            if snapshot is None:
                prediction_tensor = torch.stack(predictions, dim=1)
                snapshot = PredictionSnapshot(
                    api="",
                    config={},
                    target=target[0, :, 0].detach().cpu().tolist(),
                    prediction=prediction_tensor[0, :, 0].detach().cpu().tolist(),
                )
    model.train()
    return statistics.mean(losses), snapshot


def train_one_model(
    api: str,
    config: TimeSeriesConfig,
    train_batches: list[torch.Tensor],
    train_targets: list[torch.Tensor],
    val_batches: list[torch.Tensor],
    val_targets: list[torch.Tensor],
) -> tuple[TimeSeriesResult, PredictionSnapshot]:
    """Train one API variant on the time-series task."""
    device = torch.device(config.device)
    model = TimeSeriesMemristorModel(
        api, config, initialization_seed=config.seed + 30_000
    ).to(device)
    model.train()
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    counter = CallCounter(model.quantum)
    counter.install()

    rss_start = rss_bytes()
    rss_peak = rss_start
    synchronize_if_cuda(device)
    reset_cuda_memory_stats(device)
    cuda_memory_start = capture_cuda_memory(device)
    tracemalloc.start()
    total_start = time.perf_counter()
    train_seconds = 0.0
    validation_seconds = 0.0
    optimizer_steps = 0
    train_epoch_losses = []
    validation_epoch_losses = []
    final_snapshot: PredictionSnapshot | None = None

    for _ in range(config.epochs):
        synchronize_if_cuda(device)
        train_start = time.perf_counter()
        epoch_losses = []

        for batch, target in zip(train_batches, train_targets, strict=True):
            model.reset_sequence(batch_size=batch.shape[0])
            optimizer.zero_grad(set_to_none=True)
            loss_terms: list[torch.Tensor] = []

            for timestep in range(config.timesteps):
                prediction = model.forward_step(batch[:, timestep, :])
                loss_terms.append(criterion(prediction, target[:, timestep, :]))
                rss_peak = _sample_rss(rss_peak)

                is_chunk_end = (timestep + 1) % config.chunk_len == 0
                is_last_step = timestep + 1 == config.timesteps
                if is_chunk_end or is_last_step:
                    chunk_loss = torch.stack(loss_terms).mean()
                    epoch_losses.append(float(chunk_loss.detach().item()))
                    chunk_loss.backward()
                    model.detach_sequence_state()
                    loss_terms = []
                    rss_peak = _sample_rss(rss_peak)

            optimizer.step()
            optimizer_steps += 1
            rss_peak = _sample_rss(rss_peak)

        synchronize_if_cuda(device)
        train_seconds += time.perf_counter() - train_start
        train_epoch_losses.append(statistics.mean(epoch_losses))

        synchronize_if_cuda(device)
        validation_start = time.perf_counter()
        validation_loss, snapshot = evaluate_model(model, val_batches, val_targets)
        synchronize_if_cuda(device)
        validation_seconds += time.perf_counter() - validation_start
        validation_epoch_losses.append(validation_loss)
        final_snapshot = snapshot
        rss_peak = _sample_rss(rss_peak)

    synchronize_if_cuda(device)
    seconds = time.perf_counter() - total_start
    cuda_memory = finish_cuda_memory_snapshot(device, cuda_memory_start)
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_end = rss_bytes()

    if final_snapshot is None:
        raise RuntimeError("Validation did not produce a prediction snapshot.")
    final_snapshot.api = api
    final_snapshot.config = asdict(config)

    result = TimeSeriesResult(
        api=api,
        config=asdict(config),
        seconds=seconds,
        train_seconds=train_seconds,
        validation_seconds=validation_seconds,
        layer_output_calls=counter.layer_output_calls,
        process_calls_total=sum(counter.process_calls.values()),
        rss_start_bytes=rss_start,
        rss_peak_bytes=rss_peak,
        rss_end_bytes=rss_end,
        rss_peak_delta_bytes=(
            None if rss_start is None or rss_peak is None else rss_peak - rss_start
        ),
        rss_end_delta_bytes=(
            None if rss_start is None or rss_end is None else rss_end - rss_start
        ),
        tracemalloc_peak_bytes=tracemalloc_peak,
        **cuda_memory_fields(cuda_memory),
        optimizer_steps=optimizer_steps,
        train_epoch_losses=train_epoch_losses,
        validation_epoch_losses=validation_epoch_losses,
        final_train_loss=train_epoch_losses[-1],
        final_validation_loss=validation_epoch_losses[-1],
        trainable_parameter_count=trainable_parameter_count,
    )
    return result, final_snapshot


def summarize(results: list[TimeSeriesResult]) -> list[dict[str, Any]]:
    """Aggregate repeated measurements by API and workload."""
    grouped: dict[tuple[Any, ...], list[TimeSeriesResult]] = {}
    for result in results:
        config = result.config
        key = (
            result.api,
            config["k"],
            config["chunk_len"],
            config["timesteps"],
            config["batch_size"],
            config["n_batches"],
            config["val_batches"],
            config["epochs"],
            config["n_modes"],
            config["n_photons"],
            config["memristor_count"],
            config["device"],
        )
        grouped.setdefault(key, []).append(result)

    rows = []
    for key, group in grouped.items():
        (
            api,
            k,
            chunk_len,
            timesteps,
            batch_size,
            n_batches,
            val_batches,
            epochs,
            n_modes,
            n_photons,
            memristor_count,
            device,
        ) = key
        rows.append({
            "api": api,
            "k": k,
            "chunk_len": chunk_len,
            "timesteps": timesteps,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "val_batches": val_batches,
            "epochs": epochs,
            "n_modes": n_modes,
            "n_photons": n_photons,
            "memristor_count": memristor_count,
            "device": device,
            "seconds_mean": statistics.mean(result.seconds for result in group),
            "train_seconds_mean": statistics.mean(
                result.train_seconds for result in group
            ),
            "validation_seconds_mean": statistics.mean(
                result.validation_seconds for result in group
            ),
            "process_calls_total_mean": statistics.mean(
                result.process_calls_total for result in group
            ),
            "rss_peak_delta_bytes_max": max(
                (
                    result.rss_peak_delta_bytes
                    for result in group
                    if result.rss_peak_delta_bytes is not None
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
            "optimizer_steps_mean": statistics.mean(
                result.optimizer_steps for result in group
            ),
            "final_train_loss_mean": statistics.mean(
                result.final_train_loss for result in group
            ),
            "final_validation_loss_mean": statistics.mean(
                result.final_validation_loss for result in group
            ),
            "trainable_parameter_count": group[0].trainable_parameter_count,
        })

    return sorted(
        rows,
        key=lambda item: (
            item["k"],
            item["api"],
            item["n_modes"],
            item["n_photons"],
            item["memristor_count"],
            item["device"],
        ),
    )


def compare_apis(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare manual, blackbox, and reservoir results for each workload."""
    by_workload: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for item in summary:
        key = (
            item["k"],
            item["chunk_len"],
            item["timesteps"],
            item["batch_size"],
            item["n_batches"],
            item["val_batches"],
            item["epochs"],
            item["n_modes"],
            item["n_photons"],
            item["memristor_count"],
            item["device"],
        )
        by_workload.setdefault(key, {})[item["api"]] = item

    rows = []
    for key, items in by_workload.items():
        if "manual" not in items:
            continue
        manual = items["manual"]
        blackbox = items.get("blackbox")
        reservoir = items.get("reservoir")
        row: dict[str, Any] = {
            "k": key[0],
            "chunk_len": key[1],
            "timesteps": key[2],
            "batch_size": key[3],
            "n_batches": key[4],
            "val_batches": key[5],
            "epochs": key[6],
            "n_modes": key[7],
            "n_photons": key[8],
            "memristor_count": key[9],
            "device": key[10],
            "manual_seconds_mean": manual["seconds_mean"],
            "manual_final_validation_loss": manual["final_validation_loss_mean"],
        }
        if blackbox is not None:
            row.update({
                "blackbox_seconds_mean": blackbox["seconds_mean"],
                "blackbox_final_validation_loss": blackbox[
                    "final_validation_loss_mean"
                ],
                "seconds_ratio_blackbox_over_manual": (
                    blackbox["seconds_mean"] / manual["seconds_mean"]
                ),
                "process_call_ratio_blackbox_over_manual": (
                    blackbox["process_calls_total_mean"]
                    / manual["process_calls_total_mean"]
                ),
                "validation_loss_delta_blackbox_minus_manual": (
                    blackbox["final_validation_loss_mean"]
                    - manual["final_validation_loss_mean"]
                ),
                "blackbox_cuda_memory_peak_allocated_bytes": blackbox[
                    "cuda_memory_peak_allocated_bytes_max"
                ],
                "manual_cuda_memory_peak_allocated_bytes": manual[
                    "cuda_memory_peak_allocated_bytes_max"
                ],
            })
        if reservoir is not None:
            row.update({
                "reservoir_seconds_mean": reservoir["seconds_mean"],
                "reservoir_final_validation_loss": reservoir[
                    "final_validation_loss_mean"
                ],
                "seconds_ratio_reservoir_over_manual": (
                    reservoir["seconds_mean"] / manual["seconds_mean"]
                ),
                "validation_loss_delta_reservoir_minus_manual": (
                    reservoir["final_validation_loss_mean"]
                    - manual["final_validation_loss_mean"]
                ),
            })
        rows.append(row)

    return sorted(
        rows,
        key=lambda item: (
            item["k"],
            item["n_modes"],
            item["n_photons"],
            item["memristor_count"],
            item["device"],
        ),
    )


def _format_megabytes(value: int | None) -> str:
    """Format an optional byte count as megabytes."""
    return "n/a" if value is None else f"{value / 1_000_000:.1f}"


def print_summary(summary: list[dict[str, Any]]) -> None:
    """Print one compact row per API/workload."""
    header = (
        "api        m  p  mem  k  T  batch  epochs  sec_mean  proc_calls  "
        "train_loss  val_loss  rss_mb  py_mb  cuda_mb"
    )
    print(header)
    print("-" * len(header))
    for item in summary:
        print(
            f"{item['api']:<10} "
            f"{item['n_modes']:>2} "
            f"{item['n_photons']:>2} "
            f"{item['memristor_count']:>4} "
            f"{item['k']:>2} "
            f"{item['timesteps']:>2} "
            f"{item['batch_size']:>6} "
            f"{item['epochs']:>7} "
            f"{item['seconds_mean']:>9.3f} "
            f"{item['process_calls_total_mean']:>10.1f} "
            f"{item['final_train_loss_mean']:>10.5f} "
            f"{item['final_validation_loss_mean']:>8.5f} "
            f"{_format_megabytes(item['rss_peak_delta_bytes_max']):>6} "
            f"{item['tracemalloc_peak_bytes_max'] / 1_000_000:>6.1f} "
            f"{format_megabytes(item['cuda_memory_peak_allocated_bytes_max']):>8}"
        )


def print_comparisons(comparisons: list[dict[str, Any]]) -> None:
    """Print ratios against manual TBPTT."""
    if not comparisons:
        return
    header = (
        "m  p  mem  k  T  blackbox_time  blackbox_calls  "
        "blackbox_val_delta  reservoir_time  reservoir_val_delta"
    )
    print()
    print(header)
    print("-" * len(header))
    for item in comparisons:
        print(
            f"{item['n_modes']:>2} "
            f"{item['n_photons']:>2} "
            f"{item['memristor_count']:>4} "
            f"{item['k']:>2} "
            f"{item['timesteps']:>2} "
            f"{item.get('seconds_ratio_blackbox_over_manual', float('nan')):>13.3f} "
            f"{item.get('process_call_ratio_blackbox_over_manual', float('nan')):>14.3f} "
            f"{item.get('validation_loss_delta_blackbox_minus_manual', float('nan')):>19.3e} "
            f"{item.get('seconds_ratio_reservoir_over_manual', float('nan')):>14.3f} "
            f"{item.get('validation_loss_delta_reservoir_minus_manual', float('nan')):>19.3e}"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_figures(
    plot_dir: Path,
    summary: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    snapshots: list[PredictionSnapshot],
) -> None:
    """Write application benchmark figures when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping figures.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    if comparisons:
        labels = [
            (
                f"m{item['n_modes']}/p{item['n_photons']}/mem{item['memristor_count']}"
                f"/k{item['k']}"
            )
            for item in comparisons
        ]
        for metric, filename, ylabel in (
            (
                "seconds_ratio_blackbox_over_manual",
                "timeseries_time_ratio_blackbox_over_manual.png",
                "Blackbox / manual seconds",
            ),
            (
                "process_call_ratio_blackbox_over_manual",
                "timeseries_process_call_ratio_blackbox_over_manual.png",
                "Blackbox / manual process calls",
            ),
            (
                "validation_loss_delta_blackbox_minus_manual",
                "timeseries_validation_loss_delta_blackbox_minus_manual.png",
                "Blackbox - manual validation loss",
            ),
        ):
            values = [item.get(metric) for item in comparisons]
            if any(value is None for value in values):
                continue
            fig, axis = plt.subplots(figsize=(max(8, len(values) * 0.9), 4.5))
            axis.bar(range(len(values)), values)
            if "ratio" in metric:
                axis.axhline(1.0, color="black", linewidth=1)
            else:
                axis.axhline(0.0, color="black", linewidth=1)
            axis.set_ylabel(ylabel)
            axis.set_xticks(range(len(values)))
            axis.set_xticklabels(labels, rotation=45, ha="right")
            axis.set_title(ylabel)
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=160)
            plt.close(fig)

    if summary:
        labels = [
            (
                f"{item['api']}:m{item['n_modes']}/p{item['n_photons']}"
                f"/mem{item['memristor_count']}/k{item['k']}"
            )
            for item in summary
        ]
        for metric, filename, ylabel in (
            ("seconds_mean", "timeseries_seconds_by_api.png", "Total seconds"),
            (
                "final_validation_loss_mean",
                "timeseries_validation_loss_by_api.png",
                "Final validation loss",
            ),
        ):
            values = [item[metric] for item in summary]
            fig, axis = plt.subplots(figsize=(max(10, len(values) * 0.45), 4.8))
            axis.bar(range(len(values)), values)
            axis.set_ylabel(ylabel)
            axis.set_xticks(range(len(values)))
            axis.set_xticklabels(labels, rotation=45, ha="right")
            axis.set_title(ylabel)
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=160)
            plt.close(fig)

    first_config_key = None
    selected = []
    for snapshot in snapshots:
        config = snapshot.config
        key = (
            config["k"],
            config["n_modes"],
            config["n_photons"],
            config["memristor_count"],
        )
        if first_config_key is None:
            first_config_key = key
        if key == first_config_key:
            selected.append(snapshot)

    if selected:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(selected[0].target, label="target", color="black", linewidth=2)
        for snapshot in selected:
            axis.plot(snapshot.prediction, label=snapshot.api)
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Next value")
        axis.set_title("Validation sequence prediction")
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "timeseries_prediction_example.png", dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apis",
        nargs="+",
        choices=("reservoir", "manual", "blackbox"),
        default=["reservoir", "manual", "blackbox"],
    )
    parser.add_argument("--k", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--timesteps", nargs="+", type=int, default=[12])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[8])
    parser.add_argument("--n-batches", nargs="+", type=int, default=[4])
    parser.add_argument("--val-batches", nargs="+", type=int, default=[2])
    parser.add_argument("--epochs", nargs="+", type=int, default=[4])
    parser.add_argument("--n-modes", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--n-photons", nargs="+", type=int, default=[3])
    parser.add_argument("--input-size", nargs="+", type=int, default=[2])
    parser.add_argument("--memristor-count", nargs="+", type=int, default=[1])
    parser.add_argument("--entangling-layers", nargs="+", type=int, default=[2])
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for models and benchmark tensors: cpu, cuda, cuda:0, or auto.",
    )
    parser.add_argument(
        "--chunk-rule",
        choices=("k", "k_plus_one"),
        default="k",
        help="Use chunk_len=k or chunk_len=k+1 for manual TBPTT boundaries.",
    )
    parser.add_argument("--seed", type=int, default=9876)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    return parser.parse_args()


def iter_configs(args: argparse.Namespace) -> list[TimeSeriesConfig]:
    """Build valid workload configurations from CLI sweep arguments."""
    configs = []
    for k in args.k:
        if k < 1:
            raise ValueError("This benchmark expects positive k values.")
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
                                    for n_batches in args.n_batches:
                                        for val_batches in args.val_batches:
                                            for epochs in args.epochs:
                                                configs.append(
                                                    TimeSeriesConfig(
                                                        n_modes=n_modes,
                                                        n_photons=n_photons,
                                                        input_size=input_size,
                                                        memristor_count=memristor_count,
                                                        entangling_layers=entangling_layers,
                                                        batch_size=batch_size,
                                                        timesteps=timesteps,
                                                        n_batches=n_batches,
                                                        val_batches=val_batches,
                                                        epochs=epochs,
                                                        k=k,
                                                        chunk_len=chunk_len,
                                                        hidden_size=args.hidden_size,
                                                        learning_rate=args.learning_rate,
                                                        seed=args.seed,
                                                        device=args.device,
                                                    )
                                                )
    return configs


def _rotate_items(items: list[str], offset: int) -> list[str]:
    """Rotate API execution order to reduce fixed-order timing bias."""
    if not items:
        return []
    split = offset % len(items)
    return items[split:] + items[:split]


def main() -> None:
    """Run the time-series application benchmark."""
    args = parse_args()
    device = resolve_benchmark_device(args.device)
    args.device = str(device)
    results: list[TimeSeriesResult] = []
    snapshots: list[PredictionSnapshot] = []

    for base_config in iter_configs(args):
        for repeat in range(args.repeats):
            config = TimeSeriesConfig(
                **{**asdict(base_config), "seed": args.seed + repeat}
            )
            train_batches, train_targets = make_timeseries_batches(
                config, n_batches=config.n_batches, seed_offset=10_000
            )
            val_batches, val_targets = make_timeseries_batches(
                config, n_batches=config.val_batches, seed_offset=20_000
            )
            train_batches = move_batches_to_device(train_batches, device)
            train_targets = move_batches_to_device(train_targets, device)
            val_batches = move_batches_to_device(val_batches, device)
            val_targets = move_batches_to_device(val_targets, device)

            for _ in range(args.warmups):
                for api in args.apis:
                    train_one_model(
                        api,
                        config,
                        train_batches,
                        train_targets,
                        val_batches,
                        val_targets,
                    )
                    gc.collect()
                    synchronize_if_cuda(device)

            for api in _rotate_items(args.apis, repeat):
                result, snapshot = train_one_model(
                    api,
                    config,
                    train_batches,
                    train_targets,
                    val_batches,
                    val_targets,
                )
                results.append(result)
                snapshots.append(snapshot)
                gc.collect()
                synchronize_if_cuda(device)

    summary = summarize(results)
    comparisons = compare_apis(summary)
    print_summary(summary)
    print_comparisons(comparisons)

    payload = {
        "summary": summary,
        "comparisons": comparisons,
        "runs": [asdict(result) for result in results],
        "prediction_snapshots": [asdict(snapshot) for snapshot in snapshots],
        "notes": {
            "task": "Synthetic nonlinear delayed next-step time-series forecasting.",
            "reservoir": "num_backprop_steps=0",
            "manual": "num_backprop_steps=None with explicit detach_memristive_state at chunk boundaries",
            "blackbox": "num_backprop_steps=k with the same chunk boundaries",
        },
        "environment": benchmark_environment(device),
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.csv_output is not None:
        write_csv(args.csv_output.with_suffix(".summary.csv"), summary)
        write_csv(args.csv_output.with_suffix(".comparisons.csv"), comparisons)
        write_csv(args.csv_output.with_suffix(".runs.csv"), [asdict(r) for r in results])
    if args.plot_dir is not None:
        plot_figures(args.plot_dir, summary, comparisons, snapshots)


if __name__ == "__main__":
    main()
