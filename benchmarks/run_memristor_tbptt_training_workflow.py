"""Benchmark blackbox and manual memristor TBPTT in a training workflow.

This script complements ``run_memristor_tbptt_api_comparison.py``. The lower
level benchmark counts forward/backward graph work. This one runs an actual
sequence-regression training loop with a memristive quantum layer, a classical
readout head, optimizer steps, and fixed synthetic training data.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
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
class TrainingConfig:
    """Configuration for one training benchmark workload."""

    n_modes: int
    n_photons: int
    input_size: int
    memristor_count: int
    entangling_layers: int
    batch_size: int
    timesteps: int
    n_batches: int
    epochs: int
    k: int
    chunk_len: int
    hidden_size: int
    learning_rate: float
    seed: int
    device: str


@dataclass
class TrainingResult:
    """Measured result for one training API and workload."""

    api: str
    config: dict[str, Any]
    seconds: float
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
    first_chunk_loss: float
    last_chunk_loss: float
    epoch_losses: list[float]
    trainable_parameter_count: int


class SequenceMemristorRegressor(nn.Module):
    """Small sequence model used by the training workflow benchmark."""

    def __init__(
        self, api: str, config: TrainingConfig, initialization_seed: int
    ) -> None:
        super().__init__()
        num_backprop_steps = config.k if api == "blackbox" else None
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
        """Reset the recurrent memristive state for a new sequence batch."""
        self.quantum.reset(batch_size=batch_size)

    def detach_sequence_state(self) -> None:
        """Detach the recurrent memristive graph at a TBPTT boundary."""
        self.quantum.detach_memristive_state()

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """Run one sequence step through the quantum layer and readout."""
        return self.readout(self.quantum(x))


def make_sequence_batches(config: TrainingConfig) -> list[torch.Tensor]:
    """Create deterministic sequence batches for training."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 10_000)
    batches = []
    for _ in range(config.n_batches):
        batch = torch.randn(
            config.batch_size,
            config.timesteps,
            config.input_size,
            generator=generator,
        )
        batches.append(batch)
    return batches


def make_teacher_targets(
    config: TrainingConfig, batches: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Generate fixed sequence-regression targets from a frozen teacher model."""
    device = torch.device(config.device)
    teacher = SequenceMemristorRegressor(
        "manual", config, initialization_seed=config.seed + 20_000
    ).to(device)
    teacher.eval()

    targets = []
    with torch.no_grad():
        for batch in batches:
            teacher.reset_sequence(batch_size=batch.shape[0])
            batch_targets = []
            for timestep in range(config.timesteps):
                prediction = teacher.forward_step(batch[:, timestep, :])
                batch_targets.append(prediction.detach())
            targets.append(torch.stack(batch_targets, dim=1))
    return targets


def _sample_rss(rss_peak: int | None) -> int | None:
    """Update an RSS peak with the current process RSS."""
    current = rss_bytes()
    if current is None:
        return rss_peak
    return current if rss_peak is None else max(rss_peak, current)


def train_one_model(
    api: str,
    config: TrainingConfig,
    batches: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> TrainingResult:
    """Train one model and return timing and training diagnostics."""
    device = torch.device(config.device)
    model = SequenceMemristorRegressor(
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
    start = time.perf_counter()

    optimizer_steps = 0
    first_chunk_loss: float | None = None
    last_chunk_loss = 0.0
    epoch_losses = []

    for _ in range(config.epochs):
        epoch_loss_total = 0.0
        chunk_count = 0

        for batch, target in zip(batches, targets, strict=True):
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
                    chunk_loss_value = float(chunk_loss.detach().item())
                    if first_chunk_loss is None:
                        first_chunk_loss = chunk_loss_value
                    last_chunk_loss = chunk_loss_value
                    epoch_loss_total += chunk_loss_value
                    chunk_count += 1

                    chunk_loss.backward()
                    model.detach_sequence_state()
                    loss_terms = []
                    rss_peak = _sample_rss(rss_peak)

            optimizer.step()
            optimizer_steps += 1
            rss_peak = _sample_rss(rss_peak)

        epoch_losses.append(epoch_loss_total / max(chunk_count, 1))

    synchronize_if_cuda(device)
    seconds = time.perf_counter() - start
    cuda_memory = finish_cuda_memory_snapshot(device, cuda_memory_start)
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_end = rss_bytes()

    return TrainingResult(
        api=api,
        config=asdict(config),
        seconds=seconds,
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
        first_chunk_loss=0.0 if first_chunk_loss is None else first_chunk_loss,
        last_chunk_loss=last_chunk_loss,
        epoch_losses=epoch_losses,
        trainable_parameter_count=trainable_parameter_count,
    )


def summarize(results: list[TrainingResult]) -> list[dict[str, Any]]:
    """Aggregate repeated training measurements by API and workload."""
    grouped: dict[tuple[Any, ...], list[TrainingResult]] = {}
    for result in results:
        config = result.config
        key = (
            result.api,
            config["k"],
            config["chunk_len"],
            config["timesteps"],
            config["batch_size"],
            config["n_batches"],
            config["epochs"],
            config["n_modes"],
            config["n_photons"],
            config["memristor_count"],
            config["entangling_layers"],
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
            epochs,
            n_modes,
            n_photons,
            memristor_count,
            entangling_layers,
            device,
        ) = key
        rows.append({
            "api": api,
            "k": k,
            "chunk_len": chunk_len,
            "timesteps": timesteps,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "epochs": epochs,
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
            "first_chunk_loss_mean": statistics.mean(
                result.first_chunk_loss for result in group
            ),
            "last_chunk_loss_mean": statistics.mean(
                result.last_chunk_loss for result in group
            ),
            "final_epoch_loss_mean": statistics.mean(
                result.epoch_losses[-1] for result in group
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
            item["batch_size"],
            item["device"],
        ),
    )


def compare_apis(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare blackbox and manual training results for matching workloads."""
    by_workload: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for item in summary:
        key = (
            item["k"],
            item["chunk_len"],
            item["timesteps"],
            item["batch_size"],
            item["n_batches"],
            item["epochs"],
            item["n_modes"],
            item["n_photons"],
            item["memristor_count"],
            item["entangling_layers"],
            item["device"],
        )
        by_workload.setdefault(key, {})[item["api"]] = item

    rows = []
    for key, items in by_workload.items():
        if "blackbox" not in items or "manual" not in items:
            continue
        blackbox = items["blackbox"]
        manual = items["manual"]
        rows.append({
            "k": key[0],
            "chunk_len": key[1],
            "timesteps": key[2],
            "batch_size": key[3],
            "n_batches": key[4],
            "epochs": key[5],
            "n_modes": key[6],
            "n_photons": key[7],
            "memristor_count": key[8],
            "entangling_layers": key[9],
            "device": key[10],
            "seconds_ratio_blackbox_over_manual": (
                blackbox["seconds_mean"] / manual["seconds_mean"]
            ),
            "process_call_ratio_blackbox_over_manual": (
                blackbox["process_calls_total_mean"]
                / manual["process_calls_total_mean"]
            ),
            "blackbox_seconds_mean": blackbox["seconds_mean"],
            "manual_seconds_mean": manual["seconds_mean"],
            "blackbox_final_epoch_loss": blackbox["final_epoch_loss_mean"],
            "manual_final_epoch_loss": manual["final_epoch_loss_mean"],
            "final_epoch_loss_abs_delta": abs(
                blackbox["final_epoch_loss_mean"]
                - manual["final_epoch_loss_mean"]
            ),
            "blackbox_rss_peak_delta_bytes": blackbox[
                "rss_peak_delta_bytes_max"
            ],
            "manual_rss_peak_delta_bytes": manual["rss_peak_delta_bytes_max"],
            "blackbox_tracemalloc_peak_bytes": blackbox[
                "tracemalloc_peak_bytes_max"
            ],
            "manual_tracemalloc_peak_bytes": manual[
                "tracemalloc_peak_bytes_max"
            ],
            "blackbox_cuda_memory_peak_allocated_bytes": blackbox[
                "cuda_memory_peak_allocated_bytes_max"
            ],
            "manual_cuda_memory_peak_allocated_bytes": manual[
                "cuda_memory_peak_allocated_bytes_max"
            ],
        })

    return sorted(
        rows,
        key=lambda item: (
            item["k"],
            item["n_modes"],
            item["n_photons"],
            item["memristor_count"],
            item["batch_size"],
            item["device"],
        ),
    )


def _format_megabytes(value: int | None) -> str:
    """Format an optional byte count as a megabyte string."""
    return "n/a" if value is None else f"{value / 1_000_000:.1f}"


def print_summary(summary: list[dict[str, Any]]) -> None:
    """Print a compact training summary table."""
    header = (
        "api      m  p  mem  k  T  batch  batches  epochs  sec_mean  "
        "process_calls  first_loss  last_loss  rss_delta_mb  py_peak_mb  cuda_peak_mb"
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
            f"{item['timesteps']:>2} "
            f"{item['batch_size']:>6} "
            f"{item['n_batches']:>8} "
            f"{item['epochs']:>7} "
            f"{item['seconds_mean']:>9.3f} "
            f"{item['process_calls_total_mean']:>13.1f} "
            f"{item['first_chunk_loss_mean']:>10.5f} "
            f"{item['last_chunk_loss_mean']:>9.5f} "
            f"{_format_megabytes(item['rss_peak_delta_bytes_max']):>12} "
            f"{item['tracemalloc_peak_bytes_max'] / 1_000_000:>10.1f} "
            f"{format_megabytes(item['cuda_memory_peak_allocated_bytes_max']):>12}"
        )


def print_comparisons(comparisons: list[dict[str, Any]]) -> None:
    """Print blackbox/manual training ratios."""
    if not comparisons:
        return
    header = (
        "m  p  mem  k  T  batch  batches  epochs  time_ratio  "
        "process_ratio  loss_delta"
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
            f"{item['batch_size']:>6} "
            f"{item['n_batches']:>8} "
            f"{item['epochs']:>7} "
            f"{item['seconds_ratio_blackbox_over_manual']:>10.3f} "
            f"{item['process_call_ratio_blackbox_over_manual']:>13.3f} "
            f"{item['final_epoch_loss_abs_delta']:>10.3e}"
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
) -> None:
    """Write training figures when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping figures.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    comparison_labels = [
        (
            f"m{item['n_modes']}/p{item['n_photons']}/mem{item['memristor_count']}"
            f"/k{item['k']}/B{item['batch_size']}"
        )
        for item in comparisons
    ]

    def save_ratio(metric: str, filename: str, ylabel: str) -> None:
        values = [item[metric] for item in comparisons]
        if not values:
            return
        fig, axis = plt.subplots(figsize=(max(8, len(values) * 0.8), 4.5))
        axis.bar(range(len(values)), values)
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(comparison_labels, rotation=45, ha="right")
        axis.set_title(ylabel)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    save_ratio(
        "seconds_ratio_blackbox_over_manual",
        "training_time_ratio_blackbox_over_manual.png",
        "Blackbox / manual training seconds",
    )
    save_ratio(
        "process_call_ratio_blackbox_over_manual",
        "training_process_call_ratio_blackbox_over_manual.png",
        "Blackbox / manual process calls",
    )

    if summary:
        labels = [
            (
                f"{item['api']}:m{item['n_modes']}/p{item['n_photons']}"
                f"/mem{item['memristor_count']}/k{item['k']}/B{item['batch_size']}"
            )
            for item in summary
        ]
        for metric, filename, ylabel in (
            ("seconds_mean", "training_seconds_by_api.png", "Training seconds"),
            (
                "final_epoch_loss_mean",
                "training_final_epoch_loss_by_api.png",
                "Final epoch loss",
            ),
        ):
            values = [item[metric] for item in summary]
            fig, axis = plt.subplots(figsize=(max(10, len(values) * 0.5), 4.8))
            axis.bar(range(len(values)), values)
            axis.set_ylabel(ylabel)
            axis.set_xticks(range(len(values)))
            axis.set_xticklabels(labels, rotation=45, ha="right")
            axis.set_title(ylabel)
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=160)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--timesteps", nargs="+", type=int, default=[8])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[4])
    parser.add_argument("--n-batches", nargs="+", type=int, default=[4])
    parser.add_argument("--epochs", nargs="+", type=int, default=[3])
    parser.add_argument("--n-modes", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--n-photons", nargs="+", type=int, default=[3])
    parser.add_argument("--input-size", nargs="+", type=int, default=[2])
    parser.add_argument("--memristor-count", nargs="+", type=int, default=[1, 2])
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
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    return parser.parse_args()


def iter_training_configs(args: argparse.Namespace) -> list[TrainingConfig]:
    """Build valid training configurations from CLI sweep arguments."""
    configs = []
    for k in args.k:
        if k < 1:
            raise ValueError("This training comparison expects k >= 1.")
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
                                        for epochs in args.epochs:
                                            configs.append(
                                                TrainingConfig(
                                                    n_modes=n_modes,
                                                    n_photons=n_photons,
                                                    input_size=input_size,
                                                    memristor_count=memristor_count,
                                                    entangling_layers=entangling_layers,
                                                    batch_size=batch_size,
                                                    timesteps=timesteps,
                                                    n_batches=n_batches,
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


def main() -> None:
    """Run the training workflow benchmark."""
    args = parse_args()
    device = resolve_benchmark_device(args.device)
    args.device = str(device)
    results: list[TrainingResult] = []
    configs = iter_training_configs(args)

    for config in configs:
        batches = move_batches_to_device(make_sequence_batches(config), device)
        targets = make_teacher_targets(config, batches)

        for _ in range(args.warmups):
            train_one_model("blackbox", config, batches, targets)
            train_one_model("manual", config, batches, targets)
            gc.collect()
            synchronize_if_cuda(device)

        for repeat in range(args.repeats):
            repeat_config = TrainingConfig(
                **{**asdict(config), "seed": args.seed + repeat}
            )
            repeat_batches = (
                batches if repeat == 0 else make_sequence_batches(repeat_config)
            )
            repeat_batches = move_batches_to_device(repeat_batches, device)
            repeat_targets = (
                targets
                if repeat == 0
                else make_teacher_targets(repeat_config, repeat_batches)
            )
            results.append(
                train_one_model("blackbox", repeat_config, repeat_batches, repeat_targets)
            )
            gc.collect()
            synchronize_if_cuda(device)
            results.append(
                train_one_model("manual", repeat_config, repeat_batches, repeat_targets)
            )
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
        "notes": {
            "training_data": "Synthetic sequence batches with fixed targets generated by a frozen teacher memristive model.",
            "timing": "seconds measure the student training loop only, excluding target generation.",
            "blackbox": "num_backprop_steps=k, automatic bounded memristive window",
            "manual": "num_backprop_steps=None, user calls detach_memristive_state() at chunk boundaries",
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
        plot_figures(args.plot_dir, summary, comparisons)


if __name__ == "__main__":
    main()
