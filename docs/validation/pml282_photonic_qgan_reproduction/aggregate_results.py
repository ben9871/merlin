"""Aggregate PML-282 photonic QGAN reproduction comparison runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VALIDATION_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = VALIDATION_ROOT / "results"
DEFAULT_OUTPUT_DIR = VALIDATION_ROOT / "aggregate"


@dataclass(frozen=True)
class BackendMetric:
    """Flattened metrics for one backend in one comparison run."""

    run: str
    seed: int
    backend: str
    initial_max_abs_diff_vs_reference: float | None
    best_iteration: int
    best_ssim: float
    final_ssim: float
    first_ssim: float
    best_diversity: float
    final_diversity: float
    final_d_loss: float
    final_g_loss: float
    seconds: float


def _read_run(summary_path: Path) -> list[BackendMetric]:
    """Read one run summary into backend metric rows."""
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    config = payload["config"]
    seed = int(config["seed"])
    run = summary_path.parent.name
    initial_diffs = payload.get("initial_max_abs_diff_vs_reference", {})

    rows = []
    for summary in payload["summaries"]:
        backend = summary["backend"]
        diff_key = backend.replace("-", "_")
        if backend == "reference":
            initial_diff = None
        else:
            initial_diff = initial_diffs.get(diff_key)
        rows.append(
            BackendMetric(
                run=run,
                seed=seed,
                backend=backend,
                initial_max_abs_diff_vs_reference=initial_diff,
                best_iteration=int(summary["best_iteration"]),
                best_ssim=float(summary["best_ssim"]),
                final_ssim=float(summary["final_ssim"]),
                first_ssim=float(summary["first_ssim"]),
                best_diversity=float(summary["best_diversity"]),
                final_diversity=float(summary["final_diversity"]),
                final_d_loss=float(summary["final_d_loss"]),
                final_g_loss=float(summary["final_g_loss"]),
                seconds=float(summary["seconds"]),
            )
        )
    return rows


def _write_csv(rows: list[BackendMetric], path: Path) -> None:
    """Write flattened backend metrics to CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _aggregate(rows: list[BackendMetric]) -> dict[str, object]:
    """Return aggregate statistics and paired backend deltas."""
    by_backend: dict[str, list[BackendMetric]] = defaultdict(list)
    by_seed: dict[int, dict[str, BackendMetric]] = defaultdict(dict)
    for row in rows:
        by_backend[row.backend].append(row)
        by_seed[row.seed][row.backend] = row

    backend_stats = {}
    for backend, backend_rows in sorted(by_backend.items()):
        best = np.asarray([row.best_ssim for row in backend_rows], dtype=float)
        final = np.asarray([row.final_ssim for row in backend_rows], dtype=float)
        backend_stats[backend] = {
            "runs": len(backend_rows),
            "best_ssim_mean": float(best.mean()),
            "best_ssim_std": float(best.std(ddof=0)),
            "final_ssim_mean": float(final.mean()),
            "final_ssim_std": float(final.std(ddof=0)),
            "best_iteration_mean": float(
                np.asarray(
                    [row.best_iteration for row in backend_rows], dtype=float
                ).mean()
            ),
            "seconds_mean": float(
                np.asarray([row.seconds for row in backend_rows], dtype=float).mean()
            ),
        }

    paired_deltas = []
    for seed, seed_rows in sorted(by_seed.items()):
        reference = seed_rows.get("reference")
        merlin = seed_rows.get("merlin-count")
        if reference is None or merlin is None:
            continue
        paired_deltas.append({
            "seed": seed,
            "best_ssim_delta_merlin_minus_reference": merlin.best_ssim
            - reference.best_ssim,
            "final_ssim_delta_merlin_minus_reference": merlin.final_ssim
            - reference.final_ssim,
            "best_iteration_delta_merlin_minus_reference": merlin.best_iteration
            - reference.best_iteration,
            "initial_max_abs_diff_merlin_count": (
                merlin.initial_max_abs_diff_vs_reference
            ),
        })

    return {
        "backend_stats": backend_stats,
        "paired_deltas": paired_deltas,
    }


def _plot_summary(rows: list[BackendMetric], output_dir: Path) -> None:
    """Save compact per-seed metric comparison plots."""
    backends = sorted({row.backend for row in rows})
    seeds = sorted({row.seed for row in rows})
    metric_by_seed = {(row.seed, row.backend): row for row in rows}

    x = np.arange(len(seeds), dtype=float)
    width = 0.8 / max(len(backends), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for index, backend in enumerate(backends):
        offset = (index - (len(backends) - 1) / 2) * width
        best_values = [
            metric_by_seed[(seed, backend)].best_ssim
            if (seed, backend) in metric_by_seed
            else np.nan
            for seed in seeds
        ]
        final_values = [
            metric_by_seed[(seed, backend)].final_ssim
            if (seed, backend) in metric_by_seed
            else np.nan
            for seed in seeds
        ]
        axes[0].bar(x + offset, best_values, width=width, label=backend)
        axes[1].bar(x + offset, final_values, width=width, label=backend)

    for axis, title in zip(
        axes,
        ("Best checkpoint SSIM", "Final checkpoint SSIM"),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("seed")
        axis.set_ylabel("SSIM")
        axis.set_xticks(x)
        axis.set_xticklabels([str(seed) for seed in seeds])
        axis.set_ylim(0.0, 1.0)
        axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "ssim_by_seed.png", dpi=160)
    plt.close(fig)


def _plot_curves(
    results_root: Path, rows: list[BackendMetric], output_dir: Path
) -> None:
    """Save SSIM and loss curves grouped by backend and seed."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for row in sorted(rows, key=lambda metric: (metric.backend, metric.seed)):
        backend_dir = results_root / row.run / row.backend
        metrics_path = backend_dir / "ssim_progress.csv"
        losses_path = backend_dir / "loss_progress.csv"
        if not metrics_path.exists() or not losses_path.exists():
            continue

        metrics = np.loadtxt(metrics_path, delimiter=",")
        losses = np.loadtxt(losses_path, delimiter=",")
        if metrics.ndim == 1:
            metrics = metrics.reshape(1, -1)
        if losses.ndim == 1:
            losses = losses.reshape(1, -1)

        label = f"{row.backend} seed {row.seed}"
        axes[0].plot(metrics[:, 0], metrics[:, 3], label=label)
        axes[1].plot(losses[:, 0], label=label)
        axes[2].plot(losses[:, 1], label=label)

    axes[0].set_title("SSIM")
    axes[1].set_title("Discriminator loss")
    axes[2].set_title("Generator loss")
    for axis in axes:
        axis.set_xlabel("iteration")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("metric")
    axes[1].set_ylabel("loss")
    axes[2].set_ylabel("loss")

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves_by_seed.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate PML-282 photonic QGAN reproduction comparisons."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Aggregate all run summaries under the selected results directory."""
    args = parse_args()
    summary_paths = sorted(args.results_root.glob("*/summary.json"))
    if not summary_paths:
        raise SystemExit(f"No summary.json files found under {args.results_root}.")

    rows = [row for summary_path in summary_paths for row in _read_run(summary_path)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.output_dir / "summary_by_seed.csv")

    aggregate = _aggregate(rows)
    (args.output_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    _plot_summary(rows, args.output_dir)
    _plot_curves(args.results_root, rows, args.output_dir)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
