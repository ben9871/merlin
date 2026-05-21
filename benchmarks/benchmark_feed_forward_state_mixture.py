"""Benchmark FeedForwardBlock recombination against StateMixture recombination.

The benchmark compares the active FeedForwardBlock probability path with an
equivalent path that executes the same feed-forward stages, takes the raw
BranchState objects, and recombines them through StateMixture plus identity
probability layers. It records timing, resident-memory snapshots, and the
maximum absolute difference between the two probability tensors.

Example:

    PYTHONPATH=$PWD python benchmarks/benchmark_feed_forward_state_mixture.py \
        --label local-run \
        --json-out benchmarks/results/feed-forward-state-mixture-local.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import subprocess  # noqa: S404
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import perceval as pcvl
import torch
from perceval import BasicState, Circuit, Matrix, Unitary
from perceval.components import PERM

from merlin.algorithms.feed_forward import BranchState, FeedForwardBlock
from merlin.algorithms.layer import QuantumLayer
from merlin.core.computation_space import ComputationSpace
from merlin.core.state_mixture import StateMixture, StateMixtureBranch
from merlin.core.state_vector import StateVector
from merlin.measurement.strategies import MeasurementStrategy
from merlin.utils.combinadics import Combinadics


@dataclass(frozen=True)
class Case:
    """One feed-forward benchmark case.

    Parameters
    ----------
    name : str
        Case identifier.
    detector : str
        Detector kind for the first feed-forward measurement. Supported values
        are ``"pnr"``, ``"threshold"``, and ``"threshold_pair"``.
    """

    name: str
    detector: str


DEFAULT_CASES = (
    Case("feedforward_pnr_m4_p2", detector="pnr"),
    Case("feedforward_threshold_m4_p2", detector="threshold"),
    Case("feedforward_threshold_pair_m5_p4", detector="threshold_pair"),
)

_BASIS_CACHE: dict[tuple[int, int], list[tuple[int, ...]]] = {}


def _git_value(args: list[str], repo: Path) -> str:
    """Return git metadata for benchmark JSON, or ``unknown`` outside git."""
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fourier_unitary(dim: int) -> Unitary:
    """Return a deterministic Fourier unitary used by the benchmark circuit."""
    omega = np.exp(2j * np.pi / dim)
    matrix = np.empty((dim, dim), dtype=np.complex128)
    scale = 1 / math.sqrt(dim)
    for row in range(dim):
        for col in range(dim):
            matrix[row, col] = omega ** (row * col) * scale
    return Unitary(Matrix(matrix))


def _make_detector(kind: str) -> pcvl.Detector:
    """Return the Perceval detector configured by a benchmark case."""
    if kind == "pnr":
        return pcvl.Detector.pnr()
    if kind == "threshold":
        return pcvl.Detector.threshold()
    raise ValueError(f"Unsupported detector kind: {kind}.")


def _build_feedforward_experiment(case: Case) -> pcvl.Experiment:
    """Build the active FeedForwardBlock benchmark experiment."""
    if case.detector == "threshold_pair":
        return _build_threshold_pair_experiment()

    n_modes = 4
    input_state = [1, 1, 0, 0]

    experiment = pcvl.Experiment()
    root = Circuit(n_modes)
    root.add(0, _fourier_unitary(n_modes))
    root.add((0, 1), pcvl.BS())
    experiment.add(0, root)

    experiment.add(0, _make_detector(case.detector))

    default_branch = Circuit(n_modes - 1)
    default_branch.add(0, _fourier_unitary(n_modes - 1))

    adaptive_branch = Circuit(n_modes - 1)
    adaptive_branch.add(0, PERM([2, 1, 0]))
    adaptive_branch.add(0, _fourier_unitary(n_modes - 1))

    provider = pcvl.FFCircuitProvider(1, 0, default_branch)
    provider.add_configuration([1], adaptive_branch)
    experiment.add(0, provider)

    experiment.with_input(BasicState(input_state))
    return experiment


def _build_threshold_pair_experiment() -> pcvl.Experiment:
    """Build a two-threshold case with compatible raw branches for batching."""
    n_modes = 5
    input_state = [1, 1, 1, 1, 0]

    experiment = pcvl.Experiment()
    root = Circuit(n_modes)
    root.add(0, _fourier_unitary(n_modes))
    root.add((0, 1), pcvl.BS())
    root.add((1, 2), pcvl.BS())
    root.add((2, 3), pcvl.BS())
    experiment.add(0, root)

    experiment.add(0, pcvl.Detector.threshold())
    experiment.add(1, pcvl.Detector.threshold())

    default_branch = Circuit(n_modes - 2)
    default_branch.add(0, _fourier_unitary(n_modes - 2))

    adaptive_branch = Circuit(n_modes - 2)
    adaptive_branch.add(0, PERM([2, 1, 0]))
    adaptive_branch.add(0, _fourier_unitary(n_modes - 2))

    provider = pcvl.FFCircuitProvider(2, 0, default_branch)
    provider.add_configuration([1, 1], adaptive_branch)
    experiment.add(0, provider)

    experiment.with_input(BasicState(input_state))
    return experiment


def _basis_states(n_modes: int, n_photons: int) -> list[tuple[int, ...]]:
    """Return Fock basis states for a fixed mode and photon count."""
    cache_key = (n_modes, n_photons)
    if cache_key not in _BASIS_CACHE:
        _BASIS_CACHE[cache_key] = Combinadics(
            "fock", n_photons, n_modes
        ).enumerate_states()
    return _BASIS_CACHE[cache_key]


def _batch_probability(probability: torch.Tensor) -> torch.Tensor:
    """Return a branch probability tensor with an explicit batch dimension."""
    if probability.ndim == 0:
        return probability.reshape(1)
    if probability.ndim == 1:
        return probability
    raise ValueError(
        "Feed-forward benchmark expects scalar or one-dimensional probabilities."
    )


def _accumulate_probability(
    output: dict[tuple[int, ...], torch.Tensor],
    key: tuple[int, ...],
    probability: torch.Tensor,
) -> None:
    """Accumulate one probability column into a keyed probability map."""
    probability = _batch_probability(probability)
    output[key] = output[key] + probability if key in output else probability


def _state_mixture_branch_from_feedforward_branch(
    measurement_key: tuple[int | None, ...],
    branch: BranchState,
) -> StateMixtureBranch:
    """Convert one raw FeedForwardBlock branch into a StateMixture branch."""
    unmeasured_modes = [
        idx for idx, value in enumerate(measurement_key) if value is None
    ]
    standard_basis = _basis_states(len(unmeasured_modes), branch.remaining_n)
    amplitudes = branch.amplitudes
    if branch.basis_keys and branch.basis_keys != tuple(standard_basis):
        source_indices = {state: idx for idx, state in enumerate(branch.basis_keys)}
        reindexed = torch.zeros(
            *amplitudes.shape[:-1],
            len(standard_basis),
            dtype=amplitudes.dtype,
            device=amplitudes.device,
        )
        for target_index, state in enumerate(standard_basis):
            source_index = source_indices.get(state)
            if source_index is not None:
                reindexed[..., target_index] = amplitudes[..., source_index]
        amplitudes = reindexed

    measured_outcome = tuple(
        int(value) for value in measurement_key if value is not None
    )
    return StateMixtureBranch(
        probability=torch.nan_to_num(branch.weight, nan=0.0),
        state=StateVector(
            amplitudes,
            n_modes=len(unmeasured_modes),
            n_photons=branch.remaining_n,
        ),
        outcomes=(measured_outcome,),
    )


def _raw_feedforward_branches(
    block: FeedForwardBlock,
) -> dict[tuple[int | None, ...], list[BranchState]]:
    """Execute FeedForwardBlock stages and return raw branches before aggregation."""
    feature_tensor = block._prepare_classical_features(None)
    branches = block._run_stage(block._stage_runtimes[0], feature_tensor)
    for runtime in block._stage_runtimes[1:]:
        branches = block._propagate_future_stage(branches, runtime)
    return branches


def _identity_probability_layer(
    layer_cache: dict[tuple[int, int], QuantumLayer],
    n_modes: int,
    n_photons: int,
) -> QuantumLayer:
    """Return a cached identity probability layer for one branch shape."""
    cache_key = (n_modes, n_photons)
    layer = layer_cache.get(cache_key)
    if layer is None:
        layer = QuantumLayer(
            input_size=0,
            circuit=pcvl.Circuit(n_modes),
            input_state=[n_photons, *([0] * (n_modes - 1))],
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
        )
        layer_cache[cache_key] = layer
    return layer


def _state_mixture_probability_map(
    block: FeedForwardBlock,
    layer_cache: dict[tuple[int, int], QuantumLayer],
) -> dict[tuple[int, ...], torch.Tensor]:
    """Recombine raw FeedForwardBlock branches through StateMixture."""
    grouped_branches: dict[
        tuple[tuple[int | None, ...], int], list[StateMixtureBranch]
    ] = defaultdict(list)
    probabilities: dict[tuple[int, ...], torch.Tensor] = {}

    for measurement_key, branch_list in _raw_feedforward_branches(block).items():
        for branch in branch_list:
            if branch.remaining_n == 0:
                full_key = tuple(
                    0 if value is None else int(value) for value in measurement_key
                )
                _accumulate_probability(
                    probabilities,
                    full_key,
                    torch.nan_to_num(branch.weight, nan=0.0),
                )
                continue
            grouped_branches[(measurement_key, branch.remaining_n)].append(
                _state_mixture_branch_from_feedforward_branch(measurement_key, branch)
            )

    for (measurement_key, remaining_n), branches in grouped_branches.items():
        unmeasured_modes = [
            idx for idx, value in enumerate(measurement_key) if value is None
        ]
        mixture = StateMixture(
            branches=tuple(branches),
            measured_modes=tuple(
                idx for idx, value in enumerate(measurement_key) if value is not None
            ),
            unmeasured_modes=tuple(unmeasured_modes),
        )
        layer = _identity_probability_layer(
            layer_cache, len(unmeasured_modes), remaining_n
        )
        remaining_probabilities = layer(mixture)
        if remaining_probabilities.ndim == 1:
            remaining_probabilities = remaining_probabilities.unsqueeze(0)
        for basis_index, basis_state in enumerate(
            _basis_states(len(unmeasured_modes), remaining_n)
        ):
            full_key = list(measurement_key)
            for mode_idx, value in zip(unmeasured_modes, basis_state, strict=False):
                full_key[mode_idx] = value
            _accumulate_probability(
                probabilities,
                tuple(int(value) for value in full_key),
                remaining_probabilities[:, basis_index],
            )
    return dict(probabilities)


def _probability_map_to_tensor(
    probability_map: dict[tuple[int, ...], torch.Tensor],
    keys: list[tuple[int, ...]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a probability tensor aligned with FeedForwardBlock output keys."""
    columns = []
    for key in keys:
        probability = probability_map.get(key)
        if probability is None:
            probability = torch.zeros(1, dtype=dtype, device=device)
        columns.append(_batch_probability(probability).to(dtype=dtype, device=device))
    return torch.stack(columns, dim=1)


def _state_mixture_probability_tensor(
    block: FeedForwardBlock,
    keys: list[tuple[int, ...]],
    layer_cache: dict[tuple[int, int], QuantumLayer],
) -> torch.Tensor:
    """Return StateMixture-recombined probabilities aligned with ``keys``."""
    probability_map = _state_mixture_probability_map(block, layer_cache)
    return _probability_map_to_tensor(
        probability_map,
        keys,
        dtype=block.dtype,
        device=block.device,
    )


def _raw_branch_group_summary(block: FeedForwardBlock) -> dict[str, Any]:
    """Return raw branch counts relevant to StateMixture batching."""
    grouped_counts: defaultdict[tuple[tuple[int | None, ...], int], int] = defaultdict(
        int
    )
    raw_branch_count = 0
    vacuum_branch_count = 0
    for measurement_key, branch_list in _raw_feedforward_branches(block).items():
        for branch in branch_list:
            raw_branch_count += 1
            if branch.remaining_n == 0:
                vacuum_branch_count += 1
                continue
            grouped_counts[(measurement_key, branch.remaining_n)] += 1
    group_sizes = sorted(grouped_counts.values(), reverse=True)
    return {
        "raw_branch_count": raw_branch_count,
        "vacuum_branch_count": vacuum_branch_count,
        "state_mixture_group_count": len(group_sizes),
        "compatible_multi_branch_group_count": sum(size > 1 for size in group_sizes),
        "max_group_size": max(group_sizes, default=0),
        "group_sizes": group_sizes,
    }


def _force_materialization(result: object) -> None:
    """Force eager tensor work before timing is recorded."""
    if isinstance(result, torch.Tensor):
        _ = float(result.detach().abs().sum().cpu())


def _mean_timed_call(
    runs: int, warmups: int, callback: Callable[[], object]
) -> tuple[float, list[float], object]:
    """Return mean timing, all measured timings, and the final callback result."""
    timings: list[float] = []
    final_result: object = None
    for run_index in range(warmups + runs):
        start = time.perf_counter()
        result = callback()
        _force_materialization(result)
        elapsed = time.perf_counter() - start
        if run_index >= warmups:
            timings.append(elapsed)
        final_result = result
        gc.collect()
    return mean(timings), timings, final_result


def _rss_max_kib() -> int:
    """Return process maximum resident set size in KiB on Linux."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _timed_variant(
    runs: int,
    warmups: int,
    callback: Callable[[], object],
) -> tuple[dict[str, Any], object]:
    """Time a callback and include coarse resident-memory snapshots."""
    gc.collect()
    rss_before = _rss_max_kib()
    mean_s, times_s, output = _mean_timed_call(runs, warmups, callback)
    rss_after = _rss_max_kib()
    return (
        {
            "times_s": times_s,
            "mean_s": mean_s,
            "min_s": min(times_s),
            "max_s": max(times_s),
            "rss_before_kib": rss_before,
            "rss_after_kib": rss_after,
            "rss_delta_kib": max(rss_after - rss_before, 0),
        },
        output,
    )


def _tensor_max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the maximum absolute tensor difference."""
    if left.shape != right.shape:
        return float("inf")
    return float((left - right).detach().abs().max().item())


def _output_summary(output: torch.Tensor) -> dict[str, Any]:
    """Return compact JSON-serializable output metadata."""
    return {
        "type": "torch.Tensor",
        "shape": list(output.shape),
        "l1": float(output.detach().abs().sum().item()),
    }


def _run_case(case: Case, runs: int, warmups: int) -> dict[str, Any]:
    """Run one benchmark case and return JSON-serializable data."""
    probability_block = FeedForwardBlock(_build_feedforward_experiment(case))
    state_mixture_block = FeedForwardBlock(_build_feedforward_experiment(case))
    layer_cache: dict[tuple[int, int], QuantumLayer] = {}

    direct_metrics, direct_output = _timed_variant(runs, warmups, probability_block)
    if not isinstance(direct_output, torch.Tensor):
        raise TypeError("FeedForwardBlock probability benchmark must return a tensor.")
    output_keys = probability_block.output_keys

    state_mixture_metrics, state_mixture_output = _timed_variant(
        runs,
        warmups,
        lambda: _state_mixture_probability_tensor(
            state_mixture_block, output_keys, layer_cache
        ),
    )
    if not isinstance(state_mixture_output, torch.Tensor):
        raise TypeError("StateMixture benchmark must return a tensor.")

    return {
        **asdict(case),
        "runs": runs,
        "warmups": warmups,
        "output_size": len(output_keys),
        "raw_branch_summary": _raw_branch_group_summary(state_mixture_block),
        "feedforward_probability": direct_metrics,
        "raw_branch_state_mixture": state_mixture_metrics,
        "state_mixture_vs_feedforward_speedup": (
            direct_metrics["mean_s"] / state_mixture_metrics["mean_s"]
            if state_mixture_metrics["mean_s"] > 0
            else None
        ),
        "state_mixture_vs_feedforward_max_abs_diff": _tensor_max_abs_diff(
            direct_output, state_mixture_output
        ),
        "output": _output_summary(direct_output),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Human label for this run.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path.")
    parser.add_argument("--commit", help="Optional commit override.")
    parser.add_argument("--branch", help="Optional branch/base label override.")
    parser.add_argument(
        "--dirty",
        choices=("true", "false"),
        help="Optional dirty-state override.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in DEFAULT_CASES],
        help="Run one or more named cases. Defaults to all cases.",
    )
    return parser.parse_args()


def main() -> int:
    """Run selected cases and print/write one JSON payload."""
    args = parse_args()
    repo = Path.cwd()
    selected = set(args.case or [case.name for case in DEFAULT_CASES])
    cases = [case for case in DEFAULT_CASES if case.name in selected]
    status = _git_value(["status", "--porcelain"], repo)
    is_dirty = status not in ("", "unknown")
    if args.dirty is not None:
        is_dirty = args.dirty == "true"

    payload = {
        "label": args.label,
        "repo": str(repo),
        "commit": args.commit or _git_value(["rev-parse", "HEAD"], repo),
        "branch": args.branch
        or _git_value(["branch", "--show-current"], repo)
        or "detached",
        "is_dirty": is_dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pid": os.getpid(),
        "cases": [_run_case(case, args.runs, args.warmups) for case in cases],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
