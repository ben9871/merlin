"""Benchmark branch-mixture propagation through QuantumLayer.

The benchmark builds deterministic `QuantumLayer` instances and feeds them
classical mixtures of conditional `StateVector` branches. It measures the new
PML-315 operational path:

* `QuantumLayer(StateMixture)` branch propagation.
* `QuantumLayer(PartialMeasurement)` convenience input conversion.
* Probability-output recombination against an explicit manual branch loop.
* Amplitude and nested-partial outputs that return `StateMixture`.

Results are printed as JSON and can optionally be written to disk. The JSON
payload follows the same broad pattern as `benchmark_superposition_streaming.py`
so benchmark runs can be archived and compared across branches.

Example:

    PYTHONPATH=$PWD python benchmarks/benchmark_state_mixture.py \
        --label local-run \
        --json-out benchmarks/results/state-mixture-local.json
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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import perceval as pcvl
import torch

from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer
from merlin.core.partial_measurement import PartialMeasurement, PartialMeasurementBranch
from merlin.core.state_mixture import StateMixture, StateMixtureBranch
from merlin.core.state_vector import StateVector


@dataclass(frozen=True)
class Case:
    """One state-mixture benchmark case.

    Parameters
    ----------
    name : str
        Case identifier.
    measurement : str
        Measurement path to benchmark: ``"probabilities"``, ``"amplitudes"``,
        or ``"partial"``.
    carrier : str
        Input carrier: ``"state_mixture"`` or ``"partial_measurement"``.
    n_modes : int
        Circuit mode count.
    n_photons : int
        Photon count per branch.
    n_branches : int
        Number of mixture branches.
    batch_size : int
        Batch size stored in each branch state.
    chunk_size : int
        Forwarded to ``QuantumLayer(..., simultaneous_processes=chunk_size)``.
    """

    name: str
    measurement: str
    carrier: str
    n_modes: int
    n_photons: int
    n_branches: int
    batch_size: int
    chunk_size: int


DEFAULT_CASES = (
    Case(
        "prob_state_mixture_m4_p2_b4_batch2",
        measurement="probabilities",
        carrier="state_mixture",
        n_modes=4,
        n_photons=2,
        n_branches=4,
        batch_size=2,
        chunk_size=2,
    ),
    Case(
        "prob_partial_input_m4_p2_b4_batch2",
        measurement="probabilities",
        carrier="partial_measurement",
        n_modes=4,
        n_photons=2,
        n_branches=4,
        batch_size=2,
        chunk_size=2,
    ),
    Case(
        "amp_state_mixture_m4_p2_b4_batch2",
        measurement="amplitudes",
        carrier="state_mixture",
        n_modes=4,
        n_photons=2,
        n_branches=4,
        batch_size=2,
        chunk_size=2,
    ),
    Case(
        "nested_partial_m4_p2_b4_batch2",
        measurement="partial",
        carrier="state_mixture",
        n_modes=4,
        n_photons=2,
        n_branches=4,
        batch_size=2,
        chunk_size=2,
    ),
)


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


def _basis_size(n_modes: int, n_photons: int) -> int:
    """Return full-Fock basis size for a fixed photon number."""
    return math.comb(n_modes + n_photons - 1, n_photons)


def _make_layer(case: Case) -> QuantumLayer:
    """Build a deterministic circuit layer for a benchmark case."""
    circuit = pcvl.components.GenericInterferometer(
        case.n_modes,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    if case.measurement == "probabilities":
        strategy = MeasurementStrategy.probs(ComputationSpace.FOCK)
    elif case.measurement == "amplitudes":
        strategy = MeasurementStrategy.amplitudes(ComputationSpace.FOCK)
    elif case.measurement == "partial":
        strategy = MeasurementStrategy.partial(
            modes=[0],
            computation_space=ComputationSpace.FOCK,
        )
    else:
        raise ValueError(f"Unsupported measurement case: {case.measurement}.")
    return QuantumLayer(
        circuit=circuit,
        input_size=0,
        n_photons=case.n_photons,
        measurement_strategy=strategy,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
    )


def _normalized_complex_tensor(
    *, batch_size: int, basis_size: int, offset: int
) -> torch.Tensor:
    """Create deterministic normalized batched complex amplitudes."""
    real = torch.arange(
        1 + offset,
        1 + offset + batch_size * basis_size,
        dtype=torch.float32,
    ).reshape(batch_size, basis_size)
    imag = torch.flip(real, dims=(-1,))
    tensor = torch.complex(real, imag)
    norm = tensor.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    return tensor / norm


def _make_state_mixture(case: Case) -> StateMixture:
    """Create deterministic branch probabilities and states for a case."""
    basis_size = _basis_size(case.n_modes, case.n_photons)
    raw_probabilities = torch.arange(
        1,
        1 + case.n_branches * case.batch_size,
        dtype=torch.float32,
    ).reshape(case.n_branches, case.batch_size)
    probabilities = raw_probabilities / raw_probabilities.sum(dim=0, keepdim=True)
    branches = []
    for branch_index in range(case.n_branches):
        state_tensor = _normalized_complex_tensor(
            batch_size=case.batch_size,
            basis_size=basis_size,
            offset=branch_index * basis_size,
        )
        branches.append(
            StateMixtureBranch(
                probability=probabilities[branch_index],
                state=StateVector(
                    state_tensor,
                    n_modes=case.n_modes,
                    n_photons=case.n_photons,
                ),
                outcomes=((branch_index,),),
            )
        )
    return StateMixture(
        branches=tuple(branches),
        measured_modes=(0,),
        unmeasured_modes=tuple(range(1, case.n_modes)),
    )


def _make_partial_measurement(mixture: StateMixture) -> PartialMeasurement:
    """Create a PartialMeasurement carrying the same branch data as a mixture."""
    return PartialMeasurement(
        branches=tuple(
            PartialMeasurementBranch(
                outcome=branch.outcomes[-1] if branch.outcomes else (),
                probability=branch.probability,
                amplitudes=branch.state,
            )
            for branch in mixture
        ),
        measured_modes=mixture.measured_modes,
        unmeasured_modes=mixture.unmeasured_modes,
    )


def _case_input(case: Case, mixture: StateMixture) -> StateMixture | PartialMeasurement:
    """Return the configured input carrier for a case."""
    if case.carrier == "state_mixture":
        return mixture
    if case.carrier == "partial_measurement":
        return _make_partial_measurement(mixture)
    raise ValueError(f"Unsupported carrier: {case.carrier}.")


def _reshape_probability(
    probability: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Reshape branch probability tensors for output broadcasting."""
    if probability.ndim == 0:
        return probability
    trailing_dims = max(output.ndim - probability.ndim, 0)
    return probability.reshape((*probability.shape, *((1,) * trailing_dims)))


def _manual_probability_recombination(
    layer: QuantumLayer, mixture: StateMixture, *, chunk_size: int
) -> torch.Tensor:
    """Explicit branch loop used to validate probability-output recombination."""
    outputs = []
    for branch in mixture:
        branch_output = layer(branch.state, simultaneous_processes=chunk_size)
        outputs.append(
            _reshape_probability(branch.probability, branch_output) * branch_output
        )
    return torch.stack(outputs, dim=0).sum(dim=0)


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


def _force_materialization(result: object) -> None:
    """Force eager tensor work before timing is recorded."""
    if isinstance(result, torch.Tensor):
        _ = float(result.detach().abs().sum().cpu())
        return
    if isinstance(result, StateMixture):
        total = torch.tensor(0.0)
        for branch in result:
            total = total + branch.probability.detach().abs().sum().cpu()
            total = total + branch.state.tensor.detach().abs().sum().cpu()
        _ = float(total)


def _output_summary(output: object) -> dict[str, Any]:
    """Return compact JSON-serializable output metadata."""
    if isinstance(output, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(output.shape),
            "l1": float(output.detach().abs().sum().item()),
        }
    if isinstance(output, StateMixture):
        probability_l1 = sum(
            float(branch.probability.detach().abs().sum().item()) for branch in output
        )
        state_l1 = sum(
            float(branch.state.tensor.detach().abs().sum().item()) for branch in output
        )
        return {
            "type": "StateMixture",
            "branch_count": len(output),
            "probability_l1": probability_l1,
            "state_l1": state_l1,
            "history_lengths": [len(history) for history in output.outcome_histories],
        }
    return {"type": type(output).__name__}


def _output_values(output: object) -> dict[str, Any]:
    """Return full tensor output values for optional cross-run comparison."""
    if not isinstance(output, torch.Tensor):
        return _output_summary(output)
    flat = output.detach().cpu().reshape(-1)
    return {
        "shape": list(output.shape),
        "values": [float(value) for value in flat.tolist()],
    }


def _run_case(
    case: Case, runs: int, warmups: int, *, include_output: bool
) -> dict[str, Any]:
    """Run one benchmark case and return JSON-serializable data."""
    torch.manual_seed(1234)
    layer = _make_layer(case)
    mixture = _make_state_mixture(case)
    carrier_input = _case_input(case, mixture)

    gc.collect()
    mean_s, times_s, output = _mean_timed_call(
        runs,
        warmups,
        lambda: layer(carrier_input, simultaneous_processes=case.chunk_size),
    )
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result: dict[str, Any] = {
        **asdict(case),
        "basis_size": _basis_size(case.n_modes, case.n_photons),
        "output_size": layer.output_size,
        "runs": runs,
        "warmups": warmups,
        "times_s": times_s,
        "mean_s": mean_s,
        "min_s": min(times_s),
        "max_s": max(times_s),
        "mean_s_per_branch": mean_s / case.n_branches,
        "rss_max_kib": rss_kib,
        "output": _output_summary(output),
    }
    if case.measurement == "probabilities":
        manual = _manual_probability_recombination(
            layer, mixture, chunk_size=case.chunk_size
        )
        implicit = output
        if not isinstance(implicit, torch.Tensor):
            raise TypeError("Probability case must return a tensor.")
        result["manual_recombination_max_abs_diff"] = float(
            (implicit - manual).abs().max().item()
        )
    if include_output:
        result["output_values"] = _output_values(output)
    return result


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
        "--include-output",
        action="store_true",
        help="Include full tensor output values where available.",
    )
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
        "cases": [
            _run_case(
                case,
                args.runs,
                args.warmups,
                include_output=args.include_output,
            )
            for case in cases
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
