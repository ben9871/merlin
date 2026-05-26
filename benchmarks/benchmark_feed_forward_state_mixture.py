"""Benchmark representative StateMixture propagation through QuantumLayer.

The primary benchmark compares two ways to propagate the same conditional
branches through a nontrivial downstream ``QuantumLayer``:

* a sequential per-branch loop over ``StateVector`` inputs;
* a batched ``StateMixture`` input that lets ``QuantumLayer`` fuse compatible
  branches and recombine tensor outputs.

The feed-forward block is used only as a deterministic branch generator. Its
specialized probability path is recorded as context, but it is not the primary
baseline for the ``QuantumLayer`` batching decision.

The benchmark also keeps identity-probability paths under a diagnostic section.
Those paths validate recombination and basis expansion, but they intentionally
measure a near-worst case where there is little layer work to amortize.

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
from statistics import mean, median
from typing import Any

import numpy as np
import perceval as pcvl
import torch
from perceval import BasicState, Circuit, Matrix, Unitary
from perceval.components import PERM

from merlin.algorithms.feed_forward import BranchState, FeedForwardBlock
from merlin.algorithms.layer import QuantumLayer
from merlin.core.computation_space import ComputationSpace
from merlin.core.partial_measurement import PartialMeasurement
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
        are ``"pnr"``, ``"threshold"``, ``"threshold_pair"``, and
        ``"threshold_triple"``.
    downstream_depth : int
        Number of repeated Fourier-and-beamsplitter blocks in the downstream
        circuit used by the nontrivial propagation benchmark. Default value is
        ``1``.
    """

    name: str
    detector: str
    downstream_depth: int = 1


DEFAULT_CASES = (
    Case("feedforward_pnr_m4_p2", detector="pnr"),
    Case("feedforward_threshold_m4_p2", detector="threshold"),
    Case(
        "feedforward_threshold_pair_m5_p4_depth2",
        detector="threshold_pair",
        downstream_depth=2,
    ),
    Case(
        "feedforward_threshold_triple_m6_p6_depth4",
        detector="threshold_triple",
        downstream_depth=4,
    ),
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
        return _build_threshold_multi_experiment(
            n_modes=5,
            n_photons=4,
            n_measured_modes=2,
        )
    if case.detector == "threshold_triple":
        return _build_threshold_multi_experiment(
            n_modes=6,
            n_photons=6,
            n_measured_modes=3,
        )

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


def _build_threshold_multi_experiment(
    *, n_modes: int, n_photons: int, n_measured_modes: int
) -> pcvl.Experiment:
    """Build a multi-threshold case with compatible raw branches for batching."""
    experiment = pcvl.Experiment()
    root = Circuit(n_modes)
    root.add(0, _fourier_unitary(n_modes))
    for mode in range(min(n_modes - 1, 4)):
        root.add((mode, mode + 1), pcvl.BS())
    experiment.add(0, root)

    for mode in range(n_measured_modes):
        experiment.add(mode, pcvl.Detector.threshold())

    n_remaining_modes = n_modes - n_measured_modes
    default_branch = Circuit(n_remaining_modes)
    default_branch.add(0, _fourier_unitary(n_remaining_modes))

    adaptive_branch = Circuit(n_remaining_modes)
    adaptive_branch.add(0, PERM(list(reversed(range(n_remaining_modes)))))
    adaptive_branch.add(0, _fourier_unitary(n_remaining_modes))

    provider = pcvl.FFCircuitProvider(n_measured_modes, 0, default_branch)
    provider.add_configuration([1] * n_measured_modes, adaptive_branch)
    experiment.add(0, provider)

    input_state = [1] * n_photons + [0] * (n_modes - n_photons)
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
    """Run the identity-probability diagnostic over StateMixture branches."""
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


def _first_stage_partial_measurement_layer(block: FeedForwardBlock) -> QuantumLayer:
    """Build a partial-measurement layer matching the first feed-forward stage."""
    runtime = block._stage_runtimes[0]
    if block._base_input_state is None:
        raise ValueError(
            "Partial-measurement feed-forward benchmark requires a basis input state."
        )

    experiment = pcvl.Experiment()
    experiment.add(0, runtime.circuit.copy())
    positions = {mode: idx for idx, mode in enumerate(runtime.active_modes)}
    for global_mode, detector in runtime.detectors.items():
        if detector is not None:
            experiment.add(positions[global_mode], detector)
    experiment.with_input(block._base_input_state)

    return QuantumLayer(
        experiment=experiment,
        input_size=0,
        n_photons=block.n_photons,
        measurement_strategy=MeasurementStrategy.partial(
            list(runtime.measured_modes),
            ComputationSpace.FOCK,
        ),
        dtype=block.dtype,
        device=block.device,
    )


def _global_measurement_key_from_partial_outcome(
    block: FeedForwardBlock,
    outcome: tuple[int, ...],
) -> tuple[int | None, ...]:
    """Return a full feed-forward key skeleton for one partial outcome."""
    runtime = block._stage_runtimes[0]
    full_key: list[int | None] = [None] * block.total_modes
    for mode, value in zip(runtime.global_measured_modes, outcome, strict=True):
        full_key[mode] = int(value)
    return tuple(full_key)


def _partial_measurement_state_mixture_probability_map(
    block: FeedForwardBlock,
    partial_layer: QuantumLayer,
    probability_layer_cache: dict[tuple[int, int], QuantumLayer],
) -> dict[tuple[int, ...], torch.Tensor]:
    """Run the identity-probability diagnostic from PartialMeasurement output."""
    if len(block._stage_runtimes) != 1:
        raise ValueError(
            "PartialMeasurement StateMixture feed-forward benchmark is one-stage."
        )
    runtime = block._stage_runtimes[0]
    partial = partial_layer()
    if not isinstance(partial, PartialMeasurement):
        raise TypeError(
            "PartialMeasurement StateMixture benchmark expected a PartialMeasurement "
            f"from the measured layer, got {type(partial).__name__}."
        )

    grouped_branches: dict[
        tuple[tuple[int | None, ...], tuple[int, ...], int],
        list[StateMixtureBranch],
    ] = defaultdict(list)
    probabilities: dict[tuple[int, ...], torch.Tensor] = {}
    for branch in partial.to_state_mixture():
        outcome = branch.outcomes[-1] if branch.outcomes else ()
        global_key = _global_measurement_key_from_partial_outcome(block, outcome)
        remaining_n = branch.state.n_photons
        if remaining_n == 0:
            full_key = tuple(0 if value is None else int(value) for value in global_key)
            _accumulate_probability(probabilities, full_key, branch.probability)
            continue
        grouped_branches[(global_key, outcome, remaining_n)].append(branch)

    for (global_key, outcome, remaining_n), branches in grouped_branches.items():
        unmeasured_modes = tuple(
            idx for idx, value in enumerate(global_key) if value is None
        )
        mixture = StateMixture(
            branches=tuple(branches),
            measured_modes=runtime.global_measured_modes,
            unmeasured_modes=unmeasured_modes,
        )
        propagated = mixture
        conditional_layer = block._select_conditional_layer(
            runtime, outcome, remaining_n
        )
        if conditional_layer is not None:
            expected_dim = len(
                conditional_layer.computation_process.simulation_graph.mapped_keys
            )
            if mixture.branches[0].state.tensor.shape[-1] != expected_dim:
                raise ValueError(
                    "PartialMeasurement StateMixture benchmark cannot apply a "
                    "conditional layer with a mismatched basis dimension."
                )
            conditional_output = conditional_layer(mixture)
            if not isinstance(conditional_output, StateMixture):
                raise TypeError(
                    "PartialMeasurement StateMixture conditional layer returned "
                    f"{type(conditional_output).__name__}."
                )
            propagated = conditional_output

        probability_layer = _identity_probability_layer(
            probability_layer_cache,
            len(unmeasured_modes),
            remaining_n,
        )
        remaining_probabilities = probability_layer(propagated)
        _expand_remaining_probabilities(
            probabilities,
            global_key,
            remaining_probabilities,
            remaining_n,
        )
    return dict(probabilities)


def _partial_measurement_state_mixture_tensor(
    block: FeedForwardBlock,
    partial_layer: QuantumLayer,
    keys: list[tuple[int, ...]],
    probability_layer_cache: dict[tuple[int, int], QuantumLayer],
) -> torch.Tensor:
    """Return probabilities from the PartialMeasurement/StateMixture pipeline."""
    probability_map = _partial_measurement_state_mixture_probability_map(
        block,
        partial_layer,
        probability_layer_cache,
    )
    return _probability_map_to_tensor(
        probability_map,
        keys,
        dtype=block.dtype,
        device=block.device,
    )


def _state_mixture_operated_feedforward_probability_map(
    block: FeedForwardBlock,
    probability_layer_cache: dict[tuple[int, int], QuantumLayer],
) -> dict[tuple[int, ...], torch.Tensor]:
    """Run one-stage feed-forward diagnostics using StateMixture conditionals."""
    if len(block._stage_runtimes) != 1:
        raise ValueError("StateMixture-operated feed-forward benchmark is one-stage.")
    runtime = block._stage_runtimes[0]
    if runtime.pre_layer is None or runtime.detector_transform is None:
        raise RuntimeError("Feed-forward runtime is not fully initialized.")

    call_args: list[torch.Tensor] = []
    if runtime.initial_amplitudes is not None:
        call_args.append(runtime.initial_amplitudes)
    if runtime.classical_input_size:
        call_args.append(block._prepare_classical_features(None))
    amplitudes = runtime.pre_layer(*call_args) if call_args else runtime.pre_layer()
    measurement_data = runtime.detector_transform(amplitudes)

    grouped_branches: dict[
        tuple[tuple[int | None, ...], tuple[int, ...], int],
        list[StateMixtureBranch],
    ] = defaultdict(list)
    for remaining_n, bucket in enumerate(measurement_data):
        for measurement_key, entries in bucket.items():
            global_key = block._merge_measurement_key(None, runtime, measurement_key)
            reduced_key = block._reduce_measurement_values(
                measurement_key, runtime.measured_modes
            )
            unmeasured_modes = [
                idx for idx, value in enumerate(measurement_key) if value is None
            ]
            for probability, branch_amplitudes in entries:
                grouped_branches[(global_key, reduced_key, remaining_n)].append(
                    StateMixtureBranch(
                        probability=probability,
                        state=StateVector(
                            branch_amplitudes,
                            n_modes=len(unmeasured_modes),
                            n_photons=remaining_n,
                        ),
                        outcomes=(reduced_key,),
                    )
                )

    probabilities: dict[tuple[int, ...], torch.Tensor] = {}
    for (global_key, reduced_key, remaining_n), branches in grouped_branches.items():
        unmeasured_modes = [
            idx for idx, value in enumerate(global_key) if value is None
        ]
        mixture = StateMixture(
            branches=tuple(branches),
            measured_modes=runtime.measured_modes,
            unmeasured_modes=tuple(unmeasured_modes),
        )
        if remaining_n == 0:
            full_key = tuple(0 if value is None else int(value) for value in global_key)
            for branch in mixture:
                _accumulate_probability(probabilities, full_key, branch.probability)
            continue

        propagated = mixture
        conditional_layer = block._select_conditional_layer(
            runtime, reduced_key, remaining_n
        )
        if conditional_layer is not None:
            expected_dim = len(
                conditional_layer.computation_process.simulation_graph.mapped_keys
            )
            if mixture.branches[0].state.tensor.shape[-1] == expected_dim:
                conditional_output = conditional_layer(mixture)
                if not isinstance(conditional_output, StateMixture):
                    raise TypeError(
                        "StateMixture-operated conditional layer returned "
                        f"{type(conditional_output).__name__}."
                    )
                propagated = conditional_output

        probability_layer = _identity_probability_layer(
            probability_layer_cache,
            len(unmeasured_modes),
            remaining_n,
        )
        remaining_probabilities = probability_layer(propagated)
        _expand_remaining_probabilities(
            probabilities,
            global_key,
            remaining_probabilities,
            remaining_n,
        )

    return dict(probabilities)


def _state_mixture_operated_feedforward_tensor(
    block: FeedForwardBlock,
    keys: list[tuple[int, ...]],
    probability_layer_cache: dict[tuple[int, int], QuantumLayer],
) -> torch.Tensor:
    """Return probabilities from a StateMixture-operated feed-forward path."""
    probability_map = _state_mixture_operated_feedforward_probability_map(
        block,
        probability_layer_cache,
    )
    return _probability_map_to_tensor(
        probability_map,
        keys,
        dtype=block.dtype,
        device=block.device,
    )


def _downstream_circuit(n_modes: int, depth: int) -> pcvl.Circuit:
    """Return a deterministic nontrivial downstream circuit."""
    if depth < 1:
        raise ValueError("Downstream circuit depth must be at least one.")
    circuit = pcvl.Circuit(n_modes)
    if n_modes == 0:
        return circuit
    for _ in range(depth):
        circuit.add(0, _fourier_unitary(n_modes))
        for mode in range(n_modes - 1):
            circuit.add((mode, mode + 1), pcvl.BS())
    return circuit


def _downstream_probability_layer(
    layer_cache: dict[tuple[int, int, int], QuantumLayer],
    n_modes: int,
    n_photons: int,
    depth: int,
) -> QuantumLayer:
    """Return a cached nontrivial downstream probability layer."""
    cache_key = (n_modes, n_photons, depth)
    layer = layer_cache.get(cache_key)
    if layer is None:
        layer = QuantumLayer(
            input_size=0,
            circuit=_downstream_circuit(n_modes, depth),
            input_state=[n_photons, *([0] * (n_modes - 1))],
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
        )
        layer_cache[cache_key] = layer
    return layer


def _expand_remaining_probabilities(
    probabilities: dict[tuple[int, ...], torch.Tensor],
    measurement_key: tuple[int | None, ...],
    remaining_probabilities: torch.Tensor,
    remaining_n: int,
) -> None:
    """Expand remaining-mode probabilities back into full feed-forward keys."""
    unmeasured_modes = [
        idx for idx, value in enumerate(measurement_key) if value is None
    ]
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


def _sequential_downstream_probability_map(
    block: FeedForwardBlock,
    layer_cache: dict[tuple[int, int, int], QuantumLayer],
    *,
    depth: int,
) -> dict[tuple[int, ...], torch.Tensor]:
    """Propagate feed-forward branches through downstream layers one at a time."""
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
            mixture_branch = _state_mixture_branch_from_feedforward_branch(
                measurement_key, branch
            )
            n_modes = mixture_branch.state.n_modes
            layer = _downstream_probability_layer(
                layer_cache, n_modes, branch.remaining_n, depth
            )
            branch_probabilities = layer(mixture_branch.state)
            probability = _batch_probability(mixture_branch.probability)
            weighted = probability.reshape(-1, 1) * branch_probabilities
            _expand_remaining_probabilities(
                probabilities,
                measurement_key,
                weighted,
                branch.remaining_n,
            )
    return dict(probabilities)


def _batched_downstream_probability_map(
    block: FeedForwardBlock,
    layer_cache: dict[tuple[int, int, int], QuantumLayer],
    *,
    depth: int,
) -> dict[tuple[int, ...], torch.Tensor]:
    """Propagate only multi-branch groups as StateMixture batches."""
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
        layer = _downstream_probability_layer(
            layer_cache, len(unmeasured_modes), remaining_n, depth
        )
        if len(branches) == 1:
            branch = branches[0]
            branch_probabilities = layer(branch.state)
            if branch_probabilities.ndim == 1:
                branch_probabilities = branch_probabilities.unsqueeze(0)
            probability = _batch_probability(branch.probability)
            remaining_probabilities = probability.reshape(-1, 1) * branch_probabilities
        else:
            mixture = StateMixture(
                branches=tuple(branches),
                measured_modes=tuple(
                    idx
                    for idx, value in enumerate(measurement_key)
                    if value is not None
                ),
                unmeasured_modes=tuple(unmeasured_modes),
            )
            remaining_probabilities = layer(mixture)
        _expand_remaining_probabilities(
            probabilities,
            measurement_key,
            remaining_probabilities,
            remaining_n,
        )
    return dict(probabilities)


def _probability_map_max_abs_diff(
    left: dict[tuple[int, ...], torch.Tensor],
    right: dict[tuple[int, ...], torch.Tensor],
) -> float:
    """Return maximum absolute difference across aligned probability maps."""
    if set(left) != set(right):
        return float("inf")
    max_diff = 0.0
    for key in left:
        max_diff = max(max_diff, _tensor_max_abs_diff(left[key], right[key]))
    return max_diff


def _probability_map_summary(
    probability_map: dict[tuple[int, ...], torch.Tensor],
) -> dict[str, Any]:
    """Return compact JSON-serializable metadata for a probability map."""
    total_l1 = 0.0
    for probability in probability_map.values():
        total_l1 += float(probability.detach().abs().sum().item())
    return {
        "type": "dict[tuple[int, ...], torch.Tensor]",
        "key_count": len(probability_map),
        "l1": total_l1,
    }


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
        return
    if isinstance(result, dict):
        total = torch.tensor(0.0)
        for value in result.values():
            if isinstance(value, torch.Tensor):
                total = total + value.detach().abs().sum().cpu()
        _ = float(total)


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
    tail_times_s = times_s[len(times_s) // 2 :]
    return (
        {
            "times_s": times_s,
            "mean_s": mean_s,
            "median_s": median(times_s),
            "tail_mean_s": mean(tail_times_s),
            "tail_median_s": median(tail_times_s),
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


def _run_case(
    case: Case, runs: int, warmups: int, *, representative_only: bool
) -> dict[str, Any]:
    """Run one benchmark case and return JSON-serializable data."""
    branch_summary_block = FeedForwardBlock(_build_feedforward_experiment(case))
    sequential_downstream_block = FeedForwardBlock(_build_feedforward_experiment(case))
    batched_downstream_block = FeedForwardBlock(_build_feedforward_experiment(case))
    sequential_downstream_cache: dict[tuple[int, int, int], QuantumLayer] = {}
    batched_downstream_cache: dict[tuple[int, int, int], QuantumLayer] = {}

    sequential_downstream_metrics, sequential_downstream_output = _timed_variant(
        runs,
        warmups,
        lambda: _sequential_downstream_probability_map(
            sequential_downstream_block,
            sequential_downstream_cache,
            depth=case.downstream_depth,
        ),
    )
    if not isinstance(sequential_downstream_output, dict):
        raise TypeError(
            "Sequential downstream benchmark must return a probability map."
        )
    batched_downstream_metrics, batched_downstream_output = _timed_variant(
        runs,
        warmups,
        lambda: _batched_downstream_probability_map(
            batched_downstream_block,
            batched_downstream_cache,
            depth=case.downstream_depth,
        ),
    )
    if not isinstance(batched_downstream_output, dict):
        raise TypeError("Batched downstream benchmark must return a probability map.")

    result: dict[str, Any] = {
        **asdict(case),
        "runs": runs,
        "warmups": warmups,
        "output_size": len(sequential_downstream_output),
        "raw_branch_summary": _raw_branch_group_summary(branch_summary_block),
        "representative_downstream_layer": {
            "purpose": (
                "Compare sequential per-branch StateVector propagation against "
                "a conservative StateMixture batching policy through a "
                "nontrivial downstream QuantumLayer."
            ),
            "batch_policy": (
                "Use StateMixture only for grouped branches with at least two "
                "compatible states; propagate singleton groups directly as "
                "StateVector inputs."
            ),
            "downstream_depth": case.downstream_depth,
            "sequential_branch_loop": sequential_downstream_metrics,
            "batched_state_mixture": batched_downstream_metrics,
            "batched_vs_sequential_speedup": (
                sequential_downstream_metrics["mean_s"]
                / batched_downstream_metrics["mean_s"]
                if batched_downstream_metrics["mean_s"] > 0
                else None
            ),
            "batched_vs_sequential_median_speedup": (
                sequential_downstream_metrics["median_s"]
                / batched_downstream_metrics["median_s"]
                if batched_downstream_metrics["median_s"] > 0
                else None
            ),
            "batched_vs_sequential_tail_speedup": (
                sequential_downstream_metrics["tail_mean_s"]
                / batched_downstream_metrics["tail_mean_s"]
                if batched_downstream_metrics["tail_mean_s"] > 0
                else None
            ),
            "batched_vs_sequential_max_abs_diff": _probability_map_max_abs_diff(
                sequential_downstream_output,
                batched_downstream_output,
            ),
            "output": _probability_map_summary(sequential_downstream_output),
        },
    }
    if representative_only:
        return result

    probability_block = FeedForwardBlock(_build_feedforward_experiment(case))
    partial_measurement_block = FeedForwardBlock(_build_feedforward_experiment(case))
    partial_measurement_layer = _first_stage_partial_measurement_layer(
        partial_measurement_block
    )
    partial_measurement_cache: dict[tuple[int, int], QuantumLayer] = {}
    state_mixture_block = FeedForwardBlock(_build_feedforward_experiment(case))
    identity_layer_cache: dict[tuple[int, int], QuantumLayer] = {}
    state_mixture_operated_block = FeedForwardBlock(_build_feedforward_experiment(case))
    state_mixture_operated_cache: dict[tuple[int, int], QuantumLayer] = {}

    direct_metrics, direct_output = _timed_variant(runs, warmups, probability_block)
    if not isinstance(direct_output, torch.Tensor):
        raise TypeError("FeedForwardBlock probability benchmark must return a tensor.")
    output_keys = probability_block.output_keys

    partial_measurement_metrics, partial_measurement_output = _timed_variant(
        runs,
        warmups,
        lambda: _partial_measurement_state_mixture_tensor(
            partial_measurement_block,
            partial_measurement_layer,
            output_keys,
            partial_measurement_cache,
        ),
    )
    if not isinstance(partial_measurement_output, torch.Tensor):
        raise TypeError(
            "PartialMeasurement StateMixture feed-forward benchmark must return a tensor."
        )

    state_mixture_metrics, state_mixture_output = _timed_variant(
        runs,
        warmups,
        lambda: _state_mixture_probability_tensor(
            state_mixture_block, output_keys, identity_layer_cache
        ),
    )
    if not isinstance(state_mixture_output, torch.Tensor):
        raise TypeError("StateMixture benchmark must return a tensor.")

    state_mixture_operated_metrics, state_mixture_operated_output = _timed_variant(
        runs,
        warmups,
        lambda: _state_mixture_operated_feedforward_tensor(
            state_mixture_operated_block,
            output_keys,
            state_mixture_operated_cache,
        ),
    )
    if not isinstance(state_mixture_operated_output, torch.Tensor):
        raise TypeError("StateMixture-operated feed-forward must return a tensor.")

    result["feedforward_probability_context"] = {
        "metrics": direct_metrics,
        "output": _output_summary(direct_output),
    }
    result["diagnostic_identity_recombination"] = {
        "purpose": (
            "Validate StateMixture recombination and basis expansion with an "
            "identity probability layer. This is not the representative "
            "QuantumLayer batching benchmark."
        ),
        "partial_measurement_state_mixture": partial_measurement_metrics,
        "partial_measurement_state_mixture_vs_feedforward_speedup": (
            direct_metrics["mean_s"] / partial_measurement_metrics["mean_s"]
            if partial_measurement_metrics["mean_s"] > 0
            else None
        ),
        "partial_measurement_state_mixture_vs_feedforward_max_abs_diff": (
            _tensor_max_abs_diff(direct_output, partial_measurement_output)
        ),
        "raw_branch_state_mixture": state_mixture_metrics,
        "state_mixture_vs_feedforward_speedup": (
            direct_metrics["mean_s"] / state_mixture_metrics["mean_s"]
            if state_mixture_metrics["mean_s"] > 0
            else None
        ),
        "state_mixture_vs_feedforward_max_abs_diff": _tensor_max_abs_diff(
            direct_output, state_mixture_output
        ),
        "state_mixture_operated_feedforward": state_mixture_operated_metrics,
        "state_mixture_operated_vs_feedforward_speedup": (
            direct_metrics["mean_s"] / state_mixture_operated_metrics["mean_s"]
            if state_mixture_operated_metrics["mean_s"] > 0
            else None
        ),
        "state_mixture_operated_vs_feedforward_max_abs_diff": _tensor_max_abs_diff(
            direct_output,
            state_mixture_operated_output,
        ),
    }
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
        "--representative-only",
        action="store_true",
        help="Skip identity diagnostics and run only the nontrivial downstream benchmark.",
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
        "representative_only": bool(args.representative_only),
        "cases": [
            _run_case(
                case,
                args.runs,
                args.warmups,
                representative_only=bool(args.representative_only),
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
