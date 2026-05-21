# MIT License
#
# Copyright (c)
#
# Tests for FeedForwardBlock API.

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import perceval as pcvl
import pytest
import torch
from perceval import BasicState, Circuit, Matrix, Unitary
from perceval.algorithm import Sampler
from perceval.components import PERM
from perceval.utils import NoiseModel

from merlin.algorithms.feed_forward import BranchState, FeedForwardBlock
from merlin.algorithms.layer import QuantumLayer
from merlin.core.computation_space import ComputationSpace
from merlin.core.partial_measurement import PartialMeasurement
from merlin.core.state_mixture import StateMixture, StateMixtureBranch
from merlin.core.state_vector import StateVector
from merlin.measurement.strategies import MeasurementStrategy
from merlin.utils.combinadics import Combinadics

_BASIS_CACHE: dict[tuple[int, int], list[tuple[int, ...]]] = {}


def _basis_states(n_modes: int, n_photons: int) -> list[tuple[int, ...]]:
    cache_key = (n_modes, n_photons)
    if cache_key not in _BASIS_CACHE:
        _BASIS_CACHE[cache_key] = Combinadics(
            "fock", n_photons, n_modes
        ).enumerate_states()
    return _BASIS_CACHE[cache_key]


def _as_keyed_tensors(block: FeedForwardBlock, tensor: torch.Tensor):
    keys = block.output_keys
    mapped: dict[tuple[int, ...], torch.Tensor] = {}
    for idx, key in enumerate(keys):
        entry = tensor[:, idx]
        if entry.shape[0] == 1:
            entry = entry.squeeze(0)
        size = block.output_state_sizes[key]
        if entry.ndim > 1 and entry.shape[-1] > size:
            entry = entry[..., :size]
        mapped[key] = entry
    return mapped


def _build_balanced_feedforward_experiment():
    """Construct a small experiment with one detector and a feed-forward provider."""
    exp = pcvl.Experiment()
    root = pcvl.Circuit(3)
    root.add(0, pcvl.BS())
    exp.add(0, root)

    exp.add(0, pcvl.Detector.pnr())

    reflective = pcvl.Circuit(2)
    reflective.add(0, PERM([1, 0]))

    transmissive = pcvl.Circuit(2)
    transmissive.add(0, pcvl.BS())

    provider = pcvl.FFCircuitProvider(1, 0, reflective)
    provider.add_configuration([1], transmissive)
    exp.add(0, provider)
    return exp


def _build_two_stage_experiment():
    """Approximate the multi-level experiment from ff_perceval.py."""
    exp = pcvl.Experiment()
    root = pcvl.Circuit(4)
    root.add(0, pcvl.BS())
    exp.add(0, root)

    exp.add(0, pcvl.Detector.pnr())
    v0 = pcvl.Circuit(3) // pcvl.BS()
    v1 = pcvl.Circuit(3) // pcvl.BS()
    v2 = pcvl.Circuit(3) // pcvl.BS()
    provider1 = pcvl.FFCircuitProvider(1, 0, v0)
    provider1.add_configuration([1], v1)
    provider1.add_configuration([2], v2)
    exp.add(0, provider1)

    exp.add(3, pcvl.Detector.threshold())
    provider2 = pcvl.FFCircuitProvider(1, -1, pcvl.Circuit(2))
    provider2.add_configuration([1], pcvl.Circuit(2) // pcvl.BS())
    exp.add(3, provider2)

    for mode in (1, 2):
        exp.add(mode, pcvl.Detector.pnr())

    return exp


def test_feedforward_block2_balanced_split():
    exp = _build_balanced_feedforward_experiment()
    block = FeedForwardBlock(
        exp,
        input_state=[2, 0, 0],
    )

    outputs = block()
    distribution_map = _as_keyed_tensors(block, outputs)

    total_prob = 0.0
    measurement_probs = defaultdict(float)
    for key, probability in distribution_map.items():
        if probability.ndim:
            prob_value = probability.squeeze().item()
        else:
            prob_value = probability.item()
        total_prob += prob_value
        measurement_probs[key[0]] += prob_value

    assert math.isclose(total_prob, 1.0, rel_tol=1e-5)
    assert len(measurement_probs) == 3


def test_feedforward_block2_parses_multiple_stages():
    exp = _build_two_stage_experiment()
    block = FeedForwardBlock(exp, input_state=[1, 1, 0, 0])

    assert len(block.stages) == 2
    assert block.stages[0].measured_modes == (0,)
    assert block.stages[1].measured_modes == (3,)
    desc = block.describe()
    assert "Stage 1" in desc and "Stage 2" in desc


def test_feedforward_block_uses_experiment_input_state():
    exp_with_state = _build_balanced_feedforward_experiment()
    exp_with_state.with_input(BasicState([2, 0, 0]))
    block_from_experiment = FeedForwardBlock(exp_with_state)

    exp_reference = _build_balanced_feedforward_experiment()
    block_reference = FeedForwardBlock(exp_reference, input_state=[2, 0, 0])

    assert torch.allclose(block_from_experiment(), block_reference())


def test_feedforward_block_warns_on_conflicting_input_state():
    exp = _build_balanced_feedforward_experiment()
    exp.with_input(BasicState([2, 0, 0]))
    with pytest.warns(UserWarning):
        FeedForwardBlock(exp, input_state=[1, 1, 0])


def test_feedforward_block_rejects_noisy_experiment():
    exp = _build_balanced_feedforward_experiment()
    exp.noise = NoiseModel(brightness=0.9)
    with pytest.raises(NotImplementedError):
        FeedForwardBlock(exp, input_state=[2, 0, 0])


def test_feedforward_block2_matches_perceval_two_stage():
    exp = _build_two_stage_experiment()
    input_state = [1, 1, 0, 0]
    block = FeedForwardBlock(exp, input_state=input_state)

    block_outputs = block()
    distribution_map = _as_keyed_tensors(block, block_outputs)
    block_probs = {
        key: float(value.item() if value.ndim == 0 else value.squeeze().item())
        for key, value in distribution_map.items()
    }

    exp.with_input(pcvl.BasicState(input_state))
    processor = pcvl.Processor("SLOS", exp)
    sampler = Sampler(processor)
    perceval_results = dict(sampler.probs()["results"])
    perceval_probs = {
        tuple(int(v) for v in state): float(prob)
        for state, prob in perceval_results.items()
    }
    assert set(block_probs) == set(perceval_probs)
    for key, value in block_probs.items():
        assert math.isclose(value, perceval_probs[key], rel_tol=1e-5, abs_tol=1e-5)


def test_feedforward_block2_amplitude_strategy_matches_probabilities():
    exp = _build_balanced_feedforward_experiment()
    input_state = [2, 0, 0]
    block_prob = FeedForwardBlock(exp, input_state=input_state)
    block_amp = FeedForwardBlock(
        exp,
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.NONE,
    )

    prob_outputs = block_prob()
    prob_map = _as_keyed_tensors(block_prob, prob_outputs)
    amp_outputs = block_amp()
    assert isinstance(amp_outputs, list)

    full_probabilities = {
        key: float(value.item() if value.ndim == 0 else value.squeeze().item())
        for key, value in prob_map.items()
    }
    reconstructed: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for measurement_key, branch_prob, remaining_n, amp_tensor in amp_outputs:
        assert torch.is_complex(amp_tensor)
        prob = branch_prob
        while prob.ndim < amp_tensor.ndim:
            prob = prob.unsqueeze(-1)
        distribution = amp_tensor.abs().pow(2) * prob
        unmeasured = [idx for idx, value in enumerate(measurement_key) if value is None]
        basis = _basis_states(len(unmeasured), remaining_n)
        states = basis if basis else ((),)
        flat_distribution = distribution.reshape(-1).tolist()
        for state, prob_value in zip(states, flat_distribution, strict=False):
            full_key = list(measurement_key)
            for mode_idx, value in zip(unmeasured, state, strict=False):
                full_key[mode_idx] = value
            reconstructed[tuple(full_key)] += prob_value

    assert set(full_probabilities.keys()) == set(reconstructed.keys())
    for key, value in full_probabilities.items():
        assert math.isclose(value, reconstructed[key], rel_tol=1e-6, abs_tol=1e-6)


def test_feedforward_block2_mode_expectations():
    exp = _build_balanced_feedforward_experiment()
    input_state = [2, 0, 0]
    block_prob = FeedForwardBlock(exp, input_state=input_state)
    block_expect = FeedForwardBlock(
        exp,
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.mode_expectations(
            ComputationSpace.UNBUNCHED
        ),
    )

    prob_outputs = block_prob()
    expect_outputs = block_expect()
    prob_map = _as_keyed_tensors(block_prob, prob_outputs)
    prob_scalars = {
        key: float(value.item() if value.ndim == 0 else value.squeeze().item())
        for key, value in prob_map.items()
    }
    expectation = expect_outputs.squeeze(0)
    manual = torch.zeros_like(expectation)
    for state, probability in prob_scalars.items():
        state_tensor = torch.tensor(
            state, dtype=expectation.dtype, device=expectation.device
        )
        manual += probability * state_tensor
    assert torch.allclose(manual, expectation, atol=1e-6, rtol=1e-6)


def test_feedforward_block2_accepts_tensor_input_state():
    exp = _build_balanced_feedforward_experiment()
    block_basic = FeedForwardBlock(exp, input_state=[2, 0, 0])
    basis = _basis_states(3, 2)
    amplitudes = torch.zeros(len(basis), dtype=torch.complex64)
    amplitudes[basis.index((2, 0, 0))] = 1.0
    block_tensor = FeedForwardBlock(exp, input_state=amplitudes)

    ref_outputs = _as_keyed_tensors(block_basic, block_basic())
    tensor_outputs = _as_keyed_tensors(block_tensor, block_tensor())
    for key in block_basic.output_keys:
        assert torch.allclose(ref_outputs[key], tensor_outputs[key], atol=1e-6)


def test_feedforward_block2_accepts_state_vector_input():
    exp = _build_balanced_feedforward_experiment()
    block_basic = FeedForwardBlock(exp, input_state=[2, 0, 0])
    state_vector = pcvl.StateVector()
    state_vector += pcvl.StateVector(pcvl.BasicState([2, 0, 0])) * 1.0
    block_sv = FeedForwardBlock(exp, input_state=state_vector)

    ref_outputs = _as_keyed_tensors(block_basic, block_basic())
    sv_outputs = _as_keyed_tensors(block_sv, block_sv())
    for key in block_basic.output_keys:
        assert torch.allclose(ref_outputs[key], sv_outputs[key], atol=1e-6)


def test_feedforward_block2_input_and_trainable_parameters_backward():
    exp = pcvl.Experiment()
    root = pcvl.Circuit(2)
    root.add(0, pcvl.PS(pcvl.P("phi")))
    root.add((0, 1), pcvl.BS(theta=pcvl.P("theta_1")))
    exp.add(0, root)
    exp.add(0, pcvl.Detector.pnr())

    conditional = pcvl.Circuit(1)
    conditional.add(0, pcvl.PS(pcvl.P("theta_2")))
    provider = pcvl.FFCircuitProvider(1, 0, conditional)
    exp.add(0, provider)

    block = FeedForwardBlock(
        exp,
        input_state=[1, 0],
        input_parameters=["phi"],
        trainable_parameters=["theta"],
    )

    x = torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True)
    outputs = block(x)
    target_index = next(idx for idx, key in enumerate(block.output_keys) if key[0] == 1)
    loss = outputs[:, target_index].real.sum()
    loss.backward()

    assert x.grad is not None
    # assert torch.any(x.grad.abs() > 0)
    # assert any(
    #    parameter.grad is not None and torch.any(parameter.grad != 0)
    #    for parameter in block.parameters()
    # )


def test_feedforward_block2_forward_without_inputs_matches_explicit_tensor():
    exp = _build_balanced_feedforward_experiment()
    block = FeedForwardBlock(exp, input_state=[2, 0, 0])

    automatic = block()
    explicit = block(torch.zeros((1, 0)))
    assert torch.allclose(automatic, explicit)


def test_feedforward_block2_requires_classical_features_when_needed():
    exp = pcvl.Experiment()
    circuit = pcvl.Circuit(2)
    circuit.add(0, pcvl.PS(pcvl.P("phi")))
    exp.add(0, circuit)
    exp.add(0, pcvl.Detector.pnr())
    provider = pcvl.FFCircuitProvider(1, 0, pcvl.Circuit(1))
    exp.add(0, provider)

    block = FeedForwardBlock(exp, input_state=[1, 0], input_parameters=["phi"])

    with pytest.raises(ValueError, match="provide a feature tensor"):
        block()

    one_d = torch.tensor([0.2], dtype=torch.float32)
    block(one_d.unsqueeze(0))
    block(one_d)


def _fourier_unitary(dim: int) -> Unitary:
    omega = np.exp(2j * np.pi / dim)
    matrix = np.empty((dim, dim), dtype=np.complex128)
    scale = 1 / math.sqrt(dim)
    for row in range(dim):
        for col in range(dim):
            matrix[row, col] = omega ** (row * col) * scale
    return Unitary(Matrix(matrix))


def _build_feedforward_experiment(detector) -> tuple[pcvl.Experiment, list[int]]:
    m = 4
    input_state = [1, 1, 0, 0]

    exp = pcvl.Experiment()
    root = Circuit(m)
    root.add(0, _fourier_unitary(m))
    root.add((0, 1), pcvl.BS())
    exp.add(0, root)

    exp.add(0, detector)

    default_branch = Circuit(m - 1)
    default_branch.add(0, _fourier_unitary(m - 1))

    adaptive_branch = Circuit(m - 1)
    adaptive_branch.add(0, PERM([2, 1, 0]))
    adaptive_branch.add(0, _fourier_unitary(m - 1))

    provider = pcvl.FFCircuitProvider(1, 0, default_branch)
    provider.add_configuration([1], adaptive_branch)
    exp.add(0, provider)

    exp.with_input(BasicState(input_state))

    return exp


def _build_multi_threshold_feedforward_experiment() -> pcvl.Experiment:
    m = 5
    n_measured_modes = 3
    input_state = [1, 1, 1, 1, 1]

    exp = pcvl.Experiment()
    root = Circuit(m)
    root.add(0, _fourier_unitary(m))
    for mode in range(4):
        root.add((mode, mode + 1), pcvl.BS())
    exp.add(0, root)

    for mode in range(n_measured_modes):
        exp.add(mode, pcvl.Detector.threshold())

    n_remaining_modes = m - n_measured_modes
    default_branch = Circuit(n_remaining_modes)
    default_branch.add(0, _fourier_unitary(n_remaining_modes))

    adaptive_branch = Circuit(n_remaining_modes)
    adaptive_branch.add(0, PERM(list(reversed(range(n_remaining_modes)))))
    adaptive_branch.add(0, _fourier_unitary(n_remaining_modes))

    provider = pcvl.FFCircuitProvider(n_measured_modes, 0, default_branch)
    provider.add_configuration([1] * n_measured_modes, adaptive_branch)
    exp.add(0, provider)
    exp.with_input(BasicState(input_state))
    return exp


def _nontrivial_downstream_layer(n_modes: int, n_photons: int) -> QuantumLayer:
    circuit = Circuit(n_modes)
    circuit.add(0, _fourier_unitary(n_modes))
    for mode in range(n_modes - 1):
        circuit.add((mode, mode + 1), pcvl.BS())
    circuit.add(0, _fourier_unitary(n_modes))
    return QuantumLayer(
        input_size=0,
        circuit=circuit,
        input_state=[n_photons, *([0] * (n_modes - 1))],
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
    )


def _first_compatible_feedforward_mixture(block: FeedForwardBlock) -> StateMixture:
    for measurement_key, branch_list in _feedforward_branch_states(block).items():
        grouped: dict[int, list[BranchState]] = defaultdict(list)
        for branch in branch_list:
            if branch.remaining_n > 0:
                grouped[branch.remaining_n].append(branch)
        for _remaining_n, branches in grouped.items():
            if len(branches) < 2:
                continue
            unmeasured_modes = [
                idx for idx, value in enumerate(measurement_key) if value is None
            ]
            return StateMixture(
                branches=tuple(
                    _state_mixture_branch_from_feedforward_branch(
                        measurement_key, branch
                    )
                    for branch in branches
                ),
                measured_modes=tuple(
                    idx
                    for idx, value in enumerate(measurement_key)
                    if value is not None
                ),
                unmeasured_modes=tuple(unmeasured_modes),
            )
    raise AssertionError("Expected at least one compatible feed-forward branch group.")


def _perceval_probabilities(exp: pcvl.Experiment) -> dict[tuple[int, ...], float]:
    processor = pcvl.Processor("SLOS", exp)
    sampler = Sampler(processor)
    results = sampler.probs()["results"]
    return {
        tuple(int(value) for value in state): float(prob)
        for state, prob in results.items()
    }


def _prune_probabilities(
    distribution: dict[tuple[int, ...], float], *, atol: float = 1e-12
) -> dict[tuple[int, ...], float]:
    """Remove numerically empty entries from a probability map."""
    return {key: value for key, value in distribution.items() if abs(value) > atol}


def _block_probabilities(
    block: FeedForwardBlock, outputs: torch.Tensor
) -> dict[tuple[int, ...], float]:
    if outputs.shape[0] != 1:
        raise AssertionError("Test expects a single batch item.")
    batch = outputs.squeeze(0)
    return {block.output_keys[idx]: float(batch[idx]) for idx in range(batch.shape[0])}


def _state_mixture_branch_from_feedforward_branch(
    measurement_key: tuple[int | None, ...],
    branch: BranchState,
) -> StateMixtureBranch:
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


def _feedforward_branch_states(
    block: FeedForwardBlock,
) -> dict[tuple[int | None, ...], list[BranchState]]:
    feature_tensor = block._prepare_classical_features(None)
    branches = block._run_stage(block._stage_runtimes[0], feature_tensor)
    for runtime in block._stage_runtimes[1:]:
        branches = block._propagate_future_stage(branches, runtime)
    return branches


def _probabilities_from_raw_feedforward_branches(
    block: FeedForwardBlock,
) -> dict[tuple[int, ...], float]:
    grouped_branches: dict[
        tuple[tuple[int | None, ...], int], list[StateMixtureBranch]
    ] = defaultdict(list)
    reconstructed: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for measurement_key, branch_list in _feedforward_branch_states(block).items():
        for branch in branch_list:
            if branch.remaining_n == 0:
                full_key = list(measurement_key)
                for mode_idx, value in enumerate(full_key):
                    if value is None:
                        full_key[mode_idx] = 0
                probability = torch.nan_to_num(branch.weight, nan=0.0)
                if probability.ndim == 0:
                    reconstructed[tuple(int(value) for value in full_key)] += float(
                        probability
                    )
                    continue
                if probability.shape[0] != 1:
                    raise AssertionError("Test expects a single batch item.")
                reconstructed[tuple(int(value) for value in full_key)] += float(
                    probability[0]
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
        layer = QuantumLayer(
            input_size=0,
            circuit=pcvl.Circuit(len(unmeasured_modes)),
            input_state=[remaining_n, *([0] * (len(unmeasured_modes) - 1))],
            n_photons=remaining_n,
            measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
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
            reconstructed[tuple(int(value) for value in full_key)] += float(
                remaining_probabilities[0, basis_index]
            )
    return dict(reconstructed)


def _probabilities_from_state_mixture_feedforward_operations(
    block: FeedForwardBlock, *, require_batched_multi_branch: bool = False
) -> dict[tuple[int, ...], float]:
    if len(block._stage_runtimes) != 1:
        raise AssertionError("Test helper supports one-stage feed-forward blocks only.")
    runtime = block._stage_runtimes[0]
    if runtime.pre_layer is None or runtime.detector_transform is None:
        raise AssertionError("Feed-forward runtime is not fully initialized.")

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

    reconstructed: defaultdict[tuple[int, ...], float] = defaultdict(float)
    batched_multi_branch_groups = 0
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
                probability = branch.probability
                if probability.ndim == 0:
                    reconstructed[full_key] += float(probability)
                else:
                    reconstructed[full_key] += float(probability[0])
            continue

        conditional_layer = block._select_conditional_layer(
            runtime, reduced_key, remaining_n
        )
        propagated = mixture
        if conditional_layer is not None:
            expected_dim = len(
                conditional_layer.computation_process.simulation_graph.mapped_keys
            )
            if mixture.branches[0].state.tensor.shape[-1] == expected_dim:
                restore = None
                if require_batched_multi_branch and len(mixture) > 1:
                    batched_multi_branch_groups += 1
                    restore = conditional_layer._state_mixture_branch_outputs
                    conditional_layer._state_mixture_branch_outputs = (
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError(
                                "StateMixture feed-forward conditional propagation "
                                "used the sequential branch path."
                            )
                        )
                    )
                try:
                    conditional_output = conditional_layer(mixture)
                finally:
                    if restore is not None:
                        conditional_layer._state_mixture_branch_outputs = restore
                assert isinstance(conditional_output, StateMixture)
                propagated = conditional_output

        probability_layer = QuantumLayer(
            input_size=0,
            circuit=pcvl.Circuit(len(unmeasured_modes)),
            input_state=[remaining_n, *([0] * (len(unmeasured_modes) - 1))],
            n_photons=remaining_n,
            measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
        )
        remaining_probabilities = probability_layer(propagated)
        if remaining_probabilities.ndim == 1:
            remaining_probabilities = remaining_probabilities.unsqueeze(0)
        for basis_index, basis_state in enumerate(
            _basis_states(len(unmeasured_modes), remaining_n)
        ):
            full_key = list(global_key)
            for mode_idx, value in zip(unmeasured_modes, basis_state, strict=False):
                full_key[mode_idx] = value
            reconstructed[tuple(int(value) for value in full_key)] += float(
                remaining_probabilities[0, basis_index]
            )

    if require_batched_multi_branch and batched_multi_branch_groups == 0:
        raise AssertionError("Expected a compatible multi-branch StateMixture group.")
    return dict(reconstructed)


def _first_stage_partial_measurement_layer(block: FeedForwardBlock) -> QuantumLayer:
    runtime = block._stage_runtimes[0]
    if block._base_input_state is None:
        raise AssertionError("PartialMeasurement test requires a basis input state.")

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
    )


def _global_key_from_partial_outcome(
    block: FeedForwardBlock, outcome: tuple[int, ...]
) -> tuple[int | None, ...]:
    runtime = block._stage_runtimes[0]
    full_key: list[int | None] = [None] * block.total_modes
    for mode, value in zip(runtime.global_measured_modes, outcome, strict=True):
        full_key[mode] = int(value)
    return tuple(full_key)


def _probabilities_from_partial_measurement_state_mixture_pipeline(
    block: FeedForwardBlock,
    *,
    require_batched_multi_branch: bool = False,
) -> dict[tuple[int, ...], float]:
    runtime = block._stage_runtimes[0]
    partial_layer = _first_stage_partial_measurement_layer(block)
    partial = partial_layer()
    if not isinstance(partial, PartialMeasurement):
        raise AssertionError(
            "Expected MeasurementStrategy.partial() to return PartialMeasurement."
        )

    grouped_branches: dict[
        tuple[tuple[int | None, ...], tuple[int, ...], int],
        list[StateMixtureBranch],
    ] = defaultdict(list)
    reconstructed: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for branch in partial.to_state_mixture():
        outcome = branch.outcomes[-1] if branch.outcomes else ()
        global_key = _global_key_from_partial_outcome(block, outcome)
        remaining_n = branch.state.n_photons
        if remaining_n == 0:
            full_key = tuple(0 if value is None else int(value) for value in global_key)
            probability = branch.probability
            reconstructed[full_key] += float(
                probability if probability.ndim == 0 else probability[0]
            )
            continue
        grouped_branches[(global_key, outcome, remaining_n)].append(branch)

    batched_multi_branch_groups = 0
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
                raise AssertionError(
                    "PartialMeasurement StateMixture test cannot apply a "
                    "conditional layer with a mismatched basis dimension."
                )
            restore = None
            if require_batched_multi_branch and len(mixture) > 1:
                batched_multi_branch_groups += 1
                restore = conditional_layer._state_mixture_branch_outputs
                conditional_layer._state_mixture_branch_outputs = (
                    lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError(
                            "PartialMeasurement StateMixture propagation "
                            "used the sequential branch path."
                        )
                    )
                )
            try:
                conditional_output = conditional_layer(mixture)
            finally:
                if restore is not None:
                    conditional_layer._state_mixture_branch_outputs = restore
            assert isinstance(conditional_output, StateMixture)
            propagated = conditional_output

        probability_layer = QuantumLayer(
            input_size=0,
            circuit=pcvl.Circuit(len(unmeasured_modes)),
            input_state=[remaining_n, *([0] * (len(unmeasured_modes) - 1))],
            n_photons=remaining_n,
            measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
        )
        remaining_probabilities = probability_layer(propagated)
        if remaining_probabilities.ndim == 1:
            remaining_probabilities = remaining_probabilities.unsqueeze(0)
        for basis_index, basis_state in enumerate(
            _basis_states(len(unmeasured_modes), remaining_n)
        ):
            full_key = list(global_key)
            for mode_idx, value in zip(unmeasured_modes, basis_state, strict=False):
                full_key[mode_idx] = value
            reconstructed[tuple(int(value) for value in full_key)] += float(
                remaining_probabilities[0, basis_index]
            )

    if require_batched_multi_branch and batched_multi_branch_groups == 0:
        raise AssertionError("Expected a compatible multi-branch StateMixture group.")
    return dict(reconstructed)


def test_feedforward_block_matches_perceval_distribution():
    experiment = _build_feedforward_experiment(pcvl.Detector.pnr())
    block = FeedForwardBlock(experiment)

    classical_inputs = torch.zeros((1, 0))
    outputs = block(classical_inputs)
    block_probs = _prune_probabilities(_block_probabilities(block, outputs))

    perceval_probs = _prune_probabilities(_perceval_probabilities(experiment))

    assert set(block_probs.keys()) == set(perceval_probs.keys())
    for key, prob in block_probs.items():
        assert math.isclose(prob, perceval_probs[key], rel_tol=1e-5, abs_tol=1e-5), (
            f"Mismatch for key {key}: Merlin={prob}, Perceval={perceval_probs[key]}"
        )


@pytest.mark.parametrize(
    "detector_factory", [pcvl.Detector.pnr, pcvl.Detector.threshold]
)
def test_feedforward_block_state_mixture_recombination_matches_probability_path(
    detector_factory,
):
    probability_block = FeedForwardBlock(
        _build_feedforward_experiment(detector_factory())
    )
    state_mixture_block = FeedForwardBlock(
        _build_feedforward_experiment(detector_factory())
    )

    probability_outputs = probability_block()

    expected = _prune_probabilities(
        _block_probabilities(probability_block, probability_outputs)
    )
    reconstructed = _prune_probabilities(
        _probabilities_from_raw_feedforward_branches(state_mixture_block)
    )

    assert set(reconstructed) == set(expected)
    for key, value in expected.items():
        assert math.isclose(
            value,
            reconstructed[key],
            rel_tol=1e-5,
            abs_tol=1e-5,
        ), (
            f"Mismatch for key {key}: FeedForwardBlock={value}, StateMixture={reconstructed[key]}"
        )


def test_feedforward_block_state_mixture_operated_path_matches_probability_path():
    probability_block = FeedForwardBlock(
        _build_multi_threshold_feedforward_experiment()
    )
    state_mixture_block = FeedForwardBlock(
        _build_multi_threshold_feedforward_experiment()
    )

    probability_outputs = probability_block()

    expected = _prune_probabilities(
        _block_probabilities(probability_block, probability_outputs)
    )
    reconstructed = _prune_probabilities(
        _probabilities_from_state_mixture_feedforward_operations(
            state_mixture_block,
            require_batched_multi_branch=True,
        )
    )

    assert set(reconstructed) == set(expected)
    for key, value in expected.items():
        assert math.isclose(
            value,
            reconstructed[key],
            rel_tol=1e-5,
            abs_tol=1e-5,
        ), (
            f"Mismatch for key {key}: FeedForwardBlock={value}, StateMixture-operated={reconstructed[key]}"
        )


def test_feedforward_partial_measurement_state_mixture_pipeline_matches_probability_path():
    probability_block = FeedForwardBlock(
        _build_multi_threshold_feedforward_experiment()
    )
    partial_block = FeedForwardBlock(_build_multi_threshold_feedforward_experiment())

    probability_outputs = probability_block()

    expected = _prune_probabilities(
        _block_probabilities(probability_block, probability_outputs)
    )
    reconstructed = _prune_probabilities(
        _probabilities_from_partial_measurement_state_mixture_pipeline(
            partial_block,
            require_batched_multi_branch=True,
        )
    )

    assert set(reconstructed) == set(expected)
    for key, value in expected.items():
        assert math.isclose(
            value,
            reconstructed[key],
            rel_tol=1e-5,
            abs_tol=1e-5,
        ), (
            f"Mismatch for key {key}: FeedForwardBlock={value}, PartialMeasurement+StateMixture={reconstructed[key]}"
        )


def test_feedforward_state_mixture_batches_nontrivial_downstream_layer(monkeypatch):
    block = FeedForwardBlock(_build_multi_threshold_feedforward_experiment())
    mixture = _first_compatible_feedforward_mixture(block)
    assert len(mixture) > 1
    n_modes = mixture.branches[0].state.n_modes
    n_photons = mixture.branches[0].state.n_photons

    sequential_layer = _nontrivial_downstream_layer(n_modes, n_photons)
    sequential_outputs = sequential_layer._state_mixture_branch_outputs(
        tuple(mixture),
        shots=None,
        sampling_method=None,
        simultaneous_processes=None,
    )
    expected = sequential_layer._weighted_state_mixture_tensor(
        mixture, sequential_outputs
    )

    batched_layer = _nontrivial_downstream_layer(n_modes, n_photons)
    monkeypatch.setattr(
        batched_layer,
        "_state_mixture_branch_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("StateMixture downstream propagation used sequential path.")
        ),
    )

    output = batched_layer(mixture)

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)
