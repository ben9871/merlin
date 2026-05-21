# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import perceval as pcvl
import pytest
import torch

import merlin as ML
from merlin.core.partial_measurement import PartialMeasurement, PartialMeasurementBranch
from merlin.core.state_mixture import StateMixture, StateMixtureBranch
from merlin.core.state_vector import StateVector


def _single_photon_mixture(
    *,
    probabilities_require_grad: bool = False,
    states_require_grad: bool = False,
) -> StateMixture:
    tensor_a = torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j]], dtype=torch.complex64)
    tensor_b = torch.tensor([[0.0 + 0.0j, 1.0 + 0.0j]], dtype=torch.complex64)
    if states_require_grad:
        tensor_a.requires_grad_()
        tensor_b.requires_grad_()
    prob_a = torch.tensor([0.25], requires_grad=probabilities_require_grad)
    prob_b = torch.tensor([0.75], requires_grad=probabilities_require_grad)
    return StateMixture(
        branches=(
            StateMixtureBranch(
                probability=prob_a,
                state=StateVector(tensor_a, n_modes=2, n_photons=1),
                outcomes=((1,),),
            ),
            StateMixtureBranch(
                probability=prob_b,
                state=StateVector(tensor_b, n_modes=2, n_photons=1),
                outcomes=((0,),),
            ),
        ),
        measured_modes=(0,),
        unmeasured_modes=(1,),
    )


def _layer(strategy: ML.MeasurementStrategy) -> ML.QuantumLayer:
    return ML.QuantumLayer(
        circuit=pcvl.Circuit(2),
        input_size=0,
        n_photons=1,
        measurement_strategy=strategy,
    )


def test_probability_output_layer_accepts_state_mixture_and_recombines():
    mixture = _single_photon_mixture()
    layer = _layer(ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK))

    output = layer(mixture)

    assert output.shape == (1, 2)
    assert torch.allclose(output, torch.tensor([[0.25, 0.75]]))


def test_probability_output_layer_accepts_partial_measurement_directly():
    mixture = _single_photon_mixture()
    partial = PartialMeasurement(
        branches=tuple(
            PartialMeasurementBranch(
                outcome=branch.outcomes[-1],
                probability=branch.probability,
                amplitudes=branch.state,
            )
            for branch in mixture
        ),
        measured_modes=mixture.measured_modes,
        unmeasured_modes=mixture.unmeasured_modes,
    )
    layer = _layer(ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK))

    output = layer(partial)

    assert torch.allclose(output, torch.tensor([[0.25, 0.75]]))


def test_amplitude_output_layer_accepts_state_mixture_and_returns_mixture():
    mixture = _single_photon_mixture()
    layer = _layer(ML.MeasurementStrategy.amplitudes(ML.ComputationSpace.FOCK))

    output = layer(mixture)

    assert isinstance(output, StateMixture)
    assert len(output) == 2
    assert output.outcome_histories == mixture.outcome_histories
    assert output.branches[0].probability is mixture.branches[0].probability
    assert output.branches[1].probability is mixture.branches[1].probability
    assert torch.allclose(
        output.branches[0].state.tensor, mixture.branches[0].state.tensor
    )
    assert torch.allclose(
        output.branches[1].state.tensor, mixture.branches[1].state.tensor
    )


def test_partial_output_layer_accepts_state_mixture_and_merges_nested_outcomes():
    mixture = _single_photon_mixture()
    layer = _layer(
        ML.MeasurementStrategy.partial(
            modes=[0],
            computation_space=ML.ComputationSpace.FOCK,
        )
    )

    output = layer(mixture)

    assert isinstance(output, StateMixture)
    assert all(len(history) == 2 for history in output.outcome_histories)
    assert any(history == ((1,), (1,)) for history in output.outcome_histories)
    assert any(history == ((0,), (0,)) for history in output.outcome_histories)
    assert any(
        torch.allclose(probability, torch.tensor([0.25]))
        for probability in output.probabilities
    )
    assert any(
        torch.allclose(probability, torch.tensor([0.75]))
        for probability in output.probabilities
    )


def test_state_mixture_probability_output_keeps_gradients():
    mixture = _single_photon_mixture(
        probabilities_require_grad=True,
        states_require_grad=True,
    )
    layer = _layer(ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK))

    output = layer(mixture)
    loss = output[:, 0].sum()
    loss.backward()

    assert mixture.branches[0].probability.grad is not None
    assert mixture.branches[1].probability.grad is not None
    assert mixture.branches[0].state.tensor.grad is not None


def test_state_mixture_rejects_incompatible_branch_mode_count():
    bad_state = StateVector.from_basic_state([1, 0, 0], sparse=False)
    mixture = StateMixture(
        branches=(
            StateMixtureBranch(
                probability=torch.tensor([1.0]),
                state=bad_state,
            ),
        )
    )
    layer = _layer(ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK))

    with pytest.raises(ValueError, match="mode count"):
        layer(mixture)


def test_state_mixture_rejects_mixed_tensor_inputs():
    mixture = _single_photon_mixture()
    layer = _layer(ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK))

    with pytest.raises(TypeError, match="Cannot mix"):
        layer(mixture, torch.tensor([0.1]))
