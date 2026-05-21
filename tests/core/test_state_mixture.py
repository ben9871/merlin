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

import pytest
import torch

from merlin.core.partial_measurement import PartialMeasurement, PartialMeasurementBranch
from merlin.core.state_mixture import StateMixture, StateMixtureBranch
from merlin.core.state_vector import StateVector


def _state(batch: int = 2) -> StateVector:
    tensor = torch.randn(batch, 2, dtype=torch.complex64)
    return StateVector(tensor=tensor, n_modes=2, n_photons=1)


def test_state_mixture_branch_validates_probability_and_state_types():
    state = _state()

    with pytest.raises(TypeError, match="probability"):
        StateMixtureBranch(probability=0.5, state=state)

    with pytest.raises(TypeError, match="state"):
        StateMixtureBranch(probability=torch.tensor([0.5, 0.5]), state=object())

    with pytest.raises(TypeError, match="real-valued"):
        StateMixtureBranch(
            probability=torch.tensor([0.5 + 0.0j, 0.5 + 0.0j]),
            state=state,
        )


def test_state_mixture_branch_validates_batched_probability_shape():
    state = _state(batch=3)

    StateMixtureBranch(probability=torch.tensor([0.2, 0.3, 0.5]), state=state)

    with pytest.raises(ValueError, match="probability shape"):
        StateMixtureBranch(probability=torch.tensor([0.5, 0.5]), state=state)


def test_state_mixture_from_partial_measurement_preserves_branch_data():
    prob_a = torch.tensor([0.25, 0.40], requires_grad=True)
    prob_b = torch.tensor([0.75, 0.60], requires_grad=True)
    state_a = _state()
    state_b = _state()
    partial = PartialMeasurement(
        branches=(
            PartialMeasurementBranch(
                outcome=(1, 0),
                probability=prob_a,
                amplitudes=state_a,
            ),
            PartialMeasurementBranch(
                outcome=(0, 1),
                probability=prob_b,
                amplitudes=state_b,
            ),
        ),
        measured_modes=(0, 2),
        unmeasured_modes=(1, 3),
    )

    mixture = StateMixture.from_partial_measurement(partial)

    assert mixture.measured_modes == (0, 2)
    assert mixture.unmeasured_modes == (1, 3)
    assert mixture.probabilities[0] is prob_b
    assert mixture.probabilities[1] is prob_a
    assert mixture.states[0] is state_b
    assert mixture.states[1] is state_a
    assert mixture.outcomes == ((0, 1), (1, 0))
    assert mixture.outcome_histories == (((0, 1),), ((1, 0),))


def test_partial_measurement_to_state_mixture_delegates_to_converter():
    state = StateVector.from_basic_state([1, 0], sparse=False)
    partial = PartialMeasurement(
        branches=(
            PartialMeasurementBranch(
                outcome=(1,),
                probability=torch.tensor([1.0]),
                amplitudes=state,
            ),
        ),
        measured_modes=(0,),
        unmeasured_modes=(1,),
    )

    direct = StateMixture.from_partial_measurement(partial)
    delegated = partial.to_state_mixture()

    assert delegated.probabilities[0] is direct.probabilities[0]
    assert delegated.states[0] is direct.states[0]
    assert delegated.outcome_histories == direct.outcome_histories
    assert delegated.measured_modes == direct.measured_modes
    assert delegated.unmeasured_modes == direct.unmeasured_modes


def test_state_mixture_properties_expose_branch_data_in_order():
    state_a = StateVector.from_basic_state([1, 0], sparse=False)
    state_b = StateVector.from_basic_state([0, 1], sparse=False)
    prob_a = torch.tensor([0.4])
    prob_b = torch.tensor([0.6])
    mixture = StateMixture(
        branches=(
            StateMixtureBranch(
                probability=prob_a,
                state=state_a,
                outcomes=((1,),),
            ),
            StateMixtureBranch(
                probability=prob_b,
                state=state_b,
                outcomes=((0,), (2,)),
            ),
        ),
        measured_modes=(0,),
        unmeasured_modes=(1,),
    )

    assert len(mixture) == 2
    assert tuple(id(branch) for branch in mixture) == tuple(
        id(branch) for branch in mixture.branches
    )
    assert mixture.probabilities[0] is prob_a
    assert mixture.probabilities[1] is prob_b
    assert mixture.states[0] is state_a
    assert mixture.states[1] is state_b
    assert mixture.outcomes == ((1,), (2,))
    assert mixture.outcome_histories == (((1,),), ((0,), (2,)))


def test_state_mixture_rejects_empty_branch_list():
    with pytest.raises(ValueError, match="at least one branch"):
        StateMixture(branches=())
