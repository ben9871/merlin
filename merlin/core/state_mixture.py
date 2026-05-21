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

"""Classical mixtures of conditional state-vector branches."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .state_vector import StateVector

if TYPE_CHECKING:
    from .partial_measurement import PartialMeasurement

Outcome = tuple[int, ...]
OutcomeHistory = tuple[Outcome, ...]


def _normalize_outcome_history(outcomes: OutcomeHistory) -> OutcomeHistory:
    """Validate and normalize a branch outcome history."""
    normalized: list[Outcome] = []
    for outcome in outcomes:
        if not isinstance(outcome, tuple):
            raise TypeError("StateMixtureBranch outcomes must be tuples of integers.")
        normalized.append(tuple(_validate_int(value) for value in outcome))
    return tuple(normalized)


def _validate_int(value: int) -> int:
    """Return ``value`` as an int after rejecting booleans and non-integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("StateMixtureBranch outcomes must contain integers.")
    return value


def _state_batch_shape(state: StateVector) -> tuple[int, ...]:
    """Return the probability batch shape expected by a branch state."""
    if state.tensor.ndim <= 1:
        return (1,)
    return tuple(int(dim) for dim in state.tensor.shape[:-1])


@dataclass(frozen=True)
class StateMixtureBranch:
    """Single branch of a classical mixture of pure conditional states.

    Parameters
    ----------
    probability : torch.Tensor
        Branch probability. A scalar is accepted for unbatched states. Batched
        states require a probability tensor whose shape matches the leading
        batch dimensions of ``state``.
    state : merlin.core.state_vector.StateVector
        Conditional pure state carried by the branch.
    outcomes : tuple[tuple[int, ...], ...]
        History of measured outcomes that produced this branch. If omitted, no
        outcome history is attached. Default value is ``()``.

    Raises
    ------
    TypeError
        If ``probability`` is not a tensor, ``state`` is not a
        :class:`merlin.core.state_vector.StateVector`, or ``outcomes`` is not a
        tuple of integer tuples.
    ValueError
        If a batched probability shape is incompatible with the branch state's
        batch shape.
    """

    probability: torch.Tensor
    state: StateVector
    outcomes: OutcomeHistory = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate branch probability, state, and outcome metadata."""
        if not isinstance(self.probability, torch.Tensor):
            raise TypeError("StateMixtureBranch probability must be a torch.Tensor.")
        if self.probability.is_complex():
            raise TypeError("StateMixtureBranch probability must be real-valued.")
        if not isinstance(self.state, StateVector):
            raise TypeError("StateMixtureBranch state must be a StateVector.")
        object.__setattr__(self, "outcomes", _normalize_outcome_history(self.outcomes))
        self._validate_probability_shape()

    def _validate_probability_shape(self) -> None:
        """Validate branch probability shape against the branch state batch."""
        batch_shape = _state_batch_shape(self.state)
        probability_shape = tuple(int(dim) for dim in self.probability.shape)
        if self.probability.ndim == 0:
            if batch_shape != (1,):
                raise ValueError(
                    "Scalar branch probability is valid only for unbatched "
                    f"states; got state batch shape {batch_shape}."
                )
            return
        if probability_shape != batch_shape:
            raise ValueError(
                "Branch probability shape must match the branch state batch "
                f"shape: got probability shape {probability_shape}, expected "
                f"{batch_shape}."
            )


@dataclass(frozen=True)
class StateMixture:
    """Classical mixture of conditional :class:`merlin.core.state_vector.StateVector` branches.

    ``StateMixture`` is the propagatable carrier produced after a partial
    measurement. It stores classical branch probabilities separately from the
    conditional pure states, so downstream layers can propagate each branch and
    recombine tensor outputs by probability.

    Parameters
    ----------
    branches : tuple[StateMixtureBranch, ...]
        Branches in deterministic order.
    measured_modes : tuple[int, ...]
        Modes measured to produce the current mixture. For nested mixtures,
        appended mode tuples are stored in the local coordinate system of each
        layer. Default value is ``()``.
    unmeasured_modes : tuple[int, ...]
        Modes still represented by the branch states. Default value is ``()``.

    Raises
    ------
    TypeError
        If branches or mode metadata have invalid types.
    ValueError
        If no branches are provided.
    """

    branches: tuple[StateMixtureBranch, ...]
    measured_modes: tuple[int, ...] = field(default_factory=tuple)
    unmeasured_modes: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate branch collection and mode metadata."""
        if not self.branches:
            raise ValueError("StateMixture requires at least one branch.")
        if not all(isinstance(branch, StateMixtureBranch) for branch in self.branches):
            raise TypeError("StateMixture branches must be StateMixtureBranch objects.")
        object.__setattr__(
            self,
            "measured_modes",
            tuple(_validate_int(mode) for mode in self.measured_modes),
        )
        object.__setattr__(
            self,
            "unmeasured_modes",
            tuple(_validate_int(mode) for mode in self.unmeasured_modes),
        )

    def __iter__(self) -> Iterator[StateMixtureBranch]:
        """Iterate over branches in stored order.

        Returns
        -------
        Iterator[StateMixtureBranch]
            Iterator over mixture branches.
        """
        return iter(self.branches)

    def __len__(self) -> int:
        """Return the number of mixture branches.

        Returns
        -------
        int
            Number of branches.
        """
        return len(self.branches)

    @property
    def probabilities(self) -> tuple[torch.Tensor, ...]:
        """Return branch probabilities in branch order.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Branch probability tensors.
        """
        return tuple(branch.probability for branch in self.branches)

    @property
    def states(self) -> tuple[StateVector, ...]:
        """Return conditional states in branch order.

        Returns
        -------
        tuple[merlin.core.state_vector.StateVector, ...]
            Conditional branch states.
        """
        return tuple(branch.state for branch in self.branches)

    @property
    def outcomes(self) -> tuple[Outcome, ...]:
        """Return the latest outcome for each branch.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Latest measured outcome for each branch, or ``()`` when a branch has
            no outcome history.
        """
        return tuple(
            branch.outcomes[-1] if branch.outcomes else () for branch in self.branches
        )

    @property
    def outcome_histories(self) -> tuple[OutcomeHistory, ...]:
        """Return complete outcome histories in branch order.

        Returns
        -------
        tuple[tuple[tuple[int, ...], ...], ...]
            Branch outcome histories.
        """
        return tuple(branch.outcomes for branch in self.branches)

    @classmethod
    def from_partial_measurement(cls, partial: PartialMeasurement) -> StateMixture:
        """Convert a partial-measurement record into a propagatable mixture.

        Parameters
        ----------
        partial : merlin.core.partial_measurement.PartialMeasurement
            Partial-measurement result to convert.

        Returns
        -------
        StateMixture
            Classical mixture with one branch per partial-measurement branch.

        Raises
        ------
        TypeError
            If ``partial`` is not a
            :class:`merlin.core.partial_measurement.PartialMeasurement`.
        """
        from .partial_measurement import PartialMeasurement

        if not isinstance(partial, PartialMeasurement):
            raise TypeError("partial must be a PartialMeasurement.")
        return cls(
            branches=tuple(
                StateMixtureBranch(
                    probability=branch.probability,
                    state=branch.amplitudes,
                    outcomes=(branch.outcome,),
                )
                for branch in partial.branches
            ),
            measured_modes=partial.measured_modes,
            unmeasured_modes=partial.unmeasured_modes,
        )


__all__ = ["StateMixture", "StateMixtureBranch"]
