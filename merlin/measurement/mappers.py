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

"""
Output mapping implementations for quantum-to-classical conversion.

Quantum outputs are expected to be:
1. Per state amplitudes, if the processing was a simulation
2. Per state probabilities, if the processing was on hardware
"""

import torch
import torch.nn as nn

from merlin.core import ComputationSpace

from .strategies import (
    MeasurementKind,
    MeasurementStrategyLike,
    _resolve_measurement_kind,
)


class OutputMapper:
    """Handles mapping quantum state amplitudes or probabilities to classical outputs.

    This class provides factory methods for creating different types of output mappers
    that convert quantum state amplitudes or probabilities to classical outputs.

    Parameters
    ----------
    None
    """

    @staticmethod
    def create_mapping(
        strategy: MeasurementStrategyLike,
        computation_space: ComputationSpace = ComputationSpace.FOCK,
        keys: list[tuple[int, ...]] | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Create an output mapping for the requested measurement strategy.

        Parameters
        ----------
        strategy : :data:`~merlin.measurement.strategies.MeasurementStrategyLike`
            Measurement mapping strategy to use.
        computation_space : ComputationSpace
            Computation space for the measurement.
        keys : list[tuple[int, ...]] | None
            List of Fock states. Required for mode-expectation mappings.
            For example, keys = [(0,1,0,2), (1,0,1,0), ...]
        dtype : torch.dtype | None
            Target dtype for internal tensors.

        Returns
        -------
        torch.nn.Module
            PyTorch module mapping amplitudes or probabilities to the desired
            output representation.

        Raises
        ------
        ValueError
            If ``strategy`` is unknown or required ``keys`` are missing.
        """
        try:
            kind = _resolve_measurement_kind(strategy)
        except TypeError as exc:
            raise ValueError(f"Unknown measurement strategy: {strategy}") from exc
        if kind == MeasurementKind.PROBABILITIES:
            return Probabilities()
        elif kind == MeasurementKind.MODE_EXPECTATIONS:
            if keys is None:
                raise ValueError(
                    "When using ModeExpectations measurement strategy, keys must be provided."
                )
            return ModeExpectations(computation_space, keys, dtype=dtype)
        elif kind == MeasurementKind.AMPLITUDES:
            return Amplitudes()
        else:
            raise ValueError(f"Unknown measurement strategy: {strategy}")


class Probabilities(nn.Module):
    """Map amplitudes or probabilities to a full Fock-state distribution.

    Parameters
    ----------
    None
    """

    def __init__(self):
        """Initialize the probability mapper."""
        super().__init__()

    def forward(self, x):
        """Compute the probability distribution of possible Fock states from amplitudes or probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input amplitudes or probabilities with shape ``(num_states,)`` or
            ``(batch_size, num_states)``.

        Returns
        -------
        torch.Tensor
            Fock states probability tensor of shape (batch_size, num_states) or (num_states,)
        """
        trailing_dim = x.shape[-1]
        # Collapse any leading batch dimensions so amplitude detection works uniformly for scalars, matrices or tensors.
        leading_shape = x.shape[:-1]
        reshaped = x.reshape(-1, trailing_dim)

        # Determine if x represents amplitudes (normalized squared norm)
        norm = torch.sum(reshaped.abs() ** 2, dim=1, keepdim=True)
        is_amplitude = torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

        if is_amplitude:
            prob = reshaped.abs() ** 2
        else:
            prob = reshaped

        return prob.reshape(*leading_shape, trailing_dim)


class ModeExpectations(nn.Module):
    """Map amplitudes or probabilities to per-mode expected photon counts.

    Parameters
    ----------
    computation_space : ComputationSpace
        Computation space used to interpret the keys.
    keys : list[tuple[int, ...]]
        List of tuples describing the possible Fock states output from the circuit preceding the output
        mapping. e.g., [(0,1,0,2), (1,0,1,0), ...]
    dtype : torch.dtype | None
        Target dtype for internal tensors.
    """

    def __init__(
        self,
        computation_space: ComputationSpace,
        keys: list[tuple[int, ...]],
        *,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the mode-expectation mapper.

        Parameters
        ----------
        computation_space : ComputationSpace
            Computation space used to interpret the keys.
        keys : list[tuple[int, ...]]
            List of tuples describing the possible Fock states output from the circuit preceding the output
            mapping. e.g., [(0,1,0,2), (1,0,1,0), ...]
        dtype : torch.dtype | None
            Target dtype for internal tensors.
        """
        super().__init__()
        self.computation_space = computation_space
        self.keys = keys

        if not keys:
            raise ValueError("Keys list cannot be empty")

        if len({len(key) for key in keys}) > 1:
            raise ValueError("All keys must have the same length (number of modes)")

        # Resolve dtype (default to float32 for backward compatibility)
        resolved_dtype = dtype if dtype is not None else torch.float32

        # Create mask and register as buffer
        keys_tensor = torch.tensor(keys, dtype=torch.long)
        if computation_space.requires_postselection:
            mask = (keys_tensor >= 1).T.to(dtype=resolved_dtype)
        else:
            mask = keys_tensor.T.to(dtype=resolved_dtype)

        # Make the expected type explicit for static analysers.
        self.mask: torch.Tensor
        self.register_buffer("mask", mask)

    def marginalize_per_mode(
        self, probability_distribution: torch.Tensor
    ) -> torch.Tensor:
        """Marginalize Fock-state probabilities into per-mode expectations.

        Parameters
        ----------
        probability_distribution : torch.Tensor
            Tensor of probabilities for each Fock state.

        Returns
        -------
        torch.Tensor
            Per-mode expected photon counts.
        """
        marginalized = probability_distribution @ self.mask.T
        return marginalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-mode expectations from amplitudes or probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input amplitudes or probabilities with shape ``(num_states,)`` or
            ``(batch_size, num_states)``.

        Returns
        -------
        torch.Tensor
            Expected photon counts per mode.
        """
        # Validate input
        if x.dim() not in [1, 2]:
            raise ValueError("Input must be 1D or 2D tensor")

        # Get probabilities
        distribution_mapper = Probabilities()
        prob = distribution_mapper(x)

        # Handle both 1D and 2D inputs uniformly
        original_shape = prob.shape
        if prob.dim() == 1:
            prob = prob.unsqueeze(0)

        marginalized_probs = self.marginalize_per_mode(prob)

        if len(original_shape) == 1:
            marginalized_probs = marginalized_probs.squeeze(0)

        return marginalized_probs


class Amplitudes(nn.Module):
    """
    Output the Fock state vector (also called amplitudes) directly. This can only be done with a simulator because amplitudes cannot be retrieved
    from the per state probabilities obtained with a QPU.
    """

    def __init__(self):
        """Initialize the amplitude passthrough mapper."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fock state vector amplitudes."""
        original_shape = x.shape
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if len(original_shape) == 1:
            x = x.squeeze(0)
        return x
