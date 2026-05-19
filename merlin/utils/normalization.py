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

"""Shared normalization helpers for amplitudes and probabilities."""

from __future__ import annotations

import torch

from ..core.computation_space import ComputationSpace


def probabilities_from_amplitudes(amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Convert complex amplitudes into probabilities.

    Parameters
    ----------
    amplitudes : torch.Tensor
        Complex amplitude tensor.

    Returns
    -------
    torch.Tensor
        Probability tensor computed elementwise from ``amplitudes``.
    """
    return amplitudes.real**2 + amplitudes.imag**2


def normalize_probabilities(
    probabilities: torch.Tensor, computation_space: ComputationSpace | None
) -> torch.Tensor:
    """
    Normalize probabilities for computation spaces that require it.

    Parameters
    ----------
    probabilities : torch.Tensor
        Probability tensor to normalize.
    computation_space : ComputationSpace | None
        Computation space that determines whether normalization is required.

    Returns
    -------
    torch.Tensor
        Normalized probability tensor. For computation spaces that do not
        require renormalization, the input tensor is returned unchanged.
    """
    if computation_space is None or not computation_space.requires_postselection:
        return probabilities

    sum_probs = probabilities.sum(dim=-1, keepdim=True)
    valid_entries = sum_probs > 0
    if valid_entries.any():
        probabilities = torch.where(
            valid_entries,
            probabilities
            / torch.where(valid_entries, sum_probs, torch.ones_like(sum_probs)),
            probabilities,
        )
    return probabilities


def normalize_probabilities_and_amplitudes(
    amplitudes: torch.Tensor, computation_space: ComputationSpace | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return probabilities and renormalized amplitudes when required.

    Parameters
    ----------
    amplitudes : torch.Tensor
        Complex amplitude tensor.
    computation_space : ComputationSpace | None
        Computation space that determines whether renormalization is required.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Probability tensor and possibly renormalized amplitude tensor.
    """
    probabilities = probabilities_from_amplitudes(amplitudes)

    if computation_space is not None and computation_space.requires_postselection:
        sum_probs = probabilities.sum(dim=-1, keepdim=True)
        valid_entries = sum_probs > 0
        if valid_entries.any():
            amplitudes = torch.where(
                valid_entries,
                amplitudes
                / torch.where(
                    valid_entries, sum_probs.sqrt(), torch.ones_like(sum_probs)
                ),
                amplitudes,
            )
        probabilities = normalize_probabilities(probabilities, computation_space)

    return probabilities, amplitudes
