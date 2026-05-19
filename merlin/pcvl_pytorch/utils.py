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

"""Utilities for converting between various perceval and torch representations."""

import perceval as pcvl  # type: ignore[import]
import torch

from ..core import ComputationSpace


def pcvl_to_tensor(
    state_vector: pcvl.StateVector,
    computation_space: ComputationSpace = ComputationSpace.FOCK,
    dtype: torch.dtype = torch.complex64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convert a Perceval ``StateVector`` into a torch tensor.

    Parameters
    ----------
    state_vector : pcvl.StateVector
        Perceval state vector.
    computation_space : merlin.core.computation_space.ComputationSpace
        Computation space of the state vector following combinadics ordering. Default is ``ComputationSpace.FOCK``
    dtype : torch.dtype
        Desired torch dtype of the output tensor. Default is ``torch.complex64``.
    device : torch.device
        Desired torch device of the output tensor. Default is ``torch.device("cpu")``.

    Returns
    -------
    torch.Tensor
        Equivalent torch tensor.

    Raises
    ------
    ValueError
        If the state vector includes states with incompatible photon number for
        the specified computation space, or inconsistent photon numbers across
        the states.

    """
    # Perceval StateVector.n is a set.
    ns_set = state_vector.n
    if len(ns_set) != 1:
        raise ValueError(
            "StateVector must have a fixed number of photons for conversion to tensor."
        )
    n_photons = ns_set.pop()

    n_modes = state_vector.m

    space = ComputationSpace.coerce(computation_space)
    basis_states = space.fock_basis_states(n_modes=n_modes, n_photons=n_photons)
    state_to_index = {state: index for index, state in enumerate(basis_states)}
    tensor = torch.zeros(len(basis_states), dtype=dtype, device=device)

    # Perceval StateVector iteration yields (basic_state, amplitude)
    for bs, amplitude in state_vector:
        state = list(bs)

        # Validate constraints for restricted computation spaces
        index = state_to_index.get(tuple(state))
        if index is None:
            raise ValueError(
                f"State {tuple(state)} is not part of computation_space={space}."
            )
        tensor[index] = amplitude

    return tensor
