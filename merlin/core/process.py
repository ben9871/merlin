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
Quantum computation processes and factories.
"""

import math
from typing import Literal, overload

import perceval as pcvl
import torch

from ..pcvl_pytorch import CircuitConverter, build_slos_distribution_computegraph
from ..utils.combinadics import Combinadics
from ..utils.deprecations import raise_no_bunching_deprecated
from .base import AbstractComputationProcess
from .computation_space import ComputationSpace


class ComputationProcess(AbstractComputationProcess):
    """Handle quantum circuit computation and state evolution.

    Parameters
    ----------
    circuit : pcvl.Circuit
        Circuit used to build the unitary and simulation graphs.
    input_state : list[int] | torch.Tensor
        Input Fock state or superposition tensor.
    trainable_parameters : list[str]
        Prefixes of trainable circuit parameters.
    input_parameters : list[str]
        Prefixes of input-driven circuit parameters.
    n_photons : int | None
        Number of photons represented by the process.
    dtype : torch.dtype
        Real dtype used for internal tensor conversions.
    device : torch.device | None
        Device on which computation graphs are materialized.
    computation_space : ComputationSpace | None
        Computation space used for basis enumeration.
    no_bunching : bool | None
        Deprecated legacy parameter.
    output_map_func : Any
        Optional output mapping function.
    """

    def __init__(
        self,
        circuit: pcvl.Circuit,
        input_state: list[int] | torch.Tensor,
        trainable_parameters: list[str],
        input_parameters: list[str],
        n_photons: int = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        computation_space: ComputationSpace | None = None,
        no_bunching: bool | None = None,
        output_map_func=None,
    ):
        """Initialize a computation process.

        Parameters
        ----------
        circuit : pcvl.Circuit
            Circuit used to build the unitary and simulation graphs.
        input_state : list[int] | torch.Tensor
            Input Fock state or superposition tensor.
        trainable_parameters : list[str]
            Prefixes of trainable circuit parameters.
        input_parameters : list[str]
            Prefixes of input-driven circuit parameters.
        n_photons : int | None
            Number of photons represented by the process.
        dtype : torch.dtype
            Real dtype used for internal tensor conversions.
        device : torch.device | None
            Device on which computation graphs are materialized.
        computation_space : ComputationSpace | None
            Computation space used for basis enumeration.
        no_bunching : bool | None
            Deprecated legacy parameter.
        output_map_func : Any
            Optional output mapping function.
        """
        self.circuit = circuit
        self.input_state = input_state
        self.n_photons = n_photons
        self.trainable_parameters = trainable_parameters
        self.input_parameters = input_parameters
        self.dtype = dtype
        self.device = device

        if no_bunching is not None:
            raise_no_bunching_deprecated(stacklevel=2)

        if computation_space is None:
            computation_space = ComputationSpace.UNBUNCHED

        self.computation_space = computation_space
        self.output_map_func = output_map_func

        # Extract circuit parameters for graph building

        self.m = circuit.m  # Number of modes
        if n_photons is None:
            if type(input_state) is list:
                self.n_photons = sum(input_state)  # Total number of photons
            else:
                raise ValueError("The number of photons should be provided")
        else:
            self.n_photons = n_photons
        combinator = Combinadics(
            self.computation_space.value, self.n_photons, self.m
        )
        self.logical_keys = [tuple(state) for state in combinator.iter_states()]
        # Build computation graphs
        self._setup_computation_graphs()

        # validate initial input state shape when provided as tensor
        if isinstance(self.input_state, torch.Tensor):
            state_tensor: torch.Tensor = self.input_state
            self._validate_superposition_state_shape(state_tensor)

    def _setup_computation_graphs(self):
        """Setup unitary and simulation computation graphs."""
        # Determine parameter specs
        parameter_specs = self.trainable_parameters + self.input_parameters

        # Build unitary graph
        self.converter = CircuitConverter(
            self.circuit, parameter_specs, dtype=self.dtype, device=self.device
        )

        # Build simulation graph with correct parameters
        self.simulation_graph = build_slos_distribution_computegraph(
            m=self.m,  # Number of modes
            n_photons=self.n_photons,  # Total number of photons
            computation_space=self.computation_space,
            keep_keys=True,  # Usually want to keep keys for output interpretation
            device=self.device,
            dtype=self.dtype,
            output_map_func=self.output_map_func,
        )

    def compute(self, parameters: list[torch.Tensor]) -> torch.Tensor:
        """Compute output amplitudes for the configured input state.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Circuit parameters passed to the converter.

        Returns
        -------
        torch.Tensor
            Output amplitudes produced by the simulation graph.
        """
        # Generate unitary matrix from parameters

        unitary = self.converter.to_tensor(*parameters)
        self.unitary = unitary
        # Compute output distribution using the input state
        if isinstance(self.input_state, torch.Tensor):
            input_state = [1] * self.n_photons + [0] * (self.m - self.n_photons)
        else:
            input_state = self.input_state

        keys, amplitudes = self.simulation_graph.compute(unitary, input_state)
        return amplitudes

    @overload
    def compute_superposition_state(
        self, parameters: list[torch.Tensor], *, return_keys: Literal[True]
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]: ...

    @overload
    def compute_superposition_state(
        self, parameters: list[torch.Tensor], *, return_keys: Literal[False] = False
    ) -> torch.Tensor: ...

    def compute_superposition_state(
        self, parameters: list[torch.Tensor], *, return_keys: bool = False
    ) -> torch.Tensor | tuple[list[tuple[int, ...]], torch.Tensor]:
        prepared_state = self._prepare_superposition_tensor()
        unitary = self.converter.to_tensor(*parameters)
        changed_unitary = True

        def is_swap_permutation(t1, t2):
            if t1 == t2:
                return False
            diff = [
                (i, i) for i, (x, y) in enumerate(zip(t1, t2, strict=False)) if x != y
            ]
            if len(diff) != 2:
                return False
            i, j = diff[0][0], diff[1][0]

            return t1[i] == t2[j] and t1[j] == t2[i]

        def reorder_swap_chain(lst):
            remaining = lst[:]
            chain = [remaining.pop(0)]
            while remaining:
                for i, candidate in enumerate(remaining):
                    if is_swap_permutation(chain[-1][1], candidate[1]):
                        chain.append(remaining.pop(i))
                        break
                else:
                    chain.append(remaining.pop(0))

            return chain

        input_states = [
            (k, self.logical_keys[k])
            for k in self._active_superposition_indices(prepared_state)
        ]
        if not input_states:
            empty = torch.zeros(
                (prepared_state.shape[0], len(self.simulation_graph.mapped_keys)),
                dtype=unitary.dtype,
                device=prepared_state.device,
            )
            keys_out = list(self.simulation_graph.mapped_keys)
            if return_keys:
                return keys_out, empty
            return empty

        state_list = reorder_swap_chain(input_states)

        prev_state_index, prev_state = state_list.pop(0)

        keys, amplitude = self.simulation_graph.compute(unitary, prev_state)
        if amplitude.ndim == 2 and amplitude.shape[0] == 1:
            amplitude = amplitude.squeeze(0)
        keys_out = list(self.simulation_graph.mapped_keys)
        final_amplitudes = torch.zeros(
            (prepared_state.shape[0], len(self.simulation_graph.mapped_keys)),
            dtype=amplitude.dtype,
            device=prepared_state.device,
        )
        amplitude = amplitude / amplitude.norm(p=2).clamp_min(1e-12)
        coeffs = self._gather_superposition_coefficients(
            prepared_state, [prev_state_index]
        ).to(amplitude.dtype)
        final_amplitudes += coeffs[:, :1] * amplitude.unsqueeze(0)

        for index, fock_state in state_list:
            component = self.simulation_graph.compute_pa_inc(
                unitary,
                prev_state,
                fock_state,
                changed_unitary=changed_unitary,
            )
            if component.ndim == 2 and component.shape[0] == 1:
                component = component.squeeze(0)
            component = component / component.norm(p=2).clamp_min(1e-12)
            coeffs = self._gather_superposition_coefficients(
                prepared_state, [index]
            ).to(component.dtype)
            final_amplitudes += coeffs[:, :1] * component.unsqueeze(0)
            changed_unitary = False
            prev_state = fock_state

        if return_keys:
            return keys_out, final_amplitudes

        return final_amplitudes

    def compute_ebs_simultaneously(
        self, parameters: list[torch.Tensor], simultaneous_processes: int = 1
    ) -> torch.Tensor:
        """
        Evaluate a single circuit parametrisation against all superposed input
        states by chunking them in groups and delegating the heavy work to the
        TorchScript-enabled batch kernel.

        The method converts the trainable parameters into a unitary matrix,
        normalises the input state (if it is not already normalised), filters
        out components with zero amplitude, and then queries the simulation
        graph for batches of Fock states. Each batch feeds
        :meth:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph.compute_batch`, producing a tensor that contains
        the amplitudes of all reachable output states for the selected input
        components. The partial results are accumulated into a preallocated
        tensor and finally weighted by the complex coefficients of
        ``self.input_state`` to produce the global output amplitudes.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Differentiable parameters that encode the photonic circuit.
        simultaneous_processes : int
            Maximum number of non-zero input components propagated in a single
            call to ``compute_batch``.

        Returns
        -------
        torch.Tensor
            Superposed output amplitudes with shape
            ``[batch_size, num_output_states]``.

        Raises
        ------
        TypeError
            If ``self.input_state`` is not a ``torch.Tensor``.

        Notes
        -----
            - ``self.input_state`` is normalized in place to avoid an extra
              allocation.They are forwarded to ``self.converter`` to build the unitary matrix used during the
              simulation.
            - Zero-amplitude components are skipped to minimise the number of
              calls to ``compute_batch``.
            - The method is agnostic to the device: tensors remain on the device
              they already occupy, so callers should ensure ``parameters`` and
              ``self.input_state`` live on the same device.
        """

        # input state was validated by _prepare_superposition_tensor, ie: renormalized, typed, and converted from logical basis to fock basis (if shape did not match)
        # we don't want anymore the logical basis but normalization and typing cannot hurt even if it is a small overhead
        prepared_state = self._prepare_superposition_tensor()

        unitary = self.converter.to_tensor(*parameters)
        # Allow classical parameters to be batched: in that case the converter already returns a stack of unitaries.
        batched_parameters = unitary.dim() == 3
        if not batched_parameters:
            unitary = unitary.unsqueeze(0)
        parameter_batch = unitary.shape[0]

        input_states = [
            (k, self.logical_keys[k])
            for k in self._active_superposition_indices(prepared_state)
        ]

        # Stream each chunk directly into the final output tensor instead of
        # materializing a dense [input_basis, output_basis] intermediate.
        final_amplitudes = torch.zeros(
            (
                parameter_batch,
                prepared_state.shape[0],
                len(self.simulation_graph.mapped_keys),
            ),
            dtype=unitary.dtype,
            device=prepared_state.device,
        )

        # Process input states in batches
        for i in range(0, len(input_states), simultaneous_processes):
            batch_end = min(i + simultaneous_processes, len(input_states))
            batch_indices = []
            batch_fock_states = []

            for j in range(i, batch_end):
                idx, fock_state = input_states[j]
                batch_indices.append(idx)
                batch_fock_states.append(fock_state)

            coeffs = self._gather_superposition_coefficients(
                prepared_state, batch_indices
            ).to(unitary.dtype)
            # Compute batch amplitudes
            _, batch_amplitudes = self.simulation_graph.compute_batch(
                unitary, batch_fock_states
            )
            batch_amplitudes = batch_amplitudes / batch_amplitudes.norm(
                p=2, dim=1, keepdim=True
            ).clamp_min(1e-12)
            final_amplitudes += torch.einsum(
                "se, boe -> bso", coeffs, batch_amplitudes
            )

        if final_amplitudes.shape[0] == 1:
            final_amplitudes = final_amplitudes.squeeze(0)
        if final_amplitudes.ndim == 3 and final_amplitudes.shape[1] == 1:
            final_amplitudes = final_amplitudes.squeeze(1)

        return final_amplitudes

    def compute_with_keys(self, parameters: list[torch.Tensor]):
        """Compute output amplitudes and return them with basis keys.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Circuit parameters passed to the converter.

        Returns
        -------
        tuple[Any, torch.Tensor]
            Simulation-graph keys and corresponding amplitudes.
        """
        # Generate unitary matrix from parameters
        unitary = self.converter.to_tensor(*parameters)

        # Compute output distribution using the input state
        keys, amplitudes = self.simulation_graph.compute(unitary, self.input_state)

        return keys, amplitudes

    def _expected_superposition_size(self) -> int:
        """Expected number of Fock states given current computation space."""
        if self.n_photons < 0:
            raise ValueError("Number of photons must be non-negative.")
        if self.computation_space is ComputationSpace.DUAL_RAIL:
            if self.n_photons is None:
                raise ValueError("Dual-rail encoding requires 'n_photons'.")
            if self.m != 2 * self.n_photons:
                raise ValueError(
                    "Dual-rail encoding requires the number of modes to equal 2 * n_photons."
                )
            # Dual-rail limits to 2**n logical states (one photon per rail pair).
            return 2**self.n_photons
        if self.computation_space is ComputationSpace.UNBUNCHED:
            if self.n_photons > self.m:
                raise ValueError(
                    "Invalid configuration: ComputationSpace.UNBUNCHED requires "
                    "n_photons to be less than or equal to the number of modes."
                )
            return math.comb(self.m, self.n_photons)
        return math.comb(self.m + self.n_photons - 1, self.n_photons)

    def _validate_superposition_state_shape(self, input_state: torch.Tensor) -> None:
        """Ensure the provided superposition state matches the configured computation space."""
        if not isinstance(input_state, torch.Tensor):
            raise TypeError("Input state should be a tensor")

        if input_state.dim() == 1:
            state_dim = input_state.shape[0]
        elif input_state.dim() == 2:
            state_dim = input_state.shape[1]
        else:
            raise ValueError(
                f"Superposed input state must be 1D or 2D tensor, got shape {tuple(input_state.shape)}"
            )

        expected = self._expected_superposition_size()
        if state_dim != expected:
            if (
                self.computation_space is ComputationSpace.DUAL_RAIL
                and state_dim == len(self.simulation_graph.mapped_keys)
            ):
                return
            if self.computation_space is ComputationSpace.DUAL_RAIL:
                explanation = (
                    f"expected 2**n_photons = 2**{self.n_photons} = {expected}"
                )
            elif self.computation_space is ComputationSpace.UNBUNCHED:
                explanation = f"expected C(m, n_photons) = C({self.m}, {self.n_photons}) = {expected}"
            else:
                explanation = (
                    f"expected C(m + n_photons - 1, n_photons) = "
                    f"C({self.m + self.n_photons - 1}, {self.n_photons}) = {expected}"
                )
            raise ValueError(
                "Input state dimension mismatch for computation_space "
                f"'{self.computation_space}': got {state_dim}, {explanation}."
            )

    def _should_defer_state_validation(self, tensor: torch.Tensor) -> bool:
        """Detect amplitude tensors that will be validated after configuring dual-rail space."""
        if tensor.dim() == 1:
            state_dim = tensor.shape[0]
        elif tensor.dim() == 2:
            state_dim = tensor.shape[1]
        else:
            return False

        if self.n_photons is None or self.m is None:
            return False

        return (
            self.computation_space is ComputationSpace.UNBUNCHED
            and self.m == 2 * self.n_photons
            and state_dim == 2**self.n_photons
        )

    def _coerce_superposition_tensor_shape(
        self, tensor: torch.Tensor
    ) -> torch.Tensor | None:
        """Attempt to reconcile tensors encoded in a smaller logical basis."""
        if self.computation_space is not ComputationSpace.FOCK:
            return None

        if self.n_photons is None or self.m is None:
            return None

        if tensor.dim() == 1:
            feature_dim = tensor.shape[0]
        elif tensor.dim() == 2:
            feature_dim = tensor.shape[1]
        else:
            return None

        logical_keys = [tuple(key) for key in self.logical_keys]
        target_dim = len(logical_keys)

        mapped_keys = [
            tuple(key)
            for key in self.simulation_graph.mapped_keys  # type: ignore[attr-defined]
        ]
        if feature_dim == len(mapped_keys) and target_dim != feature_dim:
            key_to_index = {state: idx for idx, state in enumerate(logical_keys)}
            indices = []
            for state in mapped_keys:
                index = key_to_index.get(state)
                if index is None:
                    return None
                indices.append(index)
            return self._expand_superposition_tensor(tensor, indices, target_dim)

        # Detect tensors encoded in the UNBUNCHED basis and lift them to the Fock basis.
        unbunched_size = math.comb(self.m, self.n_photons)
        if feature_dim != unbunched_size:
            return None

        key_to_index = {state: idx for idx, state in enumerate(logical_keys)}

        try:
            combinator = Combinadics("unbunched", self.n_photons, self.m)
        except ValueError:
            return None

        indices: list[int] = []
        for state in combinator.iter_states():
            index = key_to_index.get(state)
            if index is None:
                return None
            indices.append(index)

        return self._expand_superposition_tensor(tensor, indices, target_dim)

    @staticmethod
    def _expand_superposition_tensor(
        tensor: torch.Tensor, indices: list[int], target_dim: int
    ) -> torch.Tensor:
        """Lift a reduced-basis superposition tensor into a larger logical basis."""
        if tensor.dim() == 1:
            if tensor.is_sparse:
                coalesced = tensor.coalesce()
                index_map = torch.tensor(
                    indices, dtype=torch.long, device=coalesced.device
                )
                mapped = index_map[coalesced.indices()[0]]
                return torch.sparse_coo_tensor(
                    mapped.unsqueeze(0),
                    coalesced.values(),
                    (target_dim,),
                    dtype=coalesced.dtype,
                    device=coalesced.device,
                ).coalesce()
            expanded = tensor.new_zeros(target_dim)
            expanded[indices] = tensor
            return expanded

        if tensor.is_sparse:
            coalesced = tensor.coalesce()
            index_map = torch.tensor(
                indices, dtype=torch.long, device=coalesced.device
            )
            mapped = index_map[coalesced.indices()[1]]
            expanded_indices = torch.stack((coalesced.indices()[0], mapped))
            return torch.sparse_coo_tensor(
                expanded_indices,
                coalesced.values(),
                (tensor.shape[0], target_dim),
                dtype=coalesced.dtype,
                device=coalesced.device,
            ).coalesce()
        expanded = tensor.new_zeros(tensor.shape[0], target_dim)
        expanded[:, indices] = tensor
        return expanded

    @staticmethod
    def _unsqueeze_superposition_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Add a batch dimension while preserving sparse COO storage."""
        if tensor.dim() != 1:
            return tensor
        if not tensor.is_sparse:
            return tensor.unsqueeze(0)

        coalesced = tensor.coalesce()
        nnz = coalesced.values().shape[0]
        batch_indices = torch.zeros(
            (1, nnz), dtype=torch.long, device=coalesced.device
        )
        indices = torch.cat((batch_indices, coalesced.indices()), dim=0)
        return torch.sparse_coo_tensor(
            indices,
            coalesced.values(),
            (1, tensor.shape[0]),
            dtype=coalesced.dtype,
            device=coalesced.device,
        ).coalesce()

    @staticmethod
    def _normalize_superposition_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize batched superposition tensors without forcing densification."""
        if not tensor.is_sparse:
            norm = tensor.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
            return tensor / norm

        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        row_indices = indices[0]
        magnitude_sq = values.real.pow(2) + values.imag.pow(2)
        norm_sq = torch.zeros(
            tensor.shape[0], dtype=magnitude_sq.dtype, device=magnitude_sq.device
        )
        norm_sq.scatter_add_(0, row_indices, magnitude_sq)
        norms = norm_sq.sqrt().clamp_min(1e-12)
        scaled_values = values / norms[row_indices]
        return torch.sparse_coo_tensor(
            indices,
            scaled_values,
            coalesced.shape,
            dtype=coalesced.dtype,
            device=coalesced.device,
        ).coalesce()

    @staticmethod
    def _active_superposition_indices(tensor: torch.Tensor) -> list[int]:
        """Return the active basis indices in a batched superposition tensor."""
        if tensor.is_sparse:
            cols = tensor.coalesce().indices()[-1].tolist()
            return list(dict.fromkeys(int(col) for col in cols))

        mask = (tensor.real.pow(2) + tensor.imag.pow(2) < 1e-13).all(dim=0)
        return [idx for idx, active in enumerate((~mask).tolist()) if active]

    @staticmethod
    def _gather_superposition_coefficients(
        tensor: torch.Tensor, basis_indices: list[int]
    ) -> torch.Tensor:
        """Extract selected basis coefficients as a small dense tensor."""
        if not basis_indices:
            return torch.zeros(
                (tensor.shape[0], 0), dtype=tensor.dtype, device=tensor.device
            )
        if not tensor.is_sparse:
            return tensor[:, basis_indices]

        coalesced = tensor.coalesce()
        lookup = {basis_idx: pos for pos, basis_idx in enumerate(basis_indices)}
        gathered = torch.zeros(
            (tensor.shape[0], len(basis_indices)),
            dtype=coalesced.dtype,
            device=coalesced.device,
        )
        indices = coalesced.indices()
        values = coalesced.values()
        for col in range(values.shape[0]):
            basis_idx = int(indices[1, col].item())
            pos = lookup.get(basis_idx)
            if pos is None:
                continue
            row = int(indices[0, col].item())
            gathered[row, pos] = values[col]
        return gathered

    def _prepare_superposition_tensor(self) -> torch.Tensor:
        """Validate, normalise, and convert the stored superposition state to the correct dtype."""
        if not isinstance(self.input_state, torch.Tensor):
            raise TypeError("Input state should be a tensor")

        tensor = self.input_state

        coerced = self._coerce_superposition_tensor_shape(tensor)
        if coerced is not None:
            tensor = coerced

        self._validate_superposition_state_shape(tensor)

        tensor = self._unsqueeze_superposition_tensor(tensor)

        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.complex64)
        elif tensor.dtype == torch.float64:
            tensor = tensor.to(torch.complex128)
        elif tensor.dtype not in (torch.complex64, torch.complex128):
            raise TypeError(
                f"Unsupported dtype for superposition state: {tensor.dtype}"
            )

        tensor = self._normalize_superposition_tensor(tensor)
        self.input_state = tensor
        return tensor


class ComputationProcessFactory:
    """Factory for creating computation processes."""

    @staticmethod
    def create(
        circuit: pcvl.Circuit,
        input_state: list[int] | torch.Tensor,
        trainable_parameters: list[str],
        input_parameters: list[str],
        computation_space: ComputationSpace | None = None,
        **kwargs,
    ) -> ComputationProcess:
        """Create a computation process.

        Parameters
        ----------
        circuit : pcvl.Circuit
            Circuit used to build the process.
        input_state : list[int] | torch.Tensor
            Input Fock state or superposition tensor.
        trainable_parameters : list[str]
            Prefixes of trainable circuit parameters.
        input_parameters : list[str]
            Prefixes of input-driven circuit parameters.
        computation_space : ComputationSpace | None
            Computation space used for basis enumeration.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`ComputationProcess`.

        Returns
        -------
        ComputationProcess
            Created computation process.
        """
        return ComputationProcess(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            computation_space=computation_space,
            **kwargs,
        )
