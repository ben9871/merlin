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

"""StateVector with combinatorial metadata and conversions.

This module provides a lightweight :class:`StateVector` wrapper that keeps the
Fock-space metadata (number of modes, number of photons, basis ordering) tied to
its amplitude tensor. It supports dense and sparse tensors, Fock ordering via
:class:`~merlin.utils.combinadics.Combinadics`, and conversion to/from
:class:`exqalibur.StateVector`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import cast

import perceval as pcvl
import torch

from ..utils.combinadics import Combinadics
from ..utils.dtypes import complex_dtype_for
from .computation_space import ComputationSpace
from .encoding_space import EncodingSpace

Scalar = float | int | complex

Basis = Combinadics


@cache
def _basis_for(n_modes: int, n_photons: int) -> Basis:
    return Combinadics("fock", n_photons, n_modes)


@cache
def _basis_size(n_modes: int, n_photons: int) -> int:
    return Combinadics("fock", n_photons, n_modes).compute_space_size()


def _to_complex_dense(
    tensor: torch.Tensor, *, dtype: torch.dtype | None, device: torch.device | None
) -> torch.Tensor:
    target_device = device or tensor.device
    target_dtype = dtype or (
        tensor.dtype if tensor.is_complex() else complex_dtype_for(torch.float32)
    )
    if tensor.is_complex():
        return tensor.to(device=target_device, dtype=target_dtype)
    if tensor.is_floating_point() or tensor.dtype in (
        torch.int32,
        torch.int64,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        real = tensor.to(
            device=target_device, dtype=torch.promote_types(tensor.dtype, torch.float32)
        )
        imag = torch.zeros_like(real)
        return torch.complex(real, imag).to(dtype=target_dtype, device=target_device)
    raise TypeError(
        "Tensor dtype is not supported for complex conversion; expected real or complex inputs."
    )


def _to_complex(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if tensor.is_sparse:
        coalesced = tensor.coalesce()
        values = _to_complex_dense(coalesced.values(), dtype=dtype, device=device)
        return torch.sparse_coo_tensor(
            coalesced.indices(),
            values,
            coalesced.shape,
            device=device or coalesced.device,
        )
    return _to_complex_dense(tensor, dtype=dtype, device=device)


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_sparse:
        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        if tensor.ndim == 1:
            norm_sq = torch.sum(values.abs().pow(2))
            if norm_sq == 0:
                return tensor
            norm = torch.sqrt(norm_sq)
            new_values = values / norm
            return torch.sparse_coo_tensor(
                indices, new_values, tensor.shape, device=tensor.device
            )

        nnz = values.shape[0]
        norm_map: dict[tuple[int, ...], torch.Tensor] = {}
        for col in range(nnz):
            batch_coords = tuple(int(v) for v in indices[:-1, col].tolist())
            contrib = values[col].abs().pow(2)
            norm_map[batch_coords] = (
                norm_map.get(
                    batch_coords,
                    torch.tensor(0.0, device=values.device, dtype=values.dtype),
                )
                + contrib
            )

        norm_map = {k: torch.sqrt(v) for k, v in norm_map.items()}
        scaled_values: list[torch.Tensor] = []
        for col in range(nnz):
            batch_coords = tuple(int(v) for v in indices[:-1, col].tolist())
            norm = norm_map.get(batch_coords)
            if norm is None or norm == 0:
                scaled_values.append(values[col])
            else:
                scaled_values.append(values[col] / norm)
        new_values_tensor = torch.stack(scaled_values)
        return torch.sparse_coo_tensor(
            indices, new_values_tensor, tensor.shape, device=tensor.device
        )

    norm = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
    norm_safe = torch.where(norm == 0, torch.ones_like(norm), norm)
    return tensor / norm_safe


def _infer_dual_rail_photon_count(logical_size: int) -> int:
    """Infer a dual-rail photon count from a compact logical basis width.

    Dual-rail has one binary choice per photon, so a valid logical tensor has a
    final dimension of ``2 ** n_photons``. The zero-photon case is not inferred
    because it would not determine a positive mode count.
    """

    if logical_size <= 1 or logical_size & (logical_size - 1):
        raise ValueError(
            "dual_rail encoding requires tensor last dimension to be a power "
            "of two greater than one when n_modes and n_photons are omitted."
        )
    return logical_size.bit_length() - 1


def _resolve_from_tensor_dimensions(
    tensor: torch.Tensor,
    *,
    encoding: EncodingSpace,
    n_modes: int | None,
    n_photons: int | None,
) -> tuple[int, int]:
    """Resolve concrete Fock dimensions for ``StateVector.from_tensor``.

    Structured encodings may infer their own dimensions, but explicit
    dimensions are validation metadata rather than a request to stretch the
    encoding layout.
    """

    if tensor.dim() == 0:
        raise ValueError("Amplitude tensor must be at least one-dimensional.")

    if encoding.kind == "dual_rail":
        if n_modes is None and n_photons is None:
            n_photons = _infer_dual_rail_photon_count(tensor.shape[-1])
            n_modes = 2 * n_photons
        elif n_modes is None:
            if (
                not isinstance(n_photons, int)
                or isinstance(n_photons, bool)
                or n_photons <= 0
            ):
                raise ValueError(
                    "dual_rail encoding requires a positive n_photons value."
                )
            n_modes = 2 * n_photons
        elif n_photons is None:
            if (
                not isinstance(n_modes, int)
                or isinstance(n_modes, bool)
                or n_modes <= 0
            ):
                raise ValueError("n_modes must be a strictly positive integer.")
            if n_modes % 2 != 0:
                raise ValueError("dual_rail requires an even n_modes value.")
            n_photons = n_modes // 2
        assert n_modes is not None
        assert n_photons is not None
        return n_modes, n_photons

    if encoding.family == "partitioned" and encoding.modes_per_photon is not None:
        resolved_modes = encoding.n_modes if n_modes is None else n_modes
        resolved_photons = encoding.n_photons if n_photons is None else n_photons
        assert resolved_modes is not None
        assert resolved_photons is not None
        return resolved_modes, resolved_photons

    if n_modes is None or n_photons is None:
        raise ValueError(f"{encoding.kind} encoding requires n_modes and n_photons.")
    return n_modes, n_photons


def _remap_last_dim(
    tensor: torch.Tensor, indices: list[int], target_dim: int
) -> torch.Tensor:
    if tensor.is_sparse:
        coalesced = tensor.coalesce()
        sparse_indices = coalesced.indices().clone()
        sparse_values = coalesced.values()
        if sparse_indices.numel() == 0:
            shape = list(coalesced.shape)
            shape[-1] = target_dim
            return torch.sparse_coo_tensor(
                sparse_indices,
                sparse_values,
                tuple(shape),
                device=coalesced.device,
            )
        index_lookup = torch.tensor(indices, dtype=torch.long, device=coalesced.device)
        sparse_indices[-1] = index_lookup[sparse_indices[-1]]
        shape = list(coalesced.shape)
        shape[-1] = target_dim
        return torch.sparse_coo_tensor(
            sparse_indices,
            sparse_values,
            tuple(shape),
            device=coalesced.device,
        )

    expanded_shape = list(tensor.shape)
    expanded_shape[-1] = target_dim
    expanded = tensor.new_zeros(tuple(expanded_shape))
    index_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    expanded.index_copy_(-1, index_tensor, tensor)
    return expanded


def embed_tensor_in_fock_basis(
    tensor: torch.Tensor,
    *,
    n_modes: int,
    n_photons: int,
    computation_space: ComputationSpace | str,
) -> torch.Tensor:
    """Embed a compact logical tensor into the full Fock basis when applicable.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense or sparse amplitude tensor in either Fock ordering or a compact
        logical ordering compatible with the provided computation space.
    n_modes : int
        Number of photonic modes.
    n_photons : int
        Total number of photons.
    computation_space : ComputationSpace | str
        Logical ordering associated with ``tensor``.

    Returns
    -------
    torch.Tensor
        Tensor expressed in the full Fock basis. If ``tensor`` is already in
        Fock ordering, it is returned unchanged.

    Raises
    ------
    ValueError
        If the tensor shape is not compatible with a supported compact input
        ordering.
    """
    if tensor.dim() == 0:
        raise ValueError("Amplitude input must be at least one-dimensional.")

    feature_dim = tensor.shape[-1]
    fock_size = _basis_size(n_modes, n_photons)
    if feature_dim == fock_size:
        return tensor
    fock_basis = _basis_for(n_modes, n_photons)

    space = ComputationSpace.coerce(computation_space)
    if space != ComputationSpace.FOCK:
        logical_size = space.basis_size(n_modes=n_modes, n_photons=n_photons)
        if feature_dim == logical_size:
            indices = list(
                space.logical_to_fock_indices(
                    n_modes=n_modes, n_photons=n_photons
                ).values()
            )
            return _remap_last_dim(tensor, indices, fock_size)

    # Preserve the current compact-input behaviour for collision-free states in
    # Fock mode until all input paths use explicit EncodingSpace selection.
    if space == ComputationSpace.FOCK and n_photons <= n_modes:
        logical_basis = Combinadics("unbunched", n_photons, n_modes)
        if feature_dim == logical_basis.compute_space_size():
            indices = [fock_basis.fock_to_index(state) for state in logical_basis]
            return _remap_last_dim(tensor, indices, fock_size)

    if space == ComputationSpace.FOCK:
        detail = (
            f"expected either Fock size {fock_size} or compact unbunched size "
            f"{Combinadics('unbunched', n_photons, n_modes).compute_space_size()}"
            if n_photons <= n_modes
            else f"expected Fock size {fock_size}"
        )
    else:
        detail = (
            f"expected compact {space.value} size "
            f"{space.basis_size(n_modes=n_modes, n_photons=n_photons)} "
            f"or full Fock size {fock_size}"
        )
    raise ValueError(
        f"Amplitude input dimension mismatch: got {feature_dim}, {detail}."
    )


def _basic_state_counts(state: Sequence[int] | pcvl.BasicState) -> tuple[int, ...]:
    if isinstance(state, pcvl.BasicState):
        return tuple(int(x) for x in state)
    return tuple(int(x) for x in state)


def _basis_index_map(basis: Basis) -> dict[tuple[int, ...], int]:
    return {state: idx for idx, state in enumerate(basis)}


def _basic_state_tuple(state: Sequence[int] | pcvl.BasicState) -> tuple[int, ...]:
    if isinstance(state, pcvl.BasicState):
        return tuple(int(x) for x in state)
    return tuple(int(x) for x in state)


@dataclass
class StateVector:
    """Amplitude tensor bundled with its Fock metadata.

    Keeps ``n_modes`` / ``n_photons`` and combinadics basis ordering alongside the
    underlying PyTorch tensor (dense or sparse).

    Parameters
    ----------
    tensor : torch.Tensor
        Dense or sparse amplitude tensor; leading dimensions (if any) are treated
        as batch axes.
    n_modes : int
        Number of modes in the Fock space.
    n_photons : int
        Total photon number represented by the state.
    encoding : EncodingSpace
        Logical encoding used to construct the stored tensor. Default value is
        EncodingSpace.FOCK.
    _normalized : bool
        Internal flag tracking whether the stored tensor is normalized. Default
        value is False.

    Notes
    -----
    This is a thin wrapper over a ``torch.Tensor``: only ``shape``, ``device``,
    ``dtype``, and ``requires_grad`` are delegated automatically, and tensor-like
    helpers ``to``, ``clone``, ``detach``, and ``requires_grad_`` are provided to
    mirror common tensor workflows while preserving metadata. Layout-changing
    operations (e.g., ``reshape``/``view``) are intentionally not exposed; perform
    those on ``tensor`` explicitly if needed and rebuild via ``from_tensor``.
    """

    tensor: torch.Tensor
    n_modes: int
    n_photons: int
    encoding: EncodingSpace = EncodingSpace.FOCK
    _normalized: bool = field(default=False)

    def __setattr__(self, name: str, value) -> None:
        if name in ("n_modes", "n_photons", "encoding") and name in self.__dict__:
            raise AttributeError(
                "n_modes, n_photons, and encoding are immutable once set"
            )
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        allowed = {"shape", "device", "dtype", "requires_grad"}
        tensor = self.__dict__.get("tensor")
        if tensor is not None and name in allowed and hasattr(tensor, name):
            return getattr(tensor, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!s}")

    @property
    def is_normalized(self) -> bool:
        """bool: Whether the stored tensor is already normalized."""
        return self._normalized

    def _normalized_tensor(self) -> torch.Tensor:
        if self._normalized:
            return self.tensor
        normalized = _normalize_tensor(self.tensor)
        self.tensor = normalized
        self._normalized = True
        return normalized

    @property
    def basis(self) -> Basis:
        """Lazy combinadics basis for ``(n_modes, n_photons)`` in Fock ordering."""
        return _basis_for(self.n_modes, self.n_photons)

    @property
    def is_sparse(self) -> bool:
        """Return True if the underlying tensor uses a sparse layout."""
        return self.tensor.is_sparse

    @property
    def basis_size(self) -> int:
        """Return the number of basis states for ``(n_modes, n_photons)``."""
        return _basis_size(self.n_modes, self.n_photons)

    def to(self, *args, **kwargs) -> StateVector:
        """Return a new state vector moved or cast via ``torch.Tensor.to``.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`torch.Tensor.to`.
        **kwargs
            Keyword arguments forwarded to :meth:`torch.Tensor.to`.

        Returns
        -------
        StateVector
            Converted state vector.
        """
        new_tensor = self.tensor.to(*args, **kwargs)
        return StateVector(
            new_tensor,
            self.n_modes,
            self.n_photons,
            encoding=self.encoding,
            _normalized=self._normalized,
        )

    def clone(self) -> StateVector:
        """Return a cloned state vector with identical metadata and normalization flag.

        Returns
        -------
        StateVector
            Cloned state vector.
        """
        return StateVector(
            self.tensor.clone(),
            self.n_modes,
            self.n_photons,
            encoding=self.encoding,
            _normalized=self._normalized,
        )

    def detach(self) -> StateVector:
        """Return a detached ``StateVector`` sharing data without gradients.

        Returns
        -------
        StateVector
            Detached state vector.
        """
        return StateVector(
            self.tensor.detach(),
            self.n_modes,
            self.n_photons,
            encoding=self.encoding,
            _normalized=self._normalized,
        )

    def requires_grad_(self, requires_grad: bool = True) -> StateVector:
        """Set ``requires_grad`` on the underlying tensor and return self.

        Parameters
        ----------
        requires_grad : bool
            Whether gradients should be tracked.

        Returns
        -------
        StateVector
            The updated instance.
        """
        self.tensor.requires_grad_(requires_grad)
        return self

    def memory_bytes(self) -> int:
        """Approximate memory footprint (bytes) of the underlying tensor data."""
        if self.tensor.is_sparse:
            coalesced = self._tensor_coalesced()
            idx = coalesced.indices()
            vals = coalesced.values()
            return int(
                idx.numel() * idx.element_size() + vals.numel() * vals.element_size()
            )
        return int(self.tensor.numel() * self.tensor.element_size())

    def logical_to_fock_map(self) -> dict[tuple[int, ...], int]:
        """Return the logical-to-Fock index map for this state vector.

        The returned dictionary exposes the exact embedding order implied by
        ``self.encoding`` for this state vector's ``n_modes`` and
        ``n_photons``. Keys are logical basis labels and values are indices in
        Merlin's canonical full-Fock basis. For ``EncodingSpace.FOCK``, keys are
        full Fock occupation tuples and values are their descending-lexicographic
        Fock indices.

        Parameters
        ----------
        None
            This method uses the state vector metadata stored on ``self``.

        Returns
        -------
        dict[tuple[int, ...], int]
            Mapping from logical basis labels to canonical Fock-basis indices.

        Raises
        ------
        ValueError
            If the stored encoding cannot resolve a basis for this state
            vector's ``n_modes`` and ``n_photons``.

        Examples
        --------
        >>> import torch
        >>> from merlin.core import EncodingSpace
        >>> from merlin.core.state_vector import StateVector
        >>> logical = torch.zeros(4, dtype=torch.complex64)
        >>> sv = StateVector.from_tensor(logical, encoding=EncodingSpace.DUAL_RAIL)
        >>> sv.logical_to_fock_map()
        {(0, 0): 2, (0, 1): 3, (1, 0): 5, (1, 1): 6}
        """

        return self.encoding.logical_to_fock_indices(
            n_modes=self.n_modes,
            n_photons=self.n_photons,
        )

    def _tensor_coalesced(self) -> torch.Tensor:
        if not self.tensor.is_sparse:
            return self.tensor
        if self.tensor.is_coalesced():
            return self.tensor
        coalesced = self.tensor.coalesce()
        self.tensor = coalesced
        return coalesced

    def _extract_single_state(self) -> tuple[tuple[int, ...], torch.Tensor] | None:
        """Detect one-hot vectors (single non-zero amplitude) and return (state, amplitude)."""
        if self.tensor.ndim != 1:
            return None
        if self.is_sparse:
            coalesced = self._tensor_coalesced()
            if coalesced._nnz() != 1:
                return None
            idx = int(coalesced.indices()[0, 0].item())
            return cast(tuple[int, ...], self.basis[idx]), coalesced.values()[0]
        non_zero = torch.nonzero(self.tensor.abs(), as_tuple=False)
        if non_zero.numel() != 1:
            return None
        idx = int(non_zero[0].item())
        return cast(tuple[int, ...], self.basis[idx]), self.tensor[idx]

    def to_perceval(self) -> pcvl.StateVector | list[pcvl.StateVector]:
        """Convert to ``pcvl.StateVector``.

        Returns
        -------
        pcvl.StateVector | list[pcvl.StateVector]
            A Perceval state for 1D tensors, or a list for batched tensors,
            with amplitudes preserved (no extra renormalization ).
        """
        basis = self.basis
        if self.tensor.ndim == 1:
            return self._perceval_from_1d(self.tensor, basis)
        if self.is_sparse:
            return self._perceval_from_sparse_batch(self.tensor, basis)
        flat = self.tensor.reshape(-1, self.tensor.shape[-1])
        result: list[pcvl.StateVector] = []
        for row in flat:
            result.append(self._perceval_from_1d(row, basis))
        return result

    @staticmethod
    def _perceval_from_1d(vector: torch.Tensor, basis: Basis) -> pcvl.StateVector:
        entries: Iterable[tuple[int, complex]]
        if vector.is_sparse:
            coalesced = vector.coalesce()
            entries = (
                (int(i), complex(val))
                for i, val in zip(
                    coalesced.indices().flatten().tolist(),
                    coalesced.values().tolist(),
                    strict=False,
                )
            )
        else:
            entries = ((i, complex(val)) for i, val in enumerate(vector.tolist()))
        mapping = [
            (pcvl.BasicState(basis[idx]), val)
            for idx, val in entries
            if val != 0 and val != 0.0
        ]
        acc: pcvl.StateVector | None = None
        for bs, amp in mapping:
            term = pcvl.StateVector(bs)
            if amp != 1:
                term = term * amp
            acc = term if acc is None else acc + term
        return acc if acc is not None else pcvl.StateVector()

    @staticmethod
    def _perceval_from_sparse_batch(
        tensor: torch.Tensor, basis: Basis
    ) -> list[pcvl.StateVector]:
        if tensor.ndim < 2:
            raise ValueError("Expected batched tensor for sparse batch conversion.")
        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        if indices.shape[0] != tensor.ndim:
            raise ValueError("Sparse indices rank does not match tensor rank.")
        batch_shape = tensor.shape[:-1]
        basis_dim = tensor.shape[-1]
        if basis_dim != len(basis):
            raise ValueError("Basis size mismatch in sparse batch conversion.")

        batch_maps: dict[tuple[int, ...], dict[int, complex]] = {}
        nnz = values.shape[0]
        for col in range(nnz):
            batch_coords = tuple(int(v) for v in indices[:-1, col].tolist())
            basis_idx = int(indices[-1, col].item())
            amp = complex(values[col].item())
            if amp == 0 or amp == 0.0:
                continue
            bucket = batch_maps.setdefault(batch_coords, {})
            bucket[basis_idx] = amp

        total_batches = 1
        for dim in batch_shape:
            total_batches *= dim

        def _coords_from_linear(linear: int) -> tuple[int, ...]:
            coords: list[int] = []
            rem = linear
            for dim in reversed(batch_shape):
                rem, idx = divmod(rem, dim)
                coords.append(idx)
            coords.reverse()
            return tuple(coords)

        result: list[pcvl.StateVector] = []
        for linear_idx in range(total_batches):
            coords = _coords_from_linear(linear_idx)
            entries = batch_maps.get(coords, {})
            if not entries:
                result.append(pcvl.StateVector())
                continue
            acc: pcvl.StateVector | None = None
            for basis_idx, amp in entries.items():
                if amp == 0 or amp == 0.0:
                    continue
                term = pcvl.StateVector(
                    pcvl.BasicState(cast(tuple[int, ...], basis[basis_idx]))
                )
                if amp != 1:
                    term = term * complex(amp)
                acc = term if acc is None else acc + term
            result.append(acc if acc is not None else pcvl.StateVector())
        return result

    @classmethod
    def from_perceval(
        cls,
        state_vector: pcvl.StateVector,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        sparse: bool | None = None,
    ) -> StateVector:
        """Build from a ``pcvl.StateVector``.

        Parameters
        ----------
        state_vector : pcvl.StateVector
            Perceval state to wrap.
        dtype : torch.dtype | None
            Optional target dtype.
        device : torch.device | None
            Optional target device.
        sparse : bool | None
            Force sparse or dense output. If ``None``, a density heuristic is
            used.

        Returns
        -------
        StateVector
            Merlin wrapper with metadata and preserved amplitudes.

        Raises
        ------
        ValueError
            If the Perceval state is empty or has inconsistent photon or mode
            counts.
        """
        items = list(state_vector)
        if not items:
            raise ValueError("Perceval StateVector is empty.")
        n_modes = len(items[0][0])
        n_photons = sum(int(v) for v in items[0][0])
        for basic, _ in items[1:]:
            if len(basic) != n_modes:
                raise ValueError("Inconsistent mode count in perceval StateVector.")
            if sum(int(v) for v in basic) != n_photons:
                raise ValueError(
                    "Perceval StateVector must have uniform photon number."
                )
        basis = _basis_for(n_modes, n_photons)
        index_map = _basis_index_map(basis)
        if sparse is None:
            basis_size = _basis_size(n_modes, n_photons)
            sparse = (len(items) / basis_size) <= 0.3
        if sparse:
            indices_list: list[int] = []
            values_list: list[complex] = []
            for basic, amplitude in items:
                idx = index_map.get(tuple(int(v) for v in basic))
                if idx is None:
                    continue
                amp_complex = complex(amplitude)
                if amp_complex == 0 or amp_complex == 0.0:
                    continue
                indices_list.append(idx)
                values_list.append(amp_complex)
            if not indices_list:
                zero = torch.zeros(
                    _basis_size(n_modes, n_photons),
                    device=device or torch.device("cpu"),
                )
                return cls(
                    _to_complex(zero, dtype=dtype, device=device), n_modes, n_photons
                )
            indices = torch.tensor([indices_list], dtype=torch.long, device=device)
            values = torch.tensor(
                values_list, dtype=complex_dtype_for(torch.float32), device=device
            )
            tensor = torch.sparse_coo_tensor(
                indices, values, (_basis_size(n_modes, n_photons),), device=device
            )
            tensor = _to_complex(tensor, dtype=dtype, device=device)
            return cls(tensor, n_modes, n_photons, _normalized=False)
        dense = torch.zeros(
            _basis_size(n_modes, n_photons),
            dtype=complex_dtype_for(torch.float32),
            device=device,
        )
        for basic, amplitude in items:
            idx = index_map.get(tuple(int(v) for v in basic))
            if idx is None:
                continue
            amp_tensor = torch.tensor(
                complex(amplitude), dtype=dense.dtype, device=dense.device
            )
            dense[idx] = amp_tensor
        dense = _to_complex(dense, dtype=dtype, device=device)
        return cls(dense, n_modes, n_photons, _normalized=False)

    @classmethod
    def from_basic_state(
        cls,
        state: Sequence[int] | pcvl.BasicState,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        sparse: bool = True,
    ) -> StateVector:
        """Create a one-hot state from a Fock occupation list/BasicState.

        Parameters
        ----------
        state : Sequence[int] | pcvl.BasicState
            Occupation numbers per mode.
        dtype : torch.dtype | None
            Optional target dtype.
        device : torch.device | None
            Optional target device.
        sparse : bool
            Whether to build a sparse tensor.

        Returns
        -------
        StateVector
            One-hot state vector.
        """
        counts = _basic_state_counts(state)
        n_modes = len(counts)
        n_photons = sum(counts)
        comb = Combinadics("fock", n_photons, n_modes)
        index = comb.fock_to_index(counts)
        basis_size = comb.compute_space_size()
        if sparse:
            indices = torch.tensor([[index]], dtype=torch.long, device=device)
            values = torch.ones(
                1, dtype=complex_dtype_for(torch.float32), device=device
            )
            tensor = torch.sparse_coo_tensor(
                indices, values, (basis_size,), device=device
            )
        else:
            tensor = torch.zeros(
                basis_size, dtype=complex_dtype_for(torch.float32), device=device
            )
            tensor[index] = 1.0
        tensor = _to_complex(tensor, dtype=dtype, device=device)
        return cls(tensor, n_modes, n_photons, _normalized=True)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        n_modes: int | None = None,
        n_photons: int | None = None,
        encoding: EncodingSpace | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> StateVector:
        """Wrap an existing tensor with explicit metadata.

        Parameters
        ----------
        tensor : torch.Tensor
            Dense or sparse amplitude tensor.
        n_modes : int | None
            Number of modes. Required for Fock and unbunched inputs. For
            structured encodings, omitted values are inferred from the
            encoding contract. If provided, the value must match that contract.
            Default value is None.
        n_photons : int | None
            Total photons. Required for Fock and unbunched inputs. For
            structured encodings, omitted values are inferred from the
            encoding contract or, for dual rail, from the tensor's final
            dimension. If provided, the value must match that contract. Default
            value is None.
        encoding : EncodingSpace | None
            Logical input encoding. When omitted, the tensor is treated as
            canonical Fock-space amplitudes and stored unchanged. Default value
            is None.
        dtype : torch.dtype | None
            Target dtype. If omitted, complex inputs keep their dtype and real
            inputs are promoted to ``torch.complex64``. Default value is None.
        device : torch.device | None
            Target device. If omitted, the input tensor's device is preserved.
            Default value is None.

        Returns
        -------
        StateVector
            Wrapped tensor with metadata.

        Raises
        ------
        ValueError
            If the tensor is scalar, if required dimensions are missing, if
            explicit dimensions do not match the encoding contract, or if the
            last dimension does not match the resolved logical basis size.
        """
        resolved_encoding = encoding or EncodingSpace.FOCK
        resolved_modes, resolved_photons = _resolve_from_tensor_dimensions(
            tensor,
            encoding=resolved_encoding,
            n_modes=n_modes,
            n_photons=n_photons,
        )
        logical_basis_size = resolved_encoding.logical_basis_size(
            n_modes=resolved_modes, n_photons=resolved_photons
        )
        if tensor.shape[-1] != logical_basis_size:
            if resolved_encoding.kind == "fock":
                raise ValueError(
                    f"Tensor last dimension {tensor.shape[-1]} does not match "
                    f"basis size {logical_basis_size}."
                )
            raise ValueError(
                "Tensor last dimension does not match the logical basis size "
                f"for encoding '{resolved_encoding.kind}': got {tensor.shape[-1]}, "
                f"expected {logical_basis_size}."
            )
        normalized = _to_complex(tensor, dtype=dtype, device=device)
        if resolved_encoding.kind != "fock":
            normalized = resolved_encoding.embed(
                normalized, n_modes=resolved_modes, n_photons=resolved_photons
            )
        return cls(
            normalized,
            resolved_modes,
            resolved_photons,
            encoding=resolved_encoding,
            _normalized=False,
        )

    def tensor_product(
        self,
        other: StateVector | Sequence[int] | pcvl.BasicState,
        *,
        sparse: bool | None = None,
    ) -> StateVector:
        """Tensor product of two states with metadata propagation.

        If any operand is dense, the result is dense. Supports one-hot fast path.
        The resulting state is normalized before returning.

        Parameters
        ----------
        other : StateVector | Sequence[int] | pcvl.BasicState
            Another state vector or a basic state / occupation list.
        sparse : bool | None
            Override sparsity of the result. By default the result remains
            dense if any input is dense.

        Returns
        -------
        StateVector
            Combined state with summed modes and photons (normalized).

        Raises
        ------
        ValueError
            If tensors are not one-dimensional.
        """
        if not isinstance(other, StateVector):
            other = StateVector.from_basic_state(
                other,
                device=self.tensor.device,
                dtype=self.tensor.dtype,
                sparse=self.is_sparse if sparse is None else sparse,
            )
        if self.tensor.ndim != 1 or other.tensor.ndim != 1:
            raise ValueError("tensor_product currently supports 1D state tensors only.")
        m_total = self.n_modes + other.n_modes
        n_total = self.n_photons + other.n_photons
        basis_total = _basis_for(m_total, n_total)
        basis_left = self.basis
        basis_right = other.basis
        left_index = _basis_index_map(basis_left)
        right_index = _basis_index_map(basis_right)
        size_total = len(basis_total)

        if sparse is None:
            sparse = self.is_sparse and other.is_sparse

        left_single = self._extract_single_state()
        right_single = other._extract_single_state()
        if left_single is not None:
            return self._product_with_basic(
                left_single, other, basic_on_left=True, sparse=sparse
            )
        if right_single is not None:
            return self._product_with_basic(
                right_single, self, basic_on_left=False, sparse=sparse
            )

        left_dense = self.to_dense()
        right_dense = other.to_dense()
        if right_dense.device != left_dense.device:
            right_dense = right_dense.to(left_dense.device)
        if right_dense.dtype != left_dense.dtype:
            right_dense = right_dense.to(left_dense.dtype)
        return self._dense_product(
            left_dense,
            right_dense,
            basis_total,
            left_index,
            right_index,
            size_total,
            m_split=self.n_modes,
            n_modes_total=m_total,
            n_photons_total=n_total,
        )

    def _product_with_basic(
        self,
        basic_entry: tuple[tuple[int, ...], torch.Tensor],
        other: StateVector,
        *,
        basic_on_left: bool,
        sparse: bool | None,
    ) -> StateVector:
        basic_state, basic_amp = basic_entry
        device = other.tensor.device
        dtype = other.tensor.dtype
        amp_scalar = basic_amp.to(device=device, dtype=dtype)
        m_total = len(basic_state) + other.n_modes
        n_total = sum(basic_state) + other.n_photons
        comb_total = Combinadics("fock", n_total, m_total)
        size_total = comb_total.compute_space_size()
        basis_other = other.basis

        use_sparse = sparse
        if use_sparse:
            coalesced = (
                other.tensor.coalesce() if other.is_sparse else other.tensor.to_sparse()
            )
            idx_list: list[int] = []
            val_list: list[torch.Tensor] = []
            flat_indices = coalesced.indices().flatten().tolist()
            values = coalesced.values().to(device=device, dtype=dtype)
            for pos, val in zip(flat_indices, values, strict=False):
                state_other = basis_other[pos]
                combined = (
                    basic_state + state_other
                    if basic_on_left
                    else state_other + basic_state
                )
                idx_total = comb_total.fock_to_index(combined)
                idx_list.append(idx_total)
                val_list.append(amp_scalar * val)
            if not idx_list:
                zero = torch.zeros(size_total, dtype=dtype, device=device)
                return StateVector(zero, m_total, n_total)
            indices = torch.tensor([idx_list], dtype=torch.long, device=device)
            values_tensor = torch.stack(val_list)
            tensor = torch.sparse_coo_tensor(
                indices, values_tensor, (size_total,), device=device
            )
            return StateVector(
                _normalize_tensor(tensor), m_total, n_total, _normalized=True
            )

        other_dense = other.to_dense().to(device=device, dtype=dtype)
        output = torch.zeros(size_total, dtype=dtype, device=device)
        for idx_other, state_other in enumerate(basis_other):
            combined = (
                basic_state + state_other
                if basic_on_left
                else state_other + basic_state
            )
            idx_total = comb_total.fock_to_index(combined)
            output[idx_total] = amp_scalar * other_dense[idx_other]
        return StateVector(
            _normalize_tensor(output), m_total, n_total, _normalized=True
        )

    def __add__(self, other: StateVector) -> StateVector:
        """Add two states without renormalization (lazy norm, like Perceval).

        Parameters
        ----------
        other : StateVector
            State vector with matching metadata.

        Returns
        -------
        StateVector
            Sum with raw amplitudes preserved.

        Raises
        ------
        ValueError
            If metadata mismatches.
        """
        if not isinstance(other, StateVector):
            return NotImplemented
        if self.n_modes != other.n_modes or self.n_photons != other.n_photons:
            raise ValueError(
                "StateVector addition requires matching n_modes and n_photons."
            )
        target_sparse = self.is_sparse and other.is_sparse
        if target_sparse:
            summed = self._tensor_coalesced() + other.tensor.coalesce()
            return StateVector(summed, self.n_modes, self.n_photons, _normalized=False)
        left = self.tensor.to_dense() if self.is_sparse else self.tensor
        right = other.tensor.to_dense() if other.is_sparse else other.tensor
        if right.device != left.device:
            right = right.to(left.device)
        if right.dtype != left.dtype:
            right = right.to(left.dtype)
        summed = left + right
        return StateVector(summed, self.n_modes, self.n_photons, _normalized=False)

    def __sub__(self, other: StateVector) -> StateVector:
        """Subtract two states without renormalization (lazy norm).

        Parameters
        ----------
        other : StateVector
            State vector with matching metadata.

        Returns
        -------
        StateVector
            Difference with raw amplitudes preserved.

        Raises
        ------
        ValueError
            If metadata mismatches.
        """
        if not isinstance(other, StateVector):
            return NotImplemented
        if self.n_modes != other.n_modes or self.n_photons != other.n_photons:
            raise ValueError(
                "StateVector subtraction requires matching n_modes and n_photons."
            )
        target_sparse = self.is_sparse and other.is_sparse
        if target_sparse:
            diff = self._tensor_coalesced() - other.tensor.coalesce()
            return StateVector(diff, self.n_modes, self.n_photons, _normalized=False)
        left = self.tensor.to_dense() if self.is_sparse else self.tensor
        right = other.tensor.to_dense() if other.is_sparse else other.tensor
        if right.device != left.device:
            right = right.to(left.device)
        if right.dtype != left.dtype:
            right = right.to(left.dtype)
        diff = left - right
        return StateVector(diff, self.n_modes, self.n_photons, _normalized=False)

    def __mul__(self, scalar: Scalar) -> StateVector:
        """Scale amplitudes by a scalar (no renormalization)."""
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        if self.is_sparse:
            return StateVector(
                self.tensor * scalar, self.n_modes, self.n_photons, _normalized=False
            )
        return StateVector(
            self.tensor * scalar, self.n_modes, self.n_photons, _normalized=False
        )

    def __rmul__(self, scalar: Scalar) -> StateVector:
        """Right scalar multiplication delegation."""
        return self.__mul__(scalar)

    def __matmul__(
        self, other: StateVector | Sequence[int] | pcvl.BasicState
    ) -> StateVector:
        """Tensor product operator alias for ``tensor_product``."""
        if isinstance(other, (StateVector, Sequence, pcvl.BasicState)):
            return self.tensor_product(other)
        return NotImplemented

    def __rmatmul__(
        self, other: StateVector | Sequence[int] | pcvl.BasicState
    ) -> StateVector:
        """Right tensor product to support BasicState/sequence @ StateVector."""
        if isinstance(other, StateVector):
            return other.tensor_product(self)
        if isinstance(other, (Sequence, pcvl.BasicState)):
            left = StateVector.from_basic_state(
                other,
                device=self.tensor.device,
                dtype=self.tensor.dtype,
                sparse=self.is_sparse,
            )
            return left.tensor_product(self)
        return NotImplemented

    def index(self, state: Sequence[int] | pcvl.BasicState) -> int | None:
        """Return basis index for the given Fock state.

        Parameters
        ----------
        state : Sequence[int] | pcvl.BasicState
            Occupation list or basic state.

        Returns
        -------
        int | None
            Basis index, or ``None`` if not present.
        """
        target = _basic_state_tuple(state)
        basis = self.basis
        try:
            idx = basis.index(target)
        except ValueError:
            return None
        if self.is_sparse:
            coalesced = self._tensor_coalesced()
            positions = (coalesced.indices()[-1] == idx).nonzero(as_tuple=False)
            return idx if positions.numel() > 0 else None
        return idx

    def __getitem__(self, state: Sequence[int] | pcvl.BasicState) -> torch.Tensor:
        """Amplitude lookup for a given Fock state.

        Parameters
        ----------
        state : Sequence[int] | pcvl.BasicState
            Occupation list or basic state.

        Returns
        -------
        torch.Tensor
            Amplitude scalar or batch-aligned tensor.

        Raises
        ------
        KeyError
            If the state is outside the basis.
        """
        target = _basic_state_tuple(state)
        basis = self.basis
        try:
            idx = basis.index(target)
        except ValueError:
            raise KeyError("State not in basis") from None

        normalized = self._normalized_tensor()

        if normalized.ndim == 1:
            if normalized.is_sparse:
                coalesced = normalized.coalesce()
                mask = coalesced.indices()[0] == idx
                positions = mask.nonzero(as_tuple=False)
                if positions.numel() == 0:
                    return torch.zeros(
                        (), dtype=normalized.dtype, device=normalized.device
                    )
                return coalesced.values()[positions[0, 0]]
            return normalized[idx]

        # Batched: gather amplitudes for each batch entry
        if normalized.is_sparse:
            coalesced = normalized.coalesce()
            indices = coalesced.indices()
            values = coalesced.values()
            batch_shape = normalized.shape[:-1]
            batch_size = 1
            for dim in batch_shape:
                batch_size *= dim
            out = torch.zeros(
                batch_size, dtype=normalized.dtype, device=normalized.device
            )
            nnz = values.shape[0]
            for col in range(nnz):
                if int(indices[-1, col].item()) != idx:
                    continue
                # map batch coords to linear
                coords = [int(v) for v in indices[:-1, col].tolist()]
                linear = 0
                for d, s in zip(coords, batch_shape, strict=False):
                    linear = linear * s + d
                out[linear] = values[col]
            return out.view(*batch_shape)

        flat = normalized.reshape(-1, normalized.shape[-1])
        gathered = flat[..., idx]
        return gathered.view(*normalized.shape[:-1])

    def _dense_product(
        self,
        left_tensor: torch.Tensor,
        right_tensor: torch.Tensor,
        basis_total: Basis,
        left_index: dict[tuple[int, ...], int],
        right_index: dict[tuple[int, ...], int],
        size_total: int,
        *,
        m_split: int,
        n_modes_total: int,
        n_photons_total: int,
    ) -> StateVector:
        device = left_tensor.device
        dtype = left_tensor.dtype
        output = torch.zeros(size_total, dtype=dtype, device=device)
        for idx_total, state in enumerate(basis_total):
            left_state = state[:m_split]
            right_state = state[m_split:]
            idx_left = left_index.get(left_state)
            idx_right = right_index.get(right_state)
            if idx_left is None or idx_right is None:
                continue
            output[idx_total] = left_tensor[idx_left] * right_tensor[idx_right]
        return StateVector(
            _normalize_tensor(output), n_modes_total, n_photons_total, _normalized=True
        )

    def to_dense(self) -> torch.Tensor:
        """Return a dense, normalized tensor view of the amplitudes."""
        normalized = self._normalized_tensor()
        return normalized.to_dense() if normalized.is_sparse else normalized

    def normalize(self) -> StateVector:
        """Normalize this state in-place and return self."""
        if self._normalized:
            return self
        normalized_tensor = _normalize_tensor(self.tensor)
        self.tensor = normalized_tensor
        self._normalized = True
        return self

    def normalized_str(self) -> str:
        """Human-friendly string of the normalized state (forces normalization for display)."""
        normalized = self.normalize()
        return f"StateVector(n_modes={normalized.n_modes}, n_photons={normalized.n_photons}, tensor={normalized.tensor})"

    def __str__(self) -> str:  # pragma: no cover - simple wrapper
        return self.normalized_str()


__all__ = ["StateVector"]
