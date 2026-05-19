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

"""Computation space definitions controlling output basis selection."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from .encoding_space import EncodingSpace

TupleInt = tuple[int, ...]


class _ComputationSpaceMeta(type):
    """Provide enum-style iteration over legacy built-in spaces."""

    def __iter__(cls) -> Iterator[ComputationSpace]:
        """Iterate over the legacy built-in computation spaces.

        Returns
        -------
        Iterator[ComputationSpace]
            Iterator yielding ``FOCK``, ``UNBUNCHED``, and ``DUAL_RAIL``.
        """
        computation_space_cls = cast(type[ComputationSpace], cls)
        return iter((
            computation_space_cls.FOCK,
            computation_space_cls.UNBUNCHED,
            computation_space_cls.DUAL_RAIL,
        ))


@dataclass(frozen=True, init=False, eq=False)
class ComputationSpace(str, metaclass=_ComputationSpaceMeta):
    """Definition of an output computation space.

    ``ComputationSpace`` describes which output Fock states are retained and how
    they are ordered for measurement. The legacy built-ins remain available as
    enum-like singleton constants. Custom partitioned and QLOQ spaces are
    additive options that reuse the same basis mapping rules as
    :class:`~merlin.core.encoding_space.EncodingSpace` while remaining a
    distinct output-side public type.
    """

    family: str
    kind: str
    modes_per_photon: tuple[int, ...] | None
    qubit_groups: tuple[int, ...] | None

    FOCK: ClassVar[ComputationSpace]
    UNBUNCHED: ClassVar[ComputationSpace]
    DUAL_RAIL: ClassVar[ComputationSpace]

    def __new__(
        cls,
        modes_per_photon: Iterable[int] | str | None = None,
        *,
        family: str | None = None,
        kind: str | None = None,
        qubit_groups: Iterable[int] | None = None,
    ) -> ComputationSpace:
        """Create the string payload carried by the computation-space object.

        Parameters
        ----------
        modes_per_photon : Iterable[int] | str | None
            Partition widths, legacy built-in string, or None. Default value is
            None.
        family : str | None
            Space family. Default value is None.
        kind : str | None
            Space kind. Default value is None.
        qubit_groups : Iterable[int] | None
            QLOQ group sizes. Default value is None.

        Returns
        -------
        ComputationSpace
            String-backed computation-space instance.
        """
        pending_modes_per_photon: tuple[int, ...] | None = None
        pending_qubit_groups: tuple[int, ...] | None = None
        if isinstance(modes_per_photon, str):
            value = modes_per_photon.lower()
        elif modes_per_photon is None:
            value = kind if kind is not None else "partitioned"
        else:
            resolved_kind = kind if kind is not None else "partitioned"
            pending_modes_per_photon = tuple(modes_per_photon)
            if qubit_groups is not None:
                pending_qubit_groups = tuple(qubit_groups)
            value = f"{resolved_kind}:{pending_modes_per_photon!r}"
            if pending_qubit_groups is not None:
                value = f"{value}:{pending_qubit_groups!r}"
        if family == "builtin" and kind is not None:
            value = kind
        instance = str.__new__(cls, value)
        if pending_modes_per_photon is not None:
            object.__setattr__(
                instance, "_pending_modes_per_photon", pending_modes_per_photon
            )
        if pending_qubit_groups is not None:
            object.__setattr__(instance, "_pending_qubit_groups", pending_qubit_groups)
        return instance

    def __init__(
        self,
        modes_per_photon: Iterable[int] | str | None = None,
        *,
        family: str | None = None,
        kind: str | None = None,
        qubit_groups: Iterable[int] | None = None,
    ) -> None:
        """Create a built-in, partitioned, or QLOQ computation space.

        Parameters
        ----------
        modes_per_photon : Iterable[int] | str | None
            Partition widths for custom spaces. A string value such as
            ``"fock"`` is accepted for compatibility with the former enum
            constructor. If omitted, ``family`` and ``kind`` must describe a
            built-in space. Default value is None.
        family : str | None
            Space family. Built-ins use ``"builtin"`` except dual rail, which
            is represented as a partitioned space. Custom spaces use
            ``"partitioned"``. Default value is None.
        kind : str | None
            Stable kind identifier. Supported values are ``"fock"``,
            ``"unbunched"``, ``"dual_rail"``, ``"partitioned"``, and
            ``"qloq"``. Default value is None.
        qubit_groups : Iterable[int] | None
            Original QLOQ group sizes. This argument is accepted only by
            :meth:`qloq`. Default value is None.

        Raises
        ------
        ValueError
            If the configuration is unsupported or internally inconsistent.
        """
        if isinstance(modes_per_photon, str):
            if family is not None or kind is not None or qubit_groups is not None:
                raise ValueError(
                    "String construction cannot be combined with family, kind, "
                    "or qubit_groups."
                )
            builtin = self.coerce(modes_per_photon)
            object.__setattr__(self, "family", builtin.family)
            object.__setattr__(self, "kind", builtin.kind)
            object.__setattr__(self, "modes_per_photon", builtin.modes_per_photon)
            object.__setattr__(self, "qubit_groups", builtin.qubit_groups)
            return

        if modes_per_photon is None:
            self._init_builtin(family=family, kind=kind, qubit_groups=qubit_groups)
            return

        encoding_space_cls = _encoding_space_cls()
        pending_modes_per_photon = getattr(self, "_pending_modes_per_photon", None)
        if pending_modes_per_photon is not None:
            modes_per_photon = pending_modes_per_photon
        pending_qubit_groups = getattr(self, "_pending_qubit_groups", None)
        if pending_qubit_groups is not None:
            qubit_groups = pending_qubit_groups

        modes_values = cast(Iterable[int], modes_per_photon)
        validated_modes = encoding_space_cls._validate_positive_int_tuple(
            modes_values, name="modes_per_photon"
        )
        validated_groups = (
            None
            if qubit_groups is None
            else encoding_space_cls._validate_positive_int_tuple(
                qubit_groups, name="qubit_groups"
            )
        )
        resolved_family = "partitioned" if family is None else family
        resolved_kind = "partitioned" if kind is None else kind
        if resolved_family != "partitioned":
            raise ValueError("Custom computation spaces must use family='partitioned'.")
        if resolved_kind not in {"partitioned", "qloq"}:
            raise ValueError(
                "Custom computation spaces must use kind='partitioned' or kind='qloq'."
            )

        object.__setattr__(self, "family", resolved_family)
        object.__setattr__(self, "kind", resolved_kind)
        object.__setattr__(self, "modes_per_photon", validated_modes)
        object.__setattr__(self, "qubit_groups", validated_groups)

    def __repr__(self) -> str:
        """Return a stable debugging representation.

        Returns
        -------
        str
            Constructor-style representation of the computation space.
        """
        if self._is_builtin_constant():
            return f"ComputationSpace.{self.name}"
        fields = [f"family={self.family!r}", f"kind={self.kind!r}"]
        if self.modes_per_photon is not None:
            fields.append(f"modes_per_photon={self.modes_per_photon!r}")
        if self.qubit_groups is not None:
            fields.append(f"qubit_groups={self.qubit_groups!r}")
        return f"ComputationSpace({', '.join(fields)})"

    def __str__(self) -> str:
        """Return an enum-style display string.

        Returns
        -------
        str
            ``ComputationSpace.<NAME>`` for built-ins, otherwise ``repr(self)``.
        """
        return repr(self)

    def __eq__(self, other: object) -> bool:
        """Compare computation spaces by their stable metadata.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            True when ``other`` represents the same computation space.
        """
        if isinstance(other, str):
            return str.__eq__(self, other)
        if not isinstance(other, ComputationSpace):
            return NotImplemented
        return (
            self.family == other.family
            and self.kind == other.kind
            and self.modes_per_photon == other.modes_per_photon
            and self.qubit_groups == other.qubit_groups
        )

    def __hash__(self) -> int:
        """Return a stable hash compatible with equality.

        Returns
        -------
        int
            Hash value for the computation-space metadata.
        """
        return str.__hash__(self)

    @property
    def value(self) -> str:
        """Return the stable string identifier for this computation space.

        Returns
        -------
        str
            The legacy enum value for built-ins, or the custom space kind.
        """
        return self.kind

    @property
    def name(self) -> str:
        """Return an enum-style uppercase name.

        Returns
        -------
        str
            Uppercase name for compatibility with enum-style call sites.
        """
        return self.kind.upper()

    @property
    def parameters(self) -> dict[str, tuple[int, ...]]:
        """Return configured parameters for introspection.

        Returns
        -------
        dict[str, tuple[int, ...]]
            Copy of configured immutable parameter tuples.
        """
        params: dict[str, tuple[int, ...]] = {}
        if self.modes_per_photon is not None:
            params["modes_per_photon"] = self.modes_per_photon
        if self.qubit_groups is not None:
            params["qubit_groups"] = self.qubit_groups
        return params

    @property
    def n_modes(self) -> int | None:
        """Return the configured mode count for partitioned spaces.

        Returns
        -------
        int | None
            Sum of ``modes_per_photon`` for partitioned spaces, otherwise None.
        """
        if self.modes_per_photon is None:
            return None
        return sum(self.modes_per_photon)

    @property
    def n_photons(self) -> int | None:
        """Return the configured photon count for partitioned spaces.

        Returns
        -------
        int | None
            Number of photon partitions for partitioned spaces, otherwise None.
        """
        if self.modes_per_photon is None:
            return None
        return len(self.modes_per_photon)

    @property
    def requires_postselection(self) -> bool:
        """Return whether the space filters the full Fock output basis.

        Returns
        -------
        bool
            False only for ``ComputationSpace.FOCK``.
        """
        return self.kind != "fock"

    @classmethod
    def default(cls, *, no_bunching: bool) -> ComputationSpace:
        """Derive the default computation space from the legacy flag.

        Parameters
        ----------
        no_bunching : bool
            Legacy flag indicating whether bunching should be disallowed.

        Returns
        -------
        ComputationSpace
            Default computation space matching the legacy behavior.
        """
        return cls.UNBUNCHED if no_bunching else cls.FOCK

    @classmethod
    def coerce(cls, value: ComputationSpace | str) -> ComputationSpace:
        """Normalize user-provided values.

        Parameters
        ----------
        value : ComputationSpace | str
            ComputationSpace instance or case-insensitive built-in string.

        Returns
        -------
        ComputationSpace
            Normalized computation space value.

        Raises
        ------
        ValueError
            If ``value`` does not match a supported built-in computation space.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.lower()
            for space in cls:
                if normalized == space.value:
                    return space
        supported = sorted(space.value for space in cls)
        raise ValueError(
            f"Invalid computation_space '{value}'. Supported values are {supported}."
        )

    @classmethod
    def qloq(cls, qubit_groups: Iterable[int]) -> ComputationSpace:
        """Create a QLOQ output computation space.

        Parameters
        ----------
        qubit_groups : Iterable[int]
            Number of qubits represented by each photon group. Each entry
            expands to ``2 ** group`` modes.

        Returns
        -------
        ComputationSpace
            Partitioned QLOQ computation space.

        Raises
        ------
        ValueError
            If ``qubit_groups`` is empty or contains non-positive integers.
        """
        encoding_space_cls = _encoding_space_cls()
        validated_groups = encoding_space_cls._validate_positive_int_tuple(
            qubit_groups, name="qubit_groups"
        )
        modes_per_photon = tuple(1 << group for group in validated_groups)
        return cls(
            modes_per_photon,
            family="partitioned",
            kind="qloq",
            qubit_groups=validated_groups,
        )

    def basis_size(
        self, *, n_modes: int | None = None, n_photons: int | None = None
    ) -> int:
        """Return the number of retained output basis states.

        Parameters
        ----------
        n_modes : int | None
            Circuit mode count used to resolve or validate the space. Default
            value is None.
        n_photons : int | None
            Circuit photon count used to resolve or validate the space. Default
            value is None.

        Returns
        -------
        int
            Number of retained output basis states.

        Raises
        ------
        ValueError
            If the dimensions are missing or incompatible with this space.
        """
        return self._as_encoding_space().basis_size(
            n_modes=n_modes, n_photons=n_photons
        )

    def resolved_modes_per_photon(
        self, *, n_modes: int | None = None, n_photons: int | None = None
    ) -> tuple[int, ...] | None:
        """Return the resolved partition layout for structured spaces.

        Parameters
        ----------
        n_modes : int | None
            Circuit mode count used to resolve dual rail. Default value is None.
        n_photons : int | None
            Circuit photon count used to resolve dual rail. Default value is
            None.

        Returns
        -------
        tuple[int, ...] | None
            Modes per photon for partitioned and dual-rail spaces, otherwise
            None.
        """
        return self._as_encoding_space().resolved_modes_per_photon(
            n_modes=n_modes, n_photons=n_photons
        )

    def logical_basis_states(
        self, *, n_modes: int | None = None, n_photons: int | None = None
    ) -> tuple[TupleInt, ...]:
        """Return compact output labels in stable order.

        Parameters
        ----------
        n_modes : int | None
            Circuit mode count used to resolve or validate the space. Default
            value is None.
        n_photons : int | None
            Circuit photon count used to resolve or validate the space. Default
            value is None.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Compact labels for retained output states.
        """
        return self._as_encoding_space().logical_basis_states(
            n_modes=n_modes, n_photons=n_photons
        )

    def fock_basis_states(
        self, *, n_modes: int | None = None, n_photons: int | None = None
    ) -> tuple[TupleInt, ...]:
        """Return retained Fock states in output order.

        Parameters
        ----------
        n_modes : int | None
            Circuit mode count used to resolve or validate the space. Default
            value is None.
        n_photons : int | None
            Circuit photon count used to resolve or validate the space. Default
            value is None.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Occupation-count states retained by this computation space.
        """
        return self._as_encoding_space().fock_basis_states(
            n_modes=n_modes, n_photons=n_photons
        )

    def logical_to_fock_map(
        self, *, n_modes: int | None = None, n_photons: int | None = None
    ) -> dict[TupleInt, TupleInt]:
        """Return compact output labels mapped to Fock occupation states.

        Parameters
        ----------
        n_modes : int | None
            Circuit mode count used to resolve or validate the space. Default
            value is None.
        n_photons : int | None
            Circuit photon count used to resolve or validate the space. Default
            value is None.

        Returns
        -------
        dict[tuple[int, ...], tuple[int, ...]]
            Mapping from compact output labels to Fock states.
        """
        return self._as_encoding_space().logical_to_fock_map(
            n_modes=n_modes, n_photons=n_photons
        )

    def logical_to_fock_indices(
        self, *, n_modes: int | None = None, n_photons: int | None = None
    ) -> dict[TupleInt, int]:
        """Return full-Fock indices for retained output labels.

        Parameters
        ----------
        n_modes : int | None
            Circuit mode count used to resolve or validate the space. Default
            value is None.
        n_photons : int | None
            Circuit photon count used to resolve or validate the space. Default
            value is None.

        Returns
        -------
        dict[tuple[int, ...], int]
            Mapping from compact output labels to canonical full-Fock indices.
        """
        return self._as_encoding_space().logical_to_fock_indices(
            n_modes=n_modes, n_photons=n_photons
        )

    def _init_builtin(
        self,
        *,
        family: str | None,
        kind: str | None,
        qubit_groups: Iterable[int] | None,
    ) -> None:
        """Initialize one of the built-in spaces from internal metadata."""
        if qubit_groups is not None:
            raise ValueError(
                "qubit_groups is only supported via ComputationSpace.qloq(...)."
            )
        if family is None or kind is None:
            raise ValueError(
                "modes_per_photon is required for custom computation spaces. "
                "Use ComputationSpace.FOCK, ComputationSpace.UNBUNCHED, "
                "ComputationSpace.DUAL_RAIL, or ComputationSpace.qloq(...)."
            )
        if family == "builtin" and kind in {"fock", "unbunched"}:
            object.__setattr__(self, "family", family)
            object.__setattr__(self, "kind", kind)
            object.__setattr__(self, "modes_per_photon", None)
            object.__setattr__(self, "qubit_groups", None)
            return
        if family == "partitioned" and kind == "dual_rail":
            object.__setattr__(self, "family", family)
            object.__setattr__(self, "kind", kind)
            object.__setattr__(self, "modes_per_photon", None)
            object.__setattr__(self, "qubit_groups", None)
            return
        raise ValueError("Invalid builtin computation-space configuration.")

    def _is_builtin_constant(self) -> bool:
        """Return whether ``self`` matches one of the legacy built-ins."""
        return (
            self.modes_per_photon is None
            and self.qubit_groups is None
            and (
                (self.family == "builtin" and self.kind in {"fock", "unbunched"})
                or (self.family == "partitioned" and self.kind == "dual_rail")
            )
        )

    def _as_encoding_space(self) -> EncodingSpace:
        """Return an internal delegate for shared basis/mapping machinery."""
        encoding_space_cls = _encoding_space_cls()
        if self.kind == "fock" and self.family == "builtin":
            return encoding_space_cls.FOCK
        if self.kind == "unbunched" and self.family == "builtin":
            return encoding_space_cls.UNBUNCHED
        if self.kind == "dual_rail" and self.family == "partitioned":
            return encoding_space_cls.DUAL_RAIL
        if self.kind == "qloq":
            assert self.qubit_groups is not None
            return encoding_space_cls.qloq(self.qubit_groups)
        assert self.modes_per_photon is not None
        return encoding_space_cls(
            self.modes_per_photon,
            family=self.family,
            kind=self.kind,
        )


def _encoding_space_cls() -> type[EncodingSpace]:
    """Import EncodingSpace lazily to avoid import cycles during package setup."""
    from .encoding_space import EncodingSpace

    return EncodingSpace


ComputationSpace.FOCK = ComputationSpace(family="builtin", kind="fock")
ComputationSpace.UNBUNCHED = ComputationSpace(family="builtin", kind="unbunched")
ComputationSpace.DUAL_RAIL = ComputationSpace(family="partitioned", kind="dual_rail")
