"""Input encoding definitions and mapping helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import ClassVar

from merlin.utils.combinadics import Combinadics

TupleInt = tuple[int, ...]


@dataclass(frozen=True, init=False)
class EncodingSpace:
    """Validated input encoding definition.

    Instances describe how a compact logical basis should be interpreted before it
    is embedded into Merlin's canonical Fock ordering.
    """

    family: str
    kind: str
    modes_per_photon: tuple[int, ...] | None
    qubit_groups: tuple[int, ...] | None

    FOCK: ClassVar[EncodingSpace]
    UNBUNCHED: ClassVar[EncodingSpace]
    DUAL_RAIL: ClassVar[EncodingSpace]

    def __init__(
        self,
        modes_per_photon: Iterable[int] | None = None,
        *,
        family: str | None = None,
        kind: str | None = None,
        qubit_groups: Iterable[int] | None = None,
    ) -> None:
        if modes_per_photon is None:
            if qubit_groups is not None:
                raise ValueError(
                    "qubit_groups is only supported via EncodingSpace.qloq(...)."
                )
            if family is None or kind is None:
                raise ValueError(
                    "modes_per_photon is required for custom encodings. "
                    "Use EncodingSpace.FOCK, EncodingSpace.UNBUNCHED, "
                    "EncodingSpace.DUAL_RAIL, or EncodingSpace.qloq(...)."
                )
            if family != "builtin" or kind not in {"fock", "unbunched", "dual_rail"}:
                raise ValueError("Invalid builtin encoding configuration.")
            object.__setattr__(self, "family", family)
            object.__setattr__(self, "kind", kind)
            object.__setattr__(self, "modes_per_photon", None)
            object.__setattr__(self, "qubit_groups", None)
            return

        validated_modes = self._validate_positive_int_tuple(
            modes_per_photon, name="modes_per_photon"
        )
        validated_groups = (
            None
            if qubit_groups is None
            else self._validate_positive_int_tuple(qubit_groups, name="qubit_groups")
        )
        resolved_family = "partitioned" if family is None else family
        resolved_kind = "partitioned" if kind is None else kind

        if resolved_family != "partitioned":
            raise ValueError("Custom encodings must use family='partitioned'.")
        if resolved_kind not in {"partitioned", "qloq"}:
            raise ValueError(
                "Custom encodings must use kind='partitioned' or kind='qloq'."
            )

        object.__setattr__(self, "family", resolved_family)
        object.__setattr__(self, "kind", resolved_kind)
        object.__setattr__(self, "modes_per_photon", validated_modes)
        object.__setattr__(self, "qubit_groups", validated_groups)

    def __repr__(self) -> str:
        fields = [f"family={self.family!r}", f"kind={self.kind!r}"]
        if self.modes_per_photon is not None:
            fields.append(f"modes_per_photon={self.modes_per_photon!r}")
        if self.qubit_groups is not None:
            fields.append(f"qubit_groups={self.qubit_groups!r}")
        return f"EncodingSpace({', '.join(fields)})"

    @property
    def parameters(self) -> dict[str, tuple[int, ...]]:
        """Return a copy of the encoding parameters for introspection."""

        params: dict[str, tuple[int, ...]] = {}
        if self.modes_per_photon is not None:
            params["modes_per_photon"] = self.modes_per_photon
        if self.qubit_groups is not None:
            params["qubit_groups"] = self.qubit_groups
        return params

    @property
    def n_modes(self) -> int | None:
        """Return the total mode count for partitioned encodings."""

        if self.modes_per_photon is None:
            return None
        return sum(self.modes_per_photon)

    @property
    def n_photons(self) -> int | None:
        """Return the photon count for partitioned encodings."""

        if self.modes_per_photon is None:
            return None
        return len(self.modes_per_photon)

    @classmethod
    def qloq(cls, qubit_groups: Iterable[int]) -> EncodingSpace:
        """Create a partitioned QLOQ encoding from qubit group sizes."""

        validated_groups = cls._validate_positive_int_tuple(
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
        """Return the number of logical basis states for this encoding."""

        if self.family == "partitioned":
            size = 1
            assert self.modes_per_photon is not None
            for width in self.modes_per_photon:
                size *= width
            return size

        resolved_modes, resolved_photons = self._resolve_dimensions(
            n_modes=n_modes, n_photons=n_photons
        )
        return Combinadics(
            self.kind, resolved_photons, resolved_modes
        ).compute_space_size()

    def logical_basis_states(
        self,
        *,
        n_modes: int | None = None,
        n_photons: int | None = None,
    ) -> tuple[TupleInt, ...]:
        """Return logical basis labels in stable embedding order."""

        if self.family == "partitioned":
            assert self.modes_per_photon is not None
            return self._product_basis_states(self.modes_per_photon)

        resolved_modes, resolved_photons = self._resolve_dimensions(
            n_modes=n_modes, n_photons=n_photons
        )
        if self.kind == "dual_rail":
            return self._product_basis_states((2,) * resolved_photons)
        if self.kind == "unbunched":
            return tuple(
                tuple(index for index, count in enumerate(state) if count)
                for state in Combinadics(
                    self.kind, resolved_photons, resolved_modes
                ).enumerate_states()
            )

        return tuple(
            Combinadics(self.kind, resolved_photons, resolved_modes).enumerate_states()
        )

    def fock_basis_states(
        self,
        *,
        n_modes: int | None = None,
        n_photons: int | None = None,
    ) -> tuple[TupleInt, ...]:
        """Return mapped Fock states in the same order as ``logical_basis_states``."""

        return tuple(
            self.logical_to_fock_map(n_modes=n_modes, n_photons=n_photons).values()
        )

    def logical_to_fock_map(
        self,
        *,
        n_modes: int | None = None,
        n_photons: int | None = None,
    ) -> dict[TupleInt, TupleInt]:
        """Return the logical-to-Fock mapping in stable order."""

        logical_states = self.logical_basis_states(n_modes=n_modes, n_photons=n_photons)
        if self.family == "partitioned":
            assert self.modes_per_photon is not None
            return self._logical_states_to_partitioned_fock_map(
                logical_states, self.modes_per_photon
            )

        if self.kind == "fock":
            return {state: state for state in logical_states}

        resolved_modes, resolved_photons = self._resolve_dimensions(
            n_modes=n_modes, n_photons=n_photons
        )
        if self.kind == "dual_rail":
            return self._logical_states_to_partitioned_fock_map(
                logical_states, (2,) * resolved_photons
            )

        if self.kind == "unbunched":
            mapping: dict[TupleInt, TupleInt] = {}
            for logical_state in logical_states:
                counts = [0] * resolved_modes
                for occupied_mode in logical_state:
                    counts[occupied_mode] = 1
                mapping[logical_state] = tuple(counts)
            return mapping

        raise ValueError(f"Unsupported encoding kind '{self.kind}'.")

    def logical_to_fock_indices(
        self,
        *,
        n_modes: int | None = None,
        n_photons: int | None = None,
    ) -> dict[TupleInt, int]:
        """Return full-Fock indices for each logical basis label."""

        resolved_modes, resolved_photons = self._resolve_dimensions(
            n_modes=n_modes, n_photons=n_photons
        )
        fock_basis = Combinadics("fock", resolved_photons, resolved_modes)
        return {
            logical_state: fock_basis.fock_to_index(fock_state)
            for logical_state, fock_state in self.logical_to_fock_map(
                n_modes=n_modes, n_photons=n_photons
            ).items()
        }

    def _resolve_dimensions(
        self,
        *,
        n_modes: int | None,
        n_photons: int | None,
    ) -> tuple[int, int]:
        if self.family == "partitioned":
            assert self.modes_per_photon is not None
            expected_modes = sum(self.modes_per_photon)
            expected_photons = len(self.modes_per_photon)
            if n_modes is not None and n_modes != expected_modes:
                raise ValueError(
                    f"EncodingSpace expects n_modes={expected_modes}, got {n_modes}."
                )
            if n_photons is not None and n_photons != expected_photons:
                raise ValueError(
                    f"EncodingSpace expects n_photons={expected_photons}, got {n_photons}."
                )
            return expected_modes, expected_photons

        resolved_modes = self._validate_mode_count(n_modes)
        resolved_photons = self._validate_photon_count(n_photons)
        if self.kind == "dual_rail":
            if resolved_modes is None and resolved_photons is not None:
                resolved_modes = 2 * resolved_photons
            if resolved_photons is None and resolved_modes is not None:
                if resolved_modes % 2 != 0:
                    raise ValueError("dual_rail requires an even n_modes value.")
                resolved_photons = resolved_modes // 2

        if resolved_modes is None or resolved_photons is None:
            raise ValueError(f"{self.kind} encoding requires n_modes and n_photons.")
        return resolved_modes, resolved_photons

    @staticmethod
    def _partition_offsets(modes_per_photon: tuple[int, ...]) -> tuple[int, ...]:
        offsets = []
        start = 0
        for width in modes_per_photon:
            offsets.append(start)
            start += width
        return tuple(offsets)

    @staticmethod
    def _product_basis_states(widths: tuple[int, ...]) -> tuple[TupleInt, ...]:
        ranges = (range(width) for width in widths)
        return tuple(tuple(state) for state in product(*ranges))

    @classmethod
    def _logical_states_to_partitioned_fock_map(
        cls,
        logical_states: tuple[TupleInt, ...],
        modes_per_photon: tuple[int, ...],
    ) -> dict[TupleInt, TupleInt]:
        offsets = cls._partition_offsets(modes_per_photon)
        total_modes = sum(modes_per_photon)
        mapping: dict[TupleInt, TupleInt] = {}
        for logical_state in logical_states:
            counts = [0] * total_modes
            for index, local_mode in enumerate(logical_state):
                counts[offsets[index] + local_mode] = 1
            mapping[logical_state] = tuple(counts)
        return mapping

    @staticmethod
    def _validate_positive_int_tuple(
        values: Iterable[int], *, name: str
    ) -> tuple[int, ...]:
        validated = tuple(values)
        if not validated:
            raise ValueError(f"{name} must contain at least one value.")
        for value in validated:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must contain integers only.")
            if value <= 0:
                raise ValueError(f"{name} must contain strictly positive integers.")
        return validated

    @staticmethod
    def _validate_mode_count(value: int | None) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("n_modes must be a strictly positive integer.")
        return value

    @staticmethod
    def _validate_photon_count(value: int | None) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("n_photons must be a non-negative integer.")
        return value


EncodingSpace.FOCK = EncodingSpace(family="builtin", kind="fock")
EncodingSpace.UNBUNCHED = EncodingSpace(family="builtin", kind="unbunched")
EncodingSpace.DUAL_RAIL = EncodingSpace(family="builtin", kind="dual_rail")
