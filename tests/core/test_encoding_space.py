import pytest

from merlin import EncodingSpace
from merlin.core import EncodingSpace as CoreEncodingSpace
from merlin.utils.combinadics import Combinadics


def test_builtin_encoding_space_constants():
    assert CoreEncodingSpace is EncodingSpace
    assert EncodingSpace.FOCK.kind == "fock"
    assert EncodingSpace.FOCK.family == "builtin"
    assert EncodingSpace.UNBUNCHED.kind == "unbunched"
    assert EncodingSpace.UNBUNCHED.family == "builtin"
    assert EncodingSpace.DUAL_RAIL.kind == "dual_rail"
    assert EncodingSpace.DUAL_RAIL.family == "partitioned"


def test_partitioned_modes_per_photon_validation():
    enc = EncodingSpace(modes_per_photon=[3, 4, 2])

    assert enc.family == "partitioned"
    assert enc.kind == "partitioned"
    assert enc.modes_per_photon == (3, 4, 2)
    assert enc.parameters == {"modes_per_photon": (3, 4, 2)}
    assert enc.n_photons == 3
    assert enc.n_modes == 9
    assert enc.basis_size() == 24


def test_qloq_helper_expands_groups_to_modes_per_photon():
    qloq = EncodingSpace.qloq(qubit_groups=[2, 1])

    assert qloq.family == "partitioned"
    assert qloq.kind == "qloq"
    assert qloq.qubit_groups == (2, 1)
    assert qloq.modes_per_photon == (4, 2)
    assert qloq.parameters == {
        "modes_per_photon": (4, 2),
        "qubit_groups": (2, 1),
    }


def test_encoding_space_equality_hashability_and_repr():
    left = EncodingSpace(modes_per_photon=[3, 2])
    right = EncodingSpace(modes_per_photon=(3, 2))

    assert left == right
    assert hash(left) == hash(right)
    assert {left: "ok"}[right] == "ok"
    assert (
        repr(EncodingSpace.qloq(qubit_groups=[2, 1]))
        == "EncodingSpace(family='partitioned', kind='qloq', "
        "modes_per_photon=(4, 2), qubit_groups=(2, 1))"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EncodingSpace(modes_per_photon=[]),
        lambda: EncodingSpace(modes_per_photon=[3, 0]),
        lambda: EncodingSpace(modes_per_photon=[True, 2]),
        lambda: EncodingSpace.qloq(qubit_groups=[]),
        lambda: EncodingSpace.qloq(qubit_groups=[2, 0]),
        lambda: EncodingSpace.qloq(qubit_groups=[False, 1]),
    ],
)
def test_invalid_encoding_space_inputs_raise_value_error(factory):
    with pytest.raises(ValueError):
        factory()


def test_partitioned_logical_to_fock_mapping_order():
    enc = EncodingSpace(modes_per_photon=[3, 2])

    expected_items = [
        ((0, 0), (1, 0, 0, 1, 0)),
        ((0, 1), (1, 0, 0, 0, 1)),
        ((1, 0), (0, 1, 0, 1, 0)),
        ((1, 1), (0, 1, 0, 0, 1)),
        ((2, 0), (0, 0, 1, 1, 0)),
        ((2, 1), (0, 0, 1, 0, 1)),
    ]

    assert list(enc.logical_to_fock_map().items()) == expected_items

    fock_basis = Combinadics("fock", 2, 5)
    assert enc.logical_to_fock_indices() == {
        logical_state: fock_basis.fock_to_index(fock_state)
        for logical_state, fock_state in expected_items
    }


def test_partitioned_two_by_two_order_matches_example():
    enc = EncodingSpace(modes_per_photon=[2, 2])

    expected_items = [
        ((0, 0), (1, 0, 1, 0)),
        ((0, 1), (1, 0, 0, 1)),
        ((1, 0), (0, 1, 1, 0)),
        ((1, 1), (0, 1, 0, 1)),
    ]

    assert enc.logical_basis_states() == tuple(
        logical_state for logical_state, _ in expected_items
    )
    assert enc.fock_basis_states() == tuple(
        fock_state for _, fock_state in expected_items
    )
    assert list(enc.logical_to_fock_map().items()) == expected_items

    fock_basis = Combinadics("fock", 2, 4)
    assert enc.logical_to_fock_indices() == {
        logical_state: fock_basis.fock_to_index(fock_state)
        for logical_state, fock_state in expected_items
    }


def test_dual_rail_builtin_mapping_order():
    mapping = EncodingSpace.DUAL_RAIL.logical_to_fock_map(n_modes=4, n_photons=2)

    assert list(mapping.items()) == [
        ((0, 0), (1, 0, 1, 0)),
        ((0, 1), (1, 0, 0, 1)),
        ((1, 0), (0, 1, 1, 0)),
        ((1, 1), (0, 1, 0, 1)),
    ]


def test_dual_rail_builtin_matches_equivalent_partitioned_engine():
    dual_rail = EncodingSpace.DUAL_RAIL
    partitioned = EncodingSpace(modes_per_photon=[2, 2, 2])

    assert dual_rail.logical_basis_states(n_modes=6, n_photons=3) == (
        partitioned.logical_basis_states()
    )
    assert dual_rail.fock_basis_states(n_modes=6, n_photons=3) == (
        partitioned.fock_basis_states()
    )
    assert dual_rail.logical_to_fock_map(n_modes=6, n_photons=3) == (
        partitioned.logical_to_fock_map()
    )
    assert dual_rail.logical_to_fock_indices(n_modes=6, n_photons=3) == (
        partitioned.logical_to_fock_indices()
    )
