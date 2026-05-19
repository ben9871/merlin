import pytest

from merlin.core import ComputationSpace, EncodingSpace


def test_builtin_computation_space_constants():
    assert list(ComputationSpace) == [
        ComputationSpace.FOCK,
        ComputationSpace.UNBUNCHED,
        ComputationSpace.DUAL_RAIL,
    ]
    assert ComputationSpace.FOCK.value == "fock"
    assert ComputationSpace.UNBUNCHED.value == "unbunched"
    assert ComputationSpace.DUAL_RAIL.value == "dual_rail"
    assert ComputationSpace.FOCK.name == "FOCK"
    assert ComputationSpace.UNBUNCHED.kind == "unbunched"
    assert ComputationSpace.DUAL_RAIL.family == "partitioned"
    assert isinstance(ComputationSpace.FOCK, str)


def test_computation_space_string_coercion_backwards_compatibility():
    assert ComputationSpace.coerce(ComputationSpace.FOCK) is ComputationSpace.FOCK
    assert ComputationSpace.coerce("fock") is ComputationSpace.FOCK
    assert ComputationSpace.coerce("UNBUNCHED") is ComputationSpace.UNBUNCHED
    assert ComputationSpace.coerce("dual_rail") is ComputationSpace.DUAL_RAIL
    assert ComputationSpace("fock") == ComputationSpace.FOCK
    assert ComputationSpace.FOCK == "fock"
    assert ComputationSpace.FOCK != "FOCK"


def test_partitioned_modes_per_photon_validation():
    space = ComputationSpace(modes_per_photon=[3, 4, 2])

    assert space.family == "partitioned"
    assert space.kind == "partitioned"
    assert space.modes_per_photon == (3, 4, 2)
    assert space.n_modes == 9
    assert space.n_photons == 3
    assert space.basis_size() == 24


def test_qloq_helper_expands_groups_to_modes_per_photon():
    qloq = ComputationSpace.qloq(qubit_groups=[2, 1])

    assert qloq.family == "partitioned"
    assert qloq.kind == "qloq"
    assert qloq.qubit_groups == (2, 1)
    assert qloq.modes_per_photon == (4, 2)
    assert qloq.basis_size() == 8


def test_computation_space_equality_and_hashability():
    left = ComputationSpace(modes_per_photon=[2, 3])
    right = ComputationSpace(modes_per_photon=(2, 3))
    qloq = ComputationSpace.qloq([1, 2])

    assert left == right
    assert len({left, right, qloq}) == 2
    assert qloq != "qloq"
    assert repr(qloq) == (
        "ComputationSpace(family='partitioned', kind='qloq', "
        "modes_per_photon=(2, 4), qubit_groups=(1, 2))"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ComputationSpace(modes_per_photon=[]),
        lambda: ComputationSpace(modes_per_photon=[2, 0]),
        lambda: ComputationSpace(modes_per_photon=[True]),
        lambda: ComputationSpace(modes_per_photon=[2], family="builtin"),
        lambda: ComputationSpace(modes_per_photon=[2], kind="fock"),
        lambda: ComputationSpace.qloq([]),
        lambda: ComputationSpace.qloq([1, -1]),
        lambda: ComputationSpace.coerce("invalid"),
    ],
)
def test_invalid_computation_space_inputs_raise_value_error(factory):
    with pytest.raises(ValueError):
        factory()


def test_computation_space_mapping_matches_encoding_space_backend():
    qloq = ComputationSpace.qloq([2, 1])
    encoding = EncodingSpace.qloq([2, 1])

    assert qloq.logical_basis_states() == encoding.logical_basis_states()
    assert qloq.fock_basis_states() == encoding.fock_basis_states()
    assert qloq.logical_to_fock_indices() == encoding.logical_to_fock_indices()


def test_encoding_space_and_computation_space_are_distinct_public_types():
    assert ComputationSpace.qloq([1]) != EncodingSpace.qloq([1])
    assert not isinstance(ComputationSpace.FOCK, EncodingSpace)
