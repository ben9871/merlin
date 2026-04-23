import pytest
import torch

from merlin.core.state_vector import StateVector
from merlin.utils.combinadics import Combinadics


def _normalized_dense(tensor: torch.Tensor) -> torch.Tensor:
    dense = tensor.to_dense() if tensor.is_sparse else tensor
    norms = torch.linalg.vector_norm(dense, dim=-1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return dense / norms


@pytest.mark.parametrize("batched", [False, True])
def test_from_tensor_default_contract_dense_preserves_fock_basis_and_raw_tensor(
    batched: bool,
):
    n_modes, n_photons = 4, 2
    basis = Combinadics("fock", n_photons, n_modes)
    basis_size = basis.compute_space_size()

    if batched:
        amps = torch.arange(1, 2 * basis_size + 1, dtype=torch.float32).reshape(
            2, basis_size
        )
    else:
        amps = torch.arange(1, basis_size + 1, dtype=torch.float32)
    amps = amps.to(torch.complex64)

    sv = StateVector.from_tensor(amps, n_modes=n_modes, n_photons=n_photons)

    assert sv.n_modes == n_modes
    assert sv.n_photons == n_photons
    assert sv.basis_size == basis_size
    assert list(sv.basis) == list(basis)
    assert not sv.tensor.is_sparse
    assert not sv.is_normalized
    assert torch.allclose(sv.tensor, amps)
    assert torch.allclose(sv.to_dense(), _normalized_dense(amps))
    assert sv.is_normalized


@pytest.mark.parametrize("batched", [False, True])
def test_from_tensor_default_contract_sparse_preserves_layout_until_dense_view(
    batched: bool,
):
    n_modes, n_photons = 4, 2
    basis_size = Combinadics("fock", n_photons, n_modes).compute_space_size()

    if batched:
        indices = torch.tensor([[0, 0, 1], [0, basis_size - 1, 1]])
        values = torch.tensor([1 + 0j, 2 + 0j, 3 + 0j], dtype=torch.complex64)
        amps = torch.sparse_coo_tensor(indices, values, (2, basis_size))
    else:
        indices = torch.tensor([[0, basis_size - 1]])
        values = torch.tensor([1 + 0j, 2 + 0j], dtype=torch.complex64)
        amps = torch.sparse_coo_tensor(indices, values, (basis_size,))

    sv = StateVector.from_tensor(amps, n_modes=n_modes, n_photons=n_photons)

    original = amps.coalesce()
    stored = sv.tensor.coalesce()

    assert sv.n_modes == n_modes
    assert sv.n_photons == n_photons
    assert sv.tensor.is_sparse
    assert not sv.is_normalized
    assert torch.equal(stored.indices(), original.indices())
    assert torch.allclose(stored.values(), original.values())
    assert torch.allclose(sv.to_dense(), _normalized_dense(amps))
    assert sv.is_normalized


def test_from_tensor_default_contract_rejects_basis_size_mismatch():
    tensor = torch.ones(9, dtype=torch.complex64)

    with pytest.raises(ValueError, match="basis size"):
        StateVector.from_tensor(tensor, n_modes=4, n_photons=2)


def test_from_tensor_default_contract_explicit_dtype_and_device_are_applied():
    tensor = torch.arange(1, 7, dtype=torch.float32)

    sv = StateVector.from_tensor(
        tensor,
        n_modes=3,
        n_photons=2,
        dtype=torch.complex128,
        device=torch.device("cpu"),
    )

    assert sv.tensor.device.type == "cpu"
    assert sv.tensor.dtype == torch.complex128
    assert torch.allclose(sv.tensor, tensor.to(torch.complex128))


def test_from_tensor_default_contract_real_inputs_default_to_complex64():
    tensor = torch.arange(1, 7, dtype=torch.float64)

    sv = StateVector.from_tensor(tensor, n_modes=3, n_photons=2)

    assert sv.tensor.dtype == torch.complex64
    assert torch.allclose(sv.tensor, tensor.to(torch.complex64))
