# QVT Backend Prototype for MerLin

## Scope

This document records the local MerLin changes used to support the
`quantum_vision_transformers` reproduction in the parent repository.

It is intentionally narrower than the larger QVT project report. The goal here
is to make the branch legible to the MerLin team:

- which source files changed
- what problem each change was addressing
- which user-facing feature became possible because of it
- what should probably be reworked more cleanly upstream

This branch currently modifies only:

- `merlin/algorithms/layer.py`
- `merlin/core/process.py`
- `merlin/measurement/strategies.py`
- `merlin/pcvl_pytorch/slos_torchscript.py`

The branch also keeps local `.pre_*` backup snapshots for comparison, but those
files are intentionally not part of the commit.

## Executive Summary

The prototype solved two practical problems for QVT-style photonic models.

First, sparse or reduced-basis `StateVector` inputs were previously routed
through a dense, incremental superposition path that was a poor fit for the
QVT workload. The prototype keeps the amplitude input sparse for longer, lifts
reduced bases only when needed, and dispatches vectorized amplitude input to
the EBS path instead of the older `compute_superposition_state(...)` route.

Second, QVT required structured post-selection on output sectors such as
"keep only states whose photon counts across mode partitions equal `(1, 1)`".
The prototype extends `MeasurementStrategy` so this can be expressed directly
and lowered into final-basis filtering through `output_map_func`.

The TorchScript kernel chunking helps, but the main performance/correctness win
came from fixing the high-level dispatch and representation path rather than
from kernel micro-optimization alone.

## Change Map

| Area | File(s) | What changed | What it enabled |
|---|---|---|---|
| Sparse amplitude handling | `layer.py`, `process.py` | `StateVector` input stays sparse/normalized longer and reduced-basis tensors are accepted | practical amplitude encoding for sparse QVT inputs |
| Superposition execution path | `layer.py`, `process.py` | vectorized amplitude inputs dispatch to EBS rather than incremental `compute_pa_inc(...)` | large runtime and memory improvement on QVT workloads |
| Partition-aware measurement | `strategies.py`, `layer.py` | `MeasurementStrategy` accepts mode partitions and allowed photon-count signatures | structured post-selection API |
| Final-basis filtering | `layer.py`, `process.py` | partition selection lowers to `output_map_func` and filtered output bases | pruned output basis without post-hoc Python masking |
| TorchScript chunking | `slos_torchscript.py` | large transition lists are processed in chunks | smaller dense temporaries and a moderate runtime win |

## 1. Sparse `StateVector` Input Handling

### Problem

QVT feeds `QuantumLayer` with deliberately sparse superpositions. The baseline
path called `to_dense()` too early, which removed the representation advantage
before the computation had even begun.

### Prototype change

In `QuantumLayer`, the amplitude path now uses the normalized tensor directly:

```python
sv_tensor = statevector_input.normalize().tensor
...
amplitude_input = self._validate_amplitude_input(amplitude_tensor)
```

The validator no longer assumes that the amplitude feature dimension must equal
only the current output basis size. It accepts either:

- the full logical basis size
- or a reduced mapped/output basis size

### Why it matters

This removes a dense-path assumption from the API. Sparse or filtered
superpositions can now flow into the computation process without being rejected
up front.

### What upstream should probably do

MerLin should treat sparse amplitude input as a first-class path rather than as
an edge case that is immediately coerced back into the dense representation.

## 2. Superposition Preparation and Reduced-Basis Expansion

### Problem

The baseline process logic assumed a single dense amplitude layout. For QVT,
two extra cases mattered:

- sparse batched superpositions
- reduced-basis tensors produced by partition filtering

### Prototype change

`ComputationProcess` now keeps a `logical_keys` basis and contains helpers for:

- sparse-aware batch unsqueezing
- sparse-aware normalization
- expanding reduced bases into the logical basis only when needed
- gathering only active superposition coefficients

Representative helper methods added in `process.py`:

```python
_expand_superposition_tensor(...)
_unsqueeze_superposition_tensor(...)
_normalize_superposition_tensor(...)
_active_superposition_indices(...)
_gather_superposition_coefficients(...)
```

### Why it matters

The simulator now pays attention to the actual active support of the input
state instead of building its work around the assumption that the whole basis
is populated.

### What upstream should probably do

These helpers are prototype-shaped, but the underlying direction is likely
correct: MerLin needs a clearer internal distinction between logical basis,
mapped/output basis, and sparse support of the current input state.

## 3. EBS Dispatch for Vectorized Amplitude Inputs

### Problem

For QVT, `StateVector -> QuantumLayer` was hitting
`compute_superposition_state(...)`, which relies on the incremental
`compute_pa_inc(...)` path. That route is semantically valid, but it is the
wrong performance profile for vectorized amplitude input.

### Prototype change

`QuantumLayer._compute_amplitudes(...)` now receives an explicit
`vectorized_amplitude_input` flag and uses that to select the EBS path:

```python
if self.amplitude_encoding or vectorized_amplitude_input:
    ...
```

### Why it matters

This was the single most important execution-path change for the QVT workload.
It moved the computation away from an incremental path that retained poor
memory/runtime behavior and onto the more appropriate batched route.

### What upstream should probably do

This should likely become the default behavior for `StateVector` amplitude
inputs rather than a QVT-specific branch condition.

## 4. Partition-Based `MeasurementStrategy`

### Problem

QVT uses structured post-selection such as:

- split modes into contiguous partitions
- keep only states whose per-partition photon counts match specific signatures

Before this prototype, that logic lived above MerLin as post-hoc masking.

### Prototype change

`MeasurementStrategy` now supports:

- `partition_blocks`
- `allowed_counts`

and exposes:

- `has_partition_selection()`
- `validate_partition_selection(...)`
- `build_output_map_func(...)`

Representative usage:

```python
MeasurementStrategy.probs(
    computation_space=ComputationSpace.FOCK,
    partition_blocks=[n_patches, d],
    allowed_counts=[(1, 1)],
)
```

### Why it matters

This gives MerLin a user-facing way to express structured output selection,
rather than forcing each application to reimplement sector filtering outside
the backend.

### What upstream should probably do

The API shape is reasonable, but it may deserve a more explicit naming scheme
or a dedicated output-selection object if the feature expands beyond contiguous
mode partitions.

## 5. Lowering Partition Selection into the Backend

### Problem

A measurement API alone is not enough if the backend still computes the full
final output and discards unwanted states afterward in Python.

### Prototype change

`QuantumLayer` now converts partition selection into `output_map_func`, which
is passed into `ComputationProcessFactory.create(...)` and then into SLOS graph
construction.

This means disallowed final output states are removed from the final output
basis itself.

### Why it matters

This is still only final-basis pruning, not full throughout-the-graph
reachability pruning. But it is materially better than post-hoc masking and it
makes the feature part of MerLin’s computation model rather than an external
wrapper concern.

### What upstream should probably do

Upstream MerLin should probably distinguish clearly between:

- output filtering/post-selection that preserves physical evolution
- computational subspace restrictions such as `UNBUNCHED`

Those are not the same operation and should not be conflated.

## 6. TorchScript Chunking

### Problem

The SLOS kernels previously processed all operations in one dense block, which
created unnecessarily large temporaries for layers with many transitions.

### Prototype change

`slos_torchscript.py` now computes a chunk size with `_resolve_op_chunk_size`
and loops through operation slices in:

- `layer_compute_vectorized(...)`
- `layer_compute_batch(...)`

Representative pattern:

```python
for start in range(0, sources.shape[0], chunk_size):
    ...
    result.scatter_add_(...)
```

### Why it matters

This reduces the size of per-layer dense temporaries and improved runtime in
the QVT benchmarks. It did not by itself solve the major memory bottleneck, but
it is a sensible kernel-level improvement.

### What upstream should probably do

This change is relatively self-contained and should be straightforward to
review independently from the higher-level dispatch work.

## Practical Outcome for QVT

The prototype made three QVT-facing capabilities practical:

1. sparse amplitude-encoded `StateVector` inputs are usable
2. structured sector readout can be expressed through `MeasurementStrategy`
3. the simulator follows a much better execution path for amplitude input

In the QVT repo, the largest measured gains came from the dispatch and sparse
handling changes. The TorchScript chunking was helpful but secondary.

## Recommended Upstream Roadmap

If these ideas were cleaned up for upstream MerLin, the most defensible order
would be:

1. land chunked TorchScript kernels as a narrow performance patch
2. make `StateVector` amplitude inputs dispatch to EBS by default
3. formalize sparse/reduced-basis superposition handling
4. upstream partition-aware output filtering as a first-class feature

The prototype code in this branch should therefore be read as a working proof
of necessity and feasibility, not as the final upstream API design.
