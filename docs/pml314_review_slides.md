# PML-314 Review Deck: ComputationSpace QLOQ And Partitioned Output Spaces

This file is the editable source companion for `docs/pml314_review_slides.pptx`.
It is review material and should not be committed unless the project explicitly
wants to keep the deck.

The deck explains the PML-314 implementation: `ComputationSpace` keeps its
legacy built-in API while adding partitioned and QLOQ output spaces. The public
meaning stays output-side; input embedding remains the responsibility of
`EncodingSpace`.

## Deck Outline

1. **PML-314: ComputationSpace QLOQ Output Selection**
   - Main message: additive API, same public meaning, richer output filters.

2. **Why This Ticket Exists**
   - `EncodingSpace` had partitioned/QLOQ options; `ComputationSpace` did not.
   - This blocked QLOQ-style output selection in `MeasurementStrategy`.

3. **Design Boundary**
   - `EncodingSpace`: input embedding contract.
   - `ComputationSpace`: output filtering and ordering contract.
   - Shared mapping logic, distinct public types.

4. **User-Facing API**
   - Built-ins remain valid.
   - New `ComputationSpace(modes_per_photon=[...])`.
   - New `ComputationSpace.qloq(qubit_groups=[...])`.

5. **Compatibility Contract**
   - `ComputationSpace.FOCK`, `.UNBUNCHED`, `.DUAL_RAIL` remain.
   - `.value`, `.name`, string coercion, `list(ComputationSpace)`, and
     `isinstance(space, str)` are preserved.

6. **Implementation Shape**
   - `ComputationSpace` is now a string-backed, enum-like value type.
   - Custom spaces carry metadata and do not collapse to plain `"qloq"`.

7. **QLOQ Mapping Example**
   - `ComputationSpace.qloq([2, 1])` expands to `(4, 2)`.
   - Eight retained output states in partition product order.

8. **Code2flow: Shared Basis Mapping**
   - Shows `ComputationSpace.fock_basis_states()` delegating through
     `_as_encoding_space()` to the existing mapping machinery.

9. **Measurement Strategy Boundary**
   - Factories now accept `ComputationSpace | str`.
   - Existing factory defaults are unchanged.

10. **Code2flow: QuantumLayer Output Selection**
    - Shows `ComputationProcess._setup_computation_graphs()` adding a custom
      output map before building the SLOS graph.

11. **How Output Selection Works**
    - Custom computation spaces build allowed Fock states.
    - SLOS still computes through the existing graph; the map keeps selected
      final states and renormalizes.

12. **Code2flow: ProbabilityDistribution Filtering**
    - Shows `ProbabilityDistribution.filter(...)` using
      `ComputationSpace.fock_basis_states(...)` instead of hardcoded branches.

13. **Conversion And Normalization Updates**
    - `pcvl_to_tensor(...)` and compact tensor embedding now use
      computation-space basis helpers.
    - Any non-FOCK computation space is treated as postselected for
      normalization.

14. **Remote Execution Boundary**
    - Current remote mapping code is based on built-in Combinadics scheme
      strings.
    - QLOQ is not inherently incompatible; remote support should use
      `space.fock_basis_states(...)` client-side.

15. **Files That Matter**
    - Core API, measurement strategy, process/SLOS, probability distribution,
      conversion helpers, docs, and tests.

16. **Tests And Validation**
    - Focused tests, SLOS/mappers/layer/kernel slices, Ruff, mypy, and strict
      Sphinx docs.

17. **What Did Not Change**
    - No `StateVector.from_tensor(...)` feature work.
    - No graph-level optimization for custom spaces.
    - Built-in semantics and defaults remain unchanged.

18. **Reviewer Checklist**
    - Check API compatibility.
    - Check QLOQ ordering.
    - Check whether remote support should be included now or follow later.

## Code2flow Commands Used

```bash
code2flow merlin/core/computation_space.py \
  --language py \
  --target-function ComputationSpace.fock_basis_states \
  --upstream-depth 2 \
  --downstream-depth 4 \
  --output docs/pml314_code2flow_basis.dot \
  --skip-parse-errors \
  --hide-legend \
  --quiet

code2flow merlin/core/process.py merlin/pcvl_pytorch/slos_torchscript.py \
  --language py \
  --target-function ComputationProcess._setup_computation_graphs \
  --upstream-depth 2 \
  --downstream-depth 4 \
  --output docs/pml314_code2flow_layer_output.dot \
  --skip-parse-errors \
  --hide-legend \
  --quiet

code2flow merlin/core/probability_distribution.py merlin/core/computation_space.py \
  --language py \
  --target-function ProbabilityDistribution.filter \
  --upstream-depth 1 \
  --downstream-depth 4 \
  --output docs/pml314_code2flow_probability_filter.dot \
  --skip-parse-errors \
  --hide-legend \
  --quiet

code2flow merlin/pcvl_pytorch/utils.py merlin/core/state_vector.py merlin/core/computation_space.py \
  --language py \
  --target-function pcvl_to_tensor \
  --upstream-depth 1 \
  --downstream-depth 4 \
  --output docs/pml314_code2flow_conversion.dot \
  --skip-parse-errors \
  --hide-legend \
  --quiet
```

Graphviz `dot` is not available in the local WSL environment, so the DOT files
were used as source material for native PowerPoint diagrams rather than rendered
as external images.
