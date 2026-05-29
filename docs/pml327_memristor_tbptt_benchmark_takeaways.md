# PML-327 Memristor TBPTT Benchmark Takeaways

This branch is a benchmarking branch for the memristive backpropagation guard.
It is meant to compare two ways of getting truncated backpropagation through a
memristive `QuantumLayer`, not to define final production documentation.

## What Was Compared

Two APIs were benchmarked against the same memristive circuits and sequence
workloads:

- Black-box finite window: `num_backprop_steps=k`.
- Manual PyTorch-style TBPTT: `num_backprop_steps=None` with explicit
  `layer.detach_memristive_state()` calls at chunk boundaries.

The finite-`k` path corresponds to the black-box approach from `layer 1.py`. It
keeps the user API simple, but internally rebuilds recent memristive state and
therefore performs extra quantum-layer/SLOS evaluations.

The manual path keeps one quantum evaluation per timestep. The user controls the
chunking boundary in the training loop, as is normal for recurrent PyTorch
models.

## Main Result

The black-box finite-`k` path is consistently slower because it performs more
process calls. On the NVIDIA L4 GPU VM, the black-box/manual wall-time ratio was
usually around `1.5x` to `2.0x`, and the process-call ratio tracked the same
structural overhead.

Increasing batch size up to 64 did not remove the overhead. Larger batches
increase CUDA work and memory, but they do not change the fact that the
black-box path performs extra evaluations.

The manual and black-box paths matched numerically in the comparison runs where
matching was expected. The difference is therefore performance and API control,
not a different training objective.

## Practical Takeaway

For this branch, the cleanest performance story is:

- keep `num_backprop_steps=0` as the conservative reservoir-like default;
- allow full-history mode with `num_backprop_steps=None`;
- use `detach_memristive_state()` for explicit user-managed TBPTT;
- treat finite positive `k` as a convenient black-box option with a documented
  performance cost.

This keeps the efficient path aligned with standard PyTorch recurrent-training
patterns while still preserving the black-box implementation for comparison.

## Where To Look

Implementation and tests:

- `merlin/algorithms/layer.py`
- `merlin/builder/circuit_builder.py`
- `tests/algorithms/test_layer.py`

Benchmark scripts:

- `benchmarks/run_memristor_tbptt_api_comparison.py`
- `benchmarks/run_memristor_tbptt_training_workflow.py`
- `benchmarks/run_memristor_timeseries_workflow.py`
- `benchmarks/memristor_benchmark_utils.py`

Benchmark notes and reproduction commands:

- `docs/pml327_memristor_gpu_benchmark_runbook.md`
- `docs/pml327_memristor_gpu_expanded_benchmark_summary.md`
- `docs/pml327_memristor_tbptt_benchmark_results.md`

Raw results and plots:

- `.benchmarks/pml327-gpu`
- `.benchmarks/pml327_memristor_*`

## Hardware Context

The expanded GPU benchmarks were run on:

- GPU: NVIDIA L4, 23034 MiB
- Python: `/home/qdmin/merlincass/.venv/bin/python`
- Torch: `2.10.0+cu128`
- Perceval: `perceval-quandela==1.2.1`
- Merlin: editable install from `/home/qdmin/merlincass`
