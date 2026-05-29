# PML-327 Expanded GPU Benchmark Summary

Remote VM:

- Host: `51.158.126.149`
- Project: `/home/qdmin/merlincass`
- Python: `/home/qdmin/merlincass/.venv/bin/python`
- GPU: NVIDIA L4, 23034 MiB
- Torch: `2.10.0+cu128`
- Merlin install: editable from `/home/qdmin/merlincass`, version forced to
  `0.0.0.dev0` because `.git` was intentionally not deployed.
- Perceval: `perceval-quandela==1.2.1`

## Artifact Locations

All benchmark data is on the VM under:

```text
/home/qdmin/merlincass/.benchmarks/pml327-gpu
```

The retrieved local copy committed with this branch is under:

```text
.benchmarks/pml327-gpu
```

Expanded JSON outputs:

```text
.benchmarks/pml327-gpu/expanded-mechanism/k_scaling_m8_p3_mem2.json
.benchmarks/pml327-gpu/expanded-mechanism/dimension_scaling_k8_mem2.json
.benchmarks/pml327-gpu/expanded-mechanism/memristor_scaling_m12_p4.json
.benchmarks/pml327-gpu/expanded-training/training_m8_p3_mem2.json
.benchmarks/pml327-gpu/expanded-training/training_m12_p4_mem4_stress.json
.benchmarks/pml327-gpu/expanded-timeseries/timeseries_m8_p3_mem2.json
.benchmarks/pml327-gpu/expanded-timeseries/timeseries_m12_p4_mem4_stress.json
.benchmarks/pml327-gpu/expanded-large-batch/mechanism_m8_p3_mem2_B32_B64.json
.benchmarks/pml327-gpu/expanded-large-batch/training_m8_p3_mem2_B32_B64.json
```

CSV summaries and plot PNGs were also generated beside each JSON file.

## Expanded Scope

The benchmark now covers:

- `k` scaling up to 32.
- Timesteps up to 64 in the mechanism benchmarks.
- Batch sizes up to 16.
- Modes up to 12.
- Photons up to 4.
- Memristors up to 6.
- Batch sizes up to 64.
- Workflow-level training and time-series application runs.
- CUDA peak allocated/reserved memory, synchronized CUDA timing, and environment
  metadata in JSON.

## Main Results

Low-level `k` scaling, fixed `m=8`, `p=3`, `mem=2`, `T=64`, `B=16`:

| k | blackbox/manual time | blackbox/manual process calls |
|---:|---:|---:|
| 2 | 1.221 | 1.500 |
| 4 | 1.618 | 1.750 |
| 8 | 1.839 | 1.875 |
| 16 | 1.922 | 1.938 |
| 32 | 2.009 | 1.969 |

Dimension scaling at `k=8`, `T=64`, `B=8`, `mem=2`:

- 12 comparisons over `m in {6, 8, 10, 12}` and `p in {2, 3, 4}`.
- Time ratio range: `1.774` to `1.839`.
- Process-call ratio: consistently `1.875`.

Memristor scaling at `m=12`, `p=4`, `T=64`, `B=8`:

- `mem in {1, 2, 4, 6}`, `k in {8, 16}`.
- Time ratio range: `1.803` to `1.981`.
- Process-call ratio range: `1.875` to `1.938`.
- Increasing memristor count did not change the structural overhead; the extra
  quantum evaluations dominate.

Training workflow, `m=8`, `p=3`, `mem=2`, `T=32`, `B=8`, 4 batches, 3 epochs:

| k | blackbox/manual time | blackbox/manual process calls |
|---:|---:|---:|
| 4 | 1.629 | 1.750 |
| 8 | 1.830 | 1.875 |
| 16 | 1.937 | 1.938 |

Training stress point, `m=12`, `p=4`, `mem=4`, `T=16`, `B=8`, 2 batches, 2 epochs:

| k | blackbox/manual time | blackbox/manual process calls |
|---:|---:|---:|
| 8 | 1.842 | 1.875 |
| 16 | 1.946 | 1.938 |

Time-series workflow, `m=8`, `p=3`, `mem=2`, `T=32`, `B=8`, 4 train batches,
2 validation batches, 3 epochs:

| k | blackbox/manual time | blackbox/manual process calls |
|---:|---:|---:|
| 4 | 1.516 | 1.500 |
| 8 | 1.693 | 1.583 |
| 16 | 1.776 | 1.625 |

Time-series stress point, `m=12`, `p=4`, `mem=4`, `T=16`, `B=8`, 2 train
batches, 1 validation batch, 2 epochs:

| k | blackbox/manual time | blackbox/manual process calls |
|---:|---:|---:|
| 8 | 1.707 | 1.583 |
| 16 | 1.792 | 1.625 |

Large-batch mechanism run, `m=8`, `p=3`, `mem=2`, `T=64`:

| k | batch | blackbox/manual time | blackbox/manual process calls |
|---:|---:|---:|---:|
| 8 | 32 | 1.817 | 1.875 |
| 8 | 64 | 1.806 | 1.875 |
| 16 | 32 | 1.922 | 1.938 |
| 16 | 64 | 1.923 | 1.938 |
| 32 | 32 | 2.005 | 1.969 |
| 32 | 64 | 1.999 | 1.969 |

Large-batch training workflow, `m=8`, `p=3`, `mem=2`, `T=32`, 2 batches,
2 epochs:

| k | batch | blackbox/manual time | blackbox/manual process calls |
|---:|---:|---:|---:|
| 8 | 32 | 1.773 | 1.875 |
| 8 | 64 | 1.835 | 1.875 |
| 16 | 32 | 1.869 | 1.938 |
| 16 | 64 | 1.902 | 1.938 |

## Interpretation

The expanded GPU benchmarks strengthen the earlier CPU-side conclusion. The
blackbox `num_backprop_steps=k` path is consistently slower because it performs
extra quantum-layer evaluations. The wall-time ratio tracks the process-call
ratio closely in the mechanism and training benchmarks. In the time-series
workflow, validation loss was effectively unchanged between manual and blackbox,
so the blackbox path is paying extra runtime without improving the measured
learning behavior in these runs.

The manual API gives the user the same training result in these tests while
keeping one quantum evaluation per timestep. That supports the design direction
of exposing `k=0` and full graph behavior, plus an explicit
`detach_memristive_state()` utility for user-managed TBPTT, rather than hiding
intermediate TBPTT windows inside the layer.

The large-batch follow-up does not change that conclusion. Moving from batch 16
to 32 or 64 increases CUDA memory, but it does not amortize the blackbox
overhead away. The ratio remains governed by the extra process calls.
