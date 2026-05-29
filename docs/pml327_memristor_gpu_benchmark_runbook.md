# PML-327 Memristor GPU Benchmark Runbook

This note records how to run the memristor TBPTT benchmarks on the isolated
GPU VM. The goal is to compare the blackbox `num_backprop_steps=k` API against
manual user-managed TBPTT on the same CUDA-backed `QuantumLayer` path.

## Remote Setup

Use the remote interpreter:

```text
/home/qdmin/merlincass/.venv/bin/python
```

The remote checkout at `/home/qdmin/merlincass` must contain the local branch
state, including uncommitted benchmark scripts and any experimental `layer.py`
changes. Since this branch is not meant to be pushed yet, deploy with PyCharm
SFTP or another direct file sync rather than relying on GitHub.

After deployment:

```bash
cd /home/qdmin/merlincass
/home/qdmin/merlincass/.venv/bin/python -m pip install --upgrade pip
/home/qdmin/merlincass/.venv/bin/python -m pip uninstall -y merlinquantum
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MERLINQUANTUM=0.0.0.dev0 \
  /home/qdmin/merlincass/.venv/bin/python -m pip install -e .
/home/qdmin/merlincass/.venv/bin/python -m pip install --upgrade perceval-quandela psutil matplotlib pandas
```

Verify CUDA:

```bash
/home/qdmin/merlincass/.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Smoke Run

Run this first to verify that deployment, editable installation, CUDA, and the
benchmark scripts all agree:

```bash
mkdir -p .benchmarks/pml327-gpu/smoke
/home/qdmin/merlincass/.venv/bin/python benchmarks/run_memristor_tbptt_api_comparison.py \
  --device cuda \
  --k 2 \
  --timesteps 8 \
  --batch-size 2 \
  --n-modes 4 \
  --n-photons 2 \
  --input-size 2 \
  --memristor-count 1 \
  --entangling-layers 2 \
  --repeats 1 \
  --warmups 1 \
  --json-output .benchmarks/pml327-gpu/smoke/api.json \
  --csv-output .benchmarks/pml327-gpu/smoke/api.csv \
  --plot-dir .benchmarks/pml327-gpu/smoke/plots
```

## Mechanism Sweep

This sweep stresses the raw graph behavior: process calls, wall time, Python
allocations, RSS, and CUDA memory.

```bash
mkdir -p .benchmarks/pml327-gpu/mechanism
/home/qdmin/merlincass/.venv/bin/python benchmarks/run_memristor_tbptt_api_comparison.py \
  --device cuda \
  --k 2 4 8 16 32 \
  --timesteps 32 64 128 \
  --batch-size 8 16 \
  --n-modes 8 10 12 \
  --n-photons 3 4 \
  --input-size 2 \
  --memristor-count 1 2 4 \
  --entangling-layers 2 3 \
  --repeats 3 \
  --warmups 1 \
  --json-output .benchmarks/pml327-gpu/mechanism/api.json \
  --csv-output .benchmarks/pml327-gpu/mechanism/api.csv \
  --plot-dir .benchmarks/pml327-gpu/mechanism/plots
```

If this is too large, reduce one axis at a time. The Fock-space size grows fast
with modes and photons, so failed or impractically slow combinations should be
recorded rather than hidden.

## Training Workflow

This tests a real optimizer loop with a quantum layer and classical readout:

```bash
mkdir -p .benchmarks/pml327-gpu/training
/home/qdmin/merlincass/.venv/bin/python benchmarks/run_memristor_tbptt_training_workflow.py \
  --device cuda \
  --k 4 8 16 32 \
  --timesteps 32 64 \
  --batch-size 8 16 \
  --n-batches 8 \
  --epochs 5 \
  --n-modes 8 10 \
  --n-photons 3 4 \
  --input-size 2 \
  --memristor-count 1 2 4 \
  --entangling-layers 2 3 \
  --repeats 3 \
  --warmups 1 \
  --json-output .benchmarks/pml327-gpu/training/training.json \
  --csv-output .benchmarks/pml327-gpu/training/training.csv \
  --plot-dir .benchmarks/pml327-gpu/training/plots
```

## Application Workflow

This tests next-step prediction on a synthetic delayed nonlinear time-series:

```bash
mkdir -p .benchmarks/pml327-gpu/timeseries
/home/qdmin/merlincass/.venv/bin/python benchmarks/run_memristor_timeseries_workflow.py \
  --device cuda \
  --apis reservoir manual blackbox \
  --k 4 8 16 32 \
  --timesteps 32 64 \
  --batch-size 8 16 \
  --n-batches 8 \
  --val-batches 4 \
  --epochs 5 \
  --n-modes 8 10 \
  --n-photons 3 4 \
  --input-size 2 \
  --memristor-count 1 2 4 \
  --entangling-layers 2 3 \
  --repeats 3 \
  --warmups 1 \
  --json-output .benchmarks/pml327-gpu/timeseries/timeseries.json \
  --csv-output .benchmarks/pml327-gpu/timeseries/timeseries.csv \
  --plot-dir .benchmarks/pml327-gpu/timeseries/plots
```

## Decision Criteria

The blackbox API is only attractive if it is close to manual TBPTT on practical
wall time and memory. The important columns are:

- `seconds_ratio_blackbox_over_manual`
- `process_call_ratio_blackbox_over_manual`
- `cuda_memory_peak_allocated_bytes_max`
- `rss_peak_delta_bytes_max`
- loss deltas in the training and time-series workflows

The process-call ratio is the structural signal. Wall time can be noisy on CUDA,
but if blackbox consistently performs extra SLOS evaluations and also costs more
time or memory on the application workflow, the simpler API should stay
`k=0` or full graph, with user-managed TBPTT through `detach_memristive_state()`.
