# PML-282 Photonic QGAN Reproduction Comparison

This directory contains the validation runner and committed artifacts for a
direct Adam photonic-QGAN comparison against the reproduced implementation.

The comparison is not a public documentation demo. It is evidence for the
implementation branch: MerLin's `PhotonicGenerator` is run on the same Optdigits
digit task as the reproduced `PatchGenerator`, using the same batches, latent
noise tensors, optimizer settings, discriminator architecture, and patch-image
mapping intent.

## Runner

Run from the repository root after cloning `merlinquantum/reproduced_papers`
under `external/reproduced_papers`:

```powershell
.\.venv\Scripts\python.exe docs\validation\pml282_photonic_qgan_reproduction\run_comparison.py --iterations 1500 --batch-size 4 --seed 0 --digit 0 --backends reference merlin-count
```

The default arguments match that command. The optional `merlin-list` backend is
available to compare `PhotonicGenerator([head_0, ...])` with
`PhotonicGenerator(head_template, count=N)`.

## Committed Runs

Committed per-seed artifacts are under:

```text
docs/validation/pml282_photonic_qgan_reproduction/results/run_20260602_164222
docs/validation/pml282_photonic_qgan_reproduction/results/seed_1_1500
docs/validation/pml282_photonic_qgan_reproduction/results/seed_2_1500_complete
```

Configuration:

| Field | Value |
| --- | --- |
| task | Optdigits digit `0` |
| iterations | `1500` |
| batch size | `4` |
| input state | `(0, 1, 0, 1, 0)` |
| generator heads | `4` |
| latent dimension | `1` |
| discriminator optimizer | Adam, lr `0.0002` |
| generator optimizer | Adam, lr `0.004` |
| update ratio | `d_steps=1`, `g_steps=3` |
| latent scale | `N(0, 2*pi)` |

Summary:

Single-seed summary for the original seed-0 run:

| Backend | Initial max abs diff vs reference | Best iteration | Best SSIM | Final SSIM | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | n/a | `132` | `0.807675` | `0.509099` | `923.54s` |
| merlin-count | `5.96e-08` | `132` | `0.799804` | `0.144471` | `812.93s` |

## Multi-Seed Metrics

Aggregate artifacts are under:

```text
docs/validation/pml282_photonic_qgan_reproduction/aggregate
```

The aggregate covers seeds `0`, `1`, and `2`, all with the same task
configuration. The initial generator outputs match the reproduced
implementation to numerical precision in every seed:

| Seed | Initial max abs diff, MerLin vs reference | Reference best SSIM | MerLin best SSIM | Reference final SSIM | MerLin final SSIM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `5.96e-08` | `0.807675` | `0.799804` | `0.509099` | `0.144471` |
| `1` | `5.96e-08` | `0.791165` | `0.788424` | `0.745019` | `0.393628` |
| `2` | `5.96e-08` | `0.823585` | `0.801664` | `0.723030` | `0.430652` |

Mean metrics:

| Backend | Runs | Best SSIM mean | Best SSIM std | Final SSIM mean | Final SSIM std | Runtime mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | `3` | `0.807475` | `0.013236` | `0.659049` | `0.106410` | `849.60s` |
| merlin-count | `3` | `0.796631` | `0.005853` | `0.322917` | `0.127082` | `1075.77s` |

The important parity checks are:

- initial generator outputs match the reproduced implementation to numerical
  precision for each seed;
- best-checkpoint SSIM is close between the two implementations across seeds;
- final-checkpoint quality is materially noisier than best-checkpoint quality,
  so final SSIM alone is not a stable reproduction metric for this GAN run;
- the MerLin path is slower on average in this local CPU run.

Key artifacts:

- `aggregate/summary_by_seed.csv`: per-seed metrics.
- `aggregate/aggregate_summary.json`: aggregate means, standard deviations,
  and paired deltas.
- `aggregate/ssim_by_seed.png`: best/final SSIM by seed.
- `aggregate/training_curves_by_seed.png`: SSIM curves across seeds.
- `*/summary.json`: full per-run configuration and metric summary.
- `*/training_curves.png`: loss and SSIM curves for both backends.
- `*/reference/fake_progress_best.png`: best reference samples.
- `*/merlin-count/fake_progress_best.png`: best MerLin samples.

## Scope

This validates the Adam digit task used for the MerLin photonic-QGAN path. It
does not claim full paper reproduction across SPSA, all digits, noisy/lossy
settings, or the full paper-scale training budget.
