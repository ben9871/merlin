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

## Committed Run

Committed artifacts are under:

```text
docs/validation/pml282_photonic_qgan_reproduction/results/run_20260602_164222
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

| Backend | Initial max abs diff vs reference | Best iteration | Best SSIM | Final SSIM | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | n/a | `132` | `0.807675` | `0.509099` | `923.54s` |
| merlin-count | `5.96e-08` | `132` | `0.799804` | `0.144471` | `812.93s` |

The important parity checks are:

- initial generator outputs match the reproduced implementation to numerical
  precision;
- the best checkpoint occurs at the same iteration for the reference and
  MerLin `count` path;
- best-checkpoint SSIM is close between the two implementations;
- final-checkpoint quality is worse for MerLin in this deterministic run, which
  is consistent with GAN instability and is why checkpoint selection is recorded.

Key artifacts:

- `summary.json`: full run configuration and metric summary.
- `training_curves.png`: loss and SSIM curves for both backends.
- `reference/fake_progress_best.png`: best reference samples.
- `merlin-count/fake_progress_best.png`: best MerLin samples.
- `reference/fake_progress_last.png`: final reference samples.
- `merlin-count/fake_progress_last.png`: final MerLin samples.

## Scope

This validates the Adam digit task used for the MerLin photonic-QGAN path. It
does not claim full paper reproduction across SPSA, all digits, noisy/lossy
settings, or multiple seeds.
