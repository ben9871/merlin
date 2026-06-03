"""Compare PhotonicGenerator against the reproduced photonic QGAN digit run.

This script is validation support for PML-282. It runs the reproduced
``PatchGenerator`` and the MerLin ``PhotonicGenerator`` path on the same
Optdigits digit-0 training batches, with the same Adam optimizer settings,
same latent noise tensors, and same discriminator architecture.

The output is intentionally written beside this script so the comparison branch
can keep its runner and committed result artifacts together.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import perceval as pcvl
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

VALIDATION_ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
REPRO_ROOT = ROOT / "external" / "reproduced_papers"
PHOTONIC_QGAN = REPRO_ROOT / "papers" / "photonic_QGAN"
DATA_PATH = REPRO_ROOT / "data" / "photonic_QGAN" / "optdigits_csv.csv"

if not PHOTONIC_QGAN.exists():
    raise SystemExit(
        "Missing external/reproduced_papers checkout. Clone it under external/ "
        "before running this validation script."
    )

sys.path.insert(0, str(PHOTONIC_QGAN))
sys.path.insert(0, str(REPRO_ROOT))

from lib.discriminator import Discriminator  # noqa: E402
from lib.generators import PatchGenerator  # noqa: E402
from papers.shared.photonic_QGAN.digits import DigitsDataset  # noqa: E402
from utils.pqc import ParametrizedQuantumCircuit  # noqa: E402

import merlin as ML  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    """Configuration for one deterministic photonic QGAN digit run."""

    seed: int
    digit: int
    iterations: int
    batch_size: int
    image_size: int
    setup: str
    input_state: tuple[int, ...]
    gen_count: int
    pnr: bool
    lossy: bool
    noise_dim: int
    arch: tuple[str, ...]
    lr_d: float
    lr_g: float
    adam_beta1: float
    adam_beta2: float
    real_label: float
    fake_label: float
    gen_target: float
    d_steps: int
    g_steps: int


@dataclass(frozen=True)
class RunSummary:
    """Metrics captured from one backend run."""

    backend: str
    seconds: float
    final_d_loss: float
    final_g_loss: float
    final_similarity: float
    final_diversity: float
    final_ssim: float
    first_ssim: float
    best_iteration: int
    best_similarity: float
    best_diversity: float
    best_ssim: float


class MerlinPatchGenerator(nn.Module):
    """PhotonicGenerator wrapper matching the reproduced PatchGenerator API."""

    def __init__(
        self,
        *,
        image_size: int,
        gen_count: int,
        gen_arch: tuple[str, ...],
        input_state: tuple[int, ...],
        pnr: bool,
        lossy: bool,
        use_count_shortcut: bool,
    ) -> None:
        super().__init__()
        if pnr:
            raise ValueError("This comparison currently targets pnr=False runs.")
        if lossy:
            raise ValueError("This comparison currently targets lossy=False runs.")

        if use_count_shortcut:
            layer = _make_merlin_layer(
                gen_arch=gen_arch,
                input_state=input_state,
                pnr=pnr,
            )
            layers: ML.QuantumLayer | list[ML.QuantumLayer] = layer
            count = gen_count
        else:
            layers = [
                _make_merlin_layer(
                    gen_arch=gen_arch,
                    input_state=input_state,
                    pnr=pnr,
                )
                for _ in range(gen_count)
            ]
            count = None

        self.generator = ML.PhotonicGenerator(
            layers=layers,
            count=count,
            output_adapter=ML.ImageAdapter(
                shape=(1, image_size, image_size),
                headwise=True,
                normalize_patches=True,
            ),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return flattened generated images."""
        return self.generator(z).reshape(z.shape[0], -1)


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable local comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_merlin_layer(
    *,
    gen_arch: tuple[str, ...],
    input_state: tuple[int, ...],
    pnr: bool,
) -> ML.QuantumLayer:
    """Build one MerLin generator head using the reproduced circuit helper."""
    pcvl_circuit = ParametrizedQuantumCircuit(len(input_state), list(gen_arch))
    return ML.QuantumLayer(
        input_size=len(pcvl_circuit.enc_param_names),
        circuit=pcvl_circuit.circuit,
        input_parameters=pcvl_circuit.enc_param_names,
        trainable_parameters=pcvl_circuit.var_param_names,
        input_state=pcvl.BasicState(input_state),
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK,
            occupancy_readout=not pnr,
        ),
    )


def _make_reference_generator(cfg: RunConfig) -> nn.Module:
    """Build the reproduced PatchGenerator backend."""
    return PatchGenerator(
        cfg.image_size,
        cfg.gen_count,
        list(cfg.arch),
        pcvl.BasicState(cfg.input_state),
        cfg.pnr,
        cfg.lossy,
    )


def _make_merlin_list_generator(cfg: RunConfig) -> nn.Module:
    """Build PhotonicGenerator with one explicitly instantiated head per patch."""
    return MerlinPatchGenerator(
        image_size=cfg.image_size,
        gen_count=cfg.gen_count,
        gen_arch=cfg.arch,
        input_state=cfg.input_state,
        pnr=cfg.pnr,
        lossy=cfg.lossy,
        use_count_shortcut=False,
    )


def _make_merlin_count_generator(cfg: RunConfig) -> nn.Module:
    """Build PhotonicGenerator through the count shortcut."""
    return MerlinPatchGenerator(
        image_size=cfg.image_size,
        gen_count=cfg.gen_count,
        gen_arch=cfg.arch,
        input_state=cfg.input_state,
        pnr=cfg.pnr,
        lossy=cfg.lossy,
        use_count_shortcut=True,
    )


def _digit_batches(cfg: RunConfig) -> list[torch.Tensor]:
    """Return the exact real-data batches consumed by every backend."""
    dataset = DigitsDataset(
        csv_file=str(DATA_PATH),
        label=cfg.digit,
        transform=None,
    )
    sampler_generator = torch.Generator().manual_seed(cfg.seed + 17)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=cfg.batch_size * cfg.iterations,
        generator=sampler_generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        sampler=sampler,
    )
    return [
        torch.as_tensor(data).reshape(data.size(0), -1).float() for data, _ in loader
    ]


def _noise_schedule(cfg: RunConfig) -> dict[str, object]:
    """Return deterministic latent noise tensors shared by every backend."""
    generator = torch.Generator().manual_seed(cfg.seed + 31)
    fixed_noise = torch.normal(
        0.0,
        2 * math.pi,
        (cfg.batch_size, cfg.noise_dim),
        generator=generator,
    )
    d_noise = [
        [
            torch.normal(
                0.0,
                2 * math.pi,
                (cfg.batch_size, cfg.noise_dim),
                generator=generator,
            )
            for _ in range(cfg.d_steps)
        ]
        for _ in range(cfg.iterations)
    ]
    g_noise = [
        [
            torch.normal(
                0.0,
                2 * math.pi,
                (cfg.batch_size, cfg.noise_dim),
                generator=generator,
            )
            for _ in range(cfg.g_steps)
        ]
        for _ in range(cfg.iterations)
    ]
    return {
        "fixed": fixed_noise,
        "d": d_noise,
        "g": g_noise,
    }


def _train_backend(
    *,
    backend: str,
    cfg: RunConfig,
    make_generator: Callable[[RunConfig], nn.Module],
    batches: list[torch.Tensor],
    noises: dict[str, object],
    output_dir: Path,
) -> RunSummary:
    """Train one backend and save comparable metrics/artifacts."""
    _seed_everything(cfg.seed)
    generator = make_generator(cfg)
    discriminator = Discriminator(cfg.image_size)

    criterion = nn.BCEWithLogitsLoss()
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg.lr_d,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )
    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=cfg.lr_g,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )

    fixed_noise = noises["fixed"]
    d_noises = noises["d"]
    g_noises = noises["g"]
    if not isinstance(fixed_noise, torch.Tensor):
        raise TypeError("fixed noise must be a tensor.")
    if not isinstance(d_noises, list) or not isinstance(g_noises, list):
        raise TypeError("noise schedules must be lists.")

    fake_progress: list[np.ndarray] = []
    losses: list[tuple[float, float]] = []
    metrics: list[tuple[int, float, float, float]] = []
    best_metric: tuple[int, float, float, float] | None = None
    best_fixed: np.ndarray | None = None

    with torch.no_grad():
        fake_progress.append(generator(fixed_noise).detach().cpu().numpy())

    start = time.perf_counter()
    real_labels = torch.full((cfg.batch_size,), cfg.real_label)
    fake_labels = torch.full((cfg.batch_size,), cfg.fake_label)
    gen_labels = torch.full((cfg.batch_size,), cfg.gen_target)

    for iteration, real_data in enumerate(batches):
        d_losses = []
        for d_step in range(cfg.d_steps):
            noise_d = d_noises[iteration][d_step]
            fake_data_d = generator(noise_d).detach()

            discriminator.zero_grad()
            out_real = discriminator(real_data).view(-1)
            out_fake = discriminator(fake_data_d).view(-1)
            d_loss = criterion(out_real, real_labels)
            d_loss = d_loss + criterion(out_fake, fake_labels)
            d_loss.backward()
            opt_d.step()
            d_losses.append(float(d_loss.detach()))

        g_losses = []
        fake_data = None
        for param in discriminator.parameters():
            param.requires_grad_(False)
        for g_step in range(cfg.g_steps):
            noise_g = g_noises[iteration][g_step]
            generator.zero_grad()
            fake_data = generator(noise_g)
            out_fake_for_g = discriminator(fake_data).view(-1)
            g_loss = criterion(out_fake_for_g, gen_labels)
            g_loss.backward()
            opt_g.step()
            g_losses.append(float(g_loss.detach()))
        for param in discriminator.parameters():
            param.requires_grad_(True)

        if fake_data is None:
            raise RuntimeError("Generator training did not produce fake data.")

        real_images = (
            real_data
            .detach()
            .cpu()
            .numpy()
            .reshape(
                cfg.batch_size,
                cfg.image_size,
                cfg.image_size,
            )
        )
        fake_images = (
            fake_data
            .detach()
            .cpu()
            .numpy()
            .reshape(
                cfg.batch_size,
                cfg.image_size,
                cfg.image_size,
            )
        )
        similarity, diversity = _get_metrics(
            np.clip(real_images, 0.0, 1.0),
            np.clip(fake_images, 0.0, 1.0),
        )
        ssim_value = similarity

        losses.append((float(np.mean(d_losses)), float(np.mean(g_losses))))
        metrics.append((iteration + 1, similarity, diversity, ssim_value))
        if best_metric is None or ssim_value > best_metric[3]:
            best_metric = (iteration + 1, similarity, diversity, ssim_value)
            with torch.no_grad():
                best_fixed = generator(fixed_noise).detach().cpu().numpy()

        if (iteration + 1) % 100 == 0 or iteration == len(batches) - 1:
            with torch.no_grad():
                fake_progress.append(generator(fixed_noise).detach().cpu().numpy())

    seconds = time.perf_counter() - start
    backend_dir = output_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)

    loss_array = np.asarray(losses, dtype=float)
    metric_array = np.asarray(metrics, dtype=float)
    final_fixed = fake_progress[-1]
    if best_metric is None or best_fixed is None:
        raise RuntimeError("Training did not capture any metric checkpoints.")

    np.savetxt(
        backend_dir / "loss_progress.csv",
        loss_array,
        delimiter=",",
        header="D_loss,G_loss",
    )
    np.savetxt(
        backend_dir / "ssim_progress.csv",
        metric_array,
        delimiter=",",
        header="iter,similarity,diversity,ssim",
    )
    np.savetxt(
        backend_dir / "fake_progress_last.csv",
        final_fixed.reshape(final_fixed.shape[0], -1),
        delimiter=",",
    )
    np.savetxt(
        backend_dir / "fake_progress_best.csv",
        best_fixed.reshape(best_fixed.shape[0], -1),
        delimiter=",",
    )
    _save_grid(
        final_fixed.reshape(cfg.batch_size, cfg.image_size, cfg.image_size),
        backend_dir / "fake_progress_last.png",
        f"{backend}: final fixed-noise samples",
    )
    _save_grid(
        best_fixed.reshape(cfg.batch_size, cfg.image_size, cfg.image_size),
        backend_dir / "fake_progress_best.png",
        f"{backend}: best fixed-noise samples at iteration {best_metric[0]}",
    )

    return RunSummary(
        backend=backend,
        seconds=seconds,
        final_d_loss=float(loss_array[-1, 0]),
        final_g_loss=float(loss_array[-1, 1]),
        final_similarity=float(metric_array[-1, 1]),
        final_diversity=float(metric_array[-1, 2]),
        final_ssim=float(metric_array[-1, 3]),
        first_ssim=float(metric_array[0, 3]),
        best_iteration=best_metric[0],
        best_similarity=best_metric[1],
        best_diversity=best_metric[2],
        best_ssim=best_metric[3],
    )


def _save_grid(images: np.ndarray, path: Path, title: str) -> None:
    """Save a compact grid of generated or real 8x8 images."""
    count = images.shape[0]
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes_array = np.asarray(axes).reshape(-1)
    for index, axis in enumerate(axes_array):
        axis.axis("off")
        if index < count:
            axis.imshow(images[index], cmap="gray", vmin=0.0, vmax=1.0)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _get_metrics(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    """Return reproduced-style similarity and diversity metrics."""
    count = len(real)
    if count < 2:
        return 0.0, 0.0

    similarity = 0.0
    diversity = 0.0
    for i in range(count):
        for j in range(count):
            similarity += _ssim(real[i], fake[j])
        for j in range(i + 1, count):
            diversity += _ssim(fake[i], fake[j])

    similarity /= count * count
    diversity /= count * (count - 1) / 2
    return similarity, 1 - diversity


def _ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Return SSIM, using scikit-image when available."""
    try:
        from skimage.metrics import structural_similarity
    except ModuleNotFoundError:
        return _global_ssim(x, y)
    return float(structural_similarity(x, y, data_range=1.0))


def _global_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Small dependency-free SSIM fallback for 8x8 validation plots."""
    c1 = 0.01**2
    c2 = 0.03**2
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    mu_x = float(x.mean())
    mu_y = float(y.mean())
    var_x = float(((x - mu_x) ** 2).mean())
    var_y = float(((y - mu_y) ** 2).mean())
    cov_xy = float(((x - mu_x) * (y - mu_y)).mean())
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return numerator / denominator


def _save_loss_plot(output_dir: Path, summaries: list[RunSummary]) -> None:
    """Save a side-by-side loss/SSIM plot for all completed backend runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for summary in summaries:
        backend_dir = output_dir / summary.backend
        losses = np.loadtxt(backend_dir / "loss_progress.csv", delimiter=",")
        metrics = np.loadtxt(backend_dir / "ssim_progress.csv", delimiter=",")
        if losses.ndim == 1:
            losses = losses.reshape(1, -1)
        if metrics.ndim == 1:
            metrics = metrics.reshape(1, -1)
        axes[0].plot(losses[:, 0], label=f"{summary.backend} D")
        axes[0].plot(losses[:, 1], label=f"{summary.backend} G", linestyle="--")
        axes[1].plot(metrics[:, 0], metrics[:, 3], label=summary.backend)

    axes[0].set_title("BCE losses")
    axes[0].set_xlabel("iteration")
    axes[0].legend(fontsize=8)
    axes[1].set_title("SSIM similarity")
    axes[1].set_xlabel("iteration")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def _initial_output_diffs(
    cfg: RunConfig,
    noises: dict[str, object],
) -> dict[str, float]:
    """Return initial fixed-noise differences against the reference backend."""
    fixed_noise = noises["fixed"]
    if not isinstance(fixed_noise, torch.Tensor):
        raise TypeError("fixed noise must be a tensor.")

    _seed_everything(cfg.seed)
    reference = _make_reference_generator(cfg)
    reference.eval()

    diffs: dict[str, float] = {}
    with torch.no_grad():
        reference_output = reference(fixed_noise)

    for name, factory in (
        ("merlin_list", _make_merlin_list_generator),
        ("merlin_count", _make_merlin_count_generator),
    ):
        _seed_everything(cfg.seed)
        candidate = factory(cfg)
        candidate.eval()
        with torch.no_grad():
            candidate_output = candidate(fixed_noise)
        diffs[name] = float((candidate_output - reference_output).abs().max())
    return diffs


def _default_config(args: argparse.Namespace) -> RunConfig:
    """Build the notebook-equivalent setup-c/01010 Adam configuration."""
    return RunConfig(
        seed=args.seed,
        digit=args.digit,
        iterations=args.iterations,
        batch_size=args.batch_size,
        image_size=8,
        setup="setup_c",
        input_state=(0, 1, 0, 1, 0),
        gen_count=4,
        pnr=False,
        lossy=False,
        noise_dim=1,
        arch=("var", "var", "enc[2]", "var", "var"),
        lr_d=0.0002,
        lr_g=0.004,
        adam_beta1=0.5,
        adam_beta2=0.99,
        real_label=0.9,
        fake_label=0.0,
        gen_target=0.9,
        d_steps=1,
        g_steps=3,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run deterministic photonic QGAN digit reproduction checks."
    )
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--digit", type=int, default=0)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["reference", "merlin-list", "merlin-count"],
        default=["reference", "merlin-count"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=VALIDATION_ROOT / "results",
    )
    return parser.parse_args()


def main() -> None:
    """Run all requested backend comparisons and write artifacts."""
    args = parse_args()
    cfg = _default_config(args)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    batches = _digit_batches(cfg)
    noises = _noise_schedule(cfg)
    real_preview = batches[0].reshape(cfg.batch_size, cfg.image_size, cfg.image_size)
    _save_grid(
        real_preview.numpy(), output_dir / "real_digit_batch.png", "real digit batch"
    )

    initial_diffs = _initial_output_diffs(cfg, noises)
    factories: dict[str, Callable[[RunConfig], nn.Module]] = {
        "reference": _make_reference_generator,
        "merlin-list": _make_merlin_list_generator,
        "merlin-count": _make_merlin_count_generator,
    }
    summaries = [
        _train_backend(
            backend=backend,
            cfg=cfg,
            make_generator=factories[backend],
            batches=batches,
            noises=noises,
            output_dir=output_dir,
        )
        for backend in args.backends
    ]
    _save_loss_plot(output_dir, summaries)

    payload = {
        "config": asdict(cfg),
        "data_path": DATA_PATH.relative_to(ROOT).as_posix(),
        "initial_max_abs_diff_vs_reference": initial_diffs,
        "summaries": [asdict(summary) for summary in summaries],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(f"wrote artifacts: {output_dir}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
