"""Strict first-step parity checks for the photonic QGAN comparison.

This diagnostic compares the reproduced ``PatchGenerator`` backend and the
MerLin ``PhotonicGenerator`` backend before long-run GAN trajectory drift can
hide the first source of difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

VALIDATION_ROOT = Path(__file__).resolve().parent
ROOT = VALIDATION_ROOT.parents[2]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

import run_comparison as comparison  # noqa: E402

DEFAULT_OUTPUT = VALIDATION_ROOT / "strict_parity" / "seed_0_summary.json"


@dataclass(frozen=True)
class StepMetric:
    """Parity metrics for one optimization step."""

    loss_abs_diff: float
    output_max_abs_diff: float
    grad_max_abs_diff: float
    grad_max_magnitude: float
    grad_median_magnitude: float
    params_after_max_abs_diff: float
    largest_param_reference_name: str
    largest_param_merlin_name: str
    largest_param_grad_reference: float
    largest_param_grad_merlin: float
    largest_param_before_reference: float
    largest_param_before_merlin: float
    largest_param_after_reference: float
    largest_param_after_merlin: float


@dataclass(frozen=True)
class StrictParitySummary:
    """Strict parity summary for one seed and one training batch."""

    seed: int
    optimizer: str
    generator_param_init_max_abs_diff: float
    discriminator_param_init_max_abs_diff: float
    initial_generator_output_max_abs_diff: float
    initial_discriminator_output_max_abs_diff: float
    discriminator_step: StepMetric
    generator_steps: list[StepMetric]


def _trainable_named_parameters(
    module: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return trainable named parameters in PyTorch traversal order."""
    return [
        (name, param)
        for name, param in module.named_parameters()
        if param.requires_grad
    ]


def _flat_params(module: torch.nn.Module) -> torch.Tensor:
    """Return trainable parameters flattened in PyTorch traversal order."""
    parts = [
        param.detach().reshape(-1) for _, param in _trainable_named_parameters(module)
    ]
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def _flat_grads(module: torch.nn.Module) -> torch.Tensor:
    """Return trainable gradients flattened in PyTorch traversal order."""
    parts = []
    for _, param in _trainable_named_parameters(module):
        if param.grad is None:
            parts.append(torch.full_like(param.detach().reshape(-1), float("nan")))
        else:
            parts.append(param.grad.detach().reshape(-1))
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def _max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return the maximum absolute difference between same-shaped tensors."""
    if first.shape != second.shape:
        raise ValueError(f"Tensor shape mismatch: {first.shape} vs {second.shape}.")
    if first.numel() == 0:
        return 0.0
    return float((first - second).detach().abs().max())


def _build_backend_pair(
    cfg: comparison.RunConfig,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Build reference and MerLin generator/discriminator pairs from same seed."""
    comparison._seed_everything(cfg.seed)
    reference_generator = comparison._make_reference_generator(cfg)
    reference_discriminator = comparison.Discriminator(cfg.image_size)

    comparison._seed_everything(cfg.seed)
    merlin_generator = comparison._make_merlin_count_generator(cfg)
    merlin_discriminator = comparison.Discriminator(cfg.image_size)

    return (
        reference_generator,
        reference_discriminator,
        merlin_generator,
        merlin_discriminator,
    )


def _make_optimizer(
    name: str,
    parameters: object,
    *,
    lr: float,
    cfg: comparison.RunConfig,
) -> torch.optim.Optimizer:
    """Create the selected optimizer with comparison hyperparameters."""
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}.")


def _largest_post_step_difference(
    reference_before: torch.Tensor,
    merlin_before: torch.Tensor,
    reference_after: torch.Tensor,
    merlin_after: torch.Tensor,
    reference_grads: torch.Tensor,
    merlin_grads: torch.Tensor,
    reference_names: list[str],
    merlin_names: list[str],
    reference_named_parameters: list[tuple[str, torch.nn.Parameter]],
    merlin_named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, float | str]:
    """Return metadata for the parameter with largest post-step difference."""
    del reference_names
    del merlin_names

    post_diff = (reference_after - merlin_after).abs()
    if post_diff.numel() == 0:
        return {
            "largest_param_reference_name": "",
            "largest_param_merlin_name": "",
            "largest_param_grad_reference": 0.0,
            "largest_param_grad_merlin": 0.0,
            "largest_param_before_reference": 0.0,
            "largest_param_before_merlin": 0.0,
            "largest_param_after_reference": 0.0,
            "largest_param_after_merlin": 0.0,
        }

    flat_index = int(post_diff.argmax())
    cursor = 0
    for (ref_name, ref_param), (mer_name, _mer_param) in zip(
        reference_named_parameters,
        merlin_named_parameters,
        strict=True,
    ):
        size = ref_param.numel()
        if cursor <= flat_index < cursor + size:
            return {
                "largest_param_reference_name": ref_name,
                "largest_param_merlin_name": mer_name,
                "largest_param_grad_reference": float(reference_grads[flat_index]),
                "largest_param_grad_merlin": float(merlin_grads[flat_index]),
                "largest_param_before_reference": float(reference_before[flat_index]),
                "largest_param_before_merlin": float(merlin_before[flat_index]),
                "largest_param_after_reference": float(reference_after[flat_index]),
                "largest_param_after_merlin": float(merlin_after[flat_index]),
            }
        cursor += size

    raise RuntimeError("Could not map flattened parameter index to a named parameter.")


def _step_metric(
    *,
    reference_loss: torch.Tensor,
    merlin_loss: torch.Tensor,
    reference_output: torch.Tensor,
    merlin_output: torch.Tensor,
    reference_module: torch.nn.Module,
    merlin_module: torch.nn.Module,
    reference_before: torch.Tensor,
    merlin_before: torch.Tensor,
) -> StepMetric:
    """Collect gradient and post-step parameter parity metrics."""
    reference_named = _trainable_named_parameters(reference_module)
    merlin_named = _trainable_named_parameters(merlin_module)
    reference_names = [name for name, _ in reference_named]
    merlin_names = [name for name, _ in merlin_named]
    reference_grads = _flat_grads(reference_module)
    merlin_grads = _flat_grads(merlin_module)
    reference_after = _flat_params(reference_module)
    merlin_after = _flat_params(merlin_module)

    grad_abs = torch.maximum(reference_grads.abs(), merlin_grads.abs())
    largest = _largest_post_step_difference(
        reference_before,
        merlin_before,
        reference_after,
        merlin_after,
        reference_grads,
        merlin_grads,
        reference_names,
        merlin_names,
        reference_named,
        merlin_named,
    )

    return StepMetric(
        loss_abs_diff=float((reference_loss.detach() - merlin_loss.detach()).abs()),
        output_max_abs_diff=_max_abs(reference_output, merlin_output),
        grad_max_abs_diff=_max_abs(reference_grads, merlin_grads),
        grad_max_magnitude=float(grad_abs.max()) if grad_abs.numel() else 0.0,
        grad_median_magnitude=float(grad_abs.median()) if grad_abs.numel() else 0.0,
        params_after_max_abs_diff=_max_abs(reference_after, merlin_after),
        largest_param_reference_name=str(largest["largest_param_reference_name"]),
        largest_param_merlin_name=str(largest["largest_param_merlin_name"]),
        largest_param_grad_reference=float(largest["largest_param_grad_reference"]),
        largest_param_grad_merlin=float(largest["largest_param_grad_merlin"]),
        largest_param_before_reference=float(largest["largest_param_before_reference"]),
        largest_param_before_merlin=float(largest["largest_param_before_merlin"]),
        largest_param_after_reference=float(largest["largest_param_after_reference"]),
        largest_param_after_merlin=float(largest["largest_param_after_merlin"]),
    )


def run_strict_parity(seed: int, optimizer_name: str) -> StrictParitySummary:
    """Run one strict parity diagnostic for a selected seed and optimizer."""
    cfg = comparison._default_config(
        Namespace(iterations=2, batch_size=4, seed=seed, digit=0)
    )
    batches = comparison._digit_batches(cfg)
    noises = comparison._noise_schedule(cfg)
    criterion = torch.nn.BCEWithLogitsLoss()
    (
        reference_generator,
        reference_discriminator,
        merlin_generator,
        merlin_discriminator,
    ) = _build_backend_pair(cfg)
    generator_param_init_max_abs_diff = _max_abs(
        _flat_params(reference_generator),
        _flat_params(merlin_generator),
    )
    discriminator_param_init_max_abs_diff = _max_abs(
        _flat_params(reference_discriminator),
        _flat_params(merlin_discriminator),
    )

    fixed_noise = noises["fixed"]
    if not isinstance(fixed_noise, torch.Tensor):
        raise TypeError("fixed noise must be a tensor.")
    d_noises = noises["d"]
    g_noises = noises["g"]
    if not isinstance(d_noises, list) or not isinstance(g_noises, list):
        raise TypeError("noise schedules must be lists.")

    with torch.no_grad():
        reference_initial_output = reference_generator(fixed_noise)
        merlin_initial_output = merlin_generator(fixed_noise)
        reference_initial_discriminator = reference_discriminator(batches[0]).view(-1)
        merlin_initial_discriminator = merlin_discriminator(batches[0]).view(-1)

    reference_d_optimizer = _make_optimizer(
        optimizer_name,
        reference_discriminator.parameters(),
        lr=cfg.lr_d,
        cfg=cfg,
    )
    merlin_d_optimizer = _make_optimizer(
        optimizer_name,
        merlin_discriminator.parameters(),
        lr=cfg.lr_d,
        cfg=cfg,
    )
    reference_g_optimizer = _make_optimizer(
        optimizer_name,
        reference_generator.parameters(),
        lr=cfg.lr_g,
        cfg=cfg,
    )
    merlin_g_optimizer = _make_optimizer(
        optimizer_name,
        merlin_generator.parameters(),
        lr=cfg.lr_g,
        cfg=cfg,
    )

    real_batch = batches[0]
    real_labels = torch.full((cfg.batch_size,), cfg.real_label)
    fake_labels = torch.full((cfg.batch_size,), cfg.fake_label)
    gen_labels = torch.full((cfg.batch_size,), cfg.gen_target)

    reference_fake_d = reference_generator(d_noises[0][0]).detach()
    merlin_fake_d = merlin_generator(d_noises[0][0]).detach()
    reference_discriminator.zero_grad()
    merlin_discriminator.zero_grad()
    reference_out_real = reference_discriminator(real_batch).view(-1)
    reference_out_fake = reference_discriminator(reference_fake_d).view(-1)
    merlin_out_real = merlin_discriminator(real_batch).view(-1)
    merlin_out_fake = merlin_discriminator(merlin_fake_d).view(-1)
    reference_d_loss = criterion(reference_out_real, real_labels) + criterion(
        reference_out_fake, fake_labels
    )
    merlin_d_loss = criterion(merlin_out_real, real_labels) + criterion(
        merlin_out_fake, fake_labels
    )
    reference_d_before = _flat_params(reference_discriminator)
    merlin_d_before = _flat_params(merlin_discriminator)
    reference_d_loss.backward()
    merlin_d_loss.backward()
    reference_d_optimizer.step()
    merlin_d_optimizer.step()
    discriminator_step = _step_metric(
        reference_loss=reference_d_loss,
        merlin_loss=merlin_d_loss,
        reference_output=reference_out_fake,
        merlin_output=merlin_out_fake,
        reference_module=reference_discriminator,
        merlin_module=merlin_discriminator,
        reference_before=reference_d_before,
        merlin_before=merlin_d_before,
    )

    for parameter in reference_discriminator.parameters():
        parameter.requires_grad_(False)
    for parameter in merlin_discriminator.parameters():
        parameter.requires_grad_(False)

    generator_steps = []
    for step_index in range(cfg.g_steps):
        reference_generator.zero_grad()
        merlin_generator.zero_grad()
        reference_fake_g = reference_generator(g_noises[0][step_index])
        merlin_fake_g = merlin_generator(g_noises[0][step_index])
        reference_out_g = reference_discriminator(reference_fake_g).view(-1)
        merlin_out_g = merlin_discriminator(merlin_fake_g).view(-1)
        reference_g_loss = criterion(reference_out_g, gen_labels)
        merlin_g_loss = criterion(merlin_out_g, gen_labels)
        reference_g_before = _flat_params(reference_generator)
        merlin_g_before = _flat_params(merlin_generator)
        reference_g_loss.backward()
        merlin_g_loss.backward()
        reference_g_optimizer.step()
        merlin_g_optimizer.step()
        generator_steps.append(
            _step_metric(
                reference_loss=reference_g_loss,
                merlin_loss=merlin_g_loss,
                reference_output=reference_fake_g,
                merlin_output=merlin_fake_g,
                reference_module=reference_generator,
                merlin_module=merlin_generator,
                reference_before=reference_g_before,
                merlin_before=merlin_g_before,
            )
        )

    return StrictParitySummary(
        seed=seed,
        optimizer=optimizer_name,
        generator_param_init_max_abs_diff=generator_param_init_max_abs_diff,
        discriminator_param_init_max_abs_diff=discriminator_param_init_max_abs_diff,
        initial_generator_output_max_abs_diff=_max_abs(
            reference_initial_output, merlin_initial_output
        ),
        initial_discriminator_output_max_abs_diff=_max_abs(
            reference_initial_discriminator, merlin_initial_discriminator
        ),
        discriminator_step=discriminator_step,
        generator_steps=generator_steps,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run strict first-step parity checks for PML-282."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["adam", "sgd"],
        default=["adam", "sgd"],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run strict parity checks and write a JSON summary."""
    args = parse_args()
    summaries = [
        asdict(run_strict_parity(args.seed, optimizer)) for optimizer in args.optimizers
    ]
    payload = {"seed": args.seed, "summaries": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
