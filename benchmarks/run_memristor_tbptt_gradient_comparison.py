"""Compare memristive TBPTT gradients across the two benchmark APIs.

This is a diagnostic companion to ``run_memristor_tbptt_api_comparison.py``.
It does not change the implementation under test. It runs the blackbox
``num_backprop_steps=k`` path and the manual ``num_backprop_steps=None`` path
with ``detach_memristive_state()`` at chunk boundaries, then reports whether
their forward outputs, input gradients, and parameter gradients match.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from benchmarks.memristor_benchmark_utils import (
    benchmark_environment,
    resolve_benchmark_device,
)
from benchmarks.run_memristor_tbptt_api_comparison import (
    WorkloadConfig,
    build_memristive_layer,
)

DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass
class TensorDifference:
    """Maximum absolute difference for a tensor collection."""

    name: str
    max_abs_diff: float
    max_reference_abs: float


@dataclass
class ChunkComparison:
    """Gradient comparison captured immediately after one backward call."""

    chunk_index: int
    end_timestep: int
    loss_abs_diff: float
    input_grad_difference: TensorDifference
    parameter_grad_difference: TensorDifference
    passed: bool


@dataclass
class GradientComparison:
    """Complete comparison result for one workload, dtype, and window length."""

    k: int
    chunk_len: int
    dtype: str
    config: dict[str, int | str]
    initial_parameter_difference: TensorDifference
    output_difference: TensorDifference
    final_input_grad_difference: TensorDifference
    final_parameter_grad_difference: TensorDifference
    final_memristive_state_difference: TensorDifference
    chunks: list[ChunkComparison]
    atol: float
    rtol: float
    passed: bool


@dataclass
class GradientTrace:
    """Raw tensors collected from one API run."""

    losses: list[float]
    outputs: list[torch.Tensor]
    input_gradients_by_chunk: list[list[torch.Tensor]]
    parameter_gradients_by_chunk: list[dict[str, torch.Tensor]]
    final_input_gradients: list[torch.Tensor]
    final_parameter_gradients: dict[str, torch.Tensor]
    final_memristive_state: list[torch.Tensor]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare memristor TBPTT gradients between blackbox and manual APIs."
    )
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--chunk-rule",
        choices=("k", "k-plus-one"),
        default="k",
        help="Manual detach chunk length. Default matches the blackbox window size.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="Sequence length. Use 0 to run 2*k + 1 timesteps for each k.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-modes", type=int, default=4)
    parser.add_argument("--n-photons", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=2)
    parser.add_argument("--memristor-count", type=int, default=2)
    parser.add_argument("--entangling-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=tuple(DTYPES),
        default=["float32"],
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Directory where comparison plots are written.",
    )
    parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Exit with status 1 if any comparison exceeds tolerance.",
    )
    return parser.parse_args()


def _grad_or_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached copy of a tensor gradient, using zeros when absent."""
    if tensor.grad is None:
        return torch.zeros_like(tensor).detach().cpu()
    return tensor.grad.detach().cpu().clone()


def _parameter_values(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return detached named parameter values."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in layer.named_parameters()
    }


def _parameter_gradients(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return detached named parameter gradients."""
    return {
        name: _grad_or_zero(parameter) for name, parameter in layer.named_parameters()
    }


def _make_base_inputs(
    config: WorkloadConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """Create deterministic input tensors shared by both API runs."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 1000)
    return [
        torch.randn(
            config.batch_size,
            config.input_size,
            generator=generator,
            dtype=dtype,
        ).to(device=device)
        for _ in range(config.timesteps)
    ]


def _max_tensor_difference(
    left: list[torch.Tensor],
    right: list[torch.Tensor],
    prefix: str,
) -> TensorDifference:
    """Return the largest absolute difference across paired tensors."""
    max_name = prefix
    max_diff = 0.0
    max_reference = 0.0
    for index, (left_tensor, right_tensor) in enumerate(zip(left, right, strict=True)):
        diff = (left_tensor - right_tensor).abs().max().item()
        reference = right_tensor.abs().max().item()
        if diff >= max_diff:
            max_name = f"{prefix}[{index}]"
            max_diff = float(diff)
            max_reference = float(reference)
    return TensorDifference(
        name=max_name,
        max_abs_diff=max_diff,
        max_reference_abs=max_reference,
    )


def _max_named_tensor_difference(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> TensorDifference:
    """Return the largest absolute difference across named tensors."""
    if left.keys() != right.keys():
        missing_left = sorted(right.keys() - left.keys())
        missing_right = sorted(left.keys() - right.keys())
        raise RuntimeError(
            "Named tensor sets differ: "
            f"missing from left={missing_left}, missing from right={missing_right}."
        )

    max_name = ""
    max_diff = 0.0
    max_reference = 0.0
    for name in sorted(left):
        diff = (left[name] - right[name]).abs().max().item()
        reference = right[name].abs().max().item()
        if diff >= max_diff:
            max_name = name
            max_diff = float(diff)
            max_reference = float(reference)
    return TensorDifference(
        name=max_name,
        max_abs_diff=max_diff,
        max_reference_abs=max_reference,
    )


def _within_tolerance(
    difference: TensorDifference,
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Return whether a tensor difference is within absolute/relative tolerance."""
    return difference.max_abs_diff <= atol + rtol * difference.max_reference_abs


def _run_api_trace(
    *,
    api: str,
    config: WorkloadConfig,
    dtype: torch.dtype,
    base_inputs: list[torch.Tensor],
) -> GradientTrace:
    """Run one API and collect gradients at every chunk boundary."""
    device = torch.device(config.device)
    num_backprop_steps = config.k if api == "blackbox" else None
    layer = build_memristive_layer(
        n_modes=config.n_modes,
        n_photons=config.n_photons,
        input_size=config.input_size,
        memristor_count=config.memristor_count,
        entangling_layers=config.entangling_layers,
        num_backprop_steps=num_backprop_steps,
        seed=config.seed,
        device=device,
        dtype=dtype,
    )
    layer.train()
    layer.reset(batch_size=config.batch_size)

    inputs = [
        input_batch.detach().clone().requires_grad_(True) for input_batch in base_inputs
    ]
    outputs: list[torch.Tensor] = []
    losses: list[float] = []
    loss_terms: list[torch.Tensor] = []
    input_gradients_by_chunk: list[list[torch.Tensor]] = []
    parameter_gradients_by_chunk: list[dict[str, torch.Tensor]] = []

    for timestep, input_batch in enumerate(inputs):
        output = layer(input_batch)
        outputs.append(output.detach().cpu().clone())
        weights = torch.linspace(
            0.1,
            1.0,
            output.shape[1],
            device=output.device,
            dtype=output.dtype,
        )
        loss_terms.append((output * weights).sum(dim=1).mean())

        is_chunk_end = (timestep + 1) % config.chunk_len == 0
        is_last_step = timestep + 1 == config.timesteps
        if is_chunk_end or is_last_step:
            layer.zero_grad(set_to_none=True)
            loss = torch.stack(loss_terms).sum()
            loss.backward()
            losses.append(float(loss.detach().cpu().item()))
            input_gradients_by_chunk.append([
                _grad_or_zero(tensor) for tensor in inputs
            ])
            parameter_gradients_by_chunk.append(_parameter_gradients(layer))
            layer.detach_memristive_state()
            loss_terms = []

    return GradientTrace(
        losses=losses,
        outputs=outputs,
        input_gradients_by_chunk=input_gradients_by_chunk,
        parameter_gradients_by_chunk=parameter_gradients_by_chunk,
        final_input_gradients=[_grad_or_zero(tensor) for tensor in inputs],
        final_parameter_gradients=_parameter_gradients(layer),
        final_memristive_state=[
            state.detach().cpu().clone() for state in layer.memristive_state
        ],
    )


def compare_gradients(
    *,
    config: WorkloadConfig,
    dtype_name: str,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> GradientComparison:
    """Compare blackbox and manual TBPTT gradients for one configuration."""
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    blackbox_layer = build_memristive_layer(
        n_modes=config.n_modes,
        n_photons=config.n_photons,
        input_size=config.input_size,
        memristor_count=config.memristor_count,
        entangling_layers=config.entangling_layers,
        num_backprop_steps=config.k,
        seed=config.seed,
        device=device,
        dtype=dtype,
    )
    torch.manual_seed(config.seed)
    manual_layer = build_memristive_layer(
        n_modes=config.n_modes,
        n_photons=config.n_photons,
        input_size=config.input_size,
        memristor_count=config.memristor_count,
        entangling_layers=config.entangling_layers,
        num_backprop_steps=None,
        seed=config.seed,
        device=device,
        dtype=dtype,
    )
    initial_difference = _max_named_tensor_difference(
        _parameter_values(blackbox_layer),
        _parameter_values(manual_layer),
    )

    base_inputs = _make_base_inputs(config, dtype, device)
    blackbox_trace = _run_api_trace(
        api="blackbox",
        config=config,
        dtype=dtype,
        base_inputs=base_inputs,
    )
    manual_trace = _run_api_trace(
        api="manual",
        config=config,
        dtype=dtype,
        base_inputs=base_inputs,
    )

    chunks = []
    for chunk_index, (blackbox_grads, manual_grads) in enumerate(
        zip(
            blackbox_trace.parameter_gradients_by_chunk,
            manual_trace.parameter_gradients_by_chunk,
            strict=True,
        )
    ):
        input_difference = _max_tensor_difference(
            blackbox_trace.input_gradients_by_chunk[chunk_index],
            manual_trace.input_gradients_by_chunk[chunk_index],
            prefix="input_grad",
        )
        parameter_difference = _max_named_tensor_difference(
            blackbox_grads,
            manual_grads,
        )
        loss_abs_diff = abs(
            blackbox_trace.losses[chunk_index] - manual_trace.losses[chunk_index]
        )
        end_timestep = min(
            (chunk_index + 1) * config.chunk_len,
            config.timesteps,
        )
        chunks.append(
            ChunkComparison(
                chunk_index=chunk_index,
                end_timestep=end_timestep,
                loss_abs_diff=loss_abs_diff,
                input_grad_difference=input_difference,
                parameter_grad_difference=parameter_difference,
                passed=(
                    loss_abs_diff <= atol
                    and _within_tolerance(input_difference, atol=atol, rtol=rtol)
                    and _within_tolerance(parameter_difference, atol=atol, rtol=rtol)
                ),
            )
        )

    output_difference = _max_tensor_difference(
        blackbox_trace.outputs,
        manual_trace.outputs,
        prefix="output",
    )
    final_input_difference = _max_tensor_difference(
        blackbox_trace.final_input_gradients,
        manual_trace.final_input_gradients,
        prefix="input_grad",
    )
    final_parameter_difference = _max_named_tensor_difference(
        blackbox_trace.final_parameter_gradients,
        manual_trace.final_parameter_gradients,
    )
    final_state_difference = _max_tensor_difference(
        blackbox_trace.final_memristive_state,
        manual_trace.final_memristive_state,
        prefix="memristive_state",
    )
    passed = (
        _within_tolerance(initial_difference, atol=atol, rtol=rtol)
        and _within_tolerance(output_difference, atol=atol, rtol=rtol)
        and _within_tolerance(final_input_difference, atol=atol, rtol=rtol)
        and _within_tolerance(final_parameter_difference, atol=atol, rtol=rtol)
        and _within_tolerance(final_state_difference, atol=atol, rtol=rtol)
        and all(chunk.passed for chunk in chunks)
    )

    return GradientComparison(
        k=config.k,
        chunk_len=config.chunk_len,
        dtype=dtype_name,
        config=asdict(config),
        initial_parameter_difference=initial_difference,
        output_difference=output_difference,
        final_input_grad_difference=final_input_difference,
        final_parameter_grad_difference=final_parameter_difference,
        final_memristive_state_difference=final_state_difference,
        chunks=chunks,
        atol=atol,
        rtol=rtol,
        passed=passed,
    )


def _print_result(result: GradientComparison) -> None:
    """Print a compact human-readable comparison summary."""
    status = "PASS" if result.passed else "DIFF"
    print(
        f"{status} dtype={result.dtype} k={result.k} "
        f"chunk_len={result.chunk_len} T={result.config['timesteps']} "
        f"output={result.output_difference.max_abs_diff:.3e} "
        f"max_input_grad={_max_chunk_input_difference(result):.3e} "
        f"max_param_grad={_max_chunk_parameter_difference(result):.3e}"
    )
    for chunk in result.chunks:
        chunk_status = "PASS" if chunk.passed else "DIFF"
        print(
            f"  {chunk_status} chunk={chunk.chunk_index} "
            f"end_t={chunk.end_timestep} "
            f"loss={chunk.loss_abs_diff:.3e} "
            f"input_grad={chunk.input_grad_difference.max_abs_diff:.3e} "
            f"param_grad={chunk.parameter_grad_difference.max_abs_diff:.3e} "
            f"param={chunk.parameter_grad_difference.name}"
        )


def _max_chunk_parameter_difference(result: GradientComparison) -> float:
    """Return the maximum parameter-gradient difference across chunks."""
    return max(
        (chunk.parameter_grad_difference.max_abs_diff for chunk in result.chunks),
        default=0.0,
    )


def _max_chunk_input_difference(result: GradientComparison) -> float:
    """Return the maximum input-gradient difference across chunks."""
    return max(
        (chunk.input_grad_difference.max_abs_diff for chunk in result.chunks),
        default=0.0,
    )


def _plot_difference_lines(
    *,
    plot_dir: Path,
    results: list[GradientComparison],
) -> None:
    """Plot gradient and output differences versus TBPTT window length."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional benchmark dependency
        raise RuntimeError(
            "matplotlib is required to write gradient comparison plots."
        ) from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    floor = 1e-16
    dtype_names = sorted({result.dtype for result in results})

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for dtype_name in dtype_names:
        dtype_results = sorted(
            (result for result in results if result.dtype == dtype_name),
            key=lambda item: item.k,
        )
        k_values = [result.k for result in dtype_results]
        parameter_diffs = [
            max(_max_chunk_parameter_difference(result), floor)
            for result in dtype_results
        ]
        input_diffs = [
            max(_max_chunk_input_difference(result), floor) for result in dtype_results
        ]
        output_diffs = [
            max(result.output_difference.max_abs_diff, floor)
            for result in dtype_results
        ]
        axis.plot(
            k_values,
            parameter_diffs,
            marker="o",
            label=f"{dtype_name} parameter gradients",
        )
        axis.plot(
            k_values,
            input_diffs,
            marker="s",
            linestyle="--",
            label=f"{dtype_name} input gradients",
        )
        axis.plot(
            k_values,
            output_diffs,
            marker="^",
            linestyle=":",
            label=f"{dtype_name} outputs",
        )

    axis.axhline(
        results[0].atol,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"absolute tolerance ({results[0].atol:g})",
    )
    axis.set_yscale("log")
    axis.set_xlabel("TBPTT window k")
    axis.set_ylabel("max absolute difference")
    axis.set_title("Blackbox k-step vs manual detach TBPTT")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "gradient_difference_vs_k.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for dtype_name in dtype_names:
        dtype_results = sorted(
            (result for result in results if result.dtype == dtype_name),
            key=lambda item: item.k,
        )
        axis.plot(
            [result.k for result in dtype_results],
            [_max_chunk_parameter_difference(result) for result in dtype_results],
            marker="o",
            label=dtype_name,
        )
    axis.axhline(
        results[0].atol,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"absolute tolerance ({results[0].atol:g})",
    )
    axis.set_xlabel("TBPTT window k")
    axis.set_ylabel("max parameter-gradient difference")
    axis.set_title("Parameter-gradient difference across chunks")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "parameter_gradient_difference_vs_k.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run gradient comparisons and optionally persist JSON results."""
    args = parse_args()
    device = resolve_benchmark_device(args.device)
    results: list[GradientComparison] = []

    for dtype_name in args.dtype:
        dtype = DTYPES[dtype_name]
        for k in args.k:
            if k < 1:
                raise ValueError("This comparison expects k >= 1.")
            chunk_len = k if args.chunk_rule == "k" else k + 1
            timesteps = args.timesteps if args.timesteps > 0 else 2 * k + 1
            config = WorkloadConfig(
                n_modes=args.n_modes,
                n_photons=args.n_photons,
                input_size=args.input_size,
                memristor_count=args.memristor_count,
                entangling_layers=args.entangling_layers,
                batch_size=args.batch_size,
                timesteps=timesteps,
                k=k,
                chunk_len=chunk_len,
                seed=args.seed,
                device=str(device),
            )
            result = compare_gradients(
                config=config,
                dtype_name=dtype_name,
                dtype=dtype,
                atol=args.atol,
                rtol=args.rtol,
            )
            results.append(result)
            _print_result(result)

    payload: dict[str, Any] = {
        "results": [asdict(result) for result in results],
        "environment": benchmark_environment(device),
        "notes": {
            "blackbox": "num_backprop_steps=k with QuantumLayer-managed finite window.",
            "manual": "num_backprop_steps=None with detach_memristive_state() at chunk boundaries.",
            "interpretation": (
                "Non-zero parameter-gradient differences mean the APIs are not "
                "strictly gradient-identical for that workload and tolerance."
            ),
        },
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.plot_dir is not None:
        _plot_difference_lines(plot_dir=args.plot_dir, results=results)

    if args.fail_on_diff and not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
