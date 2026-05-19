import logging
import threading
import time
import uuid
import warnings
import zlib
from collections.abc import Iterable
from contextlib import suppress
from typing import Any, cast

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from perceval.algorithm import Sampler
from perceval.runtime import RemoteJob, RemoteProcessor
from perceval.runtime.session import ISession
from torch.futures import Future

from ..algorithms.module import MerlinModule
from ..utils.combinadics import Combinadics
from .computation_space import ComputationSpace

logger = logging.getLogger(__name__)


class MerlinProcessor:
    """RPC-style processor for quantum execution.

    The processor provides:

    - Torch-friendly asynchronous execution via ``Future[torch.Tensor]``.
    - Cloud offload of :class:`~merlin.algorithms.module.MerlinModule` leaves (e.g. QuantumLayer).
    - Batch chunking per quantum leaf with limited concurrency.
    - Cancellation support, both per future and globally.
    - Global timeouts that cancel in-flight jobs.
    - Fresh RemoteProcessor per chunk/attempt (no shared RPC handlers across threads).
    - Descriptive cloud job names (<= 50 chars) for traceability.

    Only modules that subclass MerlinModule and implement `export_config()` are
    considered for offload. All other modules are always run locally.

    Parameters
    ----------
    remote_processor : RemoteProcessor | None
        Perceval remote processor used in the legacy path.
    session : ISession | None
        Perceval session used to build remote processors. Exactly one of
        ``remote_processor`` or ``session`` must be provided.
    microbatch_size : int
        Maximum number of inputs submitted in a single remote chunk.
    timeout : float
        Default timeout in seconds for remote calls.
    max_shots_per_call : int | None
        Maximum number of shots to request per remote sampler call.
    chunk_concurrency : int
        Maximum number of concurrent chunk submissions per quantum layer.
    token : str | None
        Optional authentication token forwarded to cloned remote processors.
    """

    DEFAULT_MAX_SHOTS: int = 100_000
    _MAX_CHUNK_RETRIES: int = 3
    _MAX_ESTIMATOR_RETRIES: int = 3
    DEFAULT_SHOTS_PER_CALL: int = 10_000
    _JOB_NAME_MAX: int = 50

    def __init__(
        self,
        remote_processor: RemoteProcessor | None = None,
        session: ISession | None = None,
        microbatch_size: int = 32,
        timeout: float = 3600.0,
        max_shots_per_call: int | None = None,
        chunk_concurrency: int = 1,
        token: str | None = None,
    ):
        """Initialize the remote Merlin processor.

        Parameters
        ----------
        remote_processor : RemoteProcessor | None
            Perceval remote processor used in the legacy path.
        session : ISession | None
            Perceval session used to build remote processors.
        microbatch_size : int
            Maximum number of inputs submitted in a single remote chunk.
        timeout : float
            Default timeout in seconds for remote calls.
        max_shots_per_call : int | None
            Maximum number of shots to request per remote sampler call.
        chunk_concurrency : int
            Maximum number of concurrent chunk submissions per quantum layer.
        token : str | None
            Optional authentication token forwarded to cloned remote
            processors.

        Raises
        ------
        TypeError
            If neither or both of ``remote_processor`` and ``session`` are
            provided, or if their types are invalid.
        ValueError
            If the remote processor path is used and no token can be resolved.
        """
        # ── Validate: exactly one of the two must be provided ──
        if remote_processor is not None and session is not None:
            raise TypeError("Provide either 'remote_processor' or 'session', not both.")
        if remote_processor is None and session is None:
            raise TypeError("One of 'remote_processor' or 'session' must be provided.")

        self.session: ISession | None = None
        self.remote_processor: RemoteProcessor | None = None
        self._token: str | None = token

        if session is not None:
            # ── ISession path ──
            if not isinstance(session, ISession):
                raise TypeError(f"Expected ISession, got {type(session)}")
            self.session = session
            self.backend_name = getattr(session, "platform_name", "unknown")

            # Command detection is not supported on ISession-based platforms
            self.available_commands = []
        else:
            # ── Legacy RemoteProcessor path ──
            assert remote_processor is not None  # for type checker
            if not isinstance(remote_processor, RemoteProcessor):
                raise TypeError(
                    f"Expected RemoteProcessor, got {type(remote_processor)}"
                )
            self.remote_processor = remote_processor
            self.backend_name = getattr(remote_processor, "name", "unknown")

            # Auto-extract the token from the RP's handler when not
            # explicitly provided, so cloned RPs inherit it.
            if self._token is None:
                self._token = self._extract_rp_token(remote_processor)

            if self._token is None:
                raise ValueError(
                    "Could not extract auth token from RemoteProcessor. "
                    "Either pass token= to MerlinProcessor or call "
                    "RemoteConfig.set_token() before constructing the "
                    "RemoteProcessor."
                )

            if hasattr(remote_processor, "available_commands"):
                self.available_commands = remote_processor.available_commands
                if not self.available_commands:
                    warnings.warn(
                        "Remote processor has no available commands. "
                        "Ensure the platform is properly configured.",
                        stacklevel=2,
                    )
            else:
                self.available_commands = []

        self.microbatch_size = microbatch_size
        self.default_timeout = float(timeout)

        self.max_shots_per_call = (
            None if max_shots_per_call is None else int(max_shots_per_call)
        )

        # Concurrency of chunk submissions inside a single quantum leaf
        self.chunk_concurrency = max(1, int(chunk_concurrency))

        # Caches & global tracking
        self._layer_cache: dict[int, dict] = {}
        self._job_history: list[RemoteJob] = []

        # Lifecycle/cancellation
        self._lock = threading.Lock()
        self._active_jobs: set[RemoteJob] = set()
        self._closed = False

    # ---------------- Small compatibility helpers ----------------

    def _get_computation_scheme(self, layer: MerlinModule) -> str:
        """Return the Combinadics scheme string for a layer's computation space.

        Returns one of ``"fock"``, ``"unbunched"``, ``"dual_rail"``.
        """
        cs = getattr(layer, "computation_space", None)
        if cs is not None:
            # ComputationSpace.value is the scheme string
            val = getattr(cs, "value", None)
            if isinstance(val, str) and val in ("fock", "unbunched", "dual_rail"):
                return val
            if isinstance(cs, ComputationSpace):
                raise ValueError(
                    "Remote execution currently supports only built-in "
                    "ComputationSpace values: fock, unbunched, and dual_rail."
                )
            # Fallback: match by enum name
            name = getattr(cs, "name", "")
            if name == "UNBUNCHED":
                return "unbunched"
            if name == "DUAL_RAIL":
                return "dual_rail"

        return "fock"

    # ---------------- Public APIs ----------------

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("MerlinProcessor is closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.cancel_all()
        finally:
            with self._lock:
                self._closed = True

    def cancel_all(self) -> None:
        """Cancel all in-flight jobs across all futures."""
        with self._lock:
            jobs = list(self._active_jobs)
        for job in jobs:
            cancel = getattr(job, "cancel", None)
            if callable(cancel):
                with suppress(Exception):
                    cancel()

    def forward(
        self,
        module: nn.Module,
        input: torch.Tensor,
        *,
        nsample: int | None = None,
        timeout: float | None = None,
    ) -> torch.Tensor:
        """Run a module synchronously, offloading eligible Merlin layers.

        Parameters
        ----------
        module : nn.Module
            Module to evaluate.
        input : torch.Tensor
            Input batch to process.
        nsample : int | None
            Optional number of samples to request from the remote backend.
        timeout : float | None
            Optional timeout in seconds for this call.

        Returns
        -------
        torch.Tensor
            Output tensor produced by ``module``.
        """
        fut = self.forward_async(module, input, nsample=nsample, timeout=timeout)
        return fut.wait()

    def forward_async(
        self,
        module: nn.Module,
        input: torch.Tensor,
        *,
        nsample: int | None = None,
        timeout: float | None = None,
    ) -> Future:
        """Asynchronous execution of a PyTorch module, offloading MerlinModule
        leaves to the configured remote processor.

        Parameters
        ----------
        module : nn.Module
            Module to evaluate.
        input : torch.Tensor
            Input batch to process.
        nsample : int | None
            Optional number of samples to request from the remote backend.
        timeout : float | None
            Optional timeout in seconds for this call.

        Returns
        -------
        Future
            ``torch.futures.Future`` with extra helpers:
            ``future.job_ids``, ``future.status()``, and
            ``future.cancel_remote()``.

        Raises
        ------
        RuntimeError
            If the processor is closed or ``module`` is still in training mode.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("MerlinProcessor is closed")

        if module.training:
            raise RuntimeError(
                "Remote quantum execution requires `.eval()` mode. "
                "Call `module.eval()` before forward."
            )

        effective_timeout = self.default_timeout if timeout is None else timeout
        deadline: float | None = (
            None
            if effective_timeout in (None, 0)
            else time.time() + float(effective_timeout)
        )

        original_device = input.device
        original_dtype = input.dtype
        layers: list[Any] = list(self._iter_layers_in_order(module))

        fut: Future = Future()
        state = {
            "cancel_requested": False,
            "current_status": None,
            "job_ids": [],
            "chunks_total": 0,
            "chunks_done": 0,
            "active_chunks": 0,
            "call_id": uuid.uuid4().hex[:8],
        }

        def _cancel_remote():
            state["cancel_requested"] = True
            self.cancel_all()
            if not fut.done():
                try:
                    from concurrent.futures import CancelledError
                except Exception:  # pragma: no cover

                    class CancelledError(RuntimeError):
                        pass

                fut.set_exception(CancelledError("Remote call was cancelled"))

        def _status():
            js = state.get("current_status")
            return {
                "state": (
                    "COMPLETE"
                    if fut.done() and not js
                    else (js.get("state") if js else "IDLE")
                ),
                "progress": js.get("progress") if js else 0.0,
                "message": js.get("message") if js else None,
                "chunks_total": state["chunks_total"],
                "chunks_done": state["chunks_done"],
                "active_chunks": state["active_chunks"],
            }

        fut.cancel_remote = _cancel_remote  # type: ignore[attr-defined]
        fut.status = _status  # type: ignore[attr-defined]
        fut.job_ids = state["job_ids"]  # type: ignore[attr-defined]

        def _run_pipeline():
            try:
                x = input
                for layer in layers:
                    # Policy: offload MerlinModule leaves; else run locally
                    if isinstance(layer, MerlinModule):
                        try:
                            # Preferred (new) signature
                            should_offload = bool(layer.should_offload())

                        except Exception:
                            should_offload = False
                    else:
                        should_offload = False

                    if state["cancel_requested"]:
                        raise self._cancelled_error()

                    if should_offload:
                        x = self._offload_quantum_layer_with_chunking(
                            layer, x, nsample, state, deadline
                        )
                    else:
                        with torch.no_grad():
                            x = layer(x)

                if not fut.done():
                    fut.set_result(x.to(device=original_device, dtype=original_dtype))
            except BaseException as e:
                if not fut.done():
                    fut.set_exception(e)

        threading.Thread(target=_run_pipeline, daemon=True).start()
        return fut

    # ---------------- Chunked offload per quantum leaf ----------------

    def _offload_quantum_layer_with_chunking(
        self,
        layer: MerlinModule,
        input_tensor: torch.Tensor,
        nsample: int | None,
        state: dict,
        deadline: float | None,
    ) -> torch.Tensor:
        """Split the batch into chunks of size <= microbatch_size,
        submit up to chunk_concurrency jobs concurrently, and stitch."""
        if input_tensor.is_cuda:
            input_tensor = input_tensor.cpu()

        cache = self._layer_cache.get(id(layer))
        if cache is None:
            config = cast(Any, layer).export_config()
            self._layer_cache[id(layer)] = {"config": config}
        else:
            config = cache["config"]

        B = input_tensor.shape[0]

        chunks: list[tuple[int, int]] = []
        start = 0
        while start < B:
            end = min(start + self.microbatch_size, B)
            chunks.append((start, end))
            start = end
        return self._run_chunks_pooled(
            layer, config, input_tensor, chunks, nsample, state, deadline
        )

    def _run_chunks_pooled(
        self,
        layer: MerlinModule,
        config: dict,
        input_tensor: torch.Tensor,
        chunks: list[tuple[int, int]],
        nsample: int | None,
        state: dict,
        deadline: float | None,
    ) -> torch.Tensor:
        """Submit chunk jobs with limited concurrency and stitch results."""
        state["chunks_total"] += len(chunks)
        outputs: list[torch.Tensor | None] = [None] * len(chunks)
        errors: list[BaseException] = []

        total_chunks = len(chunks)
        layer_name = getattr(layer, "name", layer.__class__.__name__)

        def _call(s: int, e: int, idx: int):
            try:
                base_label = (
                    f"mer:{layer_name}:{state['call_id']}:{idx + 1}/{total_chunks}"
                )
                t = self._run_chunk(
                    layer,
                    config,
                    input_tensor[s:e],
                    nsample,
                    state,
                    deadline,
                    job_base_label=base_label,
                )
                outputs[idx] = t
            except BaseException as ex:
                errors.append(ex)

        in_flight = 0
        idx = 0
        futures: list[threading.Thread] = []
        while idx < len(chunks) or in_flight > 0:
            while idx < len(chunks) and in_flight < self.chunk_concurrency:
                s, e = chunks[idx]
                with self._lock:
                    state["active_chunks"] += 1
                th = threading.Thread(target=_call, args=(s, e, idx), daemon=True)
                th.start()
                futures.append(th)
                idx += 1
                in_flight += 1

            for th in list(futures):
                if not th.is_alive():
                    futures.remove(th)
                    in_flight -= 1
                    with self._lock:
                        state["active_chunks"] = max(0, state["active_chunks"] - 1)
                        state["chunks_done"] += 1

            if deadline is not None and time.time() >= deadline:
                self.cancel_all()
                raise TimeoutError("Remote call timed out (remote cancel issued)")

            time.sleep(0.01)

        if errors:
            raise errors[0]

        return torch.cat(outputs, dim=0)  # type: ignore[arg-type]

    def _run_chunk(
        self,
        layer: MerlinModule,
        config: dict,
        input_chunk: torch.Tensor,
        nsample: int | None,
        state: dict,
        deadline: float | None,
        job_base_label: str | None = None,
    ) -> torch.Tensor:
        """Submit a single chunk job with retries and return the mapped tensor."""
        from concurrent.futures import CancelledError

        batch_size = input_chunk.shape[0]
        if self.session is None and batch_size > self.microbatch_size:
            raise ValueError(
                f"Chunk size {batch_size} exceeds microbatch {self.microbatch_size}. "
                "Please report this bug."
            )

        input_param_names = self._extract_input_params(config)
        input_np = input_chunk.detach().cpu().numpy()

        # Pre-compute iteration params (cheap, only done once).
        iteration_params: list[dict[str, float]] = []
        for i in range(batch_size):
            circuit_params = {}
            for j, param_name in enumerate(input_param_names):
                circuit_params[param_name] = (
                    float(input_np[i, j]) if j < input_chunk.shape[1] else 0.0
                )
            iteration_params.append(circuit_params)

        def _capped_name(base: str, cmd: str) -> str:
            name = f"{base}:{cmd}"
            name = "".join(ch if ch.isalnum() or ch in "-_:/=." else "_" for ch in name)
            if len(name) <= self._JOB_NAME_MAX:
                return name
            h = f"{zlib.adler32(name.encode()):08x}"
            keep = self._JOB_NAME_MAX - 1 - len(h)
            if keep < 1:
                return h[: self._JOB_NAME_MAX]
            return name[:keep] + "~" + h

        last_error: BaseException | None = None
        for attempt in range(self._MAX_CHUNK_RETRIES):
            if state.get("cancel_requested"):
                raise CancelledError("Remote call was cancelled")
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("Remote call timed out (remote cancel issued)")

            # Build a fresh RemoteProcessor and Sampler on each attempt so that
            # a corrupted RP doesn't poison retries.
            rp = self._create_fresh_rp()
            rp.set_circuit(config["circuit"])
            if config.get("input_state"):
                input_state = pcvl.BasicState(config["input_state"])
                rp.with_input(input_state)
                n_photons = sum(config["input_state"])
                rp.min_detected_photons_filter(n_photons)

            max_shots_arg = (
                self.DEFAULT_SHOTS_PER_CALL
                if self.max_shots_per_call is None
                else int(self.max_shots_per_call)
            )
            sampler = Sampler(rp, max_shots_per_call=max_shots_arg)
            sampler.clear_iterations()
            for params in iteration_params:
                sampler.add_iteration(circuit_params=params)

            job = None
            try:
                job = self._submit_job(sampler, nsample, job_base_label, _capped_name)
                with self._lock:
                    self._active_jobs.add(job)
                    self._job_history.append(job)

                return self._poll_job(job, state, deadline, batch_size, layer, nsample)
            except (CancelledError, TimeoutError, KeyboardInterrupt):
                raise
            except Exception as exc:
                last_error = exc
                if job is not None:
                    with self._lock:
                        self._active_jobs.discard(job)
                logger.warning(
                    "Chunk attempt %d/%d failed: %s",
                    attempt + 1,
                    self._MAX_CHUNK_RETRIES,
                    exc,
                )
                if attempt < self._MAX_CHUNK_RETRIES - 1:
                    time.sleep(min(1.0 * (2**attempt), 5.0))

        raise RuntimeError(
            f"Chunk failed after {self._MAX_CHUNK_RETRIES} attempts"
        ) from last_error

    def _submit_job(self, sampler, nsample, job_base_label, _capped_name):
        """Submit a single async job via the sampler."""
        if ("probs" in self.available_commands) and (
            nsample is None or int(nsample) <= 0
        ):
            job = sampler.probs
            cmd = "probs"
            if job_base_label:
                job.name = _capped_name(job_base_label, cmd)
            return job.execute_async()

        use_shots = self.DEFAULT_SHOTS_PER_CALL if nsample is None else int(nsample)

        if "sample_count" in self.available_commands:
            job = sampler.sample_count
            cmd = "sample_count"
        elif "samples" in self.available_commands:
            job = sampler.samples
            cmd = "samples"
        else:
            job = sampler.sample_count
            cmd = "sample_count"

        if job_base_label:
            job.name = _capped_name(job_base_label, cmd)
        return job.execute_async(max_samples=use_shots)

    def _poll_job(
        self,
        job: RemoteJob,
        state: dict,
        deadline: float | None,
        batch_size: int,
        layer: MerlinModule,
        nsample: int | None,
    ) -> torch.Tensor:
        """Poll a submitted job until complete/failed/timeout and return results."""
        from concurrent.futures import CancelledError

        _MAX_NON_DICT_RETRIES = 60  # 60 * 0.1s = 6s
        non_dict_retries = 0
        sleep_ms = 50
        while True:
            if state.get("cancel_requested"):
                cancel = getattr(job, "cancel", None)
                if callable(cancel):
                    with suppress(Exception):
                        cancel()
                raise CancelledError("Remote call was cancelled")

            if deadline is not None and time.time() >= deadline:
                cancel = getattr(job, "cancel", None)
                if callable(cancel):
                    with suppress(Exception):
                        cancel()
                raise TimeoutError("Remote call timed out (remote cancel issued)")

            s = getattr(job, "status", None)
            state["current_status"] = {
                "state": getattr(s, "state", None) if s else None,
                "progress": getattr(s, "progress", None) if s else None,
                "message": getattr(s, "stop_message", None) if s else None,
            }

            job_id = getattr(job, "id", None) or getattr(job, "job_id", None)
            if job_id is not None and job_id not in state["job_ids"]:
                state["job_ids"].append(job_id)

            if getattr(job, "is_failed", False):
                msg = state["current_status"].get("message")
                if msg and "Cancel requested" in str(msg):
                    with self._lock:
                        self._active_jobs.discard(job)
                    raise CancelledError("Remote call was cancelled")
                with self._lock:
                    self._active_jobs.discard(job)
                raise RuntimeError(
                    f"Remote job failed: {msg or 'unknown error'} (job_id={job_id!r})"
                )

            if getattr(job, "is_complete", False):
                try:
                    raw = job.get_results()
                except RuntimeError as ex:
                    msg = str(ex)
                    if "Results are not available" in msg:
                        time.sleep(0.05)
                        continue
                    if "Cancel requested" in msg:
                        with self._lock:
                            self._active_jobs.discard(job)
                        raise CancelledError("Remote call was cancelled")
                    raise

                if isinstance(raw, dict):
                    with self._lock:
                        self._active_jobs.discard(job)
                    return self._process_batch_results(raw, batch_size, layer, nsample)

                # The backend sometimes reports completion before the dict
                # payload is actually available.  Re-poll the same job for a
                # bounded window before giving up to the outer retry loop.
                non_dict_retries += 1
                if non_dict_retries >= _MAX_NON_DICT_RETRIES:
                    with self._lock:
                        self._active_jobs.discard(job)
                    raise RuntimeError(
                        f"Job complete but results were not a dict after "
                        f"{_MAX_NON_DICT_RETRIES} re-polls; "
                        f"job_id={job_id!r}, type={type(raw)}, value={raw!r}"
                    )
                time.sleep(0.1)
                continue

            time.sleep(sleep_ms / 1000.0)
            sleep_ms = min(sleep_ms * 2, 400)

    # ---------------- Per-call RP pool helpers ----------------

    def _create_fresh_rp(self) -> RemoteProcessor:
        """Build a fresh RemoteProcessor for each chunk/attempt.

        For the ISession path, each call to ``session.build_remote_processor()``
        returns an independent RP with its own RPC handler state, which is safe
        for concurrent chunk execution and clean retries.
        """
        if self.session is not None:
            return self.session.build_remote_processor()
        else:
            return self._clone_remote_processor(self.remote_processor)

    # ---------------- Utilities & mapping ----------------

    def _clone_remote_processor(self, rp: RemoteProcessor) -> RemoteProcessor:
        """Create a sibling RemoteProcessor with its own RPC handler (thread-safe).

        Forwards the token extracted at init time so that inline-token
        RemoteProcessors are cloned correctly.
        """
        return RemoteProcessor(
            name=rp.name,
            token=self._token,
            url=(
                rp.get_rpc_handler().url
                if hasattr(rp.get_rpc_handler(), "url")
                else None
            ),
            proxies=rp.proxies,
        )

    @staticmethod
    def _extract_rp_token(rp: RemoteProcessor) -> str | None:
        """Extract the auth token from a RemoteProcessor.

        Perceval stores the token on the RPC handler as ``handler.token``
        and also embeds it in ``handler.headers['Authorization']``.  We
        probe both locations so that inline-token and global-config
        ``RemoteProcessor`` instances are both handled.

        As a last resort, falls back to ``RemoteConfig().get_token()``.
        Returns ``None`` only if every strategy fails.
        """
        try:
            handler = rp.get_rpc_handler()
        except Exception:
            handler = None

        if handler is not None:
            # Primary: handler.token (set by RPCHandler.__init__)
            for attr in ("token", "_token", "auth_token"):
                val = getattr(handler, attr, None)
                if isinstance(val, str) and val:
                    return val

            # Fallback: parse 'Bearer <token>' from Authorization header
            headers = getattr(handler, "headers", None)
            if isinstance(headers, dict):
                auth = headers.get("Authorization", "")
                if auth.startswith("Bearer ") and len(auth) > 7:
                    return auth[7:]

        # Last resort: check the global config
        try:
            from perceval.runtime import RemoteConfig

            global_token = (RemoteConfig().get_token() or "").strip()
            if global_token:
                return global_token
        except Exception:
            logger.debug("RemoteConfig token lookup failed", exc_info=True)

        return None

    def _iter_layers_in_order(self, module: nn.Module) -> Iterable[nn.Module]:
        """Yield execution leaves in deterministic order.

        MerlinModule instances are treated as single leaves (not recursed into).
        """
        if isinstance(module, MerlinModule):
            yield module
            return
        children = list(module.children())
        if not children:
            yield module
            return
        for child in children:
            yield from self._iter_layers_in_order(child)

    def _extract_input_params(self, config: dict) -> list[str]:
        """Extract circuit parameter names that correspond to model inputs."""
        return list(config["input_param_order"])

    def _process_batch_results(
        self,
        raw_results: Any,
        batch_size: int,
        layer: MerlinModule,
        nsample: int | None = None,
    ) -> torch.Tensor:
        """Map raw cloud results dict into a [B, dist_size] probability tensor."""
        if raw_results is None:
            raise RuntimeError(
                "Remote job returned no results. This may indicate a job execution failure "
                "or an issue with the remote platform."
            )

        if not isinstance(raw_results, dict):
            raise RuntimeError(
                f"Unexpected remote results type: {type(raw_results)} (expected dict)."
            )

        dist_size, state_to_index, valid_states = self._get_state_mapping(layer)
        output_tensors: list[torch.Tensor] = []

        if "results_list" in raw_results:
            results_list = raw_results["results_list"]
            for i, result_item in enumerate(results_list):
                if i >= batch_size:
                    break
                if "results" in result_item:
                    state_counts = result_item["results"]
                    probs = torch.zeros(dist_size)
                    if state_counts:
                        if valid_states is not None:
                            filtered_counts = {}
                            for state_str, count in state_counts.items():
                                state_tuple = self._parse_perceval_state(state_str)
                                if state_tuple in valid_states:
                                    filtered_counts[state_str] = count
                            state_counts = filtered_counts

                        if not state_counts:
                            output_tensors.append(torch.zeros(dist_size))
                            continue

                        first_value = next(iter(state_counts.values()))
                        is_probability = (
                            isinstance(first_value, float) and first_value <= 1.0
                        )
                        total = 1.0 if is_probability else sum(state_counts.values())

                        for state_str, value in state_counts.items():
                            state_tuple = self._parse_perceval_state(state_str)
                            if not state_tuple:
                                continue
                            if state_to_index is not None:
                                if state_tuple not in state_to_index:
                                    continue
                                idx = state_to_index[state_tuple]
                            else:
                                continue
                            if idx < dist_size:
                                probs[idx] = (
                                    value
                                    if is_probability
                                    else (value / total if total > 0 else 0)
                                )

                        prob_sum = probs.sum()
                        if prob_sum > 0 and abs(float(prob_sum) - 1.0) > 1e-6:
                            probs = probs / prob_sum
                        output_tensors.append(probs)
                else:
                    output_tensors.append(torch.zeros(dist_size))

        while len(output_tensors) < batch_size:
            output_tensors.append(torch.zeros(dist_size))

        return torch.stack(output_tensors[:batch_size])

    def _get_state_mapping(
        self, layer: MerlinModule
    ) -> tuple[int, dict | None, set | None]:
        """Determine the output distribution size and Fock-state-to-index mapping."""
        scheme = self._get_computation_scheme(layer)
        needs_filter = scheme != "fock"

        if hasattr(layer, "computation_process") and hasattr(
            layer.computation_process, "simulation_graph"
        ):
            graph: Any = layer.computation_process.simulation_graph

            final_keys = getattr(graph, "final_keys", None)
            if final_keys:
                keys = list(final_keys)
                dist_size = len(keys)
                state_to_index = {state: idx for idx, state in enumerate(keys)}
                valid_states = set(keys) if needs_filter else None
                return dist_size, state_to_index, valid_states

            # Prefer mapped_keys if present (newer graphs)
            mapped_keys = getattr(graph, "mapped_keys", None)
            if mapped_keys:
                keys = list(mapped_keys)
                dist_size = len(keys)
                state_to_index = {state: idx for idx, state in enumerate(keys)}
                valid_states = set(keys) if needs_filter else None
                return dist_size, state_to_index, valid_states

            if hasattr(layer, "circuit") and hasattr(layer.circuit, "m"):
                n_modes = int(layer.circuit.m)  # type: ignore[arg-type]
            else:
                n_modes = int(graph.m)  # type: ignore[arg-type]

            if hasattr(layer, "input_state"):
                input_state = layer.input_state
                n_photons = int(sum(input_state))  # type: ignore[arg-type]
            else:
                n_photons = int(graph.n_photons)  # type: ignore[arg-type]

            keys = Combinadics(scheme, n_photons, n_modes).enumerate_states()
            dist_size = len(keys)
            state_to_index = {state: idx for idx, state in enumerate(keys)}
            valid_states = set(keys) if needs_filter else None

            return dist_size, state_to_index, valid_states

        if hasattr(layer, "circuit") and hasattr(layer, "input_state"):
            circuit = cast(Any, layer.circuit)
            input_state = cast(Any, layer.input_state)

            n_modes = int(circuit.m)
            n_photons = int(sum(input_state))

            keys = Combinadics(scheme, n_photons, n_modes).enumerate_states()
            dist_size = len(keys)
            state_to_index = {state: idx for idx, state in enumerate(keys)}
            valid_states = set(keys) if needs_filter else None

            return dist_size, state_to_index, valid_states

        raise RuntimeError(
            f"Cannot infer state mapping for layer of type {type(layer)!r}. "
            "Expected a MerlinModule with either a 'computation_process' + 'simulation_graph' "
            "or 'circuit' and 'input_state' attributes."
        )

    # ---- Shot estimation (no remote jobs submitted) ----

    def estimate_required_shots_per_input(
        self,
        layer: MerlinModule,
        input: torch.Tensor,
        desired_samples_per_input: int,
    ) -> list[int]:
        """Estimate required shots per input row using the platform estimator.

        Parameters
        ----------
        layer : MerlinModule
            Layer providing ``export_config()`` for remote estimation.
        input : torch.Tensor
            Input tensor with one or more rows to estimate.
        desired_samples_per_input : int
            Target number of usable samples per input row.

        Returns
        -------
        list[int]
            Estimated shots per input row. ``0`` means the target is not
            considered viable.

        Raises
        ------
        TypeError
            If ``layer`` does not provide ``export_config()``.
        ValueError
            If ``input`` is not one- or two-dimensional.
        """
        if not hasattr(layer, "export_config") or not callable(
            cast(Any, layer).export_config
        ):
            raise TypeError("layer must provide export_config() for shot estimation")

        if input.dim() == 1:
            x = input.unsqueeze(0)
        elif input.dim() == 2:
            x = input
        else:
            raise ValueError("input must be 1D or 2D tensor")

        config = cast(Any, layer).export_config()
        child_rp = self._create_fresh_rp()
        child_rp.set_circuit(config["circuit"])

        if config.get("input_state"):
            input_state = pcvl.BasicState(config["input_state"])
            child_rp.with_input(input_state)
            n_photons = sum(config["input_state"])
            child_rp.min_detected_photons_filter(n_photons)

        input_param_names = self._extract_input_params(config)

        import requests  # type: ignore[import-untyped]

        x_np = x.detach().cpu().numpy()
        estimates: list[int] = []
        for i in range(x_np.shape[0]):
            row = x_np[i]
            param_values: dict[str, float] = {}
            for j, pname in enumerate(input_param_names):
                param_values[pname] = float(row[j] * np.pi) if j < row.shape[0] else 0.0

            # Retry on transient read timeouts from the cloud estimator.
            est = None
            last_ex: Exception | None = None
            for _attempt in range(self._MAX_ESTIMATOR_RETRIES):
                try:
                    est = child_rp.estimate_required_shots(
                        desired_samples_per_input, param_values=param_values
                    )
                    break
                except requests.exceptions.ReadTimeout as ex:
                    last_ex = ex
                    time.sleep(0.2)
            if est is None and last_ex is not None:
                raise last_ex
            estimates.append(int(est) if est is not None else 0)

        return estimates

    # ---- Misc ----

    def _parse_perceval_state(self, state_str: Any) -> tuple:
        """Parse a Perceval state string like '|1,0,1>' into a tuple of ints."""
        if isinstance(state_str, str):
            if "|" in state_str and ">" in state_str:
                state_str = state_str.strip("|>")
                try:
                    return tuple(int(v) for v in state_str.split(","))
                except Exception:
                    return ()
            elif "," in state_str:
                try:
                    return tuple(int(v) for v in state_str.split(","))
                except Exception:
                    return ()
        elif hasattr(state_str, "__iter__"):
            return tuple(state_str)
        return ()

    def get_job_history(self) -> list[RemoteJob]:
        """Return all jobs observed or submitted by this instance.

        Returns
        -------
        list[RemoteJob]
            Recorded remote jobs.
        """
        return self._job_history

    def clear_job_history(self) -> None:
        """Clear the internal job history list."""
        self._job_history = []

    def _cancelled_error(self):
        """Create a CancelledError with a standard message."""
        from concurrent.futures import CancelledError

        return CancelledError("Remote call was cancelled")
