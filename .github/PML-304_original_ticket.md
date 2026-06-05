# PML-304: Replace Monkey-Patched Future API

## 1. Description

### 1.1. Context

`forward_async()` currently creates a `torch.futures.Future` and monkey-patches extra attributes onto it:

- `cancel_remote()`
- `status()`
- `job_ids`

That works at runtime, but it hides the actual async contract from code readers, IDEs, static analysis, and generated documentation.

### 1.2. Objective

Replace the monkey-patched future behavior with an explicit typed async handle.

The goal is to make the API discoverable and maintainable while preserving the current capabilities and user-facing behavior as closely as practical.

## 2. Minimal Example

Expected usage:

```python
fut = proc.forward_async(model, x, nsample=5000)
status = fut.status()
job_ids = fut.job_ids
fut.cancel_remote()
y = fut.wait()
```

## 3. Implementation Notes

The async handle should wrap or compose the underlying `torch.futures.Future` instead of modifying it at runtime.

The public handle should expose the existing async operations explicitly:

- `wait()`
- `cancel_remote()`
- `status()`
- `job_ids`

The implementation should avoid runtime monkey-patching of attributes onto `torch.futures.Future`.

## 4. Acceptance Criteria

- The async API is explicit and discoverable in code and types.
- There is no runtime monkey-patching of future attributes.
- `forward_async()` returns the declared async handle type.
- Existing user-visible async behavior remains equivalent unless a documented API adjustment is intentionally made.

## 5. Testing

Add no-cloud tests for:

- `forward_async()` return type and helper methods.
- `cancel_remote()` behavior and resulting exception propagation.
- `status()` payload shape and `job_ids` exposure.
- synchronous `forward()` behavior remaining unchanged when built on top of the new async handle.
