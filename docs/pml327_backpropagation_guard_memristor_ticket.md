# PML-327: Backpropagation guard with memristor

## Description

### 1. Description

#### 1.1. Context

If the gradient is propagated through a `QuantumLayer` with a memristive phase-shifter, it needs to go through all the history to compute the gradient. That is very expensive.

#### 1.2. Objective

The user should be able to define the maximum memristive steps accessible by the backpropagation. By default, it should be 0.

## 2. Minimal Example (Expected Usage)

```python
circ = CircuitBuilder(n_modes=3)
circ.add_memristive_ps(
    mode=1,
    update_rule=update_rule,
    initial_state=1.2,
    num_backprop_steps=5,
)
```

## 3. Implementation Notes

For each memristive update, if the memristive state is bigger than `num_backprop_steps`, detach the first tensor from the differentiation graph with the `.detach()` method.

## 4. Acceptance Criteria

The gradient backpropagates `num_backprop_steps` steps into the memristive history and is using a sliding window.

## 5. Testing

Tests:

- The `num_backprop_steps` gets passed correctly to the quantum layer.
- The backpropagation uses the correct number of memristive history states.
- A sliding window is used for the backpropagation.

## 6. References
