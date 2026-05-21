merlin.core.state_mixture
=========================

.. automodule:: merlin.core.state_mixture
   :no-members:

.. currentmodule:: merlin.core.state_mixture

StateMixtureBranch
------------------

.. autoclass:: StateMixtureBranch
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

StateMixture
------------

.. autoclass:: StateMixture
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

Notes and Examples
------------------

``StateMixture`` is the propagatable ensemble representation of a density matrix
built from conditional pure-state branches. A partial measurement still returns
``PartialMeasurement`` for inspection, and it can be converted explicitly when
manual branch manipulation is needed:

.. code-block:: python

   partial = partial_layer()
   mixture = partial.to_state_mixture()

   for probability, state, history in zip(
       mixture.probabilities,
       mixture.states,
       mixture.outcome_histories,
   ):
       print(probability, state.n_modes, history)

``QuantumLayer`` also accepts ``PartialMeasurement`` directly and converts it to
``StateMixture`` internally:

.. code-block:: python

   partial = partial_layer()
   output = next_layer(partial)

For probability and mode-expectation outputs, branch outputs are recombined with
their branch probabilities. For amplitude outputs, the layer returns another
``StateMixture`` containing the propagated branch states.

Density-Matrix Semantics
^^^^^^^^^^^^^^^^^^^^^^^^

``StateMixture`` represents an ensemble of pure conditional states, equivalent
to a density operator of the form:

.. math::

   \rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|

Downstream layers propagate each branch state and keep the same branch weights
unless a later partial measurement splits a branch further. Tensor-valued
outputs, such as probabilities and mode expectations, are recombined linearly as
``sum_i p_i * output_i``. Amplitude-valued outputs remain a ``StateMixture``
because a mixed state does not have a single coherent amplitude vector.

This carrier intentionally does not encode off-diagonal coherences between
branches. Use a single ``StateVector`` when coherent superposition amplitudes
must be preserved.

The current implementation preserves any batch dimension already present inside
each branch state. When all branches have compatible dense ``StateVector``
tensors, ``QuantumLayer`` flattens the branch axis and the existing state batch
axis into one larger batched ``StateVector`` for the downstream layer call, then
splits the output back into branch order before recombination. Stateful
memristive layers and incompatible branch layouts use the explicit sequential
branch loop.
