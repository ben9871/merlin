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

``StateMixture`` is the propagatable carrier for a classical mixture of
conditional pure states. A partial measurement still returns
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
