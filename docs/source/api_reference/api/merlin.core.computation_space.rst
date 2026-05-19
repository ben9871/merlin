merlin.core.computation_space module
=====================================

.. automodule:: merlin.core.computation_space
   :no-members:

.. currentmodule:: merlin.core.computation_space

.. rubric:: Built-ins and structured spaces

.. code-block:: python

   from merlin.core import ComputationSpace

   assert ComputationSpace.FOCK.value == "fock"
   assert ComputationSpace.UNBUNCHED.value == "unbunched"
   assert ComputationSpace.DUAL_RAIL.value == "dual_rail"

   space = ComputationSpace(modes_per_photon=[3, 4, 2])
   assert space.modes_per_photon == (3, 4, 2)

   qloq = ComputationSpace.qloq(qubit_groups=[2, 1])
   assert qloq.modes_per_photon == (4, 2)

.. autoclass:: ComputationSpace
   :members:
   :undoc-members:
   :show-inheritance:
