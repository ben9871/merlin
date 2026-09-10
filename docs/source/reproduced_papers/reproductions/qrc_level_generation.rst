:github_url: https://github.com/merlinquantum/merlin

=========================================================
Level Generation with Quantum Reservoir Computing
=========================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Level Generation with Quantum Reservoir Computing

   **Authors**: João S. Ferreira, Pierre Fromholz, Hari Shaji, James R. Wootton

   **Published**: IEEE Computer Graphics and Applications (2025)

   **DOI**: `10.1109/MCG.2025.3591956 <https://doi.org/10.1109/MCG.2025.3591956>`_

   .. merlin-citations-badge:: qrc_level_generation

   **Paper URL**: `arXiv:2505.13287 <https://arxiv.org/abs/2505.13287>`_

   **Reproduction Status**: ⚠️ Partial — the Super Mario Bros case study is reproduced; the Roblox experiments are out of scope

   **Reproducer**: Benjamin Stott

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/qrc_level_generation_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

Ferreira et al. apply Quantum Reservoir Computing to procedural game-level
generation. A small reservoir of 4-8 qubits consumes a sequence of integer
"feature" indices that encode level columns, and a classical feed-forward network
maps the measured probability vector to a next-feature distribution. Sampling at
a controllable temperature *T* generates new levels that either preserve the
original (low *T*) or drift towards randomness (high *T*). The paper evaluates a
Super Mario Bros level 1-2 case study on an ideal simulator and on noisy
backends, and a custom Roblox obby generated in real time on superconducting
hardware.

Two metrics are introduced: the **originality rate** at sequence length *L*, the
fraction of length-*L* windows in generated samples that do not occur in the
original level; and the **broken-transition rate**, the fraction of positions
violating a hand-defined game-breaking rule. Two reference generators —
i.i.d. sampling from the original feature frequencies, and an empirical Markov
chain — are used throughout as baselines.

Significance
============

The interesting claim here is not accuracy but *control*: that a single
post-training temperature knob dials the originality-versus-playability
trade-off cheaply, without retraining. That is a property a reservoir can
plausibly offer and a trained generative model generally cannot, and it is what
makes the approach worth translating to a near-term photonic platform, where the
reservoir is frozen by construction.

MerLin Implementation
=====================

The reproduction implements both backends behind one pipeline:

* ``lib/qrc_qubit.py`` — the gate-based reservoir with an optional depolarising
  channel.
* ``lib/qrc_photonic.py`` — a MerLin photonic reservoir: one entangling MZI mesh,
  an ``add_angle_encoding`` layer, and further entangling layers, with randomly
  initialised and frozen parameters.
* ``lib/qrc_pipeline.py`` — teacher forcing and autoregressive generation shared
  by both.
* ``lib/metrics.py`` — originality, broken-rate, save-point separation statistics
  and the five ported Mario rules.
* ``lib/baselines.py`` — the Markov and uncorrelated reference generators.

Experimental Results
====================

**Metrics on the authors' published sequences.** Running the reproduction's
metric implementations directly against the Moth open-data Aer sequences
validates the metric definitions independently of the reservoir.

.. list-table:: Originality and broken-rate on the published Aer sequences (6 qubits)
   :header-rows: 1
   :widths: 25 25 25 25

   * - Temperature
     - L=2 originality
     - L=10 originality
     - Broken rate (rule "2")
   * - 0.1
     - 0.033
     - 0.567
     - 0.000
   * - 1.0
     - 0.063
     - 0.695
     - 0.003
   * - 2.0
     - 0.093
     - 0.836
     - 0.028
   * - 5.0
     - 0.380
     - 0.992
     - 0.226
   * - 30.0
     - 0.849
     - 1.000
     - 0.793

The paper states that the error rate stays below 5% for temperatures as high as
*T* = 2; the reproduction measures 2.8% at *T* = 2, in agreement. This subset is
a faithful quantitative reproduction.

**Save-point separation** (paper §IV.A). The paper's save-point table applies to
the Roblox obby rather than to Mario, which the reproduction established by
sweeping every feature index against the published Aer Roblox sequences. With
feature index 11 as the save point, the paper's table reproduces exactly.

.. list-table:: Save-point separation, Aer, β = 1, 100 samples
   :header-rows: 1
   :widths: 20 40 40

   * - Qubits
     - Reproduction
     - Paper
   * - 4
     - 17.93 ± 7.92
     - 17.9 ± 7.9
   * - 5
     - 16.34 ± 2.97
     - 16.3 ± 2.9
   * - 6
     - 18.81 ± 4.13
     - 18.8 ± 4.1
   * - 7
     - 18.62 ± 3.53
     - 18.6 ± 3.5
   * - 8
     - 17.14 ± 4.17
     - 17.1 ± 4.1

**Trained reservoirs.** Both the gate-based and the photonic reservoir reproduce
the qualitative behaviour the paper reports — originality and broken-rate both
rise monotonically with *T* — but the operating point is shifted along the
temperature axis. At *T* = 1 the reproduced gate reservoir reaches L=2
originality 0.340 and the photonic reservoir 0.274, against 0.063 for the
published sequences. This is consistent with reservoir-specific calibration:
different random reservoirs produce logit distributions of different scales, and
the temperature parameter absorbs that scale.

.. figure:: ../../_static/reproduced_papers/qrc_level_generation/level_qrc_T1.png
   :alt: A Super Mario Bros level generated by the reproduced 6-qubit QRC at temperature 1
   :align: center
   :width: 95%

   The reproduced 6-qubit reservoir at *T* = 1: coherent structures, continuous
   ground and sensible pipe placement.

.. figure:: ../../_static/reproduced_papers/qrc_level_generation/level_qrc_T30.png
   :alt: The same model at temperature 30, showing fragmented ground and floating debris
   :align: center
   :width: 95%

   The same model at *T* = 30. The broken-transition regime is directly visible
   as fragmented ground and floating debris — the metric and the picture agree.

**Photonic scaling sweep.** Three sweeps over mode count, photon number and
output dimension were run at three seeds each. At matched output dimension the
photonic reservoir is competitive with, and at the largest size better than, the
gate-based reservoir.

.. list-table:: Iso-output-dimension comparison at T = 1 (3 seeds)
   :header-rows: 1
   :widths: 30 12 20 20 18

   * - Backend
     - Dim
     - Originality L=2
     - Broken rate "2"
     - Final CE loss
   * - photonic m=6, p=2
     - 15
     - 0.484 ± 0.016
     - 0.678 ± 0.006
     - 1.93
   * - qubit q=4
     - 16
     - 0.518 ± 0.011
     - 0.664 ± 0.027
     - 1.72
   * - photonic m=6, p=3
     - 20
     - 0.478 ± 0.024
     - 0.687 ± 0.013
     - 1.89
   * - qubit q=5
     - 32
     - 0.493 ± 0.029
     - 0.691 ± 0.036
     - 1.59
   * - photonic m=8, p=3
     - 56
     - 0.427 ± 0.006
     - 0.628 ± 0.016
     - 1.73
   * - qubit q=6
     - 64
     - 0.489 ± 0.013
     - 0.665 ± 0.013
     - 1.38
   * - photonic m=8, p=4
     - 70
     - 0.401 ± 0.008
     - 0.575 ± 0.030
     - 1.73

.. figure:: ../../_static/reproduced_papers/qrc_level_generation/sweep_pareto.png
   :alt: Pareto front of originality against broken-transition rate, parametric in temperature
   :align: center
   :width: 85%

   Originality against broken-rate, parametric in *T*, with all sweep
   configurations overlaid.

The photonic ``UNBUNCHED`` reservoir at dimension 70 is Pareto-dominant in both
metrics despite a *higher* training-loss floor than the gate reservoir at
dimension 64 — the reservoir that fits the training sequence best is not the one
that generates best.

Implementation Details
======================

Run from the reproduced-papers repository root. The reservoir is CPU-only; the
full Mario qubit reproduction takes under a minute and the photonic variant under
three.

.. code-block:: bash

   # Metrics on the authors' published sequences (no training)
   python implementation.py --paper qrc_level_generation --config configs/reference_eval.json

   # Gate-based QRC, temperature sweep, ideal simulator
   python implementation.py --paper qrc_level_generation --config configs/mario_qubit_paper.json

   # MerLin photonic reservoir
   python implementation.py --paper qrc_level_generation --config configs/mario_photonic.json

Hardware-Aware Settings (MerLin reservoir)
==========================================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Value
   * - Computation space
     - ``UNBUNCHED``
   * - Detector model
     - threshold
   * - Photon number
     - 3
   * - Number of modes
     - 6
   * - Input state
     - ``[0, 1, 0, 1, 1, 0]``
   * - Encoding
     - angle, all 6 modes, scale ``π``
   * - Measurement strategy
     - ``MeasurementStrategy.probs(computation_space=UNBUNCHED)``
   * - Postselection
     - none
   * - Simulator
     - MerLin CPU simulator (analytic, shots = 0)
   * - Wall-clock
     - 2.5 min
   * - Seeds
     - 42

A note on reservoir depth: for a *frozen* linear-optical reservoir, stacking
passive meshes with no data injection or nonlinearity between them is not an
expressivity choice, since consecutive meshes compose into one equivalent
interferometer. A controlled one-layer-versus-two-layer comparison confirms this
empirically — every metric agrees to within single-seed noise. ``n_post_layers``
defaults to 1; the committed photonic figures were produced at depth 2 and the
config pins that value so they stay reproducible.

Deviations and Limitations
==========================

* **The reservoir construction is underspecified in the paper.** Gate counts,
  angle conventions and the map from one-hot input and hidden state to rotation
  angles are not given. The reproduction uses 30 random gates per reservoir, a
  per-feature angle book sampled from ``U(-π, π)``, and a Gaussian random
  projection from the hidden state to the rotation angles. These choices shift
  the operating point in temperature space without changing the qualitative
  story.
* **Depolarising noise is applied globally per step** rather than per gate as in
  the paper, which changes the effective noise budget. The reproduced
  originality-versus-noise curve at *T* = 1 is therefore not monotonic, because
  the reproduced reservoir at *T* = 1 is already in a noisy regime.
* **The FakeGarnet noise model is not implemented**; it requires IQM calibration
  data. Metrics on the authors' published FakeGarnet sequences are computed in
  reference-only mode.
* **The Roblox obby experiments are out of scope.** Only feature PNGs, not level
  definitions, are in the open-data dump, and the authors' Roblox encoder is not
  released.
* The feed-forward read-out is a single linear layer trained for 200 epochs; the
  paper does not specify width or schedule.

Data and Attribution
====================

The original level sequence and the reference sequences come from the Moth
Quantum open-data release for this paper. The reproduction adds a level renderer
that rebuilds the tile atlas from the packaged level image and sequence, and
verifies it against every column, replacing the authors' unreleased private
encoder. No Roblox feature art is redistributed.

Code Access and Documentation
=============================

**GitHub Repository**: `merlinquantum/reproduced_papers (qrc_level_generation) <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/qrc_level_generation>`_

Citation
========

.. code-block:: bibtex

   @article{ferreira_level_2025,
     title={Level Generation with Quantum Reservoir Computing},
     author={Ferreira, Jo{\~a}o S. and Fromholz, Pierre and Shaji, Hari and Wootton, James R.},
     journal={IEEE Computer Graphics and Applications},
     year={2025},
     doi={10.1109/MCG.2025.3591956}
   }

Related Reproductions
=====================

* :doc:`quantum_reservoir_computing` — quantum optical reservoir computing powered
  by boson sampling, the photonic reservoir counterpart on a classification task.
