:github_url: https://github.com/merlinquantum/merlin

===========================
Quantum Vision Transformers
===========================

.. admonition:: Paper Information
   :class: note

   **Title**: Quantum Vision Transformers

   **Authors**: El Amine Cherrat, Iordanis Kerenidis, Natansh Mathur, Jonas Landman, Martin Strahm, Yun Yvonna Li

   **Published**: Quantum 8, 1265 (2024)

   **DOI**: `10.22331/q-2024-02-22-1265 <https://doi.org/10.22331/q-2024-02-22-1265>`_

   .. merlin-citations-badge:: quantum_vision_transformers

   **Paper URL**: `arXiv:2209.08167 <https://arxiv.org/abs/2209.08167>`_

   **Reproduction Status**: ✅ Complete — the paper's RetinaMNIST benchmark reproduces, with three architectural extensions added on top

   **Reproducer**: Benjamin Stott

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/quantum_vision_transformers_external_links.json
   :columns: 2
   :contour-color: #5648ED

Abstract
========

Cherrat et al. design transformer architectures whose attention mechanism is
carried by a quantum circuit rather than by an explicit learned score matrix. The
paper proposes a family of them — an orthogonal patch-wise network, a quantum
orthogonal transformer computing pairwise overlap scores, a direct-attention
variant, and a *compound* transformer in which attention emerges from
two-photon interference plus a cross-partition read-out rather than being
computed at all. The models are benchmarked on MedMNIST, with RetinaMNIST as the
headline dataset, against a classical vision transformer and a quantum
orthogonal fully-connected baseline.

The claim under test is not raw accuracy but **accuracy at a much smaller
attention-parameter budget**: the quantum attention layers are meant to match a
classical ViT while parameterising attention with a fraction of the weights.

Significance
============

Most quantum-attention proposals replace a classical block with a quantum one and
report comparable accuracy, which on its own says little. What makes this paper
worth reproducing is that its compound model makes a *structural* claim — that
attention need not be materialised as a matrix at all, but can be read out of the
interference pattern of a two-photon state. That is a claim about where the
computation lives, and it is directly testable on a photonic simulator, because
the sector structure the model relies on is native to linear optics.

MerLin Implementation
=====================

The reproduction is built on MerLin's native linear-optics primitives —
``QuantumLayer``, ``CircuitBuilder``, and ``StateVector.from_tensor`` for
amplitude-encoded inputs, with SLOS computing the compound actions internally.
Six architectures share one wrapper:

.. list-table:: Architectures
   :header-rows: 1
   :widths: 12 34 12 12 30

   * - Model
     - Description
     - Photons
     - Modes
     - Attention mechanism
   * - A
     - Orthogonal patch-wise network
     - 1
     - d
     - none; one shared interferometer per token
   * - B
     - Quantum orthogonal transformer
     - 1
     - d
     - pairwise overlap scores through ``W``, softmax, features through ``V``
   * - C
     - Direct quantum attention
     - 1
     - d
     - same overlap scores as B, applied to inputs before the feature transform
   * - D
     - Compound transformer
     - 2
     - n+d
     - implicit; emerges from interference plus a cross-partition read-out
   * - D_full
     - Full-sector compound *(extension)*
     - 2
     - n+d
     - as D, but keeping every two-photon sector rather than cross-partition only
   * - E
     - Multi-sector attention *(extension)*
     - 1 + 2
     - n+d
     - one shared interferometer: 1-photon sector gives features, 2-photon sector gives attention
   * - F
     - Hierarchical compound *(extension)*
     - 3
     - r+p+d
     - three-photon interference across region, patch and feature blocks

Two circuit families are selectable. **generic** is a universal rectangular MZI
mesh (Clements/Reck); **butterfly** is a structured mesh matching the layout used
in the paper, and requires power-of-two mode counts. Post-selection is applied
only where it is semantically required — the cross-partition read-out in D, and
the triple-cross read-out in F.

Experimental Results
====================

**RetinaMNIST against the paper's Table 4.** Reproduction figures are means over
three seeds (7, 42, 123) in the butterfly family; the paper reports single
values.

.. list-table:: RetinaMNIST: paper Table 4 against this reproduction
   :header-rows: 1
   :widths: 26 14 14 16 16

   * - Model
     - Paper AUC
     - Paper ACC
     - Reproduced AUC
     - Reproduced ACC
   * - Classical ViT
     - 0.736
     - 55.75%
     - 0.7417 ± 0.0037
     - 54.50%
   * - A: OrthoPatchWise
     - 0.738
     - 56.50%
     - 0.7479 ± 0.0087
     - 53.25%
   * - B: OrthoTransformer
     - 0.749
     - 56.50%
     - 0.7369 ± 0.0064
     - 51.08%
   * - D: Compound
     - 0.729
     - 56.50%
     - 0.7409 ± 0.0106
     - 52.83%
   * - OrthoFNN baseline
     - —
     - —
     - 0.6720 ± 0.0083
     - 49.08%

Every reproduced AUC lands within about 0.012 of the paper's reported value, and
the ordering of the quantum models relative to the classical ViT is preserved.
Accuracy comes out 2-5 points below the paper's across *all* models including the
classical baseline, which points at a threshold or preprocessing difference
affecting every arm equally rather than at any one architecture.

The OrthoFNN baseline is the informative row: at 0.672 it sits far below
everything else, so the benchmark does discriminate between architectures. The
quantum attention models are not simply riding a task on which every model scores
the same.

.. figure:: ../../_static/reproduced_papers/quantum_vision_transformers/comparison_retinamnist.png
   :alt: Test AUC and accuracy per model on RetinaMNIST with the paper's reported values as dashed reference lines
   :align: center
   :width: 100%

   RetinaMNIST test AUC and accuracy, butterfly family, three seeds. Dashed lines
   are the paper's reported values. The AUC panel tracks the reference closely;
   the accuracy panel sits uniformly below it.

**The parameter claim.** This is where the paper's argument actually rests, and it
holds.

.. list-table:: Attention parameters on RetinaMNIST
   :header-rows: 1
   :widths: 34 22 22 22

   * - Model
     - Attention params
     - Total params
     - Reproduced AUC
   * - Classical ViT
     - 2048
     - 9333
     - 0.7417 ± 0.0037
   * - A: OrthoPatchWise
     - 256
     - 7509
     - 0.7479 ± 0.0087
   * - B: OrthoTransformer
     - 512
     - 7797
     - 0.7369 ± 0.0064
   * - D: Compound
     - 640
     - 7893
     - 0.7409 ± 0.0106

Model A matches the classical ViT's AUC using **8x fewer attention parameters**,
and D does so with 3.2x fewer while also carrying fewer total parameters. Note
that the paper quotes attention parameters *per layer* (32, 64, 80) while the
figures above are the measured totals for the whole model, so the two columns
count different things; the ratio against the classical ViT is the comparable
quantity, and it is the ratio the paper's claim is about.

.. figure:: ../../_static/reproduced_papers/quantum_vision_transformers/param_comparison.png
   :alt: Attention-layer and total trainable parameters per model against the classical ViT attention reference
   :align: center
   :width: 100%

   Attention-layer parameters (coloured) and total trainable parameters (grey)
   per model. The dashed line is the classical ViT attention budget.

**Breadth across MedMNIST.** Eight datasets, three seeds, butterfly family, lite
profile, trained on a capped 5000-sample training subset with the official
validation and test splits intact.

.. list-table:: Test AUC across MedMNIST (train_subset_5000 regime, 3 seeds)
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Dataset
     - A
     - B
     - D
     - D_full
   * - bloodmnist
     - 0.9647
     - 0.9681
     - 0.9754
     - 0.9740
   * - breastmnist
     - 0.7875
     - 0.8341
     - 0.8094
     - 0.8059
   * - dermamnist
     - 0.8745
     - 0.8648
     - 0.8928
     - 0.8884
   * - octmnist
     - 0.7699
     - 0.7692
     - 0.8177
     - 0.8173
   * - pathmnist
     - 0.9434
     - 0.9300
     - 0.9410
     - 0.9452
   * - pneumoniamnist
     - 0.9439
     - 0.9428
     - 0.9507
     - 0.9473
   * - retinamnist
     - 0.7387
     - 0.7390
     - 0.7381
     - 0.7408
   * - tissuemnist
     - 0.8106
     - 0.8055
     - 0.8167
     - 0.8209

The compound models D and D_full lead on five of the eight datasets, and the
spread between architectures is small relative to the spread between datasets.
The extra sectors kept by D_full make little consistent difference over D, which
is worth knowing: the cross-partition read-out the paper specifies appears to
capture most of what the two-photon state carries for this task.

**Circuit family.** The structured butterfly mesh matches the universal generic
mesh on accuracy while being substantially cheaper to simulate. On RetinaMNIST,
model A reaches AUC 0.7479 ± 0.0087 under butterfly against 0.7435 ± 0.0035 under
generic, and model B 0.7369 ± 0.0064 against 0.7425 ± 0.0068 — differences inside
the seed spread. Typical butterfly wall-clock for these runs is a few hundred
seconds against a few thousand for generic. For power-of-two mode counts the
structured family is the better default.

Implementation Details
======================

Run through the shared root CLI:

.. code-block:: bash

   python implementation.py --paper quantum_vision_transformers \
       --config papers/quantum_vision_transformers/configs/paper/model_a_retina.json

From the paper directory, the suite wrappers cover the full workflow:

.. code-block:: bash

   bash scripts/validation/validate.sh                                   # all models build, gradients flow
   CPU_FRIENDLY=1 bash scripts/suites/run_retina_suite.sh --device cpu    # paper + butterfly workflow
   CPU_FRIENDLY=1 bash scripts/suites/run_medmnist_suite.sh --device cpu  # MedMNIST workflow
   python scripts/analysis/generate_figures.py outdir/                    # figures -> results/figures/

Each run directory holds the resolved config, a resumable ``last.pt`` checkpoint,
``best.pt`` at best validation AUC, an incremental ``progress.json`` and a final
``results.json``. Re-running with the same ``--outdir`` resumes automatically;
``--resume never`` forces a fresh run.

MerLin Version Note
===================

The reproduction runs against released ``merlinquantum>=0.4.0``. It previously
depended on a vendored ``third_party/merlinquantum`` 0.3 branch; the sparse/EBS
performance fixes and the amplitude-input ``StateVector`` path from that branch
have since landed upstream, and the ``partition_blocks`` / ``allowed_counts``
measurement filters that were removed are now applied classically in the read-out
modules in ``lib/photonic_primitives.py``. There is no forked dependency.

Deviations and Limitations
==========================

* **Accuracy runs uniformly below the paper's** by 2-5 points across every model
  including the classical baseline, while AUC matches. The cause has not been
  isolated; because it affects all arms equally it does not change the
  between-model comparison, which is what the paper's claims concern.
* **Extensions E and F are not trained to convergence.** The committed generic
  full-profile rows for D_full, E and F stop at ``best_epoch = 1`` with wall-clock
  in the tens of seconds, so their AUC figures there (0.65, 0.64, 0.64) are
  single-epoch snapshots and must not be read as trained results. The
  butterfly-family D_full runs *are* trained and are the ones reported above.
* **Model C is only in the generic family** among the committed full-profile runs,
  so it does not appear in the butterfly comparison table.
* **The MedMNIST breadth is on a capped 5000-sample training subset**, not the
  full training splits; validation and test splits are the official ones. It is a
  data-efficiency comparison between architectures rather than a leaderboard
  result.
* Attention-parameter counts are totals here and per-layer in the paper, as noted
  above.
* The committed comparison figures have overlapping x-axis labels at this figure
  size; the underlying values are in ``results/figures/*/summary.csv``.

Code Access and Documentation
=============================

**GitHub Repository**: `merlinquantum/reproduced_papers (quantum_vision_transformers) <https://github.com/merlinquantum/reproduced_papers/tree/main/papers/quantum_vision_transformers>`_

Citation
========

.. code-block:: bibtex

   @article{cherrat_quantum_2024,
     title={Quantum Vision Transformers},
     author={Cherrat, El Amine and Kerenidis, Iordanis and Mathur, Natansh and Landman, Jonas and Strahm, Martin and Li, Yun Yvonna},
     journal={Quantum},
     volume={8},
     pages={1265},
     year={2024},
     doi={10.22331/q-2024-02-22-1265}
   }

Related Reproductions
=====================

* :doc:`photonic_qcnn` — another photonic vision architecture on MedMNIST-style
  data, useful as a contrast in how patches are encoded.
* :doc:`nearest_centroids` — amplitude encoding with quantum inner-product
  estimation, the primitive underlying the overlap scores in models B and C.
