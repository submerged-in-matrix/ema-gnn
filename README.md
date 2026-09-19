# EMA-GNN — A Structural Graph Neural Network for Formation-Energy Prediction

[![Matbench Discovery](https://img.shields.io/badge/Matbench_Discovery-Live_on_Leaderboard-blue)](https://matbench-discovery.materialsproject.org/models/ema-gnn)
[![Figshare](https://img.shields.io/badge/Figshare-DOI_10.6084%2Fm9.figshare.33111509-orange)](https://doi.org/10.6084/m9.figshare.33111509)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green)](LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/Checkpoints-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

A message-passing graph neural network that predicts the formation energy per atom
of a crystal directly from an unrelaxed input structure, with no relaxation step at
inference time. Live on the
[Matbench Discovery leaderboard](https://matbench-discovery.materialsproject.org/models/ema-gnn).

The architecture is inspired by the structural GNN described in
[Merchant et al., *"Scaling deep learning for materials discovery,"*
Nature **624** (2023)](https://doi.org/10.1038/s41586-023-06735-9) — the GNoME
paper. That description was the starting point, not an assumption. This project runs
at a different scale, on a different training set, with its own hyperparameter
search over width, depth, activation and learning rate. That search converged
independently on the same configuration the paper reports. The architecture used
here is therefore both the paper's answer and the search's answer, arrived at twice
by different routes.

The paper describes two models. The entry already on the Matbench Discovery
leaderboard under the name
[GNoME](https://matbench-discovery.materialsproject.org/models/gnome) is the other
one — a NequIP-type interatomic potential that performs structure relaxation,
submitted as `IS2RE-SR` with targets `EF_G`. The structural GNN benchmarked here
predicts energy directly from an unrelaxed input and had not previously been
submitted.

---

## Results

WBM test set, 256,963 structures. Metrics below are
**ingestion-recomputed by the Matbench Discovery pipeline** and are authoritative
(they supersede any earlier figures from local evaluation).

| Subset            | F1    | DAF   | Precision | Recall | MAE (eV/atom) | RMSE  | R²    |
| ----------------- | ----- | ----- | --------- | ------ | ------------- | ----- | ----- |
| Full test set     | 0.566 | 2.522 | 0.433     | 0.818  | 0.084         | 0.141 | 0.387 |
| Unique prototypes | 0.558 | 2.753 | 0.421     | 0.826  | 0.086         | 0.141 | 0.412 |

Confusion matrix, full test set: TP 36,061 · FP 47,274 · TN 165,597 · FN 8,031 ·
26 missing predictions.

### Ranking among IS2E direct-prediction models

The Matbench Discovery leaderboard includes models of all types (IS2RE-SR with
structure relaxation, IS2E direct predictors, etc.). The overall ranking (45th of
~52) mixes models that perform relaxation at inference — scoring F1 0.80–0.93 — with
direct predictors that skip relaxation entirely.

Within the **IS2E direct-prediction cohort** (`test_task: IS2E`, `targets: E`), the
picture is different. These six models all predict formation energy from an
unrelaxed input with no relaxation step:

| Model       | F1    | MAE (eV/atom) | R²     | Leaderboard |
| ----------- | ----- | ------------- | ------ | ----------- |
| ESNet       | 0.568 | 0.107         | −0.148 | [link](https://matbench-discovery.materialsproject.org/models/esnet) |
| **EMA-GNN** | **0.566** | **0.084** | **0.387** | [link](https://matbench-discovery.materialsproject.org/models/ema-gnn) |
| ALIGNN      | 0.565 | 0.092         | 0.274  | [link](https://matbench-discovery.materialsproject.org/models/alignn) |
| MEGNet      | 0.513 | 0.128         | −0.277 | [link](https://matbench-discovery.materialsproject.org/models/megnet) |
| CGCNN       | 0.510 | 0.135         | −0.624 | [link](https://matbench-discovery.materialsproject.org/models/cgcnn) |
| Voronoi RF  | 0.344 | 0.141         | −0.316 | [link](https://matbench-discovery.materialsproject.org/models/voronoi-rf) |

EMA-GNN ranks **2nd by F1** (within 0.003 of both ESNet and ALIGNN) and **1st by
MAE and R²** in this cohort as of Aug, 2026. Every value in this table is sourced from the
corresponding model's YAML in the
[Matbench Discovery repository](https://github.com/janosh/matbench-discovery/tree/main/models)
and can be independently verified.

---

## Model

| Property     | Value                                      |
| ------------ | ------------------------------------------ |
| Task         | `IS2E` — initial structure to energy       |
| Target       | Formation energy per atom (`E`)            |
| Training set | Materials Project 2022, relaxed structures |
| Parameters   | 2,209,793                                  |
| Ensemble     | 6 independently seeded models              |

Each message-passing block updates edge, node and global features in sequence.
Edge-to-node messages are normalized by the dataset-average adjacency, so message
magnitude does not scale with coordination number.

```text
hidden_dim              256
n_layers                3
graph cutoff            4.0 Å
edge features           64 Gaussian RBF, r_min 0.0 Å, r_max 5.0 Å
node features           one-hot element encoding, 100 elements
activation              SiLU
readout                 linear projection of final global feature to scalar
```

### Training recipe

```text
optimizer               Adam
learning rate           5.5e-4, LinearLR to 0.1× final
epochs                  500
batch size              128 × 2 accumulation steps (effective 256)
gradient clip           1.0
loss                    L1 on standardized targets
EMA decay               0.999
early stopping          disabled
weights evaluated       EMA
```

### Inference

Twenty-point isotropic volume test-time augmentation per structure (0.8–1.2×
volume, linear in volume), aggregated by **minimum** per model, then by **median**
across the six models. This is the paper's protocol.

---

## How the project got here

### 1. Baseline — a plain 200-epoch run

A single model, 200 epochs, no additions. The purpose was a working end-to-end
pipeline and a first number to measure everything else against.

### 2. Hyperparameter search on a stratified subset

Rather than assume the paper's architecture transferred to a smaller dataset, an
anchored coordinate search — one axis at a time, carrying the current best forward
— was run on Kaggle T4×2 GPUs over a stratified subset built specifically for
search. WBM was withheld throughout, and the stratified set never entered final
model training.

Activation, MAE meV/atom: SiLU **31.5** · GELU 33.0 · Mish 34.3.
Learning rate: 5.5e-4 beat 1e-4, 3e-4 and 1e-3.
Depth: 3 layers won. Width: one hidden layer per MLP won.

Every winning value matched the configuration reported in Merchant et al., despite
this search running on a different, reduced-scale dataset. That convergence is the
basis for calling this model GNoME-inspired rather than GNoME-derived: the
configuration was tested here on its own terms and independently selected.

### 3. Ablating the architecture

Three single-seed runs at 200 epochs on the earlier training loop (no EMA, no
gradient accumulation, early stopping enabled), each removing one component with
everything else fixed. Held-out Materials Project MAE, meV/atom:

| Configuration                   | MAE   | Parameters |
| ------------------------------- | ----- | ---------- |
| Full configuration              | 24.42 | 2,209,793  |
| Two message-passing layers      | 25.14 | 1,487,361  |
| Hidden dimension 128            | 25.19 | 564,225    |
| Adjacency normalization removed | 25.44 | 2,209,793  |

Every removal degraded accuracy, and removing adjacency normalization was the most
damaging despite leaving parameter count unchanged. The margins are small, however,
and each ablation is a single seed — comparable in size to the 1.1 meV spread across
the final ensemble's six seeds. The ordering is consistent with the components being
useful; individual differences should not be read as significant.

### 4. Scaling down deliberately — 6 seeds, 500 epochs

The paper trains a 10-member ensemble for 1,000 epochs each. This project had a
single RTX 4070 Ti, so the ensemble is **6 seeds at 500 epochs**. This is the
largest scope reduction in the project and is stated plainly rather than buried.

### 5. Adding EMA weight averaging

Exponential moving average over the weights is **an addition, not part of the
published recipe**. A single-seed 500-epoch run recorded a best EMA validation MAE
of 23.77 meV/atom at epoch 386 against a final-epoch raw MAE of 24.31 — a gain of
0.54 meV/atom. EMA continued improving past the point where the raw weights
plateaued, which is why early stopping was disabled for the final ensemble. Decay
0.999 was compared against 0.99 and won.

### 6. Test-time augmentation — min, not mean

The paper's 20-point volume-scaled TTA was implemented, initially aggregating the
20 predictions per structure by **mean**, which is the intuitive choice. The methods
text specifies the **minimum**. Switching to min flips the systematic bias and
substantially improves the discovery metrics. The lesson was to read the methods
section twice.

### 7. Training the ensemble

All six seeds completed serially on the 4070 Ti with no multi-GPU tricks. Per-seed
held-out MAE, meV/atom: **23.32 / 24.02 / 24.09 / 24.33 / 23.78 / 24.41** — a spread
of 1.1 meV with no diverged run.

### 8. WBM evaluation — and a second aggregation bug

The first full ensemble evaluation combined the six models' predictions with a
**mean**. The paper's methods specify a **median**, chosen because a GNN can be
badly wrong on out-of-distribution inputs and a single failure poisons a mean in a
way it cannot poison a median. The bug was found, fixed, and the evaluation re-run.

### 9. Diagnosing the low classification score

Three distinct causes were identified, rather than a single "the model is bad":

- **Domain shift** — trained on relaxed structures, evaluated on unrelaxed ones.
  Min-aggregated TTA narrows this but does not close it.
- A meaningful share of false positives are structures far above the hull, where the
  model is confidently wrong rather than marginally wrong.
- False-positive rate rises with element count, consistent with training data that
  is thinner on quaternary and quinary compositions.

Post-hoc bias correction — shifting all predictions by a constant — was tested and
does not meaningfully improve the classification score. This is a retraining
problem, not a recalibration problem.

### 10. Finding the scoring error

The classification metric itself turned out to be wrong. See
[Note on scoring](#note-on-scoring) below. Correcting it moved discovery F1 from
approximately 0.34 to 0.56 with no change to the model.

### 11. MLIP pre-relaxation

Relaxing the WBM inputs with a pretrained universal interatomic potential before
prediction, as a way to attack the domain gap at the input level rather than by
retraining. Both MACE-MP-0 and CHGNet were evaluated. Results below.

### 12. Leaderboard submission

Predictions, checkpoints and metadata submitted to the Matbench Discovery
leaderboard via [PR #387](https://github.com/janosh/matbench-discovery/pull/387),
reviewed and merged on 2026-08-12. The prediction file was independently regenerated
on a second machine from the archived checkpoints and re-scored, reproducing the
submitted metrics to four decimal places.

---

## Note on scoring

Two ways of turning a formation-energy prediction into a stability classification
exist, and they produce very different numbers.

**Naive thresholding** classifies a material as stable whenever the predicted
formation energy is negative (`e_form_pred <= 0`). This conflates thermodynamic
stability with exothermicity. Roughly 88% of WBM has negative formation energy,
while only about 16.7% actually lies on or below the convex hull, so the criterion
produces near-unity recall and precision near the base rate regardless of model
quality.

**Hull-displacement scoring** — used by the Matbench Discovery benchmark and by all
figures in this README — holds the DFT convex hull fixed and lets the
formation-energy error shift each material across the stability line:

```
each_pred = each_true + e_form_pred - e_form_dft
```

The reference columns are the MP2020-corrected ones
(`e_above_hull_mp2020_corrected_ppd_mp`, `e_form_per_atom_mp2020_corrected`), not the
WBM-native columns. Using the native columns silently produces wrong numbers because
the MP2020 corrections shift the hull itself.

The implementation is `scripts/eval_discovery_mbd.py`, which calls
`matbench_discovery.metrics.stable_metrics` directly. An earlier script in this
repository (`scripts/f1_wbm.py`) used the naive method and is superseded.

---

## MLIP pre-relaxation

Relaxing the WBM inputs with a pretrained universal interatomic potential before
prediction was tested as a way to narrow the relaxed-train / unrelaxed-test domain
gap. Both **MACE-MP-0** and **CHGNet** were evaluated as the relaxation stage. This
is a preprocessing step in front of the unchanged ensemble
(`scripts/relax_wbm_with_mlip.py`); the model itself is not retrained or modified.

Relaxed-structure predictions are committed under `runs/ensemble_relaxed/`.

| Input                 | MAE (eV/atom) | R²    | F1    |
| --------------------- | ------------- | ----- | ----- |
| Unrelaxed (submitted) | 0.084         | 0.387 | 0.566 |
| MACE-MP-0 relaxed     | 0.079         | 0.485 | 0.560 |
| CHGNet relaxed        | 0.079         | 0.499 | 0.558 |

Pre-relaxation improves regression substantially — MAE drops from 0.084 to 0.079
and R² rises from 0.387 to 0.485–0.499 — while leaving F1 essentially unchanged at
approximately 0.56. The domain gap therefore limits energy accuracy without being
the binding constraint on stability classification. Both MLIPs produce nearly
identical downstream results despite different architectures, which suggests the
improvement saturates at the quality of the relaxed geometry rather than depending
on the relaxation method.

Only the unrelaxed predictions are submitted to the leaderboard, since the benchmark
task is `IS2E` — energy from the initial structure.

---

## Deviations from the reference description

- **Six ensemble members rather than ten**, and **500 training epochs rather than
  1000.** Both are compute-driven and are the largest scope reductions here.
- **EMA weight averaging is an addition**, not present in the published recipe.
- Training uses Materials Project 2022 relaxed structures. Trajectory data does not
  enter this model.
- The hyperparameter configuration was independently searched on this project's own
  data rather than adopted from the paper, and converged to the same values.

---

## Repository layout

```
analysis/               Per-stage analysis scripts (run(repo_root, show) -> summary dict)
  stage1_baseline/
  stage1_stageA/
  stage1_ensemble/
configs/
models/
  ema-gnn/              Matbench Discovery submission metadata
notebooks/
  analysis_gnome_struct.ipynb       Main analysis notebook
  bias_correction_diagnostic.ipynb
  wbm_FP_diagnosis.ipynb
results/                Saved plots and summary tables per stage
runs/
  ensemble/             6-seed checkpoints, predictions, metrics
  ensemble_relaxed/     MLIP-relaxed input predictions
scripts/
  eval_discovery_mbd.py             Official Matbench Discovery scoring
  eval_wbm.py                       Single-model WBM evaluation, 20-pt volume TTA
  eval_wbm_ensemble.py              6-seed ensemble WBM evaluation (median aggregation)
  eval_wbm_ensemble_from_preds.py   Fast re-aggregation from existing per-seed CSVs
  relax_wbm_with_mlip.py            MLIP pre-relaxation of WBM structures
  f1_wbm.py                         SUPERSEDED — see Note on scoring
src/gnome/
  train_ensemble_seed.py
  graphs.py, model.py
run_ensemble.ps1        Orchestrates 6-seed ensemble training
```

---

## Artifacts and citation

Checkpoints and the submitted prediction file are archived on Figshare:

**[https://doi.org/10.6084/m9.figshare.33111509](https://doi.org/10.6084/m9.figshare.33111509)**

- Prediction CSV — WBM discovery predictions, `e_form_per_atom`, 256,963 rows
- Checkpoint archive — six EMA-averaged seed checkpoints (`seed_0/best.pt` …
  `seed_5/best.pt`)

There is no accompanying paper. The Figshare item is the citable artifact for this
work; please cite the DOI above if you use the checkpoints or predictions.

Checkpoints are released under CC-BY-4.0. Code in this repository is MIT-licensed.

Loading a checkpoint requires `strict=False`, because `avg_adjacency` is registered
as a buffer but supplied through the constructor from the checkpoint's `stats` dict
and is therefore absent from `model_state`.

---

## Links

| Resource | URL |
| --- | --- |
| Leaderboard page | [matbench-discovery.materialsproject.org/models/ema-gnn](https://matbench-discovery.materialsproject.org/models/ema-gnn) |
| Merged PR | [janosh/matbench-discovery#387](https://github.com/janosh/matbench-discovery/pull/387) |
| Figshare DOI | [10.6084/m9.figshare.33111509](https://doi.org/10.6084/m9.figshare.33111509) |
| Reference paper | [Merchant et al., Nature 624 (2023)](https://doi.org/10.1038/s41586-023-06735-9) |
| Matbench Discovery paper | [Riebesell et al., Nat Mach Intell 7 (2025)](https://doi.org/10.1038/s42256-025-01055-1) |

---

## Compute

- RTX 4070 Ti — ensemble training, one seed at a time
- Kaggle T4×2 — hyperparameter search and MLIP relaxation
- Colab Pro — overflow and resumed runs
- Local machine — analysis and notebooks
- Quantum Espresso (PBE) — planned for the active-learning stage, as no VASP
  license is available

---

## Next steps

- **Retraining on MPtrj trajectory data.** The MLIP pre-relaxation result indicates
  the relaxed-train / unrelaxed-test gap is not correctable at the input level, since
  relaxing the inputs improves regression accuracy while leaving stability
  classification essentially unchanged. Exposing the model to intermediate,
  non-relaxed geometries during training addresses the gap where it originates.
- **An active-learning loop.** Generate candidate structures, filter with the
  ensemble, verify survivors with DFT (Quantum Espresso), fold results back into
  training. Scoped to a narrow chemical system.

---

## Acknowledgements

[Rajib Chandra Das](https://github.com/RajibTheKing), Doctoral Researcher in the
Real-Time / Embedded Systems Group at Christian-Albrechts-Universität zu Kiel,
provided the GPU on which the ensemble was trained, ran the independent verification
of the submitted predictions, and contributed debugging support across the training
and evaluation pipelines — including catching issues in the ensemble training and
evaluation scripts. This project would have moved considerably more slowly without
that GPU and that second pair of eyes.
