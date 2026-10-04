# WRAM: Wiener Residual Audio Matching

Code accompanying a manuscript submitted to ICASSP 2027 (under review).

A reproducible frozen-BEATs anomalous sound detection system and its matched controls. This version uses **floor-only Wiener prediction**: `H = E[near * conj(far)] / (Pfar + 1e-12)`. There is no power-dependent loading coefficient in the primary implementation.

The primitives are established methods. Our implementation studies their integration and the coupled effects of representation, reference selection, scale fitting, and matching. **This is research code for retrospective DCASE 2026 comparisons, not an official challenge submission or an untouched evaluation.**

## Pipeline

1. Read synchronized near/far 16-kHz stereo audio (first10s; right-pad if needed).
2. Estimate the complex far-to-near predictor within each recording, subtract its prediction, and reconstruct the waveform with iSTFT.
3. Run original pretrained **BEATs iter3**, frozen, retaining the native62×8×768 grid.
4. Pool time using existing RDP with exponent4, then normalize each of8 band descriptors.
5. Concatenate the8 descriptors for deterministic **unprojected FPS**. Select128 actual recording IDs shared across bands.
6. Fit self-excluded normal-neighbor scales inside the selected bank. Score each query by the minimum distance/scale per band, then average8 band scores.

Near/far coherence is not a clean noise label: the removed component may contain machine sound. The encoder is not trained on DCASE query data.

## Setup

Python3.11 and matching PyTorch/torchaudio2.5.x. Install a PyTorch build suitable for your machine, then:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
wram --help
```

`requirements-tested.txt` records the actual tested versions. CUDA BF16, batch16, padded tails and float64 signal processing/pooling match the reference execution. CPU/float32 can be explicitly configured for portability but are **not** certified numerically equivalent to that CUDA execution. `configs/floor_only.json` pins the pretrained checkpoint SHA256 and current recipe.

Download original **BEATs_iter3** from the [official BEATs repository](https://github.com/microsoft/unilm/tree/master/beats). Do not substitute the AudioSet-finetuned or iter3+ checkpoint. Obtain the DCASE2026 Task2 recordings and official labels separately from the [challenge](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring). Audio, checkpoints, labels and feature caches are not distributed here.

## Data manifests

Inference uses JSONL records, preserving row order:

```json
{"id":"Dev/fan/train/record.wav","path":"/external/data/fan/train/record.wav","split":"Dev","machine":"fan","part":"normal","domain":"source"}
{"id":"Dev/fan/test/record.wav","path":"/external/data/fan/test/record.wav","split":"Dev","machine":"fan","part":"query"}
```

Relative audio paths resolve relative to the manifest. Query labels are rejected from inference manifests. Training `part=normal` must contain only official normal recordings. There must be990 source+10 target normals and200 queries per machine. Filenames and historical metadata may themselves expose labels; manifest separation does not claim researcher blinding.

Create manifests from `root/machine/{train,test}/*.wav`:

```bash
wram index --root /external/development --split Dev --output dev.jsonl
wram index --root /external/evaluation --split Eval --output eval.jsonl
cat dev.jsonl eval.jsonl > recordings.jsonl
```

## Execute

The example script derives the repository location and works from another directory:

```bash
bash experiments/001_floor_only/run.sh /external/recordings.jsonl /external/BEATs_iter3.pt /external/wram-run-001
```

Or use the installed CLI:

```bash
wram smoke --manifest recordings.jsonl --checkpoint /external/BEATs_iter3.pt --config configs/floor_only.json --output artifacts/smoke-001
wram run --manifest recordings.jsonl --checkpoint /external/BEATs_iter3.pt --config configs/floor_only.json --output artifacts/run-001
```

Outputs must be new directories. Every run stores hashes, the executed config, an executable source snapshot, full pooled descriptors, selected IDs, fitted scales, prediction traces and a prediction freeze. All normal models are fitted before query inference. `run` does not accept labels and computes no accuracy metrics. Large artifacts are Git-ignored.

The default ablation suite includes:

- Near/Wiener/feature-residual × mean/RDP4; feature residual is `unit(p_near - 0.5 p_far)`.
- All1000 and FPS128; Wiener/RDP4 additionally FPS64, FPS256 and domain-count-matched Random128 over three fixed seeds.
- Raw distance, subtractive normalization, ratio k=1 and k=4; inherited scales, fixed-winner refit and rematching.
- Frozen-ID mean/RDP4 crossover with separately refitted scales and winners.

Set `suite` to `main` in a copied config for a smaller Wiener-only run. FPS is recomputed per representation. The same selected IDs are used for scorer contrasts. This is a fixed control suite, not an automatic method-selection loop.

## Evaluate and analyze

Labels use a separate CSV with columns `id,split,machine,domain,label` (normal0, anomalous1). `scripts/labels_from_official.py` converts the official metadata using a strict filename join.

```bash
python scripts/labels_from_official.py --manifest recordings.jsonl --dev-labels /external/dev-official --eval-labels /external/eval-official --output labels.csv
wram evaluate --run artifacts/run-001 --labels labels.csv --output artifacts/evaluation-001
wram analyze --run artifacts/run-001 --labels labels.csv --output artifacts/geometry-001
```

Source/target AUC uses normals of that domain against **all** anomalies. Standardized pAUC uses maxFPR0.1. The aggregate is the harmonic mean over21 Development or15 Evaluation components. A separate rank/ROC implementation cross-checks sklearn. Numeric checks do not constitute complete external audit certification.

`analyze` is explicitly post-hoc: CKA,10-NN overlap/purity, normal-centroid separation and5-fold logistic probes. Probes use query labels for diagnostic cross-validation; they never fit or replace the ASD system.

## FPS, coreset, and compact references

A coreset is the retained subset; FPS is the greedy selection algorithm; the compact reference bank contains the selected real recording descriptors. There are no learned prototype vectors. Current FPS uses the full6144-D concatenation, CPUfloat32 distance calculations, a centroid-nearest start and deterministic first-index tie resolution.128 is the number of retained recordings, **not** a projection dimension. Historical code used a6144→256 Gaussian projection; this implementation does not. See [method history](docs/METHOD_HISTORY.md) and [attribution](docs/ATTRIBUTION.md).

The [PatchCore implementation](https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/sampler.py) provides projected greedy coreset methods. The primary unprojected FPS policy has its own centroid start; the separate official sampler control below executes PatchCore's unmodified approximate selection code. Neither policy is a new coreset algorithm, and coverage does not guarantee the best detection score.

## Reproducibility and limits

The floor-only default was adopted **after** observing the four-level loading ablation on public Dev/Eval. That decision is recorded rather than presented as an originally preregistered rule. Prior loaded results remain historical; they must not be silently relabeled as floor-only. The main checkpoint and data are external; all scientific outputs come from the package code, not from imported private score files.

The installed package was executed from outside this repository on 2026-09-23: **Development 67.690781 / Evaluation 67.360102** for Wiener/RDP4, FPS128, ratio k=1. Full raw-audio execution took 283.2 seconds on one H100 NVL. The [result bundle](results/floor_only_v1/README_KO.md) records all matched controls and the independent numeric and historical parity checks. Transfer encoders from older manuscripts were not rerun here.

## Actual-data diagram assets

Optional plotting dependencies: `python -m pip install -e '.[visuals]'`.

```bash
python scripts/prepare_reference_visuals.py --run artifacts/run-001 --output artifacts/fan-tiles-001
python scripts/render_reference_tiles.py --folder artifacts/fan-tiles-001
```

This makes text-free transparent PNGs plus labelled contact sheets, source hashes, selection IDs and actual arrays. The projected control applies the historical Gaussian 6144-to-256 selection recipe to the same normal descriptors. The plot's PCA is a separate display operation. Projection chooses recording IDs; matching still uses their original band descriptors. These images do not imply that projected coreset selection is part of the unprojected primary system.

The completed [fixed projected control](results/projected_coreset_v1/README_KO.md) scored 67.380961 / 67.775599 at ratio k=1 (Development / Evaluation), compared with 67.690781 / 67.360102 for unprojected FPS128. This is one post-hoc seed, not a promotion of the projected policy. It can be scored from cached descriptors without any encoder forwards:

```bash
python scripts/score_projected_coreset.py --run artifacts/run-001 --output artifacts/projected-001
wram evaluate --run artifacts/projected-001 --labels labels.csv --output artifacts/projected-evaluation-001
```

New code: MIT. Vendored BEATs retains Microsoft's license, notices and exact local snapshot hashes. See `docs/BEATS_SOURCE.json`. No remote publication is performed by the scripts.

## Matched reference-budget controls

[Original-space versus projected-space FPS at K=64/128/256](results/reference_budgets_v1/README_KO.md) uses the same frozen Wiener/RDP4 descriptors, one fixed historical projection seed, and ratio k=1. Scales and matches are recomputed within each selected bank; the primary setting is unchanged.

```bash
python scripts/score_reference_budgets.py --run artifacts/run-001 --output artifacts/budget-001
wram evaluate --run artifacts/budget-001 --labels labels.csv --output artifacts/budget-evaluation-001
```

## Official PatchCore sampler control

The [official PatchCore approximate greedy sampler comparison](results/patchcore_official_v1/README_KO.md) uses Amazon Science's pinned, unmodified sampler code under `vendor/patchcore-inspection/` with its Apache-2.0 license. Its default 128-dimensional random linear projection and ten random starting points select normal recording IDs from the same 6,144-dimensional WRAM descriptors. Both RNGs use seed 20260613 per machine. Scales and BEAM matching are recalculated in the original eight-band descriptor space. This replaces neither the main original-space FPS policy nor the historical 256-dimensional Gaussian control.

From this repository, install the optional progress-bar dependency with `pip install -e '.[patchcore]'`, then run:

```bash
python scripts/score_patchcore_budgets.py --run artifacts/run-001 --output artifacts/patchcore-001
wram evaluate --run artifacts/patchcore-001 --labels labels.csv --output artifacts/patchcore-evaluation-001
```

The new script and vendored sampler live in the repository.

## Code style

Authored Python code follows a 79-character Black layout with Ruff PEP 8
checks. Vendor code, generated artifacts and historical execution snapshots
are excluded. From the repository root:

```bash
black --check .
ruff check .
python -m pytest -q
```
