# WRAM

Code accompanying a manuscript submitted to ICASSP 2027 (under review).

WRAM is a normal-reference anomalous sound detector for DCASE 2026 Task 2. It uses synchronized near and far recordings to form a Wiener residual, extracts descriptors with a frozen BEATs encoder and RDP4 pooling, selects a compact normal reference bank, and scores queries with band-wise local-density scales.

## Requirements

- Python 3.11 and a matching PyTorch/torchaudio 2.5.x installation
- [DCASE 2026 Task 2 recordings and labels](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
- The original `BEATs_iter3.pt` checkpoint from [BEATs](https://github.com/microsoft/unilm/tree/master/beats)

The audio, labels, and checkpoint are not included in this repository.

## Run

Install the package, then index the downloaded data. Each data root should contain `machine/train/*.wav` and `machine/test/*.wav` folders.

```bash
python -m pip install -e .
wram index --root /path/to/development --split Dev --output dev.jsonl
wram index --root /path/to/evaluation --split Eval --output eval.jsonl
cat dev.jsonl eval.jsonl > recordings.jsonl
wram run --manifest recordings.jsonl --checkpoint /path/to/BEATs_iter3.pt --config configs/floor_only.json --output artifacts/run-001
```

To calculate the official metrics, join the separately downloaded labels after the run:

```bash
python scripts/labels_from_official.py --manifest recordings.jsonl --dev-labels /path/to/development-labels --eval-labels /path/to/evaluation-labels --output labels.csv
wram evaluate --run artifacts/run-001 --labels labels.csv --output artifacts/evaluation-001
```

Use new output paths for each run. `wram run` does not read labels.

## Results

The recorded Wiener/RDP4/FPS128 run achieved harmonic-mean scores of **67.69%** on development and **67.36%** on evaluation. These are retrospective comparisons, not an official challenge submission. Matched controls and detailed results are in the [result bundle](results/floor_only_v1/README.md).

Our code is MIT-licensed. Third-party code retains its original licenses; see [attribution](docs/ATTRIBUTION.md).
