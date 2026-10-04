# Normal-reference selection with the official PatchCore sampler

This comparison replaces the historical 256-dimensional Gaussian RP-FPS control with the official PatchCore selection algorithm. It uses Amazon Science's unmodified `ApproximateGreedyCoresetSampler` from `patchcore-inspection` commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad` (`vendor/patchcore-inspection/sampler.py`, SHA256 `39612cd2b486865ece304348f740f4bc700a1c2e8684fbab3c2ed016c7801f2d`). The original Apache-2.0 license is included.

| Normal-reference selection | K | Dev hmean (%) | Eval hmean (%) |
|---|---:|---:|---:|
| Original-space FPS | 64 | 65.826965 | 64.946719 |
| Official PatchCore sampler | 64 | 66.734871 | 64.981959 |
| Original-space FPS | 128 | **67.690781** | **67.360102** |
| Official PatchCore sampler | 128 | 67.022330 | 66.883851 |
| Original-space FPS | 256 | 67.011995 | 66.845477 |
| Official PatchCore sampler | 256 | 65.791411 | 65.844041 |

At K=128, the official sampler scored 0.668452 points below the primary FPS setting on Development and 0.476250 points below on Evaluation. It did not meet the previously recorded criterion of a gain of at least two percentage points on both splits, so the primary setting was retained. This one-seed, post-hoc comparison does not establish a universal ranking of selection methods.

## Execution conditions

- Input: the same 6,144-dimensional concatenation of eight 768-dimensional normal-recording descriptors from floor-only Wiener residuals, the original frozen BEATs iter3 encoder, and RDP4 pooling; 1,000 recordings per machine. Existing query descriptors were reused.
- Sampler: the official default 128-dimensional random `torch.nn.Linear(bias=False)` projection, ten random starting points, and approximate greedy updates. PyTorch and NumPy RNG states were set to seed 20260613 per machine. Execution used CPU, torch 2.5.1+cu121, and NumPy 2.1.3. Selected recording IDs were saved using the official source's `_reduce_features` and `_compute_greedy_coreset_indices` methods.
- Bank sizes: the first 64, 128, and 256 selections in a 256-recording order. Separate runs with the same RNG state produce the same prefixes.
- Scoring: selection uses concatenated descriptors, while local scales and band-wise BEAM scores use the original eight 768-dimensional descriptors. Scales and query matches are recomputed for every selected bank. This is not a reproduction of the full PatchCore image detector.
- Procedure: all normal banks were fixed before reading queries. A separate evaluator read labels after predictions for 2,400 queries were fixed. There was no training or encoder rerun; the public Development and Evaluation splits had already been observed.

Running the official sampler's `run()` on actual Development/fan normal data at K=128 yielded exactly the same 128 IDs as the stored selection. Checks covered 24 bank prefixes (K=64 and K=128 across 12 banks), unique recording IDs, harmonic means for 36 machine-component rows, and the aggregate harmonic means over 21 Development and 15 Evaluation components. The maximum difference between an independent ROC calculation and scikit-learn was 2.22e-16. These are reproduction checks for the current code and hashes, not a complete external audit.

`MACHINE_METRICS.csv`, `AGGREGATES.csv`, `VALIDATION.json`, and `EXECUTION.json` are the result files. The full frozen model, predictions, and source snapshot are in `artifacts/patchcore_official_001/`; label-based evaluation is in `artifacts/patchcore_official_evaluation_001/`. The v37 RP-FPS results remain a separate historical experiment.
