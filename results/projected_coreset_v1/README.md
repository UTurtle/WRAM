# Fixed projected-coreset comparison

This comparison reuses the same floor-only Wiener/RDP4 embeddings and applies a 6,144-to-256-dimensional Gaussian projection before FPS128. It uses seed 20260613, a centroid-nearest start, and exact greedy FPS. Scales and matches are recomputed using the **original 8-by-768 descriptors** of the recordings selected in projected space.

| Selection policy | Ratio k | Development | Evaluation |
|---|---:|---:|---:|
| Original-space FPS128 | 1 | 67.690781 | 67.360102 |
| Original-space FPS128 | 4 | 68.110483 | 66.486401 |
| Projected 256-D FPS128 | 1 | 67.380961 | 67.775599 |
| Projected 256-D FPS128 | 4 | 68.347419 | 67.433381 |

At the primary k=1 setting, projection changed Development by -0.310 and Evaluation by +0.415 percentage points. At k=4, the changes were +0.237 and +0.947 points. This post-hoc comparison uses one projection seed and does not establish general superiority or untouched-test performance. The primary configuration was not changed based on these results.

Comparison predictions were computed in 26.0 seconds without additional encoder passes. All normal selections and scales were fixed before reading queries; labels were read in a separate evaluation after predictions were fixed. The public Development and Evaluation labels had already been observed in earlier work.

Run `python scripts/score_projected_coreset.py --run SOURCE_RUN --output NEW_RUN`, then `wram evaluate --run NEW_RUN --labels labels.csv --output NEW_EVALUATION`.
