# Reference selection and bank-size comparison

All configurations use the same floor-only Wiener/RDP4 descriptors and ratio scoring with k=1. Only the selection space and number of references change; scales and matches are recomputed within each selected bank.

| Selection policy | K | Development | Evaluation |
|---|---:|---:|---:|
| Original-space FPS | 64 | 65.826965 | 64.946719 |
| Original-space FPS | 128 | 67.690781 | 67.360102 |
| Original-space FPS | 256 | 67.011995 | 66.845477 |
| Projected 256-D FPS | 64 | 66.917282 | 67.717714 |
| Projected 256-D FPS | 128 | 67.380961 | 67.775599 |
| Projected 256-D FPS | 256 | 65.625967 | 67.683206 |

The projected policy uses the historical local 6,144-to-256-dimensional Gaussian projection, seed 20260613, and exact greedy FPS with a centroid-nearest start. For each policy, the K=64 and K=128 banks are prefixes of the K=256 selection order. Projection selects recording IDs only; scoring uses their original 8-by-768 descriptors. This does not reproduce the full PatchCore detector or its approximate sampler.

All 12 normal banks were fixed before queries were read. Labels were read in a separate evaluation after predictions for 2,400 queries were fixed. Development and Evaluation had already been observed. There was no additional training, encoder pass, or seed search, and the primary configuration was not changed.

The original-space FPS64/128/256 and projected FPS128 results matched 48 existing prediction vectors with maximum absolute error 6.062e-14. Twenty-four nested bank-prefix checks passed; the maximum difference between an independent ROC calculation and scikit-learn was 2.220e-16. These are numeric and reproduction checks, not a complete external audit.

## Interpretation

- In original space, K=128 has the highest aggregate on both splits. The projected policy also peaks at K=128, although its three Evaluation bank sizes are close.
- Similar aggregates can conceal opposite machine-level changes. From projected K=64 to K=128, ToothBrush drops from 61.75 to 56.92 while ToyDrone rises from 69.58 to 72.53.
- The effects of K and selection space vary by machine. This one-seed, post-hoc comparison does not establish a universally best policy or bank size.

## Reproduce

```bash
python scripts/score_reference_budgets.py --run SOURCE_FLOOR_ONLY_RUN --output NEW_RUN
wram evaluate --run NEW_RUN --labels labels.csv --output NEW_EVALUATION
```
