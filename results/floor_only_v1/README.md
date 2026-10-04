# Floor-only reproduction with the public implementation

On 2026-09-23, the installed public package processed 14,400 raw audio recordings in a fresh run outside the development repository. Metrics were computed separately using the official labels.

## Main results

| Configuration | Development | Evaluation |
|---|---:|---:|
| Near / mean / All1000 | 57.934026 | 62.166333 |
| Wiener / mean / FPS128 | 66.590913 | 65.412164 |
| Wiener / RDP4 / All1000 | 64.406574 | 66.139873 |
| Wiener / RDP4 / FPS128 | 67.690781 | 67.360102 |
| Wiener / RDP4 / FPS64 | 65.826965 | 64.946719 |
| Wiener / RDP4 / FPS256 | 67.011995 | 66.845477 |

All rows use ratio scoring with k=1, scales fitted within the selected bank, and rematching. The CSV files contain the full set of configurations and three metric components per machine.

## Reproduction checks

- The full raw-audio run took 283.227 seconds on one H100 NVL. No new model was trained.
- All 2,400 query descriptors matched the earlier c=0 control bit for bit.
- FPS128 selected the same recording IDs for all 12 machines.
- The maximum final-score difference was 9.93e-14; an independent ROC calculation differed from scikit-learn by less than 2.23e-16.
- All normal banks were fixed before processing queries; labels were read by a separate evaluator only after predictions were fixed.
- Seven core checks passed in the installed package. These numeric checks are not a complete external audit.

## Interpretation and scope

- Floor-only prediction was adopted after viewing an earlier loading ablation on Development and Evaluation. These are not untouched test results.
- Only the power-dependent loading term was removed from the denominator of H; the numerical epsilon of 1e-12 remains.
- Near to Wiener increased the point estimates of anomaly-probe performance and neighbor purity for 11 of 12 machines, and normal-centroid separation margin for 10 of 12.
- Wiener mean to RDP4 had median linear CKA of 0.99849, a smaller representation change; probe performance did not improve on every machine.
- Probes are separate post-hoc diagnostics. They did not train the ASD model or reference bank.
- Only BEATs was rerun. Earlier manuscript tables for other encoders should not be updated from these results.
- Projected coreset selection is a separate comparison and does not change the primary configuration.
