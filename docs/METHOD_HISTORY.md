# Method history and selection disclosure

- The earlier reference-selection framework used Gaussian projection6144→256 followed by centroid-start farthest-first selection of128 recording IDs. Selection used projected vectors; scoring used original bands.
- The current Wiener/BEATs framework uses unprojected FPS in6144-D. The primitive is still greedy k-center/farthest-first coreset selection.
- The historical Wiener control inherited diagonal loading `max(0.005*(Pn+Pf),1e-12)` from a fixed MVDR experiment. RDP4, FPS128 and ratio k=1 were retained after prior Dev/Eval inspection.
- A subsequent four-condition loading ablation produced:

| c | Development | Evaluation |
|---:|---:|---:|
|0|67.690781|67.360102|
|0.0005|67.665818|67.308004|
|0.005|67.779719|67.023654|
|0.05|66.529233|67.257690|

These are archived results used to decide to simplify the filter. The other three coefficients remain historical measurements. The new default removes the power-dependent term and retains additive epsilon1e-12. On 2026-09-23 the installed public package re-extracted all 14,400 recordings and reproduced the c=0 result: Development 67.690781 and Evaluation 67.360102. All 2,400 query descriptors were bit-identical to the archived c=0 control, all selected recording IDs agreed, and the maximum prediction difference was 9.93e-14. See results/floor_only_v1/ for the new measurements and validation. The change does not establish that c=0 is universally optimal or independently selected from evaluation data.

All/Random/FPS compare reference policies. Inherited/fixed/rematch compare normal-scale and query-assignment policies. A fixed-ID representation crossover changes descriptors while fixing the recording IDs. These are distinct interventions; do not label all of them merely FPS.

The historical results above were obtained before this code cleanup. Existing manuscript versions and result bundles remain unchanged.
