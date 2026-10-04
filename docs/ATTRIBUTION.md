# Attribution

WRAM integrates established operators. Algorithmic attribution is distinct from code provenance.

- Wiener residual: per-record linear far-to-near prediction/subtraction. Related DCASE2026 systems include [Chang et al.](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Chang_110_t2.pdf) and [Moradi/Shokriazar et al.](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Moradi_97_t2.pdf). These are prior examples, not a claim that their source code or exact old loading coefficient was copied.
- [BEATs](https://github.com/microsoft/unilm/tree/master/beats): Microsoft original iter3 checkpoint and inference architecture. The three vendored architecture files are unchanged local reproduction snapshots. License/notice text is retained. Current upstream master need not be byte-identical to the snapshot.
- [RDP pooling](https://arxiv.org/abs/2603.04605): existing relative-deviation weighting, gamma4. Mean and RDP operate on the same tokens.
- [BEAM](https://arxiv.org/abs/2603.13749): bandwise matching and localized score aggregation context.
- Greedy k-center/FPS: established farthest-first coverage selection. [PatchCore](https://arxiv.org/abs/2106.08265) is a related coreset-memory system, not the source of a new WRAM selection primitive. This package uses centroid-start unprojected FPS over recording descriptors rather than PatchCore's projected patch-feature sampler.
- Official PatchCore sampler control: [Amazon Science's `ApproximateGreedyCoresetSampler`](https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/sampler.py) is copied byte-for-byte at commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad` in `vendor/patchcore-inspection/sampler.py`, with its Apache-2.0 license. It selects recording IDs from WRAM descriptors; the audio representation and anomaly scorer remain WRAM's. The original-space FPS method remains the primary system.
- Subtractive variance-minimization control: Matsumoto et al., [Adjusting Bias in Anomaly Scores via Variance Minimization for Domain-Generalized Discriminative Anomalous Sound Detection](https://dcase.community/documents/workshop2025/proceedings/DCASE2025Workshop_Matsumoto_12.pdf), DCASE Workshop 2025. This implementation uses an explicitly donor-deleted normal-only calibration control.
- Linear CKA: Kornblith et al., [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414). Other geometry/probe diagnostics are post-hoc analyses.
- [DCASE2026 evaluator](https://github.com/nttcslab/dcase2026_task2_evaluator): metric definitions. This package independently implements and cross-checks AUC/pAUC; it does not redistribute or claim certification by that evaluator.

No component is renamed to imply primitive novelty. The repository contains the concrete integration and reproducible controlled comparisons.
