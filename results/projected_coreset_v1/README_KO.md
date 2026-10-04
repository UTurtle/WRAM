# 투영 coreset 고정 대조군

같은 floor-only Wiener/RDP4 임베딩을 재사용해 Gaussian 6144→256 투영 후 FPS128을 비교했습니다. seed 20260613, centroid-nearest start, exact greedy FPS. 투영으로 선택한 ID의 **원래 8×768 descriptor**로 scale을 재추정하고 매칭했습니다.

| 선택 방식 | ratio k | Development | Evaluation |
|---|---:|---:|---:|
| 원래 공간 FPS128 | 1 | 67.690781 | 67.360102 |
| 원래 공간 FPS128 | 4 | 68.110483 | 66.486401 |
| 256D 투영 후 FPS128 | 1 | 67.380961 | 67.775599 |
| 256D 투영 후 FPS128 | 4 | 68.347419 | 67.433381 |

주 비교 k=1에서 Development −0.310점, Evaluation +0.415점입니다. k=4에서는 각각 +0.237점, +0.947점입니다. 단일 투영 seed의 사후 비교이며 보편적 우월성이나 독립 test 성능을 증명하지 않습니다. 이 결과로 현재 주 설정을 자동 변경하지 않았습니다.

추가 encoder forward 없이 26.0초에 비교 예측을 계산했습니다. 모든 정상 선택과 scale을 동결한 후 query를 읽었고, 모든 예측을 동결한 후 별도 평가에서 정답을 읽었습니다. 전체 정답은 이전 작업에서 이미 관측된 공개 Dev/Eval입니다.

실행: `python scripts/score_projected_coreset.py --run SOURCE_RUN --output NEW_RUN` 후 `wram evaluate --run NEW_RUN --labels labels.csv --output NEW_EVALUATION`.
