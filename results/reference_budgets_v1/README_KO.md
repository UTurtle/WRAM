# 선택 방식 × reference budget 비교

동일한 floor-only Wiener/RDP4 descriptor와 ratio k=1 scoring을 사용했다. 선택 공간과 참조 개수만 바꾸고, 각 선택 bank 안에서 scale refit 및 rematching을 수행했다.

| 선택 방식 | K | Development | Evaluation |
|---|---:|---:|---:|
| 원래 공간 FPS | 64 | 65.826965 | 64.946719 |
| 원래 공간 FPS | 128 | 67.690781 | 67.360102 |
| 원래 공간 FPS | 256 | 67.011995 | 66.845477 |
| 256D 투영 후 FPS | 64 | 66.917282 | 67.717714 |
| 256D 투영 후 FPS | 128 | 67.380961 | 67.775599 |
| 256D 투영 후 FPS | 256 | 65.625967 | 67.683206 |

투영은 과거 로컬 구현 그대로 6144→256 Gaussian, seed 20260613, centroid-nearest exact greedy FPS를 썼다. 각 방식에서 K=64와128은 K=256 선택 순서의 prefix이다. 투영은 ID 선택에만 사용하고 score는 원래 8×768 descriptor로 계산한다. PatchCore의 전체 detector 또는 approximate sampler 재현이 아니다.

모든 12개 정상 bank를 동결한 후 query를 읽었고, 2,400개 query의 예측 동결 이후에 정답을 읽는 별도 평가를 실행했다. Dev/Eval은 이전에 이미 관측되었다. 추가 학습·encoder forward·seed 탐색은 없고 주 설정도 바꾸지 않았다.

원래 FPS64/128/256 및 투영 FPS128의 기존 48개 예측 벡터와 최대 절대오차 6.062e-14로 일치했다. 24개 nested bank-prefix 검사를 통과했고, 독립 ROC 구현과 sklearn의 최대 오차는 2.220e-16이다. 이는 해당 수치·재현 검사이지 전체 auditor 인증이 아니다.

## 해석

- 원래 공간에서는 K=128이 두 split aggregate 중 가장 높다. 투영 방식은 K=128이 가장 높지만 Eval에서는 세 budget의 차이가 작다.
- aggregate가 비슷해도 머신별 변화는 상쇄될 수 있다. 예를 들어 투영 방식 K=64→128에서 ToothBrush는61.75→56.92로 낮아지고 ToyDrone은69.58→72.53으로 높아진다.
- K와 선택 공간의 영향은 머신마다 다르므로 단일 budget 또는 방식의 보편적 우월성을 주장하지 않는다. 단일 투영 seed의 사후 비교이다.

## 재현

```bash
python scripts/score_reference_budgets.py --run SOURCE_FLOOR_ONLY_RUN --output NEW_RUN
wram evaluate --run NEW_RUN --labels labels.csv --output NEW_EVALUATION
```
