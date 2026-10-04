# 공식 PatchCore sampler를 사용한 정상 참조 선택 비교

요청에 따라 역사적 256차원 Gaussian RP-FPS를 공식 PatchCore 선택 알고리즘으로 대체하여 다시 비교했다. 코드는 Amazon Science `patchcore-inspection` commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`의 `ApproximateGreedyCoresetSampler` 원본(`vendor/patchcore-inspection/sampler.py`, SHA256 `39612cd2b486865ece304348f740f4bc700a1c2e8684fbab3c2ed016c7801f2d`)을 수정 없이 사용한다. Apache-2.0 원본 라이선스를 함께 보관한다.

| 정상 참조 선택 | K | Dev hmean (%) | Eval hmean (%) |
|---|---:|---:|---:|
| 원래 공간 FPS | 64 | 65.826965 | 64.946719 |
| 공식 PatchCore sampler | 64 | 66.734871 | 64.981959 |
| 원래 공간 FPS | 128 | **67.690781** | **67.360102** |
| 공식 PatchCore sampler | 128 | 67.022330 | 66.883851 |
| 원래 공간 FPS | 256 | 67.011995 | 66.845477 |
| 공식 PatchCore sampler | 256 | 65.791411 | 65.844041 |

K=128에서 공식 sampler는 주설정 FPS보다 Dev 0.668452점, Eval 0.476250점 낮다. 사전 기록한 주설정 검토 기준(양쪽 모두 +2pp)에 못 미쳐 주설정을 바꾸지 않았다. 이 결과는 단일 고정 시드의 사후 비교이며 선택법의 보편적 서열을 입증하지 않는다.

## 실행 조건

- 입력: 동일한 floor-only Wiener residual + frozen original BEATs iter3 + RDP4의 정상녹음 8×768 descriptor를 이어 붙인 6,144차원 벡터, 각 머신1,000개. 기존 query descriptor를 그대로 사용했다.
- 공식 sampler: 기본 128차원 무작위 `torch.nn.Linear(bias=False)` 투영, 무작위 시작점 10개와 공식 approximate greedy 갱신. PyTorch와 NumPy 난수 상태를 머신별로 seed20260613에 설정한다. CPU, torch2.5.1+cu121, numpy2.1.3. 공식 소스의 `_reduce_features` 및 `_compute_greedy_coreset_indices`를 직접 호출해 선택 ID를 저장한다.
- 크기: 한 번 선택한256개 순서에서 처음64/128/256개를 사용한다. 각각 동일 코드를 같은 난수 상태에서 별도 실행해도 접두사 선택이 된다.
- 점수: 선택은 합친 벡터에서 수행하지만 local scale과 bandwise BEAM 점수는 원래8×768 descriptor에서 계산한다. 각 K의 정상 뱅크가 scale을 다시 맞추고 query별로 다시 match한다. PatchCore의 이미지 detector 전체를 재현한다는 주장은 아니다.
- 절차: 모든 정상 bank를 동결한 후 query를 읽고, 2,400개 query 예측을 동결한 후 별도 evaluator가 label을 읽었다. 학습과 encoder 재실행은 없었다. Dev/Eval은 이미 관측된 공개 split이다.

공식 sampler의 `run()`을 실제 Dev/fan 정상데이터에 K128로 실행한 결과와 저장된128개 ID가 정확히 일치한다. 12개 뱅크의 64/128 prefix 24개, 고유 녹음ID, 머신 성분36행의 조화평균, Dev/Eval 각 21/15 성분의 집계 조화평균을 확인했다. 독립 ROC와 sklearn의 최대 오차는2.22e-16이다. 수치와 현재 해시의 재현 검사이며 외부의 전체 감사 인증은 아니다.

`MACHINE_METRICS.csv`, `AGGREGATES.csv`, `VALIDATION.json`, `EXECUTION.json`이 전달용 결과이며, 전체 frozen model/prediction과 source snapshot은 `artifacts/patchcore_official_001/`, label 평가 결과는 `artifacts/patchcore_official_evaluation_001/`에 있다. v37의 RP-FPS 결과는 서로 다른 과거 실험으로 보존한다.
