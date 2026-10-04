# 공개 구현의 floor-only 재현 결과

2026-09-23. 원시 음성 14,400개를 공개 패키지 설치본으로 새로 처리했습니다. 개발 저장소 밖에서 실행했으며, 별도 공식 정답으로 지표를 계산했습니다.

## 주 결과

| 조건 | Development | Evaluation |
|---|---:|---:|
| Near / mean / All1000 | 57.934026 | 62.166333 |
| Wiener / mean / FPS128 | 66.590913 | 65.412164 |
| Wiener / RDP4 / All1000 | 64.406574 | 66.139873 |
| Wiener / RDP4 / FPS128 | 67.690781 | 67.360102 |
| Wiener / RDP4 / FPS64 | 65.826965 | 64.946719 |
| Wiener / RDP4 / FPS256 | 67.011995 | 66.845477 |

모두 ratio k=1, 해당 bank 내부 scale + rematch. 전체 조건과 기계별 세 성분은 CSV에 기록했습니다.

## 재현 검증

- 전체 원시음성 실행 283.227초, H100 NVL 한 장. 새 모델 학습 없음.
- 기존 c=0 대조군의 전체 2,400 query descriptor와 bit-exact 일치.
- 12개 machine의 FPS128 recording IDs 모두 일치.
- 최종 점수 최대 차이 9.93e-14; 공식 정의 지표의 별도 ROC 계산과 sklearn 차이 2.23e-16 미만.
- 모든 정상 bank를 동결한 후 query를 처리하고, 모든 예측을 동결한 후 별도 평가에서 정답을 읽음.
- 설치본의 핵심 검증 테스트 7개 통과. 수치 검증은 외부 auditor의 전체 인증을 뜻하지 않음.

## 해석과 범위

- 이 설정은 과거 Dev/Eval loading ablation을 본 뒤 단순화를 위해 채택한 사후 선택입니다. 독립 미관측 test라고 주장하지 않습니다.
- H 분모의 power-dependent loading 항만 제거했고, 수치 epsilon 1e-12는 남겼습니다.
- Near→Wiener: anomaly probe 11/12, 이웃 purity 11/12, normal-centroid separation margin 10/12에서 점추정치 상승.
- Wiener mean→RDP4: median linear CKA 0.99849. 작은 representation 변경이며 모든 기계의 probe가 상승하지 않습니다.
- probe는 별도 사후 진단용 학습이며 ASD 모델이나 bank를 학습하지 않았습니다.
- BEATs만 새로 재현했습니다. 다른 encoder의 기존 원고 표를 이 결과로 자동 갱신하면 안 됩니다.
- 투영 coreset은 별도의 비교이며 현재 주 설정을 바꾸지 않습니다.
