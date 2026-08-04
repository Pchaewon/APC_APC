# 모델 설명

Wire Saw 공정 BOW 품질 예측 및 온도 Recipe 추천 모델 (`train_inverse_rolling.py` 기준)

---

## 모델 구조

두 단계로 구성된 **Forward + Inverse** 구조입니다.

**Forward 모델 (BOW 예측)**
- 알고리즘: Ridge Regression (L2 정규화 선형 회귀)
- 역할: 현재 run의 온도 recipe + 직전 run 평균 상태 → 현재 run BOW 예측
- 장비 개체 효과를 one-hot으로 흡수 (Simpson's Paradox 방지)

**Inverse 최적화 (Recipe 역산)**
- 알고리즘: SLSQP 제약 최적화
- 역할: 목표 BOW → 이를 달성하는 온도 프로파일 역산
- 학습된 Forward 모델을 뒤집어 사용 (재학습 없음)

```
[학습] 온도 + 직전평균 조건 + 장비더미  →(Ridge)→  BOW
[역산] 목표 BOW + 직전평균 조건 + 장비고정  →(SLSQP)→  온도 프로파일
```

배포 현실 반영: 현재 run은 아직 가공 전이라 값이 없으므로, **같은 wire 내 직전 run들의 평균 상태**를 조건으로 사용합니다.

---

## Input

**Forward 학습 입력 (X)**

| 그룹 | 컬럼 | 개수 | 성격 |
| :--- | :--- | :---: | :--- |
| 온도 recipe | set_frame_temp_0pct ~ 100pct | 12 | 최적화 대상 |
| 직전 평균 조건 | roll_fdc_set_tension | 1 | 고정 조건 |
| | roll_fdc_wait_time | 1 | 고정 조건 |
| | roll_fdc_ingot_len | 1 | 고정 조건 |
| | roll_range_slurry_temp_10_0 | 1 | 고정 조건 |
| 장비 식별 | eqp_* (one-hot) | 장비 수 | 개체 효과 |

- `roll_` 접두어 = 같은 wire 내 직전 최대 10 run(2 run 지연)의 평균
- **roll_BOW(직전 BOW 평균)는 제외** — 자기상관 leakage 방지 (아래 Feature Engineering 참조)

**Inverse 입력**
- 목표 BOW (기본 1.75, 스펙 중앙)
- 직전 평균 조건 (roll_ 값)
- 대상 장비명 (해당 장비 더미를 1로 고정)

---

## Output

**Forward 출력**
- 예측 BOW (avg_bow_bf_total), 연속값

**Inverse 출력**
- 온도 프로파일 12개 (set_frame_temp_0pct ~ 100pct)
- 예측 BOW (역산 recipe 적용 시)
- 장비별 CSV로 저장 (rec_ 접두어 = 추천값, actual_ = 실측값)

---

## Feature Engineering

**1. 직전 run 평균 (Rolling)**

같은 wire(장비 + wire_id) 안에서 시간순 정렬 후, 현재 run 기준 직전 run들의 평균을 계산합니다.

- `lag = 2`: 데이터 지연 반영 (15번째 run 가공 시 13번째까지만 존재)
- `window = 10`: 최대 10 run 평균
- `min_runs = 3`: 최소 3 run 있어야 평균 산출 (부족하면 제외)

이 평균은 **직전 run들로만** 계산되어(현재 run 미포함) 배포 시 사전값으로 활용 가능합니다.

**2. roll_BOW 제외 (핵심 결정)**

직전 BOW 평균(roll_avg_bow_bf_total)을 feature에 넣으면 R²가 0.83으로 급등하지만, 이는 **BOW 자기상관을 복사**하는 leakage입니다. 진단 결과:

| 구성 | R² | 온도 계수 |
| :--- | :---: | :---: |
| 온도만 | 0.137 | 0.607 |
| 온도 + roll_조건 | **0.232** | **0.521** |
| 온도 + roll_BOW | 0.830 | 0.112 (붕괴) |

roll_BOW 포함 시 온도 계수가 0.61→0.11로 붕괴하여 온도 제어력을 상실합니다. Recipe 추천이 목적이므로 **roll_BOW를 제외**하고 roll_조건만 사용합니다.

**3. 장비 one-hot**

장비별 BOW 레벨 차이가 전체 상관을 왜곡(Simpson's Paradox)합니다. 장비 더미 미적용 시 온도 계수가 공정 지식과 반대(+) 부호로 나타나므로, one-hot 인코딩으로 개체 효과를 흡수합니다.

**4. 제외한 변수**
- range_wire_guide: 가공 후 측정되는 사후값 (배포 시 미지) → 신뢰도 판정용으로만 별도 활용
- 요약 통계 feature (평균/기울기/peak): position 12개보다 예측력 낮고, 프로파일 복원 불가로 inverse 불가

---

## Data Split

**시간 분할 (Time-based Split)**

- 날짜순 정렬 후 앞 80%를 train, 뒤 20%를 test로 사용 (`split_ratio = 0.8`)
- 랜덤 분할이 아닌 시간 분할을 사용하는 이유: 랜덤 분할은 유사 시점 데이터가 train/test에 섞여 성능이 과대평가됨. 배포는 "과거 학습 → 미래 예측"이므로 시간 분할이 실제 배포 성능에 가까움

**Leakage 제거**
- 검증용 test 장비의 test 기간 데이터를 학습에서 제외 (해당하는 경우)
- 나머지 장비는 전 기간 유지

**최종 배포 모델**
- 성능 평가는 시간 분할로 수행하되, 최종 배포 모델은 **전체 데이터로 재학습**하여 최신 정보를 모두 반영

---

## 학습 방법

1. 원본 데이터 로드 및 process_time 필터 (13.3Hr)
2. wire별 직전 run 평균 feature 생성 (roll_)
3. 직전 평균이 존재하는 행만 학습 대상 (초반 run 제외)
4. 장비 one-hot 인코딩
5. StandardScaler로 표준화
6. 시간 분할 후 Ridge 학습 → 시간 분할 Test R² 측정
7. 전체 데이터로 재학습 (배포용)
8. 모델·스케일러·메타 저장 (model.pkl, scaler.pkl, meta.json)

**성능 (시간 분할)**: Test R² ≈ 0.23, MAE ≈ 0.69

이 값은 데이터의 이론적 R² 상한(0.20~0.24, 미측정 변수로 인한 구조적 한계) 범위 내의 정직한 성능입니다.

---

## Hyperparameter

| 파라미터 | 값 | 설명 |
| :--- | :---: | :--- |
| ridge_alpha | 5.0 | Ridge L2 정규화 강도. 시간 이동에 대한 강건성 확보 |
| lag | 2 | 데이터 지연 (run 단위) |
| window | 10 | 직전 run 평균 범위 (최대) |
| min_runs | 3 | 평균 산출 최소 run 수 |
| split_ratio | 0.8 | train/test 시간 분할 비율 |
| lambda_smooth | 0.0 | Inverse 온도 프로파일 smoothness 페널티 |
| include_roll_bow | False | roll_BOW 제외 (leakage 방지) |
| use_eqp_dummy | True | 장비 one-hot 사용 |

**lambda_smooth = 0 근거**: smoothness 페널티를 조금만 넣어도(0.01) 온도 프로파일이 평평하게 눌려 실측 재현이 붕괴(r 0.98→0.03). 실측 프로파일이 이미 매끄러워 별도 제약이 불필요하며, λ=0에서 역산 온도가 실측을 거의 완벽히 재현(r≈0.98).

---

## Feature Importance

Ridge 회귀 계수(표준화 기준)로 해석합니다.

**주요 계수 방향 (공정 지식 대조)**

| Feature | 계수 방향 | 해석 |
| :--- | :---: | :--- |
| set_frame_temp (장비 중심화 후) | 음(−) | 온도↑ → BOW↓ (공정 지식 일치) |
| roll_fdc_wait_time | 양(+) | 대기시간↑ → BOW↑ |
| roll_fdc_ingot_len | 양(+) | ingot 길이↑ → BOW↑ |
| fdc_set_tension | ≈0 | 유의한 레버 아님 |

**핵심 관찰**
- 온도가 주 제어 레버 (계수 절대값 평균 ≈ 0.52 유지)
- 직전 조건(roll_)이 예측력 기여 (R² 0.137 → 0.232)
- tension은 계수 ≈ 0으로 recipe 제어 인자가 아님

**장비 내 온도-BOW 상관** (검증): 6개 장비 전부에서 음의 상관이 통계적으로 유의 (−0.22 ~ −0.82), 온도 레버 작동을 확인.

---

## Recipe Optimization Logic

**목적함수 (목표값 도달)**

BOW 최소화가 아니라 **목표 BOW 도달**을 최적화합니다.

```
minimize   (predict(온도) - 목표BOW)²  +  λ · smoothness(온도)

제약: 각 온도 pct는 학습 분포 1~99% 분위 범위 내 (외삽 방지)
       λ = 0 (smoothness 페널티 미적용)
```

**최적화 절차**
1. 저장된 모델·스케일러·메타 로드
2. 고정값 세팅: 직전 평균 조건(roll_), 대상 장비 더미 = 1
3. 온도 12개만 최적화 변수로 설정
4. 현재 온도(있으면)를 시작점으로 SLSQP 실행
5. 목표 BOW에 가장 가까운 온도 프로파일 반환

**장비별 추천**
- 학습은 전체 데이터로 수행 (장비더미 포함)
- 추천 시 지정 장비(BSWS38, 42, 44)의 더미만 1로 고정하여 각 장비 조건에서 역산
- 장비별로 서로 다른 온도 프로파일 산출

**목표 BOW**: 기본 1.75 (양품 범위 1.5~2.0의 중앙). 스펙 이탈 lot을 중앙으로 유도.

---

## 최종 Pipeline

**학습 단계 (오프라인)**
```
1. 데이터 로드 + process_time 필터
2. wire별 직전 run 평균 생성 (lag=2, window=10)
3. roll_조건 feature 구성 (roll_BOW 제외)
4. 장비 one-hot + 표준화
5. Ridge 학습 (alpha=5)
6. 전체 재학습 → model.pkl, scaler.pkl, meta.json 저장
```

**추천 단계 (배포, run 단위)**
```
1. 현재 run(예: 15번째) 가공 전
2. 같은 wire 내 직전 run(4~13번째) 평균 계산 (roll_조건)
3. 대상 장비(BSWS38/42/44) 더미 = 1 고정
4. 목표 BOW(1.75) 입력
5. SLSQP로 온도 12개 역산
6. 신뢰도 태그 부착 (직전 WG 상태: HIGH_VAR/LOW_VAR)
7. 장비별 추천 온도 프로파일 출력 (inverse_by_eqp.csv)
8. 엔지니어 검토 후 반영 결정
```

**신뢰도 계층화** (별도 모듈)
- 직전 WG 고변동(HIGH_VAR, ~47%): 온도-BOW 관계 뚜렷 (부분 R² ≈ 0.44) → 적극 반영 권장
- 직전 WG 저변동(LOW_VAR, ~53%): 관계 약함 → 참고용

**실행 명령**
```bash
python train_inverse_rolling.py train    # 학습
python train_inverse_rolling.py by_eqp   # 장비별 역산 (BSWS38/42/44)
```

---

## 요약

| 항목 | 내용 |
| :--- | :--- |
| Forward | Ridge (alpha=5) + 장비 one-hot |
| Inverse | SLSQP, 목표 BOW 도달, λ=0 |
| Feature | 온도 12 + 직전평균 조건 4 + 장비더미 |
| 제외 | roll_BOW(leakage), wire_guide(사후값), 요약통계(inverse 불가) |
| 성능 | 시간분할 R² ≈ 0.23 (상한 0.24 내 정직한 값) |
| 배포 | 직전 run 평균 조건, 장비별 추천, 신뢰도 태그 |
