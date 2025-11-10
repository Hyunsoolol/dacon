# v19 모델 정리 (LB 0.14987)

운전 적성 검사 예측 문제에서 **현재 기준이 되는 v19(Base+Delta+log_ratio) 모델** 정리입니다.  
이 문서는 팀원들이 전체 파이프라인과 모델 구조를 한 번에 이해하고,  
앞으로의 개선 방향(Feature / Model / Hyperparameter)을 논의하기 위한 자료입니다.

> 참고  
> - 기존 Base 모델 LB: **0.1536969**  
> - v19 모델 LB: **0.14987**  
> → Combined Score 기준 **개선(↓)**, 특히 **B(자격 유지) 구간에서 이득**이 큼.

---

## 0. v19에서 달라진 점 요약 (vs Base)

### 0.1 Feature 측면

- 기존 *_cost(차이) → **로그 비율 기반 *_log_ratio** 로 변경
  - 예:
    - `A4_stroop_rt_cost = incongruent - congruent`
    - → `A4_stroop_rt_log_ratio = log((incongruent+eps) / (congruent+eps))`
    - `B4_flanker_acc_cost` → `B4_flanker_acc_log_ratio` 등
- **Delta(B-only) 피처 추가**
  - 같은 PrimaryKey의 **직전 검사와의 차이**를 `delta_*` 컬럼으로 추가
  - 예: `delta_B3_rt_mean`, `delta_B4_flanker_acc_log_ratio` …
  - **B 검사에서만 활성화** (Test == 'B' 인 row에만 값, 나머지는 NaN)

### 0.2 모델 구조 측면

- **A 모델**
  - 구조는 Base와 동일
  - PK Stats 사용 안 함
  - Delta 피처는 학습 전 명시적으로 제거
- **B 모델**
  - **PK Stats + Delta + log_ratio** 모두 사용
  - 하이퍼파라미터 별도 튜닝
    - `depth=3, learning_rate=0.03, l2_leaf_reg=5`
    - `random_strength=3.0, bagging_temperature=1.0, border_count=128`

### 0.3 검증/추론 파이프라인

- 여전히 **StratifiedKFold(5-fold)** 기반 (GroupKFold 미사용)
- Fold별 **CatBoost + Isotonic Regression** 구조 유지
- 전처리 단계에서 **Age_num / YearMonthIndex에 대한 z-score** 생성
  - 통계를 `normalization_stats.pkl`로 저장
  - inference 시 test 데이터에도 동일하게 적용

---

## 1. 전체 파이프라인 개요

파이프라인은 크게 두 단계로 나뉩니다.

1. **전처리 스크립트**: `1_Preprocess_delta_logratio.py`
   - PDF 명세 + 도메인 지식 기반 Feature Engineering
   - A/B 검사 raw CSV → 도메인 피처 / 요약 인덱스 생성
   - **log_ratio 2차 피처** 추가
   - **PK 히스토리 피처 + Age/YearMonthIndex 정규화 점수 + Delta(B-only)** 생성
   - `./data/all_train_data.feather` 및 `./model/normalization_stats.pkl` 저장

2. **모델 학습 스크립트**: `2_Train_Models_v18_delta_logratio.py` (v19 학습용)
   - `all_train_data.feather` 로드
   - StratifiedKFold(5-fold) 기반 A/B 분리 학습
   - Fold별로 **PK Stats(PrimaryKey level 집계)** 생성 (log_ratio 기반)
   - CatBoost + Isotonic Regression (fold별 calibration)
   - 최종 모델/보정기/PK Stats 저장 → `submit.zip` 구성에 사용

---

## 2. 입력 데이터 설명

### 2.1 메타 데이터

- 파일: `./data/train.csv`
- 주요 컬럼
  - `Test_id` : 검사 단위 ID (A/B 한 번 시행이 하나의 row)
  - `PrimaryKey` : 사람 ID (같은 사람이 여러 번 검사)
  - `Test` : `'A'` (신규 자격) / `'B'` (자격 유지)
  - `Label` : 타깃 (0/1)

### 2.2 A/B 원본 데이터

- A 검사: `./data/train/A.csv`
- B 검사: `./data/train/B.csv`
- 특징:
  - `"1,2,1,3,..."` 형식의 **trial-level 시퀀스**가 많음

예시 (A):

- `A1-1, A1-2, A1-3, A1-4` (조건/정답/RT 등)
- …
- `A9-1` ~ `A9-5`

예시 (B):

- `B1-1` ~ `B1-3`
- …
- `B10-1` ~ `B10-6`

전처리에서 이 시퀀스들을 **평균, 표준편차, 조건별 accuracy, log_ratio 지표**로 요약해서 사용합니다.

---

## 3. 1_Preprocess_delta_logratio.py: Feature Engineering

### 3.1 공통 유틸

```python
def convert_age(val):
    # "25a" → 25, "25b" → 30
    ...

def split_testdate(val):
    # 202401 → (2024, 1)
    ...

def seq_mean(series):
    # "1,2,3" → np.mean
    ...

def seq_std(series):
    # "1,2,3" → np.std
    ...

def masked_operation(cond_series, val_series, target_conds, operation='mean'/'std'/'rate'):
    # 조건(cond)이 특정 값일 때 val의 mean / std / correct rate 계산
    ...

def _log_ratio(num, den, eps=1e-6):
    # 로그 비율: log((num+eps) / (den+eps))
    ...
```

추가로 A3/B1/B2/B4/B3/B5~B8 정답률 계산용 seq_rate_* 함수들이 있습니다.
(v18과 동일한 구조)

### 3.2 A 검사: 1차 도메인 피처 (preprocess\_A)

* **기본 파생**
    * `Age` → `Age_num`
    * `TestDate` → `Year`, `Month`
* **A1 (속도 예측)**
    * 전체 RT: `A1_rt_mean`, `A1_rt_std`
    * 방향/속도 조건별 RT: `A1_rt_left`, `A1_rt_right`, `A1_rt_slow`, `A1_rt_norm`, `A1_rt_fast`
    * 속도 조건별 accuracy: `A1_acc_slow`, `A1_acc_norm`, `A1_acc_fast`
* **A2 (정지 예측)**
    * `A2_rt_mean`, `A2_rt_std`
    * 속도 조건별 RT: `A2_rt_slow_c1`, `A2_rt_norm_c1`, `A2_rt_fast_c1`
    * 가속도 조건별 RT: `A2_rt_slow_c2`, `A2_rt_norm_c2`, `A2_rt_fast_c2`
    * accuracy: `A2_acc_slow`, `A2_acc_norm`, `A2_acc_fast`
* **A3 (주의 전환)**
    * valid/invalid accuracy: `A3_valid_acc`, `A3_invalid_acc`
    * RT 요약: `A3_rt_mean`, `A3_rt_std`, `A3_rt_small`, `A3_rt_big`, `A3_rt_left`, `A3_rt_right`
* **A4 (Stroop)**
    * RT: `A4_rt_mean`, `A4_rt_std`, `A4_rt_congruent`, `A4_rt_incongruent`
    * accuracy: `A4_acc_congruent`, `A4_acc_incongruent`
* **A5 (변화 탐지)**
    * 조건별 accuracy: `A5_acc_nonchange`, `A5_acc_pos_change`, `A5_acc_color_change`, `A5_acc_shape_change`
* **A6, A7 (문제풀이)**
    * `A6_correct_count`, `A7_correct_count`
* **A8, A9 (질문지)**
    * `A8-1`, `A8-2`
    * `A9-1` ~ `A9-5`

원본 시퀀스 컬럼(`A1-1` ~ `A9-5`)은 요약 피처 생성 후 drop 합니다.

### 3.3 B 검사: 1차 도메인 피처 (preprocess\_B)

* **기본 파생**
    * `Age_num`, `Year`, `Month` 동일
* **B1, B2 (시야각)**
    * task accuracy: `B1_task1_acc`, `B2_task1_acc`
    * RT: `B1_rt_mean`, `B1_rt_std`, `B2_rt_mean`, `B2_rt_std`
    * change vs non-change accuracy: `B1_change_acc`, `B1_nonchange_acc`, `B2_change_acc`, `B2_nonchange_acc`
* **B3 (신호등)**
    * `B3_acc_rate`, `B3_rt_mean`, `B3_rt_std`
* **B4 (Flanker)**
    * `B4_congruent_acc`, `B4_incongruent_acc`, `B4_rt_mean`, `B4_rt_std`
* **B5~B8**
    * `B5_acc_rate`, `B5_rt_mean`, `B5_rt_std`
    * `B6_acc_rate`, `B7_acc_rate`, `B8_acc_rate`
* **B9, B10 (점수형)**
    * `B9-1` ~ `B9-5`
    * `B10-1` ~ `B10-6`

원본 시퀀스 컬럼(`B1-1` ~ `B10-6`)은 drop 합니다.

### 3.4 A/B 2차 피처: log\_ratio + 기타 파생 (add\_features\_A, add\_features\_B)

#### 3.4.1 공통

* **검사 시점 인덱스:**
    * `YearMonthIndex = Year * 12 + Month`
* **Speed–Accuracy Tradeoff:**
    * **A:**
        * `A1_speed_acc_tradeoff = A1_rt_mean / A1_acc_norm`
        * `A2_speed_acc_tradeoff = A2_rt_mean / A2_acc_norm`
        * `A4_speed_acc_tradeoff = A4_rt_mean / A4_acc_congruent`
    * **B:**
        * `B1_speed_acc_tradeoff` ~ `B5_speed_acc_tradeoff` (RT / accuracy)
* **RT 변동성 (CV):**
    * **A:** `A1_rt_cv` ~ `A5_rt_cv`
    * **B:** `B1_rt_cv` ~ `B5_rt_cv`

#### 3.4.2 `*_log_ratio` (중요 변화 포인트)

기존 Base에서는 단순 차이(cost) 를 썼다면, v19에서는 대부분을 로그 비율로 전환했습니다.

* **A1:** 속도 조건 간
    * `A1_rt_speed_log_ratio = log(A1_rt_fast / A1_rt_slow)`
    * `A1_acc_speed_log_ratio = log(A1_acc_fast / A1_acc_slow)`
* **A2:**
    * `A2_rt_speed_log_ratio_c1 = log(A2_rt_fast_c1 / A2_rt_slow_c1)`
    * `A2_acc_speed_log_ratio = log(A2_acc_fast / A2_acc_slow)`
* **A3:**
    * `A3_rt_size_log_ratio = log(A3_rt_big / A3_rt_small)`
    * `A3_acc_attention_log_ratio = log(A3_valid_acc / A3_invalid_acc)`
* **A4 (Stroop):**
    * `A4_stroop_rt_log_ratio = log(A4_rt_incongruent / A4_rt_congruent)`
    * `A4_stroop_acc_log_ratio = log(A4_acc_congruent / A4_acc_incongruent)`
* **A5:**
    * `A5_acc_pos_log_ratio = log(A5_acc_nonchange / A5_acc_pos_change)`
    * `A5_acc_color_log_ratio = log(A5_acc_nonchange / A5_acc_color_change)`
* **B1/B2:**
    * `B1_acc_log_ratio = log(B1_nonchange_acc / B1_change_acc)`
    * `B2_acc_log_ratio = log(B2_nonchange_acc / B2_change_acc)`
* **B4:**
    * `B4_flanker_acc_log_ratio = log(B4_congruent_acc / B4_incongruent_acc)`

#### 3.4.3 고수준 인덱스/종합 점수

v18에서 사용하던 인지 지표/위험 점수 계열은 그대로 유지

* **A 쪽:**
    * `A_working_memory_index`
    * `A_cog_sum`, `A_cog_mean`
    * `RiskScore` (Stroop/log_ratio, attention, RT CV, 변화탐지 관련 피처를 가중합)
* **B 쪽:**
    * `B_visuomotor_variability`
    * `B_reaction_overall`
    * `B_executive_control_index`
    * `B9_sum`, `B9_mean`, `B10_sum`, `B10_mean`
    * `B_cog_sum`, `B_cog_mean`
    * `RiskScore_B`

(구체 수식은 v18 구현과 거의 동일하므로 코드 참고)


### 3.5 PK 히스토리 피처 + 정규화 점수 (전략 2, 3)

#### 3.5.1 PK 히스토리 (시간축 기반)

* `PrimaryKey` 기준, (`Year`, `Month`, `Test_id`)로 정렬 후 groupby
* **생성되는 피처:**
    * `pk_hist_total_count`: 해당 검사 직전까지의 누적 검사 횟수
    * `pk_hist_A_count`: 직전까지의 A 검사 누적 횟수
    * `pk_hist_B_count`: 직전까지의 B 검사 누적 횟수
    * `pk_hist_prev_ym`: 직전 검사 시점(YearMonthIndex)
    * `pk_hist_gap_from_prev`: 현재 시점 - 직전 검사 시점

#### 3.5.2 Age / YearMonthIndex 정규화 점수

* 전체 train 기준 평균/표준편차:
    * `Age_num_mean`, `Age_num_std`
    * `YearMonthIndex_mean`, `YearMonthIndex_std`
* 각 row에 대해:
    * `Age_num_z = (Age_num - mean) / std`
    * `YearMonthIndex_z = (YearMonthIndex - mean) / std`
* 이 통계는 `./model/normalization_stats.pkl` 에 저장되고,
    inference 시 test 데이터에도 동일하게 적용됩니다.

### 3.6 Delta(B-only) 피처

v19에서 새로 추가된 핵심 변화입니다.

* **함수:** `add_delta_features_pk`
* **대상 컬럼:**
    * 접두어가 (`"B1_"`, `"B2_"`, `"B3_"`, `"B4_"`, `"B5_"`, `"B6_"`, `"B7_"`, `"B8_"`, `"B9-"`, `"B10-"`)
    * 수치형 컬럼만 사용
* **로직:**
    * `PrimaryKey`, `Year`, `Month`, `Test_id` 기준으로 정렬
    * 같은 사람 안에서 각 피처의 직전 값(`shift(1)`)을 구한 뒤
    * **현재 값 - 직전 값 = Delta**
    * 이 값은 `Test == 'B'`인 row에서만 사용,
    * `Test == 'A'`인 row에는 `NaN`을 채우도록 마스킹
* **컬럼명:**
    * 원본 `B3_rt_mean` → `delta_B3_rt_mean`
    * 원본 이름에 `-`가 있으면 `_`로 치환 (예: `B9-1` → `delta_B9_1`)
* **A/B 모델에서의 사용:**
    * **A 모델:** 학습 시점에 모든 `delta_*` 컬럼을 drop (A는 Delta 미사용)
    * **B 모델:** Delta 컬럼을 그대로 사용 (PK Stats와 함께)

### 3.7 결과 저장

* **최종 train 테이블:**
    * `./data/all_train_data.feather`
* **정규화 통계:**
    * `./model/normalization_stats.pkl`

---

## 4. `2_Train_Models_v18_delta_logratio.py`: 모델 학습

### 4.1 데이터 로드

```python
FEATURE_SAVE_PATH = "./data/all_train_data.feather"
all_train_df = pd.read_feather(FEATURE_SAVE_PATH)
all_train_df['Label'] = all_train_df['Label'].fillna(0)
```

* A/B가 통합된 단일 테이블에서 학습합니다.

### 4.2 평가 지표 & Combined Score
* ECE (Expected Calibration Error):
    * 10개 bin 기준의 classic ECE 구현 (Base와 동일)
* Combined Score:
    * score = 0.5 * (1 - AUC) + 0.25 * Brier + 0.25 * ECE
    * 목적: Combined Score 최소화
    * AUC는 높을수록 좋으므로 (1 - AUC)로 뒤집어서 사용
    * Brier, ECE는 원래 낮을수록 좋은 값
* Fold별로 A/B 각각에 대해 비보정 / 보정 후 점수를 찍어보고, A-only / B-only / 전체(A+B) 동향을 관찰합니다.

---

## 5. K-Fold 구성

```python
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
```
* 전체 row 단위로 StratifiedKFold(5-fold) 적용
* `Label` 비율을 각 fold마다 비슷하게 유지
* 아직은 PrimaryKey 기준 GroupKFold는 미사용
→ 같은 사람이 train/val에 동시에 들어갈 수 있어,
검증 점수가 다소 optimistic 할 수 있음 (추후 개선 포인트).

---

## 6. 피처 세팅 (학습 시)

```python
CAT_FEATURES    = ['Age', 'PrimaryKey']
DROP_COLS_TRAIN = [
    'Test_id', 'Test_x', 'Test_y', 'Label',
    'TestDate', 'Year', 'Month', 'base_index',
]
```
* 범주형 피처:
    * Age: 문자열로 캐스팅해서 CatBoost category로 사용
    * PrimaryKey: 사람 ID, 개인 특성을 캡처하는 카테고리 피처
* 드롭하는 컬럼 (학습에서 제외):
    * ID/메타: Test_id, Test_x, Test_y, TestDate, Year, Month, base_index
타깃: Label
A 모델은 여기서 한 번 더 delta_* 컬럼을 제거해서 순수 base+hist+norm+log_ratio만 사용.
---

## 7. PK Stats (PrimaryKey level 집계 피처, log\_ratio 버전)

Fold 루프 안에서, 각 Fold의 **train 데이터만** 사용하여 사람 단위 통계를 만듭니다.

* Base에서 `A4_stroop_rt_cost`, `B4_flanker_acc_cost`를 쓰던 부분을
    v19에서는 `A4_stroop_rt_log_ratio`, `B4_flanker_acc_log_ratio`로 교체했습니다.
* 나머지 컨셉은 동일:
    * 나이/검사 시점 분포
    * A/B 핵심 도메인 피처의 mean/std/max
    * `RiskScore`/`RiskScore_B` 요약
    * `pk_test_total_count`: 검사 횟수
* 검사 타입별 횟수:
    * `pk_test_type_count_fold` 로직을 통해 `pk_test_A_count`, `pk_test_B_count` 생성

Fold별로 생성된 `pk_stats_fold`를 리스트에 저장 → 모든 Fold 종료 후:
`all_pk_stats_df`로 concat 한 뒤, `PrimaryKey` 기준으로 평균(`mean`)을 내어
`./model/pk_stats_final.csv`로 저장합니다.

---

## 8. 모델 구조 및 하이퍼파라미터

### 8.1 공통 CatBoost 설정

* A/B 공통 옵션:
    * `iterations=3000`
    * `loss_function='Logloss'`
    * `eval_metric='AUC'`
    * `random_seed=42`
    * `early_stopping_rounds=100`
* 카테고리 변환:
    * `CAT_FEATURES` (`Age`, `PrimaryKey`)에 대해 `.fillna('nan').astype(str)` 처리

### 8.2 A 모델 (신규 자격, no PK, no Delta)

* Fold마다 `Test_x == 'A'`인 데이터를 분리합니다.
* `delta_*` 컬럼을 명시적으로 제거한 뒤 학습합니다.
* **하이퍼파라미터 (Base와 동일):**
    * `"depth": 6`
    * `"learning_rate": 0.05`
    * `"l2_leaf_reg": 3`
* A 모델은 PK Stats, Delta를 사용하지 않고
    1차/2차 도메인 피처 + PK 히스토리 + z-score + `log_ratio`만으로 학습합니다.
* **fold별 산출물:**
    * `./model/catboost_A_fold{0~4}.pkl`
    * `./model/calibrator_A_fold{0~4}.pkl`

### 8.3 B 모델 (자격 유지, PK + Delta + log\_ratio)

* Fold마다 `Test_x == 'B'`인 데이터를 분리합니다.
* 해당 Fold의 `pk_stats_fold`를 `PrimaryKey` 기준으로 merge 합니다.
* `delta_*` 컬럼을 포함한 전체 피처로 학습합니다.
* **하이퍼파라미터 (v19에서 새로 튜닝):**
    * `"depth": 3`
    * `"learning_rate": 0.03`
    * `"l2_leaf_reg": 5`
    * `"random_strength": 3.0`
    * `"bagging_temperature": 1.0`
    * `"border_count": 128`
* B 모델은 PK Stats + Delta + `log_ratio` + 히스토리 + z-score 등 수치형 피처 전체를 사용합니다.
* Delta 컬럼 개수는 Fold마다 로그에 출력되며, 현재 기준 약 30~40개 수준입니다.
* **fold별 산출물:**
    * `./model/catboost_B_fold{0~4}.pkl`
    * `./model/calibrator_B_fold{0~4}.pkl`
* (B 데이터가 너무 적거나 단일 클래스일 경우 해당 Fold B 모델은 `None`으로 저장)

---

## 9. 최종 산출물 및 submit 구성

학습이 끝나면 `./model` 폴더에 아래 리소스들이 생성됩니다.

* **모델**
    * `catboost_A_fold0.pkl` ~ `catboost_A_fold4.pkl`
    * `catboost_B_fold0.pkl` ~ `catboost_B_fold4.pkl`
* **캘리브레이터**
    * `calibrator_A_fold0.pkl` ~ `calibrator_A_fold4.pkl`
    * `calibrator_B_fold0.pkl` ~ `calibrator_B_fold4.pkl`
* **PK Stats**
    * `pk_stats_final.csv`
* **정규화 통계**
    * `normalization_stats.pkl`

이 리소스 + inference 코드(`script.py`) + `requirements.txt`를 묶어서
`submit.zip`을 구성합니다.

평가 서버에서는 `./data`에 실제 test 데이터가 들어오고,
`script.py`를 실행해 `./output/submission.csv`를 생성하는 구조입니다.

(v19 현재는 1-seed x 5-fold 기준; 5-seed 앙상블은 아직 미적용)

---

## 10. v19 성능 요약

* **Base (v18 계열) LB: 0.1536969**
* **v19 (Delta + log_ratio + B 튜닝) LB: 0.14987**

대략적인 경향:

* A-only Combined Score: Base와 거의 비슷 (소폭 악화/개선 혼재)
* B-only Combined Score: 거의 모든 Fold에서 **0.003~0.01 수준 개선**
* 전체(A+B) 기준 Combined Score 및 LB가 동시에 개선 →
    Delta(B-only) + `log_ratio` + B 튜닝 조합이 유효한 것으로 판단.

---

## 11. 앞으로의 계획 (v19 기준)

이제부터 모든 실험은 v19를 baseline으로 사용합니다.

* 항상 아래 세 가지를 트래킹
    * A-only Combined Score
    * B-only Combined Score
    * 전체(A+B) Combined Score
* 리더보드 점수와의 괴리도도 함께 체크해서 overfitting/underfitting 판단

**개선 아이디어 (초안)**

* **Ensemble**
    * v19 구조 그대로 multi-seed(예: 5 seeds) x 5-fold 앙상블
    * 용량 제한(10GB 이하)을 고려한 모델 수/압축 전략 필요
* **CV 전략**
    * `StratifiedGroupKFold`(PrimaryKey 기준) 도입 여부 검토
    * 시간축 기반 validation (최근 검사 hold-out 등)
* **Feature**
    * Delta를 A 영역까지 확장할지 여부
    * PK Stats에 더 많은 `log_ratio` / longitudinal 피처 추가
    * 특정 연령대/검사 패턴에 대한 interaction feature
* **Model / Hyperparameter**
    * B 모델 `depth`/`learning_rate` 추가 튜닝
    * 다른 트리 모델(LightGBM, XGBoost) + stacking/blending 실험

---

## 12. 요약

**v19**는
Base 모델의 도메인 피처/PK Stats 구조를 유지하면서,
핵심 `cost` 피처를 **`log_ratio`**로 바꾸고,
B 검사에 대해 **`Delta`(직전 대비 변화량) + 전용 CatBoost 튜닝**을 적용한 버전이며,
그 결과 LB 기준 **0.1537 → 0.1499** 수준으로 의미 있는 개선을 달성한 현재 기준 베이스라인입니다.
