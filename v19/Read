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
