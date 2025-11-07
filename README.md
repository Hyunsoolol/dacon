# Base 모델 정리 (LB 0.1536969035)

> 운전 적성 검사 예측 문제에서 현재 기준이 되는 **Base 모델** 정리입니다.  
> 이 문서는 팀원들이 전체 파이프라인과 모델 구조를 한 번에 이해하고,  
> 앞으로의 개선 방향(Feature / Model / Hyperparameter)을 논의하기 위한 자료입니다.

---

## 1. 전체 파이프라인 개요

파이프라인은 크게 두 단계로 나뉩니다.

1. **전처리 노트북**: `1_Preprocess.ipynb`
   - PDF 명세 + 도메인 지식 기반 Feature Engineering
   - A/B 검사 raw CSV → 도메인 피처 / 요약 인덱스 생성
   - `all_train_data.feather` 저장

2. **모델 학습 노트북**: `2_Train_Models.ipynb`
   - `all_train_data.feather` 로드
   - StratifiedKFold(5-fold) 기반 A/B 분리 학습
   - CatBoost + Isotonic Regression (fold별 calibration)
   - PK Stats(B 전용 group-level 피처) 생성 및 사용
   - 최종 모델/보정기/PK Stats 저장 → `submit.zip` 구성

---

## 2. 입력 데이터 설명

### 2.1 메타 데이터

- 파일: `./data/train.csv`
- 주요 컬럼:
  - `Test_id` : 검사 단위 ID (A/B 한 번 시행이 하나의 row)
  - `PrimaryKey` : 사람 ID (같은 사람이 여러 번 검사)
  - `Test` : `'A'` (신규 자격) / `'B'` (자격 유지)
  - `Label` : 타깃 (0/1)

### 2.2 A/B 원본 데이터

- A 검사: `./data/train/A.csv`
- B 검사: `./data/train/B.csv`
- 특징:
  - `"1,2,1,3,..."` 형식의 **trial-level 시퀀스**가 많음
  - 예시 (A 쪽):
    - `A1-1, A1-2, A1-3, A1-4` (조건/정답/RT 등)
    - …
    - `A9-1` ~ `A9-5`
  - 예시 (B 쪽):
    - `B1-1` ~ `B1-3`
    - …
    - `B10-1` ~ `B10-6`

전처리에서 이 시퀀스들을 **평균, 표준편차, 조건별 accuracy, cost 지표**로 요약해서 사용합니다.

---

## 3. 1_Preprocess.ipynb: Feature Engineering

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
