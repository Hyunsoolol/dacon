## ----------------------------------------------------------------
## [Jupyter용] 2_Train_Models_v18_delta_logratio.py
##  - v18 구조 유지
##    * A: Base + Hist + Norm (PK Stats 미사용, Delta는 명시적으로 제외)
##    * B: Base + PK + Hist + Norm + Delta(B-only)
##  - [변경] *_cost → *_log_ratio 기반 PK Stats 사용
## ----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import joblib
import warnings
import pickle  
warnings.filterwarnings('ignore')

print("Big-Tech ML Engineer (v18+Delta+log_ratio): K-Fold + Leakage Fix + Domain Features 시작.")

## 0. Hyperparameter 설정
BEST_A_PARAMS = {
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
}

BEST_B_PARAMS = {
    "depth": 3,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5,
    "random_strength": 3.0,
    "bagging_temperature": 1.0,
    "border_count": 128,
}

## 1. 로컬 경로 및 데이터 로드
BASE_DIR = "./data"
MODEL_SAVE_DIR = "./model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

FEATURE_SAVE_PATH = os.path.join(BASE_DIR, "all_train_data.feather")

# --- [이 부분 추가] ---
# 원본 로그 재현을 위해 1단계에서 저장한 피처 순서 로드
try:
    with open(os.path.join(MODEL_SAVE_DIR, "original_a_features.pkl"), "rb") as f:
        ORIGINAL_A_FEATURES = pickle.load(f)
    with open(os.path.join(MODEL_SAVE_DIR, "original_b_features.pkl"), "rb") as f:
        ORIGINAL_B_FEATURES = pickle.load(f)
    print(f"[재현] 원본 A 피처 순서 로드 완료 ({len(ORIGINAL_A_FEATURES)}개)")
    print(f"[재현] 원본 B 피처 순서 로드 완료 ({len(ORIGINAL_B_FEATURES)}개)")
except FileNotFoundError:
    print("="*50)
    print("경고: 원본 피처 순서 파일(original_..._features.pkl)을 찾을 수 없습니다.")
    print("재현이 불가능하며, sort() 기반으로 실행됩니다.")
    print("먼저 1단계(피처 순서 추출/저장)를 실행하세요.")
    print("="*50)
    ORIGINAL_A_FEATURES = None
    ORIGINAL_B_FEATURES = None

try:
    print("전처리된 메인 피처 로드 중...")
    all_train_df = pd.read_feather(FEATURE_SAVE_PATH)
except FileNotFoundError as e:
    print(f"경고: {e}")
    print("먼저 1_Preprocess_delta_logratio.py를 실행하여 all_train_data.feather를 생성해야 합니다.")
    raise
print("데이터 로드 완료.")

## 2. ECE 및 Combined Score 유틸
def expected_calibration_error(y_true, y_prob, n_bins=10):
    if len(y_true) == 0 or len(y_prob) == 0:
        return 0.0
    y_prob = np.nan_to_num(y_prob, nan=0.0)

    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0]  = -0.001
    bin_edges[-1] = 1.001

    df['y_prob'] = np.clip(df['y_prob'], 0, 1)
    df['bin'] = pd.cut(df['y_prob'], bins=bin_edges, right=True)

    bin_stats = df.groupby('bin', observed=True).agg(
        bin_total=('y_prob', 'count'),
        prob_true=('y_true', 'mean'),
        prob_pred=('y_prob', 'mean')
    )
    non_empty = bin_stats[bin_stats['bin_total'] > 0]
    if len(non_empty) == 0:
        return 0.0

    weights   = non_empty['bin_total'] / len(y_prob)
    prob_true = non_empty['prob_true']
    prob_pred = non_empty['prob_pred']
    ece = np.sum(weights * np.abs(prob_true - prob_pred))
    return ece

def combined_score(y_true, y_prob):
    if len(y_true) == 0 or len(y_prob) == 0 or np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        print("  AUC: N/A (단일 클래스), Brier: N/A, ECE: N/A (No data)")
        return 1.0

    y_prob = np.nan_to_num(y_prob, nan=0.0)
    mean_auc   = roc_auc_score(y_true, y_prob)
    mean_brier = mean_squared_error(y_true, y_prob)
    mean_ece   = expected_calibration_error(y_true, y_prob)
    score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece

    print(f"  AUC: {mean_auc:.4f}, Brier: {mean_brier:.4f}, ECE: {mean_ece:.4f}")
    print(f"  Combined Score: {score:.5f}")
    return score

## 3. K-Fold 분리
print("\n[INFO] K-Fold 교차 검증 분리 시작...")
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

train_indices_list = []
val_indices_list   = []

all_train_df['Label'] = all_train_df['Label'].fillna(0)

for train_idx, val_idx in skf.split(all_train_df, all_train_df['Label']):
    train_indices_list.append(train_idx)
    val_indices_list.append(val_idx)

print(f"{N_SPLITS}-Fold 분리 완료.")

CAT_FEATURES    = ['Age', 'PrimaryKey']
DROP_COLS_TRAIN = ['Test_id', 'Test_x', 'Test_y', 'Label', 'TestDate', 'Year', 'Month', 'base_index']

def _is_delta_column(col: str) -> bool:
    return col.startswith("delta_")

## 4. 공통 CatBoost 입력 구성 함수
def _build_cb_matrices(X_train, X_val):
    """
    numeric + categorical 분리 및 cat feature 인덱스 계산
    (A: PK Stats 없이, B: PK Stats merge 후 호출)
    """
    numeric_cols = list(set(X_train.columns) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))
    common_numeric_cols = list(set(numeric_cols) & set(X_val.columns))

    cb_X_train = X_train[common_numeric_cols + CAT_FEATURES].copy()
    cb_X_val   = X_val[common_numeric_cols + CAT_FEATURES].copy()

    for col in CAT_FEATURES:
        if col in cb_X_train.columns:
            cb_X_train[col] = cb_X_train[col].fillna('nan').astype(str)
            cb_X_val[col]   = cb_X_val[col].fillna('nan').astype(str)

    cat_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]
    return cb_X_train, cb_X_val, cat_indices

## 5. 모델 학습 함수
def train_model_A(X_train, y_train, X_val, y_val, group_label="A"):
    """
    A 모델: [재현용 수정] _build_cb_matrices로 생성 후, ORIGINAL_A_FEATURES 순서로 강제 재정렬
    """
    # delta_* 열 제거 (A는 Delta 미사용)
    drop_delta = [c for c in X_train.columns if _is_delta_column(c)]
    if drop_delta:
        X_train = X_train.drop(columns=drop_delta, errors="ignore")
        X_val   = X_val.drop(columns=drop_delta, errors="ignore")

    # [수정] 원본의 set()을 사용한 함수 호출 (변수명 변경: _unordered)
    cb_X_train_unordered, cb_X_val_unordered, _ = _build_cb_matrices(X_train, X_val)

    # --- [이 부분 추가: 재현용 강제 정렬] ---
    if ORIGINAL_A_FEATURES is not None:
        print("[재현] 'ORIGINAL_A_FEATURES' 순서로 강제 재정렬...")
        try:
            # 원본 순서 리스트에서, 현재 데이터프레임에 실제 있는 컬럼들만 필터링
            current_original_order = [f for f in ORIGINAL_A_FEATURES if f in cb_X_train_unordered.columns]
            
            cb_X_train = cb_X_train_unordered[current_original_order].copy()
            cb_X_val = cb_X_val_unordered[current_original_order].copy()
            
            # Cat feature 인덱스도 이 순서 기준으로 다시 계산
            cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]

        except Exception as e:
            print(f"!!! [재현 오류] 원본 피처 순서 재정렬 실패: {e}. sort() 버전으로 강제 실행됩니다.")
            # 실패 시 sort() 버전으로라도 실행
            numeric_cols = sorted([c for c in cb_X_train_unordered.columns if c not in CAT_FEATURES])
            final_cols = numeric_cols + [c for c in CAT_FEATURES if c in cb_X_train_unordered.columns]
            cb_X_train = cb_X_train_unordered[final_cols].copy()
            cb_X_val = cb_X_val_unordered[final_cols].copy()
            cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]
    
    else: # 원본 피처 리스트 파일 로드에 실패한 경우 (sort() 기반으로 실행)
        print("[재현] sort() 기반으로 정렬 실행...")
        numeric_cols = sorted([c for c in cb_X_train_unordered.columns if c not in CAT_FEATURES])
        final_cols = numeric_cols + [c for c in CAT_FEATURES if c in cb_X_train_unordered.columns]
        cb_X_train = cb_X_train_unordered[final_cols].copy()
        cb_X_val = cb_X_val_unordered[final_cols].copy()
        cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]
    # --- [수정 끝] ---
    
    print(f"\n[{group_label}] CatBoost (Base+Hist+Norm+log_ratio, no PK, no Delta) 학습 시작... (피처 {len(cb_X_train.columns)}개)")
    cat_base_model = cb.CatBoostClassifier(
        iterations=3000,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        thread_count=-1,
        early_stopping_rounds=100,
        verbose=1000,
        **BEST_A_PARAMS,
    )
    cat_base_model.fit(
        cb_X_train, y_train,
        eval_set=[(cb_X_val, y_val)],
        cat_features=cat_features_indices
    )

    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic) 시작...")
    pred_uncal = cat_base_model.predict_proba(cb_X_val)[:, 1]
    print(f"[{group_label}] 비보정 단독 점수:")
    _ = combined_score(y_val, pred_uncal)

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(pred_uncal, y_val)

    pred_cal = calibrator.predict(pred_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수:")
    _ = combined_score(y_val, pred_cal)

    return cat_base_model, calibrator


def train_model_B(X_train, y_train, X_val, y_val, pk_stats_fold, group_label="B"):
    """
    B 모델: [재현용 수정] PK Stats 머지 후, ORIGINAL_B_FEATURES 순서로 강제 재정렬
    """
    # PK Stats merge
    X_train = X_train.merge(pk_stats_fold, on='PrimaryKey', how='left')
    X_val   = X_val.merge(pk_stats_fold,   on='PrimaryKey', how='left')

    # [수정] 원본의 set()을 사용한 함수 호출 (변수명 변경: _unordered)
    cb_X_train_unordered, cb_X_val_unordered, _ = _build_cb_matrices(X_train, X_val)

    # --- [이 부분 추가: 재현용 강제 정렬] ---
    if ORIGINAL_B_FEATURES is not None:
        print("[재현] 'ORIGINAL_B_FEATURES' 순서로 강제 재정렬...")
        try:
            # 원본 순서 리스트에서, 현재 데이터프레임에 실제 있는 컬럼들만 필터링
            current_original_order = [f for f in ORIGINAL_B_FEATURES if f in cb_X_train_unordered.columns]

            cb_X_train = cb_X_train_unordered[current_original_order].copy()
            cb_X_val = cb_X_val_unordered[current_original_order].copy()
            
            # Cat feature 인덱스도 이 순서 기준으로 다시 계산
            cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]

        except Exception as e:
            print(f"!!! [재현 오류] 원본 피처 순서 재정렬 실패: {e}. sort() 버전으로 강제 실행됩니다.")
            # 실패 시 sort() 버전으로라도 실행
            numeric_cols = sorted([c for c in cb_X_train_unordered.columns if c not in CAT_FEATURES])
            final_cols = numeric_cols + [c for c in CAT_FEATURES if c in cb_X_train_unordered.columns]
            cb_X_train = cb_X_train_unordered[final_cols].copy()
            cb_X_val = cb_X_val_unordered[final_cols].copy()
            cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]
    
    else: # 원본 피처 리스트 파일 로드에 실패한 경우 (sort() 기반으로 실행)
        print("[재현] sort() 기반으로 정렬 실행...")
        numeric_cols = sorted([c for c in cb_X_train_unordered.columns if c not in CAT_FEATURES])
        final_cols = numeric_cols + [c for c in CAT_FEATURES if c in cb_X_train_unordered.columns]
        cb_X_train = cb_X_train_unordered[final_cols].copy()
        cb_X_val = cb_X_val_unordered[final_cols].copy()
        cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]
    # --- [수정 끝] ---

    # Delta 사용 열 수 체크
    n_delta_train = sum(_is_delta_column(c) for c in cb_X_train.columns)
    print(f"\n[{group_label}] Delta 사용 열 수: {n_delta_train}")

    print(f"[{group_label}] CatBoost (Base+PK+Hist+Norm+Delta+log_ratio) 학습 시작... (피처 {len(cb_X_train.columns)}개)")
    cat_base_model = cb.CatBoostClassifier(
        iterations=3000,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        thread_count=-1,
        early_stopping_rounds=100,
        verbose=1000,
        **BEST_B_PARAMS,
    )
    cat_base_model.fit(
        cb_X_train, y_train,
        eval_set=[(cb_X_val, y_val)],
        cat_features=cat_features_indices
    )

    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic) 시작...")
    pred_uncal = cat_base_model.predict_proba(cb_X_val)[:, 1]
    print(f"[{group_label}] 비보정 단독 점수:")
    _ = combined_score(y_val, pred_uncal)

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(pred_uncal, y_val)

    pred_cal = calibrator.predict(pred_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수:")
    _ = combined_score(y_val, pred_cal)

    return cat_base_model, calibrator


## 6. K-Fold 루프 실행 (PK Stats 생성 + A/B 모델 학습)
all_pk_stats_folds = []

for fold in range(N_SPLITS):
    print(f"\n=== Fold {fold+1}/{N_SPLITS} 학습 시작 ===")

    train_idx = train_indices_list[fold]
    val_idx   = val_indices_list[fold]

    train_df_fold = all_train_df.iloc[train_idx].copy()
    val_df_fold   = all_train_df.iloc[val_idx].copy()

    # --- PK Stats: train_df_fold에서만 생성 (Leakage 방지) ---
    print(f"\n[Fold {fold+1}] K-Fold Target Encoding (PK Stats) 생성...")
    agg_funcs = {
        'Age_num': ['mean', 'min', 'max'],
        'YearMonthIndex': ['mean', 'std', 'min', 'max'],
        'A1_rt_mean': ['mean', 'std'],
        'A4_acc_congruent': ['mean', 'std'],
        'A4_acc_incongruent': ['mean', 'std'],
        # cost → log_ratio
        'A4_stroop_rt_log_ratio': ['mean', 'std'],
        'RiskScore': ['mean', 'std', 'max'],
        'B1_change_acc': ['mean', 'std'],
        'B1_nonchange_acc': ['mean', 'std'],
        'B3_rt_mean': ['mean', 'std'],
        # cost → log_ratio
        'B4_flanker_acc_log_ratio': ['mean', 'std'],
        'B4_rt_mean': ['mean', 'std'],
        'RiskScore_B': ['mean', 'std', 'max'],
        'Test_id': ['count']
    }
    valid_agg_funcs = {col: funcs for col, funcs in agg_funcs.items() if col in train_df_fold.columns}

    pk_stats_fold = train_df_fold.groupby('PrimaryKey').agg(valid_agg_funcs)
    pk_stats_fold.columns = ['_'.join(col).strip() for col in pk_stats_fold.columns.values]
    pk_stats_fold.rename(columns={'Test_id_count': 'pk_test_total_count'}, inplace=True)

    # PK별 A/B 시험 횟수 (train fold 내부에서만)
    if "Test_x" in train_df_fold.columns:
        test_col = "Test_x"
    elif "Test" in train_df_fold.columns:
        test_col = "Test"
    else:
        raise KeyError("PK Stats 생성을 위해 'Test' 혹은 'Test_x' 컬럼이 필요합니다.")

    pk_test_type_count_fold = train_df_fold.groupby('PrimaryKey')[test_col].value_counts().unstack(fill_value=0)
    if 'A' not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold['A'] = 0
    if 'B' not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold['B'] = 0
    pk_test_type_count_fold = pk_test_type_count_fold[['A', 'B']]
    pk_test_type_count_fold.columns = ['pk_test_A_count', 'pk_test_B_count']

    pk_stats_fold = pk_stats_fold.join(pk_test_type_count_fold, how='left').reset_index()
    all_pk_stats_folds.append(pk_stats_fold)

    # --- A/B 데이터 분리 ---
    X_train_A = train_df_fold[train_df_fold[test_col] == 'A'].copy()
    y_train_A = X_train_A['Label'].values
    X_val_A   = val_df_fold[val_df_fold[test_col] == 'A'].copy()
    y_val_A   = X_val_A['Label'].values

    X_train_B = train_df_fold[train_df_fold[test_col] == 'B'].copy()
    y_train_B = X_train_B['Label'].values
    X_val_B   = val_df_fold[val_df_fold[test_col] == 'B'].copy()
    y_val_B   = X_val_B['Label'].values

    # --- 모델 A 학습 ---
    print("\n--- 모델 A (PK Stats 미사용, Delta 제외, log_ratio 포함) 학습 ---")
    cat_A, calib_A = train_model_A(X_train_A, y_train_A, X_val_A, y_val_A, group_label="A")
    joblib.dump(cat_A,   os.path.join(MODEL_SAVE_DIR, f"catboost_A_fold{fold}.pkl"))
    joblib.dump(calib_A, os.path.join(MODEL_SAVE_DIR, f"calibrator_A_fold{fold}.pkl"))

    # --- 모델 B 학습 ---
    print("\n--- 모델 B (PK Stats + Delta + log_ratio, tuned) 학습 ---")
    if len(X_train_B) > 0 and len(X_val_B) > 0 and len(np.unique(y_train_B)) > 1:
        cat_B, calib_B = train_model_B(X_train_B, y_train_B, X_val_B, y_val_B, pk_stats_fold, group_label="B")
        joblib.dump(cat_B,   os.path.join(MODEL_SAVE_DIR, f"catboost_B_fold{fold}.pkl"))
        joblib.dump(calib_B, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_fold{fold}.pkl"))
    else:
        print(f"[Fold {fold+1}] B모델 학습/검증 데이터가 부족하여 이 Fold는 건너뜁니다.")
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"catboost_B_fold{fold}.pkl"))
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_fold{fold}.pkl"))

## 7. Fold별 PK Stats 평균 -> 최종 PK Stats 저장
print("\n[INFO] K-Fold PK Stats 병합...")
if all_pk_stats_folds:
    all_pk_stats_df = pd.concat(all_pk_stats_folds, ignore_index=True)
    final_pk_stats  = all_pk_stats_df.groupby('PrimaryKey').mean().reset_index()
    final_pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    final_pk_stats.to_csv(final_pk_stats_path, index=False)
else:
    print("경고: 유효한 PK Stats가 생성되지 않았습니다. 빈 파일을 생성합니다.")
    final_pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    pd.DataFrame().to_csv(final_pk_stats_path, index=False)

print("\n[INFO] '최종 보정' CatBoost 모델 10개 및 최종 PK 통계 피처 저장 완료:")
print(f"  - PK Stats: {final_pk_stats_path}")
print("Big-Tech ML Engineer (v18+Delta+log_ratio): 미션 완료.")
