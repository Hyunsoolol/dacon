## ----------------------------------------------------------------
## [Jupyter용] 2_Tune_CatBoost.ipynb
##  - 1_Preprocess.ipynb 실행 후, base 파이프라인 고정 상태에서
##    CatBoost 하이퍼파라미터 튜닝 (fold0 기준 랜덤 서치)
## ----------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm
import warnings
import itertools
import random

warnings.filterwarnings('ignore')
tqdm.pandas()

print("Big-Tech ML Engineer (Tuning): Base 파이프라인 고정 + CatBoost 하이퍼파라미터 튜닝 시작.")

# ----------------------------------------------------------------
# 1. 전처리된 Feather 로드 (1_Preprocess.ipynb가 생성한 파일)
# ----------------------------------------------------------------
BASE_DIR = "./data"
FEATURE_SAVE_PATH = os.path.join(BASE_DIR, "all_train_data.feather")

try:
    print(f"전처리된 메인 피처 로드 중... ({FEATURE_SAVE_PATH})")
    all_train_df = pd.read_feather(FEATURE_SAVE_PATH)
except FileNotFoundError as e:
    print(f"경고: {e}")
    print("먼저 1_Preprocess.ipynb를 실행하여 all_train_data.feather를 생성해야 합니다.")
    raise

print("데이터 로드 완료:", all_train_df.shape)

# ----------------------------------------------------------------
# 2. ECE / Combined Score 함수 (base 모델과 동일 정의)
# ----------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    if len(y_true) == 0 or len(y_prob) == 0:
        return 0.0

    y_prob = np.nan_to_num(y_prob, nan=0.0)
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0] = -0.001
    bin_edges[-1] = 1.001

    df['y_prob'] = np.clip(df['y_prob'], 0, 1)
    df['bin'] = pd.cut(df['y_prob'], bins=bin_edges, right=True)

    bin_stats = df.groupby('bin', observed=True).agg(
        bin_total=('y_prob', 'count'),
        prob_true=('y_true', 'mean'),
        prob_pred=('y_prob', 'mean')
    )

    non_empty_bins = bin_stats[bin_stats['bin_total'] > 0]
    if len(non_empty_bins) == 0:
        return 0.0

    bin_weights = non_empty_bins['bin_total'] / len(y_prob)
    prob_true = non_empty_bins['prob_true']
    prob_pred = non_empty_bins['prob_pred']

    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece


def combined_score(y_true, y_prob):
    if (
        len(y_true) == 0 or len(y_prob) == 0 or
        np.sum(y_true) == 0 or np.sum(y_true) == len(y_true)
    ):
        print("  AUC: N/A (단일 클래스), Brier: N/A, ECE: N/A (No data)")
        return 1.0  # 최악 점수로 취급

    y_prob = np.nan_to_num(y_prob, nan=0.0)

    mean_auc = roc_auc_score(y_true, y_prob)
    mean_brier = mean_squared_error(y_true, y_prob)
    mean_ece = expected_calibration_error(y_true, y_prob)

    score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    print(f"  AUC: {mean_auc:.4f}, Brier: {mean_brier:.4f}, ECE: {mean_ece:.4f}")
    print(f"  Combined Score: {score:.5f}")
    return score

# ----------------------------------------------------------------
# 3. Stratified K-Fold 생성 (base와 동일: 5-Fold, random_state=42)
# ----------------------------------------------------------------
print("\n[INFO] K-Fold 교차 검증 분리 시작...")
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

train_indices_list = []
val_indices_list = []

all_train_df['Label'] = all_train_df['Label'].fillna(0)

for train_idx, val_idx in skf.split(all_train_df, all_train_df['Label']):
    train_indices_list.append(train_idx)
    val_indices_list.append(val_idx)

print(f"{N_SPLITS}-Fold 분리 완료.")

# base 파이프라인 공통 설정
CAT_FEATURES = ['Age', 'PrimaryKey']
DROP_COLS_TRAIN = ['Test_id', 'Test_x', 'Test_y', 'Label', 'TestDate', 'Year', 'Month', 'base_index']

# ----------------------------------------------------------------
# 4. 튜닝용 CatBoost 기본 파라미터 + 탐색 공간 정의
# ----------------------------------------------------------------
BASE_CB_PARAMS = {
    "iterations": 2000,            # 튜닝 단계에서는 2000으로 줄여 속도 확보
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "thread_count": -1,
    "early_stopping_rounds": 100,
    "verbose": 200,
}

# A / B 모델 각각 탐색 공간 (필요시 직접 줄여도 됨)
param_search_space_A = {
    "depth": [4, 5, 6, 7],
    "learning_rate": [0.03, 0.05, 0.07],
    "l2_leaf_reg": [2, 3, 5, 7, 10],
    "random_strength": [0.5, 1.0, 2.0],
    "bagging_temperature": [0, 0.5, 1.0],
    "border_count": [128, 254],
}

param_search_space_B = {
    "depth": [4, 5, 6, 7],
    "learning_rate": [0.03, 0.05, 0.07],
    "l2_leaf_reg": [2, 3, 5, 7, 10],
    "random_strength": [0.5, 1.0, 2.0],
    "bagging_temperature": [0, 0.5, 1.0],
    "border_count": [128, 254],
}

def generate_param_candidates(search_space, n_samples=30, seed=42):
    """
    search_space의 전체 조합에서 랜덤하게 n_samples개 샘플링
    """
    keys = list(search_space.keys())
    all_combos = list(itertools.product(*[search_space[k] for k in keys]))
    random.seed(seed)
    random.shuffle(all_combos)
    for combo in all_combos[:n_samples]:
        yield {k: v for k, v in zip(keys, combo)}

# ----------------------------------------------------------------
# 5. A 모델 튜닝 (fold0 기준)
# ----------------------------------------------------------------
def evaluate_params_A_on_fold0(all_train_df, train_indices_list, val_indices_list, params):
    fold = 0
    train_idx = train_indices_list[fold]
    val_idx = val_indices_list[fold]

    train_df_fold = all_train_df.iloc[train_idx]
    val_df_fold = all_train_df.iloc[val_idx]

    X_train_A = train_df_fold[train_df_fold['Test_x'] == 'A'].copy()
    y_train_A = X_train_A['Label'].values
    X_val_A = val_df_fold[val_df_fold['Test_x'] == 'A'].copy()
    y_val_A = X_val_A['Label'].values

    # Label 분포 체크 (혹시 단일 클래스면 튜닝 불가)
    if len(np.unique(y_train_A)) < 2 or len(np.unique(y_val_A)) < 2:
        print("[A-fold0] 경고: 단일 클래스 Fold, 이 조합은 건너뜀")
        return 1.0

    # numeric / cat 분리 (PK Stats는 A에서 사용하지 않음)
    base_cols = [col for col in X_train_A.columns if not col.startswith('pk_')]
    numeric_cols = list(set(base_cols) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))

    cb_X_train = X_train_A[numeric_cols + CAT_FEATURES].copy()
    cb_X_val = X_val_A[numeric_cols + CAT_FEATURES].copy()

    for col in CAT_FEATURES:
        cb_X_train[col] = cb_X_train[col].fillna('nan').astype(str)
        cb_X_val[col] = cb_X_val[col].fillna('nan').astype(str)

    cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]

    cb_params = BASE_CB_PARAMS.copy()
    cb_params.update(params)

    print(f"\n[A-fold0] 튜닝 파라미터: {params}")
    model = cb.CatBoostClassifier(**cb_params)

    model.fit(
        cb_X_train, y_train_A,
        eval_set=[(cb_X_val, y_val_A)],
        cat_features=cat_features_indices
    )

    pred_val = model.predict_proba(cb_X_val)[:, 1]
    score = combined_score(y_val_A, pred_val)
    return score


print("\n================ A 모델 튜닝 시작 (fold0 기준) ================")
N_TRIALS_A = 30   # 필요하면 10~20으로 줄여도 됨

results_A = []
for i, params in enumerate(generate_param_candidates(param_search_space_A, n_samples=N_TRIALS_A, seed=42)):
    print(f"\n=== [A] 튜닝 {i+1}/{N_TRIALS_A} ===")
    score = evaluate_params_A_on_fold0(all_train_df, train_indices_list, val_indices_list, params)
    results_A.append((score, params))

results_A_sorted = sorted(results_A, key=lambda x: x[0])
print("\n=== [A] 모델 튜닝 결과 TOP 5 (Combined Score 낮을수록 좋음) ===")
for s, p in results_A_sorted[:5]:
    print(f"Score={s:.5f}, params={p}")

best_A_score, best_A_params = results_A_sorted[0]
print(f"\n>>> [A] Best Score={best_A_score:.5f}, Best Params={best_A_params}")

# ----------------------------------------------------------------
# 6. B 모델 튜닝 (fold0 기준, PK Stats 포함)
# ----------------------------------------------------------------
def evaluate_params_B_on_fold0(all_train_df, train_indices_list, val_indices_list, params):
    fold = 0
    train_idx = train_indices_list[fold]
    val_idx = val_indices_list[fold]

    train_df_fold = all_train_df.iloc[train_idx].copy()
    val_df_fold = all_train_df.iloc[val_idx].copy()

    # --- PK Stats 생성 (base 코드와 동일 로직) ---
    agg_funcs = {
        'Age_num': ['mean', 'min', 'max'], 'YearMonthIndex': ['mean', 'std', 'min', 'max'],
        'A1_rt_mean': ['mean', 'std'],
        'A4_acc_congruent': ['mean', 'std'], 'A4_acc_incongruent': ['mean', 'std'], 'A4_stroop_rt_cost': ['mean', 'std'],
        'RiskScore': ['mean', 'std', 'max'],
        'B1_change_acc': ['mean', 'std'], 'B1_nonchange_acc': ['mean', 'std'],
        'B3_rt_mean': ['mean', 'std'],
        'B4_flanker_acc_cost': ['mean', 'std'], 'B4_rt_mean': ['mean', 'std'],
        'RiskScore_B': ['mean', 'std', 'max'],
        'Test_id': ['count']
    }
    valid_agg_funcs = {col: funcs for col, funcs in agg_funcs.items() if col in train_df_fold.columns}
    pk_stats_fold = train_df_fold.groupby('PrimaryKey').agg(valid_agg_funcs)
    pk_stats_fold.columns = ['_'.join(col).strip() for col in pk_stats_fold.columns.values]
    pk_stats_fold.rename(columns={'Test_id_count': 'pk_test_total_count'}, inplace=True)

    pk_test_type_count_fold = train_df_fold.groupby('PrimaryKey')['Test_x'].value_counts().unstack(fill_value=0)
    if 'A' not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold['A'] = 0
    if 'B' not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold['B'] = 0
    pk_test_type_count_fold = pk_test_type_count_fold[['A', 'B']]
    pk_test_type_count_fold.columns = ['pk_test_A_count', 'pk_test_B_count']
    pk_stats_fold = pk_stats_fold.join(pk_test_type_count_fold, how='left').reset_index()

    # --- B 데이터셋 분리 ---
    X_train_B = train_df_fold[train_df_fold['Test_x'] == 'B'].copy()
    y_train_B = X_train_B['Label'].values
    X_val_B = val_df_fold[val_df_fold['Test_x'] == 'B'].copy()
    y_val_B = X_val_B['Label'].values

    if len(np.unique(y_train_B)) < 2 or len(np.unique(y_val_B)) < 2:
        print("[B-fold0] 경고: 단일 클래스 Fold, 이 조합은 건너뜀")
        return 1.0

    # PK Stats merge
    X_train_B = X_train_B.merge(pk_stats_fold, on='PrimaryKey', how='left')
    X_val_B = X_val_B.merge(pk_stats_fold, on='PrimaryKey', how='left')

    numeric_cols = list(set(X_train_B.columns) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))
    common_numeric_cols = list(set(X_train_B[numeric_cols].columns) & set(X_val_B[numeric_cols].columns))

    cb_X_train = X_train_B[common_numeric_cols + CAT_FEATURES].copy()
    cb_X_val = X_val_B[common_numeric_cols + CAT_FEATURES].copy()

    for col in CAT_FEATURES:
        cb_X_train[col] = cb_X_train[col].fillna('nan').astype(str)
        cb_X_val[col] = cb_X_val[col].fillna('nan').astype(str)

    cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]

    cb_params = BASE_CB_PARAMS.copy()
    cb_params.update(params)

    print(f"\n[B-fold0] 튜닝 파라미터: {params}")
    model = cb.CatBoostClassifier(**cb_params)

    model.fit(
        cb_X_train, y_train_B,
        eval_set=[(cb_X_val, y_val_B)],
        cat_features=cat_features_indices
    )

    pred_val = model.predict_proba(cb_X_val)[:, 1]
    score = combined_score(y_val_B, pred_val)
    return score


print("\n================ B 모델 튜닝 시작 (fold0 기준) ================")
N_TRIALS_B = 30   # 필요하면 10~20으로 줄여도 됨

results_B = []
for i, params in enumerate(generate_param_candidates(param_search_space_B, n_samples=N_TRIALS_B, seed=777)):
    print(f"\n=== [B] 튜닝 {i+1}/{N_TRIALS_B} ===")
    score = evaluate_params_B_on_fold0(all_train_df, train_indices_list, val_indices_list, params)
    results_B.append((score, params))

results_B_sorted = sorted(results_B, key=lambda x: x[0])
print("\n=== [B] 모델 튜닝 결과 TOP 5 (Combined Score 낮을수록 좋음) ===")
for s, p in results_B_sorted[:5]:
    print(f"Score={s:.5f}, params={p}")

best_B_score, best_B_params = results_B_sorted[0]
print(f"\n>>> [B] Best Score={best_B_score:.5f}, Best Params={best_B_params}")

print("\n[INFO] 튜닝 완료.")
print(" - 위에서 출력된 best_A_params, best_B_params를 2_Train_Models.ipynb의")
print("   CatBoost 설정(iterations=3000 등)에 반영한 뒤, 전체 5-Fold 재학습하면 됨.")
