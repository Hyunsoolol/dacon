## ----------------------------------------------------------------
## [Jupyter용] 2_Train_Models_v21_cal.py
##  - v18 구조 유지
##    * A: Base + Hist + Norm (PK Stats 미사용, Delta 제외)
##    * B: Base + PK + Hist + Norm + Delta(B-only)
##  - *_cost → *_log_ratio 기반 도메인 피처 가정
##  - [GPU 기본] task_type='GPU' 기본, 장치 자동 감지. 없으면 CPU 폴백
##  - [GPU binning] max_bin 기본 254 (환경변수로 변경 가능)
##  - [추가] 학습 단계에서 캘린더/주기 피처 생성(외부 CSV 없음)
## ----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import catboost as cb
from catboost.utils import get_gpu_device_count
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Big-Tech ML Engineer (v20 시작.)")

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
    # (GPU에서는 border_count 대신 max_bin을 사용)
}

# ---- GPU/CPU 선택 옵션 ----
# 기본 GPU, 장치 미검출 시에만 CPU 폴백
CAT_TASK_TYPE = "CPU"
try:
    _ngpu = get_gpu_device_count()
    if _ngpu <= 0:
        print("[WARN] GPU 장치를 찾지 못했습니다. CPU로 폴백합니다.")
        CAT_TASK_TYPE = "CPU"
except Exception:
    print("[WARN] GPU 장치 수 확인 실패. CPU로 폴백합니다.")
    CAT_TASK_TYPE = "CPU"

CAT_DEVICES = os.getenv("CAT_DEVICES", "0")         # 예: "0" 또는 "0-1"
CAT_MAX_BIN = int(os.getenv("CAT_MAX_BIN", "254"))  # GPU binning 해상도

print(f"[INFO] CatBoost task_type={CAT_TASK_TYPE}, devices={CAT_DEVICES}, max_bin={CAT_MAX_BIN}")

## 1. 로컬 경로 및 데이터 로드
BASE_DIR = "./data"
MODEL_SAVE_DIR = "./model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

FEATURE_SAVE_PATH = os.path.join(BASE_DIR, "all_train_data.feather")

print("전처리된 메인 피처 로드 중...")
all_train_df = pd.read_feather(FEATURE_SAVE_PATH)
print("데이터 로드 완료.")

# ---------- (신규) 외부 CSV 없이도 캘린더/주기 피처 생성 ----------
def _make_calendar_feats(df):
    if ("Year" not in df.columns) or ("Month" not in df.columns):
        return df
    out = df.copy()
    out["Month_sin"] = np.sin(2 * np.pi * (out["Month"] - 1) / 12.0)
    out["Month_cos"] = np.cos(2 * np.pi * (out["Month"] - 1) / 12.0)
    out["Quarter"] = ((out["Month"] - 1) // 3 + 1).astype(int)
    out["is_year_end"] = (out["Month"] == 12).astype(int)
    ym = out["Year"] * 100 + out["Month"]
    out["is_covid"] = ((ym >= 202003) & (ym <= 202205)).astype(int)
    return out

def _attach_external_monthly(df_all):
    out = df_all.copy()
    if {"Year","Month"}.issubset(out.columns):
        out["yyyymm"] = out["Year"] * 100 + out["Month"]
    elif "YearMonthIndex" in out.columns:
        year = (out["YearMonthIndex"] // 12).astype(int)
        month = (out["YearMonthIndex"] % 12).astype(int)
        month = month.replace(0, 12)
        year = np.where(month == 12, year - 1, year)
        out["yyyymm"] = year * 100 + month
    else:
        print("[WARN] yyyymm 키 생성 불가 → 캘린더/주기 피처 스킵")
        return out
    out = _make_calendar_feats(out)
    print("[INFO] 외부 CSV 없이 캘린더/주기 피처 추가 완료")
    return out

all_train_df = _attach_external_monthly(all_train_df)

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

train_indices_list, val_indices_list = [], []
all_train_df['Label'] = all_train_df['Label'].fillna(0)

for train_idx, val_idx in skf.split(all_train_df, all_train_df['Label']):
    train_indices_list.append(train_idx)
    val_indices_list.append(val_idx)
print(f"{N_SPLITS}-Fold 분리 완료.")

CAT_FEATURES    = ['Age', 'PrimaryKey']
DROP_COLS_TRAIN = ['Test_id', 'Test_x', 'Test_y', 'Label', 'TestDate', 'Year', 'Month', 'base_index', 'yyyymm']

def _is_delta_column(col: str) -> bool:
    return col.startswith("delta_")

## 4. 공통 CatBoost 입력 구성 함수 (컬럼 순서 보존 + 숫자형만)
def _build_cb_matrices(X_train, X_val):
    candidate = [c for c in X_train.columns if c not in CAT_FEATURES + DROP_COLS_TRAIN]
    numeric_cols = [c for c in candidate if pd.api.types.is_numeric_dtype(X_train[c])]
    common_numeric_cols = [c for c in numeric_cols if (c in X_val.columns) and pd.api.types.is_numeric_dtype(X_val[c])]
    cb_X_train = X_train[common_numeric_cols + [c for c in CAT_FEATURES if c in X_train.columns]].copy()
    cb_X_val   = X_val[common_numeric_cols + [c for c in CAT_FEATURES if c in X_val.columns]].copy()
    for col in CAT_FEATURES:
        if col in cb_X_train.columns:
            cb_X_train[col] = cb_X_train[col].fillna('nan').astype(str)
    for col in CAT_FEATURES:
        if col in cb_X_val.columns:
            cb_X_val[col] = cb_X_val[col].fillna('nan').astype(str)
    cat_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]
    return cb_X_train, cb_X_val, cat_indices

def _inject_gpu_bins(params: dict) -> dict:
    if CAT_TASK_TYPE.upper() == "GPU":
        return {**params, "max_bin": CAT_MAX_BIN}
    return params

## 5. 모델 학습 함수
def train_model_A(X_train, y_train, X_val, y_val, group_label="A"):
    drop_delta = [c for c in X_train.columns if _is_delta_column(c)]
    if drop_delta:
        X_train = X_train.drop(columns=drop_delta, errors="ignore")
        X_val   = X_val.drop(columns=drop_delta, errors="ignore")
    cb_X_train, cb_X_val, cat_features_indices = _build_cb_matrices(X_train, X_val)
    params = _inject_gpu_bins(BEST_A_PARAMS)

    print(f"\n[{group_label}] CatBoost (Base+Hist+Norm+log_ratio, no PK, no Delta) 학습 시작... (피처 {len(cb_X_train.columns)}개)")
    cat_base_model = cb.CatBoostClassifier(
        iterations=3000,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        thread_count=-1,
        early_stopping_rounds=100,
        verbose=1000,
        task_type=CAT_TASK_TYPE,
        devices=CAT_DEVICES,
        auto_class_weights='Balanced',
        **params,
    )
    cat_base_model.fit(cb_X_train, y_train, eval_set=[(cb_X_val, y_val)], cat_features=cat_features_indices)

    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic) 시작...")
    pred_uncal = cat_base_model.predict_proba(cb_X_val)[:, 1]
    print(f"[{group_label}] 비보정 단독 점수:")
    _ = combined_score(y_val, pred_uncal)
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(pred_uncal, y_val)
    pred_cal = calibrator.predict(pred_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수:")
    _ = combined_score(y_val, pred_cal)
    return cat_base_model, calibrator

def train_model_B(X_train, y_train, X_val, y_val, pk_stats_fold, group_label="B"):
    X_train = X_train.merge(pk_stats_fold, on='PrimaryKey', how='left')
    X_val   = X_val.merge(pk_stats_fold,   on='PrimaryKey', how='left')
    cb_X_train, cb_X_val, cat_features_indices = _build_cb_matrices(X_train, X_val)
    n_delta_train = sum(_is_delta_column(c) for c in cb_X_train.columns)
    print(f"\n[{group_label}] Delta 사용 열 수: {n_delta_train}")
    params = _inject_gpu_bins(BEST_B_PARAMS)

    print(f"[{group_label}] CatBoost (Base+PK+Hist+Norm+Delta+log_ratio) 학습 시작... (피처 {len(cb_X_train.columns)}개)")
    cat_base_model = cb.CatBoostClassifier(
        iterations=3000,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        thread_count=-1,
        early_stopping_rounds=100,
        verbose=1000,
        task_type=CAT_TASK_TYPE,
        devices=CAT_DEVICES,
        auto_class_weights='Balanced',
        **params,
    )
    cat_base_model.fit(cb_X_train, y_train, eval_set=[(cb_X_val, y_val)], cat_features=cat_features_indices)

    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic) 시작...")
    pred_uncal = cat_base_model.predict_proba(cb_X_val)[:, 1]
    print(f"[{group_label}] 비보정 단독 점수:")
    _ = combined_score(y_val, pred_uncal)
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(pred_uncal, y_val)
    pred_cal = calibrator.predict(pred_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수:")
    _ = combined_score(y_val, pred_cal)
    return cat_base_model, calibrator

## 6. K-Fold 루프 실행 (PK Stats 생성 + A/B 모델 학습)
all_pk_stats_folds = []
for fold in range(N_SPLITS):
    print(f"\n=== Fold {fold+1}/{N_SPLITS} 학습 시작 ===")
    train_idx, val_idx = train_indices_list[fold], val_indices_list[fold]
    train_df_fold = all_train_df.iloc[train_idx].copy()
    val_df_fold   = all_train_df.iloc[val_idx].copy()

    # --- PK Stats (누설 방지: train fold에서만) ---
    print(f"\n[Fold {fold+1}] K-Fold Target Encoding (PK Stats) 생성...")
    agg_funcs = {
       'B3_rt_mean': ['mean', 'std'],
       'B1_acc_log_ratio': ['mean', 'std'],
       'B2_acc_log_ratio': ['mean', 'std'],
       'Test_id': ['count'],
    }
    valid_agg_funcs = {col: funcs for col, funcs in agg_funcs.items() if col in train_df_fold.columns}
    pk_stats_fold = train_df_fold.groupby('PrimaryKey').agg(valid_agg_funcs)
    pk_stats_fold.columns = ['_'.join(col).strip() for col in pk_stats_fold.columns.values]
    pk_stats_fold.rename(columns={'Test_id_count': 'pk_test_total_count'}, inplace=True)

    if "Test_x" in train_df_fold.columns:
        test_col = "Test_x"
    elif "Test" in train_df_fold.columns:
        test_col = "Test"
    else:
        raise KeyError("PK Stats 생성을 위해 'Test' 혹은 'Test_x' 컬럼이 필요합니다.")

    pk_test_type_count_fold = train_df_fold.groupby('PrimaryKey')[test_col].value_counts().unstack(fill_value=0)
    if 'A' not in pk_test_type_count_fold.columns: pk_test_type_count_fold['A'] = 0
    if 'B' not in pk_test_type_count_fold.columns: pk_test_type_count_fold['B'] = 0
    pk_test_type_count_fold = pk_test_type_count_fold[['A', 'B']]
    pk_test_type_count_fold.columns = ['pk_test_A_count', 'pk_test_B_count']

    pk_stats_fold = pk_stats_fold.join(pk_test_type_count_fold, how='left').reset_index()
    all_pk_stats_folds.append(pk_stats_fold)

    # --- A/B 데이터 분리 ---
    X_train_A = train_df_fold[train_df_fold[test_col] == 'A'].copy(); y_train_A = X_train_A['Label'].values
    X_val_A   = val_df_fold[val_df_fold[test_col] == 'A'].copy();   y_val_A   = X_val_A['Label'].values
    X_train_B = train_df_fold[train_df_fold[test_col] == 'B'].copy(); y_train_B = X_train_B['Label'].values
    X_val_B   = val_df_fold[val_df_fold[test_col] == 'B'].copy();   y_val_B   = X_val_B['Label'].values

    # --- 모델 A ---
    print("\n--- 모델 A (PK Stats 미사용, Delta 제외, log_ratio 포함) 학습 ---")
    cat_A, calib_A = train_model_A(X_train_A, y_train_A, X_val_A, y_val_A, group_label="A")
    joblib.dump(cat_A,   os.path.join(MODEL_SAVE_DIR, f"catboost_A_fold{fold}.pkl"))
    joblib.dump(calib_A, os.path.join(MODEL_SAVE_DIR, f"calibrator_A_fold{fold}.pkl"))

    # --- 모델 B ---
    print("\n--- 모델 B (PK Stats + Delta + log_ratio, tuned) 학습 ---")
    if len(X_train_B) > 0 and len(X_val_B) > 0 and len(np.unique(y_train_B)) > 1:
        cat_B, calib_B = train_model_B(X_train_B, y_train_B, X_val_B, y_val_B, pk_stats_fold, group_label="B")
        joblib.dump(cat_B,   os.path.join(MODEL_SAVE_DIR, f"catboost_B_fold{fold}.pkl"))
        joblib.dump(calib_B, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_fold{fold}.pkl"))
    else:
        print(f"[Fold {fold+1}] B모델 학습/검증 데이터 부족으로 스킵")
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"catboost_B_fold{fold}.pkl"))
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_fold{fold}.pkl"))

## 7. Fold별 PK Stats 평균 -> 최종 저장
print("\n[INFO] K-Fold PK Stats 병합...")
if all_pk_stats_folds:
    all_pk_stats_df = pd.concat(all_pk_stats_folds, ignore_index=True)
    final_pk_stats  = all_pk_stats_df.groupby('PrimaryKey').mean(numeric_only=True).reset_index()
    final_pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    final_pk_stats.to_csv(final_pk_stats_path, index=False)
else:
    print("경고: 유효한 PK Stats가 생성되지 않았습니다. 빈 파일을 생성합니다.")
    final_pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    pd.DataFrame().to_csv(final_pk_stats_path, index=False)

print("\n[INFO] '최종 보정' CatBoost 모델 10개 및 최종 PK 통계 피처 저장 완료:")
print(f"  - PK Stats: {final_pk_stats_path}")
print("Big-Tech ML Engineer (v21+cal): 미션 완료.")
