## ----------------------------------------------------------------
## [Jupyter용] 2_Train_Models.ipynb (빠른 재실행용, 최종 수정본)
##
## 목적: 1번 노트북이 저장한 Feather를 불러와 모델 학습/보정만 수행
## [수정] Data Leakage를 완벽히 수정한 K-Fold 로직 적용
## ----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold # K-Fold
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error
from sklearn.calibration import calibration_curve 
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')
print("Big-Tech ML Engineer: K-Fold + Leakage Fix + Domain Features 최종 전략 시작.")
## 1. 로컬 경로 설정 및 "전처리된 데이터" 로드
BASE_DIR = "./data"
MODEL_SAVE_DIR = "./model" 
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
FEATURE_SAVE_PATH = os.path.join(BASE_DIR, "all_train_data.feather")
try:
    print(f"전처리된 메인 피처 로드 중... (매우 빠름)")
    all_train_df = pd.read_feather(FEATURE_SAVE_PATH)
except FileNotFoundError as e:
    print(f"경고: {e}")
    print("먼저 1_Preprocess.ipynb를 실행하여 all_train_data.feather를 생성해야 합니다.")
    raise
print("데이터 로드 완료.")
# ECE 계산기 (버그 수정됨)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    if len(y_true) == 0 or len(y_prob) == 0: return 0.0
    y_prob = np.nan_to_num(y_prob, nan=0.0)
    
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0] = -0.001 
    bin_edges[-1] = 1.001 
    
    # pd.cut에서 NA가 발생하지 않도록 y_prob의 범위를 강제로 [0, 1]로 클리핑
    df['y_prob'] = np.clip(df['y_prob'], 0, 1)
    df['bin'] = pd.cut(df['y_prob'], bins=bin_edges, right=True)
    
    # .groupby(..., observed=True) to handle potential empty bins
    bin_stats = df.groupby('bin', observed=True).agg(
        bin_total=('y_prob', 'count'),
        prob_true=('y_true', 'mean'),
        prob_pred=('y_prob', 'mean')
    )
    
    non_empty_bins = bin_stats[bin_stats['bin_total'] > 0]
    if len(non_empty_bins) == 0: return 0.0
    
    bin_weights = non_empty_bins['bin_total'] / len(y_prob)
    prob_true = non_empty_bins['prob_true']
    prob_pred = non_empty_bins['prob_pred']
    
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece
# 최종 점수 계산기
def combined_score(y_true, y_prob):
    if len(y_true) == 0 or len(y_prob) == 0 or np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        print("  AUC: N/A (단일 클래스), Brier: N/A, ECE: N/A (No data)")
        return 1.0 
    
    y_prob = np.nan_to_num(y_prob, nan=0.0) 
    
    mean_auc = roc_auc_score(y_true, y_prob)
    mean_brier = mean_squared_error(y_true, y_prob)
    mean_ece = expected_calibration_error(y_true, y_prob) 
    score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    print(f"  AUC: {mean_auc:.4f}, Brier: {mean_brier:.4f}, ECE: {mean_ece:.4f}")
    print(f"  Combined Score: {score:.5f}")
    return score
## 6. K-Fold 분리 (오류 수정)
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
CAT_FEATURES = ['Age', 'PrimaryKey'] 
# [수정] 'Test' -> 'Test_x' (1_Preprocess에서 병합 시 생성된 이름)
DROP_COLS_TRAIN = ['Test_id', 'Test_x', 'Test_y', 'Label', 'TestDate', 'Year', 'Month', 'base_index']
## 7. Model A (CatBoost Only) - [수정] PK Stats 제거
def train_model_A(X_train, y_train, X_val, y_val, group_label="A"):
    
    # 피처 정의 (PK Stats 관련 피처 제외)
    base_cols = [col for col in X_train.columns if not col.startswith('pk_')]
    numeric_cols = list(set(base_cols) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))
    
    cb_X_train = X_train[numeric_cols + CAT_FEATURES]
    cb_X_val = X_val[numeric_cols + CAT_FEATURES]
    
    for col in CAT_FEATURES:
        cb_X_train[col] = cb_X_train[col].fillna('nan').astype(str)
        cb_X_val[col] = cb_X_val[col].fillna('nan').astype(str)
        
    cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train]
    
    print(f"\n[{group_label}] CatBoost (Base) 학습 시작... (피처 {len(cb_X_train.columns)}개)")
    cat_base_model = cb.CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05, 
        depth=6, 
        l2_leaf_reg=3, 
        loss_function='Logloss', 
        eval_metric='AUC',
        random_seed=42,
        thread_count=-1,
        early_stopping_rounds=100, 
        verbose=1000
    )
    cat_base_model.fit(
        cb_X_train, y_train,
        eval_set=[(cb_X_val, y_val)],
        cat_features=cat_features_indices
    )
    
    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic) 시작...")
    pred_cat_uncal = cat_base_model.predict_proba(cb_X_val)[:,1]
    print(f"[{group_label}] 비보정 단독 점수:")
    _ = combined_score(y_val, pred_cat_uncal)
    
    final_calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    final_calibrator.fit(pred_cat_uncal, y_val) 
    
    pred_cat_calibrated = final_calibrator.predict(pred_cat_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수:")
    _ = combined_score(y_val, pred_cat_calibrated)
    
    return cat_base_model, final_calibrator
## 7. Model B (CatBoost Only)
def train_model_B(X_train, y_train, X_val, y_val, pk_stats_fold, group_label="B"):
    
    # --- [Leakage 수정] PK Stats 병합 ---
    X_train = X_train.merge(pk_stats_fold, on='PrimaryKey', how='left')
    X_val = X_val.merge(pk_stats_fold, on='PrimaryKey', how='left')
    # --- 피처 정의 ---
    numeric_cols = list(set(X_train.columns) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))
    common_numeric_cols = list(set(X_train[numeric_cols].columns) & set(X_val[numeric_cols].columns))
    
    cb_X_train = X_train[common_numeric_cols + CAT_FEATURES]
    cb_X_val = X_val[common_numeric_cols + CAT_FEATURES]
    
    for col in CAT_FEATURES:
        cb_X_train[col] = cb_X_train[col].fillna('nan').astype(str)
        cb_X_val[col] = cb_X_val[col].fillna('nan').astype(str)
        
    cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train]
    
    # --- CatBoost (Base) 학습 ---
    print(f"\n[{group_label}] CatBoost (Base) 학습 시작... (피처 {len(cb_X_train.columns)}개)")
    cat_base_model = cb.CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05, 
        depth=6, 
        l2_leaf_reg=3, 
        loss_function='Logloss', 
        eval_metric='AUC',
        random_seed=42,
        thread_count=-1,
        early_stopping_rounds=100, 
        verbose=1000
    )
    cat_base_model.fit(
        cb_X_train, y_train,
        eval_set=[(cb_X_val, y_val)],
        cat_features=cat_features_indices
    )
    
    # --- 보정기 학습 ---
    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic) 시작...")
    pred_cat_uncal = cat_base_model.predict_proba(cb_X_val)[:,1]
    print(f"[{group_label}] 비보정 단독 점수:")
    _ = combined_score(y_val, pred_cat_uncal)
    
    final_calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    final_calibrator.fit(pred_cat_uncal, y_val) 
    
    pred_cat_calibrated = final_calibrator.predict(pred_cat_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수:")
    _ = combined_score(y_val, pred_cat_calibrated)
    
    return cat_base_model, final_calibrator
# K-Fold 루프 실행
all_pk_stats_folds = [] 
for fold in range(N_SPLITS):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} 학습 시작 ---")
    
    train_idx = train_indices_list[fold]
    val_idx = val_indices_list[fold]
    
    # 1. 전체 데이터에서 Train/Val 인덱스로 분리
    train_df_fold = all_train_df.iloc[train_idx]
    val_df_fold = all_train_df.iloc[val_idx]
    # 2. [Leakage 수정] PK Stats를 'train_df_fold'로만 생성
    print(f"\n[Fold {fold+1}] K-Fold Target Encoding (PK Stats) 생성...")
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
    
    # [!!!!! KeyError 수정: 'Test_x' 사용 !!!!!]
    pk_test_type_count_fold = train_df_fold.groupby('PrimaryKey')['Test_x'].value_counts().unstack(fill_value=0)
    if 'A' not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold['A'] = 0
    if 'B' not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold['B'] = 0
    pk_test_type_count_fold = pk_test_type_count_fold[['A', 'B']]
    pk_test_type_count_fold.columns = ['pk_test_A_count', 'pk_test_B_count']
    pk_stats_fold = pk_stats_fold.join(pk_test_type_count_fold, how='left').reset_index()
    
    all_pk_stats_folds.append(pk_stats_fold) # Fold별 PK Stats 저장
    # 3. A모델용 데이터 생성
    # [!!!!! KeyError 수정: 'Test_x' 사용 !!!!!]
    X_train_A = train_df_fold[train_df_fold['Test_x'] == 'A'].copy()
    y_train_A = X_train_A['Label'].values
    X_val_A = val_df_fold[val_df_fold['Test_x'] == 'A'].copy()
    y_val_A = X_val_A['Label'].values
    # 4. B모델용 데이터 생성
    X_train_B = train_df_fold[train_df_fold['Test_x'] == 'B'].copy()
    y_train_B = X_train_B['Label'].values
    X_val_B = val_df_fold[val_df_fold['Test_x'] == 'B'].copy()
    y_val_B = X_val_B['Label'].values
    # 5. 모델 A 학습 및 저장
    print("\n--- 모델 A (신규 자격) 학습 ---")
    cat_A, calibrator_A = train_model_A(X_train_A, y_train_A, X_val_A, y_val_A)
    
    joblib.dump(cat_A, os.path.join(MODEL_SAVE_DIR, f"catboost_A_fold{fold}.pkl"))
    joblib.dump(calibrator_A, os.path.join(MODEL_SAVE_DIR, f"calibrator_A_fold{fold}.pkl"))
    # 6. 모델 B 학습 및 저장
    print("\n--- 모델 B (자격 유지) 학습 ---")
    if len(X_train_B) > 0 and len(X_val_B) > 0 and len(np.unique(y_train_B)) > 1:
        cat_B, calibrator_B = train_model_B(X_train_B, y_train_B, X_val_B, y_val_B, pk_stats_fold)
        
        joblib.dump(cat_B, os.path.join(MODEL_SAVE_DIR, f"catboost_B_fold{fold}.pkl"))
        joblib.dump(calibrator_B, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_fold{fold}.pkl"))
    else:
        print(f"[{fold+1} Fold] B모델 학습/검증 데이터가 부족하여 이 Fold를 건너뜁니다.")
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"catboost_B_fold{fold}.pkl"))
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_fold{fold}.pkl"))
# [신규] K-Fold로 생성된 PK Stats를 하나로 합침 (평균)
print("\n[INFO] K-Fold PK Stats 병합...")
if all_pk_stats_folds:
    all_pk_stats_df = pd.concat(all_pk_stats_folds)
    final_pk_stats = all_pk_stats_df.groupby('PrimaryKey').mean().reset_index()
    final_pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    final_pk_stats.to_csv(final_pk_stats_path, index=False)
else:
    print("경고: 유효한 PK Stats가 생성되지 않았습니다. 빈 파일을 생성합니다.")
    pd.DataFrame().to_csv(os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv"), index=False)
print(f"\n[INFO] '최종 보정' CatBoost 모델 10개 및 최종 통계 피처 1개 저장 완료:")
print(f"  {final_pk_stats_path}")
print("Big-Tech ML Engineer: 미션 완료. 이 11개의 파일로 `submit.zip`을 재구성하십시오.")
