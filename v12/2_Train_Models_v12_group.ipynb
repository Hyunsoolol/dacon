## ----------------------------------------------------------------
## [Jupyter용] 2_Train_Models_v12_group.ipynb
##  - 1_Preprocess가 만든 Feather 로드
##  - StratifiedGroupKFold(PrimaryKey) + CatBoost(v11 하이퍼파라미터)
##  - PK Stats (운전자 통계 피처)
##  - OOF 기반 Global Isotonic 보정기 (A/B 각 1개) + Fold별 보정기
## ----------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import joblib
import warnings

warnings.filterwarnings('ignore')

print("Big-Tech ML Engineer v12(group): GroupKFold + v11 CatBoost + OOF Isotonic 전략 시작.")

## 1. 로컬 경로 및 데이터 로드
BASE_DIR = "./data"
MODEL_SAVE_DIR = "./model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

FEATURE_SAVE_PATH = os.path.join(BASE_DIR, "all_train_data.feather")

try:
    print("전처리된 메인 피처 로드 중... (매우 빠름)")
    all_train_df = pd.read_feather(FEATURE_SAVE_PATH)
except FileNotFoundError as e:
    print(f"경고: {e}")
    print("먼저 1_Preprocess.ipynb를 실행하여 all_train_data.feather를 생성해야 합니다.")
    raise

print("데이터 로드 완료:", all_train_df.shape)


## 2. Metric (AUC + Brier + ECE)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    # NaN 방어
    y_prob = np.nan_to_num(y_prob, nan=0.0)

    # sklearn calibration_curve 사용 (공식 평가 코드와 동일한 방식)
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob,
        n_bins=n_bins,
        strategy='uniform'
    )

    # bin별 weight 계산
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_totals = np.histogram(y_prob, bins=bin_edges, density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals[non_empty_bins] / len(y_prob)

    # 방어 로직: 길이가 다르면 최소 길이에 맞춰 자르기
    m = min(len(bin_weights), len(prob_true), len(prob_pred))
    if m == 0:
        return 0.0

    bin_weights = bin_weights[:m]
    prob_true = prob_true[:m]
    prob_pred = prob_pred[:m]

    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece


def combined_score(y_true, y_prob):
    if (len(y_true) == 0 or len(y_prob) == 0 or
        np.sum(y_true) == 0 or np.sum(y_true) == len(y_true)):
        print("  AUC: N/A (단일 클래스), Brier: N/A, ECE: N/A (No data)")
        return 1.0

    y_prob = np.nan_to_num(y_prob, nan=0.0)

    auc = roc_auc_score(y_true, y_prob)
    brier = mean_squared_error(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    score = 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece

    print(f"  AUC: {auc:.4f}, Brier: {brier:.4f}, ECE: {ece:.4f}")
    print(f"  Combined Score: {score:.5f}")
    return score


## 3. GroupKFold 분리 (PrimaryKey 기준)
print("\n[INFO] Group K-Fold 교차 검증 분리 시작...")

N_SPLITS = 5
all_train_df['Label'] = all_train_df['Label'].fillna(0)

groups = all_train_df['PrimaryKey'].values
y_full = all_train_df['Label'].values

gkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_indices_list = []
val_indices_list = []

for train_idx, val_idx in gkf.split(all_train_df, y_full, groups=groups):
    train_indices_list.append(train_idx)
    val_indices_list.append(val_idx)

print("StratifiedGroupKFold 사용 (라벨 분포 + 그룹 동시 고려).")
print(f"{N_SPLITS}-Fold 분리 완료. (PrimaryKey 기준 그룹 분리)")

CAT_FEATURES = ['Age', 'PrimaryKey']
DROP_COLS_TRAIN = ['Test_id', 'Test_x', 'Test_y', 'Label', 'TestDate', 'Year', 'Month', 'base_index']


## 4. Model A/B 학습 함수 (OOF raw prob 반환)
def train_model_A(X_train, y_train, X_val, y_val, group_label="A"):
    # PK Stats 없는 버전 (A는 PK 통계 안 씀)
    base_cols = [c for c in X_train.columns if not c.startswith('pk_')]
    numeric_cols = list(set(base_cols) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))

    cb_X_train = X_train[numeric_cols + CAT_FEATURES].copy()
    cb_X_val = X_val[numeric_cols + CAT_FEATURES].copy()

    for col in CAT_FEATURES:
        if col in cb_X_train.columns:
            cb_X_train[col] = cb_X_train[col].fillna('nan').astype(str)
            cb_X_val[col] = cb_X_val[col].fillna('nan').astype(str)

    cat_idx = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]

    print(f"\n[{group_label}] CatBoost (Base, v11 세팅) 학습 시작... (피처 {len(cb_X_train.columns)}개)")
    model = cb.CatBoostClassifier(
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
    model.fit(
        cb_X_train, y_train,
        eval_set=[(cb_X_val, y_val)],
        cat_features=cat_idx
    )

    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic, Fold 내부용) 시작...")
    pred_uncal = model.predict_proba(cb_X_val)[:, 1]
    print(f"[{group_label}] 비보정 단독 점수:")
    _ = combined_score(y_val, pred_uncal)

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(pred_uncal, y_val)
    pred_cal = calibrator.predict(pred_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수:")
    _ = combined_score(y_val, pred_cal)

    # pred_uncal: OOF global 보정용 raw prob
    return model, calibrator, pred_uncal


def train_model_B(X_train, y_train, X_val, y_val, pk_stats_fold, group_label="B"):
    # Leakage 방지를 위해 fold train에서만 pk_stats 생성 후 병합
    X_train = X_train.merge(pk_stats_fold, on='PrimaryKey', how='left')
    X_val   = X_val.merge(pk_stats_fold, on='PrimaryKey', how='left')

    numeric_cols = list(set(X_train.columns) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))
    common_numeric_cols = list(set(X_train[numeric_cols].columns) & set(X_val[numeric_cols].columns))

    cb_X_train = X_train[common_numeric_cols + CAT_FEATURES].copy()
    cb_X_val   = X_val[common_numeric_cols + CAT_FEATURES].copy()

    for col in CAT_FEATURES:
        if col in cb_X_train.columns:
            cb_X_train[col] = cb_X_train[col].fillna('nan').astype(str)
            cb_X_val[col]   = cb_X_val[col].fillna('nan').astype(str)

    cat_idx = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train.columns]

    print(f"\n[{group_label}] CatBoost (Tuned for B, v13) 학습 시작... (피처 {len(cb_X_train.columns)}개)")
    model = cb.CatBoostClassifier(
        # 더 부드럽게 학습시키고 early stopping에 맡기는 세팅
        iterations=8000,
        learning_rate=0.02,

        # 깊이는 그대로, 규제는 살짝 완화해서 표현력 확보
        depth=6,
        l2_leaf_reg=5,

        loss_function='Logloss',
        # AUC 기준으로 모델 선택 (분류력 극대화)
        eval_metric='AUC',

        # B 쪽은 클래스 불균형 있을 가능성이 높아서 그대로 유지
        auto_class_weights='Balanced',

        # 샘플/피처 서브샘플링은 살짝 덜 공격적으로
        bootstrap_type='Bayesian',
        bagging_temperature=1.5,
        rsm=0.8,          # 0.7 → 0.8 (더 많은 피처 활용)

        # 너무 랜덤하게 만들진 말고 약간만
        random_strength=1.0,

        random_seed=42,
        thread_count=-1,

        # 더 길게 보면서도, 필요하면 중간에 멈추게
        early_stopping_rounds=300,
        verbose=500
    )
    model.fit(
        cb_X_train, y_train,
        eval_set=[(cb_X_val, y_val)],
        cat_features=cat_idx
    )

    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic, Fold 내부용) 시작...")
    pred_uncal = model.predict_proba(cb_X_val)[:, 1]
    print(f"[{group_label}] 비보정 단독 점수:")
    _ = combined_score(y_val, pred_uncal)

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(pred_uncal, y_val)
    pred_cal = calibrator.predict(pred_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수:")
    _ = combined_score(y_val, pred_cal)

    return model, calibrator, pred_uncal


## 5. K-Fold 루프 + PK Stats + OOF 저장
all_pk_stats_folds = []

n_samples = len(all_train_df)
oof_predA = np.zeros(n_samples, dtype=float)
oof_predB = np.zeros(n_samples, dtype=float)
oof_hasA = np.zeros(n_samples, dtype=bool)
oof_hasB = np.zeros(n_samples, dtype=bool)

for fold in range(N_SPLITS):
    print(f"\n=== Fold {fold+1}/{N_SPLITS} 학습 시작 ===")

    train_idx = train_indices_list[fold]
    val_idx = val_indices_list[fold]

    train_df_fold = all_train_df.iloc[train_idx]
    val_df_fold = all_train_df.iloc[val_idx]

    # 5-1. PK Stats (fold train 기준)
    print(f"\n[Fold {fold+1}] K-Fold Target Encoding (PK Stats) 생성...")
    agg_funcs = {
        'Age_num': ['mean', 'min', 'max'],
        'YearMonthIndex': ['mean', 'std', 'min', 'max'],
        'A1_rt_mean': ['mean', 'std'],
        'A4_acc_congruent': ['mean', 'std'],
        'A4_acc_incongruent': ['mean', 'std'],
        'A4_stroop_rt_cost': ['mean', 'std'],
        'RiskScore': ['mean', 'std', 'max'],
        'B1_change_acc': ['mean', 'std'],
        'B1_nonchange_acc': ['mean', 'std'],
        'B3_rt_mean': ['mean', 'std'],
        'B4_flanker_acc_cost': ['mean', 'std'],
        'B4_rt_mean': ['mean', 'std'],
        'RiskScore_B': ['mean', 'std', 'max'],
        'Test_id': ['count'],
        'A_selective_attention_index': ['mean', 'std'],
        'A_working_memory_index': ['mean', 'std'],
        'A_cog_sum': ['mean', 'std'],
        'B_visuomotor_variability': ['mean', 'std'],
        'B_reaction_overall': ['mean', 'std'],
        'B_executive_control_index': ['mean', 'std'],
        'B_cog_sum': ['mean', 'std'],
        }
    valid_agg_funcs = {col: fs for col, fs in agg_funcs.items() if col in train_df_fold.columns}

    pk_stats_fold = train_df_fold.groupby('PrimaryKey').agg(valid_agg_funcs)
    pk_stats_fold.columns = ['_'.join(col).strip() for col in pk_stats_fold.columns.values]
    pk_stats_fold.rename(columns={'Test_id_count': 'pk_test_total_count'}, inplace=True)

    # Test 타입별 count (A/B)
    pk_test_type_count = train_df_fold.groupby('PrimaryKey')['Test_x'].value_counts().unstack(fill_value=0)
    if 'A' not in pk_test_type_count.columns:
        pk_test_type_count['A'] = 0
    if 'B' not in pk_test_type_count.columns:
        pk_test_type_count['B'] = 0
    pk_test_type_count = pk_test_type_count[['A', 'B']]
    pk_test_type_count.columns = ['pk_test_A_count', 'pk_test_B_count']

    pk_stats_fold = pk_stats_fold.join(pk_test_type_count, how='left').reset_index()
    all_pk_stats_folds.append(pk_stats_fold)

    # 5-2. A/B 분리
    X_train_A = train_df_fold[train_df_fold['Test_x'] == 'A'].copy()
    y_train_A = X_train_A['Label'].values
    X_val_A = val_df_fold[val_df_fold['Test_x'] == 'A'].copy()
    y_val_A = X_val_A['Label'].values

    X_train_B = train_df_fold[train_df_fold['Test_x'] == 'B'].copy()
    y_train_B = X_train_B['Label'].values
    X_val_B = val_df_fold[val_df_fold['Test_x'] == 'B'].copy()
    y_val_B = X_val_B['Label'].values

    # 5-3. 모델 A 학습
    print("\n--- 모델 A (신규 자격) 학습 ---")
    if len(X_train_A) > 0 and len(X_val_A) > 0 and len(np.unique(y_train_A)) > 1:
        cat_A, calib_A, val_predA_uncal = train_model_A(X_train_A, y_train_A, X_val_A, y_val_A)

        joblib.dump(cat_A, os.path.join(MODEL_SAVE_DIR, f"catboost_A_fold{fold}.pkl"))
        joblib.dump(calib_A, os.path.join(MODEL_SAVE_DIR, f"calibrator_A_fold{fold}.pkl"))

        val_A_idx = X_val_A.index
        oof_predA[val_A_idx] = val_predA_uncal
        oof_hasA[val_A_idx] = True
    else:
        print(f"[Fold {fold+1}] A모델 학습/검증 데이터 부족. 이 Fold는 A모델 건너뜁니다.")
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"catboost_A_fold{fold}.pkl"))
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"calibrator_A_fold{fold}.pkl"))

    # 5-4. 모델 B 학습
    print("\n--- 모델 B (자격 유지) 학습 ---")
    if len(X_train_B) > 0 and len(X_val_B) > 0 and len(np.unique(y_train_B)) > 1:
        cat_B, calib_B, val_predB_uncal = train_model_B(X_train_B, y_train_B, X_val_B, y_val_B, pk_stats_fold)

        joblib.dump(cat_B, os.path.join(MODEL_SAVE_DIR, f"catboost_B_fold{fold}.pkl"))
        joblib.dump(calib_B, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_fold{fold}.pkl"))

        val_B_idx = X_val_B.index
        oof_predB[val_B_idx] = val_predB_uncal
        oof_hasB[val_B_idx] = True
    else:
        print(f"[Fold {fold+1}] B모델 학습/검증 데이터 부족. 이 Fold는 B모델 건너뜁니다.")
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"catboost_B_fold{fold}.pkl"))
        joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_fold{fold}.pkl"))


## 6. PK Stats Fold 평균 → 최종 파일
print("\n[INFO] K-Fold PK Stats 병합...")
if all_pk_stats_folds:
    all_pk_stats_df = pd.concat(all_pk_stats_folds, axis=0)
    final_pk_stats = all_pk_stats_df.groupby('PrimaryKey').mean().reset_index()
    final_pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    final_pk_stats.to_csv(final_pk_stats_path, index=False)
else:
    print("경고: 유효한 PK Stats가 생성되지 않았습니다. 빈 파일을 생성합니다.")
    final_pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    pd.DataFrame().to_csv(final_pk_stats_path, index=False)


## 7. Global Isotonic (OOF 기반 보정기)
print("\n[INFO] Global Isotonic Recalibration (OOF 기반) 시작...")

y_all = all_train_df['Label'].values

# A용 Global Isotonic
mask_A = (all_train_df['Test_x'] == 'A') & oof_hasA
if mask_A.sum() > 0 and len(np.unique(y_all[mask_A])) > 1:
    cal_A_global = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    cal_A_global.fit(oof_predA[mask_A], y_all[mask_A])
    joblib.dump(cal_A_global, os.path.join(MODEL_SAVE_DIR, "calibrator_A_global.pkl"))
    print(f"[INFO] Global Isotonic A 보정기 학습/저장 완료. (샘플 {mask_A.sum()}개)")
else:
    print("[경고] Global Isotonic A 보정기를 학습할 수 없어 None으로 저장합니다.")
    joblib.dump(None, os.path.join(MODEL_SAVE_DIR, "calibrator_A_global.pkl"))

# B용 Global Isotonic
mask_B = (all_train_df['Test_x'] == 'B') & oof_hasB
if mask_B.sum() > 0 and len(np.unique(y_all[mask_B])) > 1:
    cal_B_global = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    cal_B_global.fit(oof_predB[mask_B], y_all[mask_B])
    joblib.dump(cal_B_global, os.path.join(MODEL_SAVE_DIR, "calibrator_B_global.pkl"))
    print(f"[INFO] Global Isotonic B 보정기 학습/저장 완료. (샘플 {mask_B.sum()}개)")
else:
    print("[경고] Global Isotonic B 보정기를 학습할 수 없어 None으로 저장합니다.")
    joblib.dump(None, os.path.join(MODEL_SAVE_DIR, "calibrator_B_global.pkl"))

print(f"\n[INFO] 최종 산출물:")
print(f"  - CatBoost A 모델 5개: catboost_A_fold0~4.pkl")
print(f"  - CatBoost B 모델 5개: catboost_B_fold0~4.pkl")
print(f"  - Fold별 보정기 10개: calibrator_A/B_fold0~4.pkl")
print(f"  - Global 보정기 2개: calibrator_A_global.pkl, calibrator_B_global.pkl")
print(f"  - 최종 PK Stats 1개: {final_pk_stats_path}")
print("Big-Tech ML Engineer v12(group): 미션 완료.")

## 8. OOF 기반 전체 성능 평가 (제출 전 의사결정용)
print("\n[INFO] OOF 기반 전체 성능 평가 시작...")

y_all = all_train_df['Label'].values.astype(float)

# 8-1. A / B 각각에 대해 "최종 보정된" OOF 예측 생성
#      (Global Isotonic이 있으면 그걸 쓰고, 없으면 fold-내부 raw 사용)

# A
y_prob_A = np.zeros_like(y_all, dtype=float)
if mask_A.sum() > 0:
    if 'cal_A_global' in locals() and cal_A_global is not None:
        y_prob_A[mask_A] = cal_A_global.predict(oof_predA[mask_A])
    else:
        y_prob_A[mask_A] = oof_predA[mask_A]

# B
y_prob_B = np.zeros_like(y_all, dtype=float)
if mask_B.sum() > 0:
    if 'cal_B_global' in locals() and cal_B_global is not None:
        y_prob_B[mask_B] = cal_B_global.predict(oof_predB[mask_B])
    else:
        y_prob_B[mask_B] = oof_predB[mask_B]

# 8-2. A / B / 전체 각각 점수 계산
print("\n[OOF] A 검사만 (Test_x == 'A') 점수:")
maskA_only = (all_train_df['Test_x'] == 'A') & oof_hasA
_ = combined_score(y_all[maskA_only], y_prob_A[maskA_only])

print("\n[OOF] B 검사만 (Test_x == 'B') 점수:")
maskB_only = (all_train_df['Test_x'] == 'B') & oof_hasB
_ = combined_score(y_all[maskB_only], y_prob_B[maskB_only])

print("\n[OOF] A+B 전체 통합 점수:")
# A / B 예측은 서로 disjoint이므로 더해도 됨
y_prob_all = np.zeros_like(y_all, dtype=float)
y_prob_all[maskA_only] = y_prob_A[maskA_only]
y_prob_all[maskB_only] = y_prob_B[maskB_only]
_ = combined_score(y_all, y_prob_all)

print("\n[INFO] OOF 기반 평가 완료. 이 점수를 보고 제출 여부를 결정하세요.")
