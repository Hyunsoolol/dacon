# ----------------------------------------------------------------
# [script.py] v20 Inference (5-seeds 지원)
#  - v20 규칙:
#      * A 모델: AgeGroup, YM_is_early 사용
#      * B 모델: AgeGroup, YM_is_early 없이 학습됨
#  - 학습 스크립트: 2_Train_Models_v20_5seeds.py 와 호환
#  - 모델 파일명:
#      * catboost_A_seed{seed}_fold{fold}.pkl
#      * calibrator_A_seed{seed}_fold{fold}.pkl
#      * catboost_B_seed{seed}_fold{fold}.pkl
#      * calibrator_B_seed{seed}_fold{fold}.pkl
#  - predict_group()에서 seed×fold 전체 예측 평균
# ----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import joblib
import catboost as cb

BASE_DIR = "./data"
MODEL_DIR = "./model"
OUTPUT_DIR = "./output"

N_SPLITS = 5
# 학습 스크립트와 맞춰야 함
SEEDS = [42, 43, 44, 45, 46]

CAT_FEATURES = ["Age", "AgeGroup", "PrimaryKey"]
DROP_COLS_TRAIN = [
    "Test_id",
    "Test_x",
    "Test_y",
    "Label",
    "TestDate",
    "Year",
    "Month",
    "base_index",
]

# v19+age_group+ym_early 에서 사용했던 것과 동일 threshold 이어야 함
EARLY_YM_THRESHOLD = 24234


# =========================
# 1. 공통 유틸 함수들
# =========================
def convert_age(val):
    if pd.isna(val):
        return np.nan
    try:
        s = str(val)
        base = int(s[:-1])
        return base if s[-1] == "a" else base + 5
    except Exception:
        return np.nan


def split_testdate(val):
    try:
        v = int(val)
        return v // 100, v % 100
    except Exception:
        return np.nan, np.nan


def seq_mean(series):
    return series.fillna("").apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )


def seq_std(series):
    return series.fillna("").apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )


def masked_operation(cond_series, val_series, target_conds, operation="mean"):
    """
    cond_series에서 target_conds에 해당하는 위치의 val_series에 대해
    mean / std / rate 계산
    """
    cond_df = (
        cond_series.fillna("")
        .str.split(",", expand=True)
        .replace("", np.nan)
        .to_numpy(dtype=float)
    )
    val_df = (
        val_series.fillna("")
        .str.split(",", expand=True)
        .replace("", np.nan)
        .to_numpy(dtype=float)
    )

    if isinstance(target_conds, (list, set, tuple)):
        mask = np.isin(cond_df, list(target_conds))
    else:
        mask = cond_df == target_conds

    masked_vals = np.where(mask, val_df, np.nan)

    with np.errstate(invalid="ignore"):
        if operation == "mean":
            sums = np.nansum(masked_vals, axis=1)
            counts = np.sum(mask, axis=1)
            out = sums / np.where(counts == 0, np.nan, counts)
        elif operation == "std":
            out = np.nanstd(masked_vals, axis=1)
        elif operation == "rate":
            corrects = np.nansum(np.where(masked_vals == 1, 1, 0), axis=1)
            total = np.sum(mask, axis=1)
            out = corrects / np.where(total == 0, np.nan, total)
        else:
            out = np.nan
    return pd.Series(out, index=cond_series.index)


# === PDF 명세 기반 rate 유틸 ===
def seq_rate_A3(series, target_codes):
    def calc(x):
        if not x:
            return np.nan
        s = x.split(",")
        correct = sum(s.count(code) for code in target_codes if code in ["1", "3"])
        incorrect = sum(s.count(code) for code in target_codes if code in ["2", "4"])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan

    return series.fillna("").apply(calc)


def seq_rate_B1_B2(series, target_codes):
    def calc(x):
        if not x:
            return np.nan
        s = x.split(",")
        correct = sum(s.count(code) for code in target_codes if code in ["1", "3"])
        incorrect = sum(s.count(code) for code in target_codes if code in ["2", "4"])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan

    return series.fillna("").apply(calc)


def seq_rate_B4(series, target_codes):
    def calc(x):
        if not x:
            return np.nan
        s = x.split(",")
        correct = sum(s.count(code) for code in target_codes if code in ["1", "3", "5"])
        incorrect = sum(s.count(code) for code in target_codes if code in ["2", "4", "6"])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan

    return series.fillna("").apply(calc)


def seq_rate_simple(series):
    """B3, B5, B6, B7, B8용: 1=정답, 2=오답"""
    def calc(x):
        if not x:
            return np.nan
        s = x.split(",")
        correct = s.count("1")
        incorrect = s.count("2")
        total = correct + incorrect
        return correct / total if total > 0 else np.nan

    return series.fillna("").apply(calc)


def _has(df, cols):
    return all(c in df.columns for c in cols)


def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def _log_ratio(num, den, eps=1e-6):
    """로그 비율: log((num+eps) / (den+eps))"""
    return np.log((num + eps) / (den + eps))


def _is_delta_column(col: str) -> bool:
    return col.startswith("delta_")


def _delta_colname(c: str) -> str:
    # 하이픈 있는 원본(B9-1 등)을 안전한 이름으로 변환
    return f"delta_{c.replace('-', '_')}"


# =========================
# 2. A/B 도메인 피처 생성
# =========================
def preprocess_A(df: pd.DataFrame) -> pd.DataFrame:
    print("[TEST/A] 1차 도메인 피처 생성 중...")

    df = df.copy()
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    # A1 (속도 예측)
    feats["A1_rt_mean"] = seq_mean(df["A1-4"])
    feats["A1_rt_std"] = seq_std(df["A1-4"])
    feats["A1_rt_left"] = masked_operation(df["A1-1"], df["A1-4"], 1, "mean")
    feats["A1_rt_right"] = masked_operation(df["A1-1"], df["A1-4"], 2, "mean")
    feats["A1_rt_slow"] = masked_operation(df["A1-2"], df["A1-4"], 1, "mean")
    feats["A1_rt_norm"] = masked_operation(df["A1-2"], df["A1-4"], 2, "mean")
    feats["A1_rt_fast"] = masked_operation(df["A1-2"], df["A1-4"], 3, "mean")
    feats["A1_acc_slow"] = masked_operation(df["A1-2"], df["A1-3"], 1, "rate")
    feats["A1_acc_norm"] = masked_operation(df["A1-2"], df["A1-3"], 2, "rate")
    feats["A1_acc_fast"] = masked_operation(df["A1-2"], df["A1-3"], 3, "rate")

    # A2 (정지 예측)
    feats["A2_rt_mean"] = seq_mean(df["A2-4"])
    feats["A2_rt_std"] = seq_std(df["A2-4"])
    feats["A2_rt_slow_c1"] = masked_operation(df["A2-1"], df["A2-4"], 1, "mean")
    feats["A2_rt_norm_c1"] = masked_operation(df["A2-1"], df["A2-4"], 2, "mean")
    feats["A2_rt_fast_c1"] = masked_operation(df["A2-1"], df["A2-4"], 3, "mean")
    feats["A2_rt_slow_c2"] = masked_operation(df["A2-2"], df["A2-4"], 1, "mean")
    feats["A2_rt_norm_c2"] = masked_operation(df["A2-2"], df["A2-4"], 2, "mean")
    feats["A2_rt_fast_c2"] = masked_operation(df["A2-2"], df["A2-4"], 3, "mean")
    feats["A2_acc_slow"] = masked_operation(df["A2-1"], df["A2-3"], 1, "rate")
    feats["A2_acc_norm"] = masked_operation(df["A2-1"], df["A2-3"], 2, "rate")
    feats["A2_acc_fast"] = masked_operation(df["A2-1"], df["A2-3"], 3, "rate")

    # A3 (주의 전환)
    feats["A3_valid_acc"] = seq_rate_A3(df["A3-5"], ["1", "2"])
    feats["A3_invalid_acc"] = seq_rate_A3(df["A3-5"], ["3", "4"])
    feats["A3_rt_mean"] = seq_mean(df["A3-7"])
    feats["A3_rt_std"] = seq_std(df["A3-7"])
    feats["A3_rt_small"] = masked_operation(df["A3-1"], df["A3-7"], 1, "mean")
    feats["A3_rt_big"] = masked_operation(df["A3-1"], df["A3-7"], 2, "mean")
    feats["A3_rt_left"] = masked_operation(df["A3-3"], df["A3-7"], 1, "mean")
    feats["A3_rt_right"] = masked_operation(df["A3-3"], df["A3-7"], 2, "mean")

    # A4 (Stroop)
    feats["A4_rt_mean"] = seq_mean(df["A4-5"])
    feats["A4_rt_std"] = seq_std(df["A4-5"])
    feats["A4_rt_congruent"] = masked_operation(df["A4-1"], df["A4-5"], 1, "mean")
    feats["A4_rt_incongruent"] = masked_operation(df["A4-1"], df["A4-5"], 2, "mean")
    feats["A4_acc_congruent"] = masked_operation(df["A4-1"], df["A4-3"], 1, "rate")
    feats["A4_acc_incongruent"] = masked_operation(df["A4-1"], df["A4-3"], 2, "rate")

    # A5 (변화 탐지)
    feats["A5_acc_nonchange"] = masked_operation(df["A5-1"], df["A5-2"], 1, "rate")
    feats["A5_acc_pos_change"] = masked_operation(df["A5-1"], df["A5-2"], 2, "rate")
    feats["A5_acc_color_change"] = masked_operation(df["A5-1"], df["A5-2"], 3, "rate")
    feats["A5_acc_shape_change"] = masked_operation(df["A5-1"], df["A5-2"], 4, "rate")

    # A6, A7 (문제풀이)
    feats["A6_correct_count"] = df["A6-1"]
    feats["A7_correct_count"] = df["A7-1"]

    # A8, A9 (질문지)
    feats["A8-1"] = df["A8-1"]
    feats["A8-2"] = df["A8-2"]
    feats["A9-1"] = df["A9-1"]
    feats["A9-2"] = df["A9-2"]
    feats["A9-3"] = df["A9-3"]
    feats["A9-4"] = df["A9-4"]
    feats["A9-5"] = df["A9-5"]

    seq_cols = [
        "A1-1", "A1-2", "A1-3", "A1-4",
        "A2-1", "A2-2", "A2-3", "A2-4",
        "A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6", "A3-7",
        "A4-1", "A4-2", "A4-3", "A4-4", "A4-5",
        "A5-1", "A5-2", "A5-3",
        "A6-1", "A7-1", "A8-1", "A8-2",
        "A9-1", "A9-2", "A9-3", "A9-4", "A9-5",
    ]
    out = pd.concat([df, feats], axis=1).drop(columns=seq_cols, errors="ignore")
    print("[TEST/A] 1차 전처리 완료.")
    return out


def preprocess_B(df: pd.DataFrame) -> pd.DataFrame:
    print("[TEST/B] 1차 도메인 피처 생성 중...")

    df = df.copy()
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    # B1, B2 (시야각)
    feats["B1_task1_acc"] = seq_rate_simple(df["B1-1"])
    feats["B1_rt_mean"] = seq_mean(df["B1-2"])
    feats["B1_rt_std"] = seq_std(df["B1-2"])
    feats["B1_change_acc"] = seq_rate_B1_B2(df["B1-3"], ["1", "2"])
    feats["B1_nonchange_acc"] = seq_rate_B1_B2(df["B1-3"], ["3", "4"])

    feats["B2_task1_acc"] = seq_rate_simple(df["B2-1"])
    feats["B2_rt_mean"] = seq_mean(df["B2-2"])
    feats["B2_rt_std"] = seq_std(df["B2-2"])
    feats["B2_change_acc"] = seq_rate_B1_B2(df["B2-3"], ["1", "2"])
    feats["B2_nonchange_acc"] = seq_rate_B1_B2(df["B2-3"], ["3", "4"])

    # B3 (신호등)
    feats["B3_acc_rate"] = seq_rate_simple(df["B3-1"])
    feats["B3_rt_mean"] = seq_mean(df["B3-2"])
    feats["B3_rt_std"] = seq_std(df["B3-2"])

    # B4 (Flanker)
    feats["B4_congruent_acc"] = seq_rate_B4(df["B4-1"], ["1", "2"])
    feats["B4_incongruent_acc"] = seq_rate_B4(df["B4-1"], ["3", "4", "5", "6"])
    feats["B4_rt_mean"] = seq_mean(df["B4-2"])
    feats["B4_rt_std"] = seq_std(df["B4-2"])

    # B5~B8
    feats["B5_acc_rate"] = seq_rate_simple(df["B5-1"])
    feats["B5_rt_mean"] = seq_mean(df["B5-2"])
    feats["B5_rt_std"] = seq_std(df["B5-2"])
    feats["B6_acc_rate"] = seq_rate_simple(df["B6"])
    feats["B7_acc_rate"] = seq_rate_simple(df["B7"])
    feats["B8_acc_rate"] = seq_rate_simple(df["B8"])

    # B9, B10 (다중과제)
    feats["B9-1"] = df["B9-1"]
    feats["B9-2"] = df["B9-2"]
    feats["B9-3"] = df["B9-3"]
    feats["B9-4"] = df["B9-4"]
    feats["B9-5"] = df["B9-5"]
    feats["B10-1"] = df["B10-1"]
    feats["B10-2"] = df["B10-2"]
    feats["B10-3"] = df["B10-3"]
    feats["B10-4"] = df["B10-4"]
    feats["B10-5"] = df["B10-5"]
    feats["B10-6"] = df["B10-6"]

    seq_cols = [
        "B1-1", "B1-2", "B1-3",
        "B2-1", "B2-2", "B2-3",
        "B3-1", "B3-2",
        "B4-1", "B4-2",
        "B5-1", "B5-2",
        "B6", "B7", "B8",
        "B9-1", "B9-2", "B9-3", "B9-4", "B9-5",
        "B10-1", "B10-2", "B10-3", "B10-4", "B10-5", "B10-6",
    ]
    out = pd.concat([df, feats], axis=1).drop(columns=seq_cols, errors="ignore")
    print("[TEST/B] 1차 전처리 완료.")
    return out


# =========================
# 3. 2차 파생 (log_ratio 등)
# =========================
def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6

    if _has(feats, ["Year", "Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    # Speed-Accuracy Tradeoffs
    if _has(feats, ["A1_rt_mean", "A1_acc_norm"]):
        feats["A1_speed_acc_tradeoff"] = _safe_div(feats["A1_rt_mean"], feats["A1_acc_norm"], eps)
    if _has(feats, ["A2_rt_mean", "A2_acc_norm"]):
        feats["A2_speed_acc_tradeoff"] = _safe_div(feats["A2_rt_mean"], feats["A2_acc_norm"], eps)
    if _has(feats, ["A4_rt_mean", "A4_acc_congruent"]):
        feats["A4_speed_acc_tradeoff"] = _safe_div(feats["A4_rt_mean"], feats["A4_acc_congruent"], eps)

    # CV
    for k in ["A1", "A2", "A3", "A4", "A5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    # A1: fast vs slow
    if _has(feats, ["A1_rt_fast", "A1_rt_slow"]):
        feats["A1_rt_speed_log_ratio"] = _log_ratio(feats["A1_rt_fast"], feats["A1_rt_slow"], eps)
    if _has(feats, ["A1_acc_fast", "A1_acc_slow"]):
        feats["A1_acc_speed_log_ratio"] = _log_ratio(feats["A1_acc_fast"], feats["A1_acc_slow"], eps)

    # A2
    if _has(feats, ["A2_rt_fast_c1", "A2_rt_slow_c1"]):
        feats["A2_rt_speed_log_ratio_c1"] = _log_ratio(feats["A2_rt_fast_c1"], feats["A2_rt_slow_c1"], eps)
    if _has(feats, ["A2_acc_fast", "A2_acc_slow"]):
        feats["A2_acc_speed_log_ratio"] = _log_ratio(feats["A2_acc_fast"], feats["A2_acc_slow"], eps)

    # A3
    if _has(feats, ["A3_rt_big", "A3_rt_small"]):
        feats["A3_rt_size_log_ratio"] = _log_ratio(feats["A3_rt_big"], feats["A3_rt_small"], eps)
    if _has(feats, ["A3_valid_acc", "A3_invalid_acc"]):
        feats["A3_acc_attention_log_ratio"] = _log_ratio(feats["A3_valid_acc"], feats["A3_invalid_acc"], eps)

    # A4
    if _has(feats, ["A4_rt_incongruent", "A4_rt_congruent"]):
        feats["A4_stroop_rt_log_ratio"] = _log_ratio(feats["A4_rt_incongruent"], feats["A4_rt_congruent"], eps)
    if _has(feats, ["A4_acc_congruent", "A4_acc_incongruent"]):
        feats["A4_stroop_acc_log_ratio"] = _log_ratio(feats["A4_acc_congruent"], feats["A4_acc_incongruent"], eps)

    # A5
    if _has(feats, ["A5_acc_nonchange", "A5_acc_pos_change"]):
        feats["A5_acc_pos_log_ratio"] = _log_ratio(feats["A5_acc_nonchange"], feats["A5_acc_pos_change"], eps)
    if _has(feats, ["A5_acc_nonchange", "A5_acc_color_change"]):
        feats["A5_acc_color_log_ratio"] = _log_ratio(feats["A5_acc_nonchange"], feats["A5_acc_color_change"], eps)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6

    if _has(feats, ["Year", "Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    # Speed-Accuracy Tradeoffs
    if _has(feats, ["B1_rt_mean", "B1_task1_acc"]):
        feats["B1_speed_acc_tradeoff"] = _safe_div(feats["B1_rt_mean"], feats["B1_task1_acc"], eps)
    if _has(feats, ["B2_rt_mean", "B2_task1_acc"]):
        feats["B2_speed_acc_tradeoff"] = _safe_div(feats["B2_rt_mean"], feats["B2_task1_acc"], eps)
    if _has(feats, ["B3_rt_mean", "B3_acc_rate"]):
        feats["B3_speed_acc_tradeoff"] = _safe_div(feats["B3_rt_mean"], feats["B3_acc_rate"], eps)
    if _has(feats, ["B4_rt_mean", "B4_congruent_acc"]):
        feats["B4_speed_acc_tradeoff"] = _safe_div(feats["B4_rt_mean"], feats["B4_congruent_acc"], eps)
    if _has(feats, ["B5_rt_mean", "B5_acc_rate"]):
        feats["B5_speed_acc_tradeoff"] = _safe_div(feats["B5_rt_mean"], feats["B5_acc_rate"], eps)

    # CV
    for k in ["B1", "B2", "B3", "B4", "B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    # B1/B2
    if _has(feats, ["B1_change_acc", "B1_nonchange_acc"]):
        feats["B1_acc_log_ratio"] = _log_ratio(feats["B1_nonchange_acc"], feats["B1_change_acc"], eps)
    if _has(feats, ["B2_change_acc", "B2_nonchange_acc"]):
        feats["B2_acc_log_ratio"] = _log_ratio(feats["B2_nonchange_acc"], feats["B2_change_acc"], eps)

    # B4
    if _has(feats, ["B4_congruent_acc", "B4_incongruent_acc"]):
        feats["B4_flanker_acc_log_ratio"] = _log_ratio(feats["B4_congruent_acc"], feats["B4_incongruent_acc"], eps)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


# =========================
# 4. PK 히스토리 + Delta + AgeGroup/YM_early
# =========================
def add_pk_history_and_age_features(all_df: pd.DataFrame, norm_stats: dict) -> pd.DataFrame:
    df = all_df.copy()

    # base_index: 원래 row 순서 기록 (모델에서는 DROP_COLS_TRAIN로 제거되는 컬럼)
    df["base_index"] = np.arange(len(df))

    # YearMonthIndex가 없으면 생성
    if "YearMonthIndex" not in df.columns and {"Year", "Month"}.issubset(df.columns):
        df["YearMonthIndex"] = df["Year"] * 12 + df["Month"]

    # --- PK 히스토리 ---
    print("[TEST] PK 히스토리 피처 생성...")
    sort_cols = ["PrimaryKey"]
    if "Year" in df.columns:
        sort_cols.append("Year")
    if "Month" in df.columns:
        sort_cols.append("Month")
    sort_cols.append("Test_id")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    if "Test_x" in df.columns:
        test_col = "Test_x"
    elif "Test" in df.columns:
        test_col = "Test"
    else:
        raise KeyError("히스토리 피처 생성을 위해 'Test' 혹은 'Test_x' 컬럼이 필요합니다.")

    grp = df.groupby("PrimaryKey", sort=False)

    df["pk_hist_total_count"] = grp.cumcount()

    df["_is_A"] = (df[test_col] == "A").astype(int)
    df["_is_B"] = (df[test_col] == "B").astype(int)

    df["pk_hist_A_count"] = grp["_is_A"].cumsum().shift(1).fillna(0).astype(int)
    df["pk_hist_B_count"] = grp["_is_B"].cumsum().shift(1).fillna(0).astype(int)

    if "YearMonthIndex" in df.columns:
        df["pk_hist_prev_ym"] = grp["YearMonthIndex"].shift(1)
        df["pk_hist_gap_from_prev"] = df["YearMonthIndex"] - df["pk_hist_prev_ym"]

    df.drop(columns=["_is_A", "_is_B"], inplace=True)

    # --- Age_num / YearMonthIndex 정규화 + AgeGroup + YM_is_early ---
    print("[TEST] Age_num / YearMonthIndex z-score 및 AgeGroup/YM_is_early 생성...")
    age_mean = norm_stats.get("Age_num_mean", None)
    age_std = norm_stats.get("Age_num_std", None)
    ym_mean = norm_stats.get("YearMonthIndex_mean", None)
    ym_std = norm_stats.get("YearMonthIndex_std", None)

    if "Age_num" in df.columns and age_mean is not None and age_std is not None:
        df["Age_num_z"] = (df["Age_num"] - age_mean) / (age_std + 1e-6)

    if "YearMonthIndex" in df.columns and ym_mean is not None and ym_std is not None:
        df["YearMonthIndex_z"] = (df["YearMonthIndex"] - ym_mean) / (ym_std + 1e-6)

    # YM_is_early
    if "YearMonthIndex" in df.columns:
        df["YM_is_early"] = (df["YearMonthIndex"] <= EARLY_YM_THRESHOLD).astype(int)

    # AgeGroup
    if "Age_num" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age_num"],
            bins=[0, 50, 60, 70, 100],
            labels=["<50", "50-59", "60-69", "70+"],
            right=False,
        )

    # 나이 비선형 항
    if "Age_num_z" in df.columns:
        df["Age_num_z2"] = df["Age_num_z"] ** 2
        df["Age_num_z3"] = df["Age_num_z"] ** 3

    # Delta(B-only)
    print("[TEST] Delta(B-only) 피처 생성...")
    df = add_delta_features_pk(df, test_col_name=test_col)

    return df


def add_delta_features_pk(df: pd.DataFrame, test_col_name: str) -> pd.DataFrame:
    df = df.copy()

    b_prefixes = ("B1_", "B2_", "B3_", "B4_", "B5_", "B6_", "B7_", "B8_", "B9-", "B10-")
    candidates = [
        c for c in df.columns
        if (c.startswith(b_prefixes)) and (df[c].dtype != "O")
    ]

    sort_cols_local = ["PrimaryKey"]
    if "Year" in df.columns:
        sort_cols_local.append("Year")
    if "Month" in df.columns:
        sort_cols_local.append("Month")
    sort_cols_local.append("Test_id")

    df = df.sort_values(sort_cols_local, kind="mergesort").reset_index(drop=True)
    g = df.groupby("PrimaryKey", sort=False)

    for c in candidates:
        prev = g[c].shift(1)
        delta = df[c] - prev
        use_mask = df[test_col_name] == "B"
        df[_delta_colname(c)] = np.where(use_mask, delta, np.nan)

    print(f"[TEST] 생성된 delta_* 피처 수: {sum(col.startswith('delta_') for col in df.columns)}")
    return df


# =========================
# 5. 모델 입력 준비 & 예측
# =========================
def prepare_cb_input(df: pd.DataFrame, model: cb.CatBoostClassifier) -> pd.DataFrame:
    """
    학습 시 사용한 feature_names_ 순서에 맞게 컬럼 정렬 + 범주형을 str로 캐스팅

    - A 모델은 Age, AgeGroup, PrimaryKey 를 cat으로 사용
    - B 모델은 Age, PrimaryKey 만 cat으로 사용 (AgeGroup은 학습 때 drop됨)
    - 여기서는 공통 CAT_FEATURES에서, 실제로 모델 feature에 존재하는 컬럼만 object로 변환
    """
    feat_cols = model.feature_names_
    X = df[feat_cols].copy()

    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("object").fillna("nan").astype(str)

    return X


def predict_group(test_df: pd.DataFrame, group_label: str) -> np.ndarray:
    """
    A 또는 B 그룹에 대해 (fold × seed) 모델+calibrator의 예측을 평균.
    - 우선 seed 버전(catboost_{group}_seed{seed}_fold{fold}.pkl)을 찾고
    - 없으면 예전 단일 버전(catboost_{group}_fold{fold}.pkl)으로 fallback
    """
    preds_list = []
    used_models = 0

    # 1) seed 버전 탐색
    for fold in range(N_SPLITS):
        fold_used_seed = False
        for seed in SEEDS:
            cat_path = os.path.join(MODEL_DIR, f"catboost_{group_label}_seed{seed}_fold{fold}.pkl")
            cal_path = os.path.join(MODEL_DIR, f"calibrator_{group_label}_seed{seed}_fold{fold}.pkl")

            if not (os.path.exists(cat_path) and os.path.exists(cal_path)):
                continue

            cat_model = joblib.load(cat_path)
            calibrator = joblib.load(cal_path)

            if cat_model is None or calibrator is None:
                continue

            X = prepare_cb_input(test_df, cat_model)
            pred_uncal = cat_model.predict_proba(X)[:, 1]
            pred_cal = calibrator.predict(pred_uncal)

            preds_list.append(pred_cal)
            used_models += 1
            fold_used_seed = True

        # seed 버전 없으면 fold 단일 버전 fallback
        if not fold_used_seed:
            cat_path_single = os.path.join(MODEL_DIR, f"catboost_{group_label}_fold{fold}.pkl")
            cal_path_single = os.path.join(MODEL_DIR, f"calibrator_{group_label}_fold{fold}.pkl")

            if not (os.path.exists(cat_path_single) and os.path.exists(cal_path_single)):
                continue

            cat_model = joblib.load(cat_path_single)
            calibrator = joblib.load(cal_path_single)

            if cat_model is None or calibrator is None:
                continue

            X = prepare_cb_input(test_df, cat_model)
            pred_uncal = cat_model.predict_proba(X)[:, 1]
            pred_cal = calibrator.predict(pred_uncal)

            preds_list.append(pred_cal)
            used_models += 1

    if used_models == 0:
        # 안전장치: 모델이 전혀 없으면 0.5로 채움
        return np.full(len(test_df), 0.5, dtype=float)

    preds_array = np.vstack(preds_list)
    return preds_array.mean(axis=0)


# =========================
# 6. 메인 파이프라인
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1) 테스트 메타 및 원시 데이터 로드 ---
    print("테스트 데이터 로드 중...")
    test_meta = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
    test_A_raw = pd.read_csv(os.path.join(BASE_DIR, "test", "A.csv"))
    test_B_raw = pd.read_csv(os.path.join(BASE_DIR, "test", "B.csv"))
    print("테스트 데이터 로드 완료.")

    # --- 2) A/B 도메인 피처 + 2차 파생 (log_ratio 등) ---
    test_A_features = preprocess_A(test_A_raw)
    test_A_features = add_features_A(test_A_features)

    test_B_features = preprocess_B(test_B_raw)
    test_B_features = add_features_B(test_B_features)

    # --- 3) 메타와 도메인 피처 merge ---
    meta_test_A = test_meta[test_meta["Test"] == "A"].reset_index(drop=True)
    meta_test_B = test_meta[test_meta["Test"] == "B"].reset_index(drop=True)

    test_A = meta_test_A.merge(test_A_features, on="Test_id", how="left")
    test_B = meta_test_B.merge(test_B_features, on="Test_id", how="left")

    all_test_df = pd.concat([test_A, test_B], sort=False).reset_index(drop=True)

    # --- 4) 정규화 통계 로드 (Age_num / YearMonthIndex) ---
    norm_path = os.path.join(MODEL_DIR, "normalization_stats.pkl")
    if os.path.exists(norm_path):
        norm_stats = joblib.load(norm_path)
    else:
        norm_stats = {
            "Age_num_mean": None,
            "Age_num_std": None,
            "YearMonthIndex_mean": None,
            "YearMonthIndex_std": None,
        }
        print("[WARN] normalization_stats.pkl이 없어 z-score 피처는 생성되지 않습니다.")

    # --- 5) PK 히스토리 + AgeGroup/YM_is_early + Delta(B-only) ---
    all_test_df = add_pk_history_and_age_features(all_test_df, norm_stats)

    # --- 6) A/B로 다시 분리 ---
    if "Test_x" in all_test_df.columns:
        test_col = "Test_x"
    else:
        test_col = "Test"

    test_A_df = all_test_df[all_test_df[test_col] == "A"].copy()
    test_B_df = all_test_df[all_test_df[test_col] == "B"].copy()

    # --- 7) B에 PK Stats merge ---
    pk_stats_path = os.path.join(MODEL_DIR, "pk_stats_final.csv")
    if os.path.exists(pk_stats_path):
        pk_stats_final = pd.read_csv(pk_stats_path)
        test_B_df = test_B_df.merge(pk_stats_final, on="PrimaryKey", how="left")
    else:
        print("[WARN] pk_stats_final.csv가 없어 B모델에서 PK Stats를 사용하지 않습니다.")

    # --- 8) A/B 예측 ---
    print("모델 예측 시작 (A/B, 5-seeds 평균)...")
    pred_A = predict_group(test_A_df, "A")
    pred_B = predict_group(test_B_df, "B")

    # --- 9) Test_id 기준으로 하나의 DataFrame으로 합치기 ---
    pred_A_df = pd.DataFrame({"Test_id": test_A_df["Test_id"].values, "Label": pred_A})
    pred_B_df = pd.DataFrame({"Test_id": test_B_df["Test_id"].values, "Label": pred_B})
    pred_all = pd.concat([pred_A_df, pred_B_df], ignore_index=True)

    # sample_submission 순서 맞추기
    sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)

    submission = sample_sub[["Test_id"]].merge(pred_all, on="Test_id", how="left")

    # 혹시 모를 NaN은 0.5로 채우고 [0,1]로 clip
    submission["Label"] = submission["Label"].fillna(0.5)
    submission["Label"] = submission["Label"].clip(0.0, 1.0)

    # --- 10) 저장 ---
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"{out_path} 저장 완료.")


if __name__ == "__main__":
    main()
