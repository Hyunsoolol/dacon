1) B 전용 “A→B 전이(feature transfer)” 강화 (최우선)

목표: “직전 A 값”을 그대로 쓰지 말고, 시간·횟수·안정성 신호를 함께 주입합니다.

핵심 피처 설계

has_prev_A: 직전 A 존재 여부(0/1). 둘 다 본 3.3%만 1이므로 정보량 큼.

gap_AB: B시점 − 직전 A시점(월·일 단위). 짧을수록 전이 신뢰↑.

recency_weight = exp(−λ·gap_AB). λ는 0.05~0.15 범위 그리드.

A_signal_stable: 직전 A와 직전-1 A(두 번 이상 본 사람만)의 차이(변동성). 안정적일수록 신뢰↑.

A_agg_hist: A를 여러 번 본 사람의 누적 통계(평균/표준편차/최댓값/횟수).

A×B 상호작용: (직전 A 지표)×(B의 동종 지표), 예: A4_stroop_rt_log_ratio_from_last_A × B4_flanker_acc_log_ratio.

mask & winsor: 직전 A가 너무 과거이거나(gap_AB>G_MAX), A/B 단위가 맞지 않는 경우 마스킹하고, 극단치는 윈저라이즈.

참고 코드 조각

# B 행마다 “직전 A”를 붙일 때: 동일 PK, A의 YearMonthIndex < B의 YearMonthIndex
# gap_AB, has_prev_A, recency_weight, A_agg_hist, A_signal_stable까지 생성

G_MAX = 36  # 월 기준 상한(권장: 18~36 그리드)
LAMBDA = 0.1

def attach_last_A_features(dfA, dfB):
    # dfA/dfB: PrimaryKey, YearMonthIndex, A_* / B_* 컬럼 존재
    # 1) A 이력 정렬
    dfA = dfA.sort_values(['PrimaryKey','YearMonthIndex'])
    # 2) A 누적 통계(횟수/평균/표준편차/최대값) – 월별로 up-to-(t-1) 까지 rolling이 이상적이나,
    #    간단히 “직전 A까지” 집계를 만들려면 merge 전에 groupby.transform 사용
    a_stats = (dfA
        .groupby('PrimaryKey')
        .agg(A_cnt=('YearMonthIndex','count'))
        .reset_index())

    # 3) B마다 직전 A 찾기
    #    효율을 위해 PK별로 merge_asof 사용 (Pandas 2.x 권장)
    dfA_key = dfA[['PrimaryKey','YearMonthIndex','A4_stroop_rt_log_ratio','A3_acc_attention_log_ratio']]\
                .rename(columns={'YearMonthIndex':'YM_A'})
    dfB_key = dfB[['PrimaryKey','YearMonthIndex','B4_flanker_acc_log_ratio']]\
                .rename(columns={'YearMonthIndex':'YM_B'})
    dfA_key = dfA_key.sort_values(['PrimaryKey','YM_A'])
    dfB_key = dfB_key.sort_values(['PrimaryKey','YM_B'])

    merged = pd.merge_asof(
        dfB_key, dfA_key,
        left_on='YM_B', right_on='YM_A',
        by='PrimaryKey', direction='backward', allow_exact_matches=False
    )

    # 4) 파생
    merged['gap_AB'] = merged['YM_B'] - merged['YM_A']
    merged['has_prev_A'] = (~merged['YM_A'].isna()).astype(int)
    merged['recency_weight'] = np.exp(-LAMBDA * merged['gap_AB'].clip(lower=0).fillna(G_MAX))
    merged.loc[(merged['has_prev_A']==1) & (merged['gap_AB']>G_MAX), ['has_prev_A']] = 0  # 너무 오래된 A는 무시

    # 5) 극단치 처리
    def winsor(s, q=0.01):
        lo, hi = s.quantile(q), s.quantile(1-q)
        return s.clip(lo, hi)

    for col in ['A4_stroop_rt_log_ratio','A3_acc_attention_log_ratio','B4_flanker_acc_log_ratio']:
        if col in merged.columns:
            merged[col] = winsor(merged[col])

    # 6) 상호작용
    merged['A4xB4_interact'] = merged['A4_stroop_rt_log_ratio'] * merged['B4_flanker_acc_log_ratio']
    # 7) 신뢰가중 A신호
    for col in ['A4_stroop_rt_log_ratio','A3_acc_attention_log_ratio']:
        if col in merged.columns:
            merged[f'{col}_recency'] = merged[col] * merged['recency_weight']

    # 8) 필요한 컬럼만 반환하여 원본 B에 merge
    use_cols = ['PrimaryKey','YM_B','has_prev_A','gap_AB','recency_weight',
                'A4_stroop_rt_log_ratio','A3_acc_attention_log_ratio',
                'A4xB4_interact','A4_stroop_rt_log_ratio_recency','A3_acc_attention_log_ratio_recency']
    return dfB.merge(merged[use_cols], left_on=['PrimaryKey','YearMonthIndex'], right_on=['PrimaryKey','YM_B'], how='left')\
              .drop(columns=['YM_B'])


운영 팁

결측(=has_prev_A=0)은 0으로 안전 채움 + “_mask”로 구분.

CV는 반드시 PK-그룹 및 time-aware 블록을 유지(누설 방지).

2) B 모델을 “두 코호트”로 나누기 (효과 큼)

Cohort 1: has_prev_A=1 (둘 다 본 사람) → 전이 피처를 풀로 사용하는 B₁ 모델

Cohort 2: has_prev_A=0 (B-only) → 전이 피처가 없으므로 PK 히스토리/연령/시간 피처를 강화한 B₂ 모델

추론 시 조건부 라우팅: has_prev_A==1이면 B₁, 아니면 B₂ 점수를 사용.

기대효과: 3.3%만 보이지만, 난이도 높은 케이스에서 AUC/Brier 개선 + 전체 스코어의 안정화.

3) 반복 응시자(특히 B) 처리 고도화 (효과 중간)

B는 2회 이상이 6.7만명으로 많음 → PK별 B 누적/최근 통계(cnt, mean, std, last_gap_BB), 최근 k회 모드/추세(slope) 추가.

단, 누설 방지를 위해 fold-train 내부의 과거 관측만 사용(merge_asof/rolling).

예시 피처

pk_B_cnt_sofar, pk_B_last_gap, pk_B_mean_rt/std, pk_B_trend_rt(k=2~3)

A에도 동일 구조(빈도는 낮지만 일관성 신호에 도움).

4) 손실·가중치·규제 (효과 중간)

소수 서브그룹(예: A의 21–30/≤20, B의 early)에 sample_weight 가중치(1.1~1.3) 부여.

B 모델 규제 강화(이미 제안): depth 3→4, l2_leaf_reg 5→8, random_strength 3→4.

PK를 cat_features에서 제외(가능하면)하고 pk_hist_*로 대체. 극단 FP 클러스터(고연령×late YM) 억제.

5) 보정과 서브그룹 리베이스 (효과 작지만 안전)

이미 ECE≈0이므로 전역 Platt(OoF)로 일원화 + AgeBin×YM_bin 리베이스로 미세 편향 제거.

임계값을 코호트별로 따로 최적화하면(F1/Youden), FP 스파이크 추가 완화.
