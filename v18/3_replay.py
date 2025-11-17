import joblib
import pickle  # <-- 추가
import os      # <-- 추가

# 1. 원본 모델 파일 경로 (압축 푼 경로로 수정하세요)
model_a_path = "./model/catboost_A_fold0.pkl"
model_b_path = "./model/catboost_B_fold0.pkl"

# ★ 저장할 경로 ★
SAVE_DIR = "./model"
os.makedirs(SAVE_DIR, exist_ok=True)

try:
    print("원본 A 모델 로드 중...")
    original_model_A = joblib.load(model_a_path)
    ORIGINAL_A_FEATURES = original_model_A.feature_names_
    print(f"[A 모델] 원본 피처 {len(ORIGINAL_A_FEATURES)}개 순서 확보 완료.")

    print("\n원본 B 모델 로드 중...")
    original_model_B = joblib.load(model_b_path)
    ORIGINAL_B_FEATURES = original_model_B.feature_names_
    print(f"[B 모델] 원본 피처 {len(ORIGINAL_B_FEATURES)}개 순서 확보 완료.")

    # --- [이 부분 추가] ---
    # 피처 리스트를 별도 pkl 파일로 저장
    a_save_path = os.path.join(SAVE_DIR, "original_a_features.pkl")
    b_save_path = os.path.join(SAVE_DIR, "original_b_features.pkl")

    with open(a_save_path, "wb") as f:
        pickle.dump(ORIGINAL_A_FEATURES, f)
    with open(b_save_path, "wb") as f:
        pickle.dump(ORIGINAL_B_FEATURES, f)
    
    print(f"\n[성공] 원본 피처 순서 리스트를 '{SAVE_DIR}'에 저장했습니다.")
    # --- [추가 끝] ---

except FileNotFoundError:
    print(f"오류: '{model_a_path}' 또는 '{model_b_path}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"모델 로드/저장 중 오류 발생: {e}")
