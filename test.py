import pandas as pd
import numpy as np
import glob, os

# ── 1. 저장된 결과 CSV 불러오기
resdir = r"D:\chaewon\APC\wire_saw\apc_code\add_web_data\result\frame_to_bow"
files = glob.glob(os.path.join(resdir, "bow_range_8point_by_EQP_Group_Recipe_*.csv"))
latest = max(files, key=os.path.getmtime)
print("불러온 파일:", latest)

df = pd.read_csv(latest, encoding="utf-8-sig")

# ── 2. 프로파일 문자열을 8개 온도 컬럼으로 분할하는 함수
N = 8
def split_profile(series, prefix, n=N):
    def _parse(s):
        if not isinstance(s, str) or s in ("-", "None"):
            return [np.nan] * n
        vals = [v.strip() for v in s.split("→")]
        out = []
        for v in vals[:n]:
            try:
                out.append(round(float(v), 1))
            except (ValueError, TypeError):
                out.append(np.nan)
        out += [np.nan] * (n - len(out))   # 길이 부족 시 패딩
        return out
    cols = [f"{prefix}_{i+1}" for i in range(n)]
    return pd.DataFrame(series.apply(_parse).tolist(), columns=cols, index=series.index)

# ── 3. 분할 실행 후 원본에 붙이기
rec_split = split_profile(df["Recommended_Profile"], "predict_temp")
base_split = split_profile(df["Base_Profile"], "temp")
df_out = pd.concat([df, rec_split, base_split], axis=1)

# ── 4. 확인
predict_cols = [f"predict_temp_{i+1}" for i in range(N)]
base_cols = [f"temp_{i+1}" for i in range(N)]
print("\n[추천 온도]")
print(df_out[["EQP", "Group"] + predict_cols].head())
print("\n[기존 온도]")
print(df_out[["EQP", "Group"] + base_cols].head())

# ── 5. (선택) 테스트 결과 다시 저장
out_path = os.path.join(resdir, "test_split_result.csv")
df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\n저장:", out_path)
