# -*- coding: utf-8 -*-
"""
process_time 파생 진단 (정확 컬럼 버전)
─────────────────────────────────────────
기존 버그: rid[0] 하나만 찍어서, 첫 컬럼(RECIP_ID)의 값이
          진짜 RECIPE_ID 값인 것처럼 오인됨.
수정: recipe 계열 3개 컬럼(RECIP_ID / RECIPE_ID_3200 / RECIPE_ID)을
      '각각' 분포로 보여줌 → 어느 컬럼이 시간 코드를 담는지 확정.

사용:
  python diagnose_process_time.py ./data/WireSaw_Field_Test_preprocessed.csv
"""
import sys
import pandas as pd
from preprocess_adapter import adapt_columns

csv = sys.argv[1] if len(sys.argv) > 1 else './data/WireSaw_Field_Test_preprocessed.csv'
try:
    df = pd.read_csv(csv, encoding='cp949', encoding_errors='replace')
except Exception:
    df = pd.read_csv(csv, encoding='utf-8', encoding_errors='replace')
print(f"입력: {csv} ({len(df)}행)\n")

# ── 1) recipe 계열 컬럼 '각각' 분포 ──
recip_cols = [c for c in df.columns if 'RECIP' in c.upper()]
print(f"[recipe 계열 컬럼 전체] {recip_cols}\n")
for c in recip_cols:
    print(f"=== {c} 분포 (상위 15) ===")
    print(df[c].value_counts(dropna=False).head(15).to_string())
    print(f"  (고유값 {df[c].nunique(dropna=False)}개, 결측 {df[c].isna().sum()}행)\n")

# ── 2) 0으로 덮은 범인 후보 확인 ──
print("[process_time placeholder 후보 확인]")
found_placeholder = False
for c in ['PROCESS_TIME', 'RUNTIME_MINUTES']:
    hit = [col for col in df.columns if col.upper() == c]
    if hit:
        col = hit[0]
        print(f"  · {col} 존재 → 고유값 {df[col].unique()[:10]}, dtype={df[col].dtype}")
        found_placeholder = True
if not found_placeholder:
    print("  · PROCESS_TIME / RUNTIME_MINUTES 컬럼 없음 "
          "(0은 어댑터 rename이 아니라 다른 경로)")
print()

# ── 3) 어댑터 통과 후 process_time ──
print("--- 어댑터 통과 ---")
out = adapt_columns(df, verbose=True)
print("\n--- 어댑터 후 process_time 결과 ---")
if 'process_time' in out.columns:
    print(f"고유값: {out['process_time'].unique()[:10]}")
    print(f"타입: {out['process_time'].dtype}")
    print(out['process_time'].value_counts(dropna=False).to_string())
else:
    print("process_time 컬럼 생성 안 됨")
