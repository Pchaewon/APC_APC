# -*- coding: utf-8 -*-
"""process_time 파생 진단 — preprocess CSV → 어댑터 → process_time 확인"""
import sys
import pandas as pd
from preprocess_adapter import adapt_columns

csv = sys.argv[1] if len(sys.argv) > 1 else './data/WireSaw_Field_Test_preprocessed.csv'
print(f"입력: {csv}")

# 원본 로드
try:
    df = pd.read_csv(csv, encoding='cp949', encoding_errors='replace')
except Exception:
    df = pd.read_csv(csv, encoding='utf-8', encoding_errors='replace')

# RECIPE_ID 확인
rid = [c for c in df.columns if 'recipe_id' in c.lower() or 'recip_id' in c.lower()]
print(f"\nRECIPE_ID 관련 컬럼: {rid}")
if rid:
    print(f"RECIPE_ID 샘플 값: {df[rid[0]].dropna().unique()[:8].tolist()}")

# 어댑터 통과
print(f"\n--- 어댑터 통과 ---")
out = adapt_columns(df, verbose=True)

# process_time 결과
print(f"\n--- process_time 결과 ---")
if 'process_time' in out.columns:
    print(f"고유값: {out['process_time'].unique()[:10]}")
    print(f"타입: {out['process_time'].dtype}")
    print(out['process_time'].value_counts(dropna=False))
else:
    print("process_time 컬럼 생성 안 됨")
