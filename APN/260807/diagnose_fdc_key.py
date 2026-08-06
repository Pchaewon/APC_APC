# -*- coding: utf-8 -*-
"""
diagnose_fdc_key.py
─────────────────────────────────────────
기존 FDC 테이블(DW_FA_EQ_FDCXWSNEW_S, 권한 있음)의 LOT_ID가
3200의 BLK_NO와 매칭되는지 확인.
→ DBA 방식(BLK_NO+EQP 직접 조회)을 기존 테이블에 적용 가능한지 판단.

돌리는 법:
  python diagnose_fdc_key.py
"""
import os
import trino
import pandas as pd
from datetime import datetime, timedelta

SEARCH_HOURS_AGO = 3

conn = trino.dbapi.connect(
    host=DB_CONFIG['host'], port=DB_CONFIG['port'], user=DB_CONFIG['user'],
    http_scheme='https',
    auth=trino.auth.BasicAuthentication(DB_CONFIG['user'], DB_CONFIG['password']),
    catalog='iceberg', schema='ibg_lake', verify=False,
)
cur = conn.cursor()
print("DB connected\n")

now = datetime.now()
start = (now - timedelta(hours=SEARCH_HOURS_AGO)).strftime('%Y%m%d%H%M%S')
end = now.strftime('%Y%m%d%H%M%S')

# ── 1) Y에서 최근 LOT_ID 몇 개 ──
print("=== 1) Y 검사 LOT_ID ===")
cur.execute(f"""
    SELECT DISTINCT T1.LOT_ID
    FROM ORACLE.OGGZMGR.ODB_DOPE_HIS T1
    WHERE T1.HIS_REGIST_DTTM BETWEEN '{start}' AND '{end}'
      AND T1.OPE_ID = '3305' AND T1.HIS_CAT = 'OC'
    LIMIT 30
""")
lot_ids = [r[0] for r in cur.fetchall()]
print(f"  LOT_ID {len(lot_ids)}개: {lot_ids[:5]}...\n")

if not lot_ids:
    print("  Y 데이터 없음 — 기간 늘려서 재시도")
    cur.close(); conn.close(); exit()

# ── 2) 3200에서 BLK_NO, EQP_ID ──
print("=== 2) 3200 → BLK_NO, EQP_ID ===")
lot_in = "', '".join(lot_ids)
cur.execute(f"""
    SELECT A.LOT_ID, A.BLK_NO, A.EQP_ID
    FROM ORACLE.OGGZMGR.ODB_DOPE_HIS A
    WHERE A.OPE_ID = '3200' AND A.LOT_ID IN ('{lot_in}')
""")
df_3200 = pd.DataFrame(cur.fetchall(), columns=['LOT_ID', 'BLK_NO', 'EQP_ID'])
blk_nos = df_3200['BLK_NO'].dropna().astype(str).unique().tolist()
eqp_ids = df_3200['EQP_ID'].dropna().astype(str).unique().tolist()
print(f"  BLK_NO {len(blk_nos)}개: {blk_nos[:5]}")
print(f"  EQP_ID {len(eqp_ids)}개: {eqp_ids[:5]}\n")

# ── 3) 기존 FDC 테이블에서 최근 데이터의 LOT_ID, EQP_NM 샘플 ──
print("=== 3) 기존 FDC(DW_FA_EQ_FDCXWSNEW_S) 키 샘플 ===")
recent_dt = (now - timedelta(days=7)).strftime('%Y%m%d')
cur.execute(f"""
    SELECT DISTINCT "LOT_ID", "EQP_NM", "NEW_WIRE_ID"
    FROM iceberg.ibg_lake.DW_FA_EQ_FDCXWSNEW_S
    WHERE BASE_DT >= '{recent_dt}'
    LIMIT 30
""")
df_fdc = pd.DataFrame(cur.fetchall(), columns=['LOT_ID', 'EQP_NM', 'NEW_WIRE_ID'])
fdc_lots = df_fdc['LOT_ID'].dropna().astype(str).unique().tolist()
fdc_eqps = df_fdc['EQP_NM'].dropna().astype(str).unique().tolist()
print(f"  FDC LOT_ID 샘플: {fdc_lots[:5]}")
print(f"  FDC EQP_NM 샘플: {fdc_eqps[:5]}")
print(f"  FDC NEW_WIRE_ID 샘플: {df_fdc['NEW_WIRE_ID'].dropna().astype(str).unique()[:5].tolist()}\n")

# ── 4) 교집합 확인: 어느 키가 맞나 ──
print("=== 4) 키 매칭 확인 ===")
blk_set = set(blk_nos)
fdc_lot_set = set(fdc_lots)
print(f"  3200.BLK_NO ∩ FDC.LOT_ID: {len(blk_set & fdc_lot_set)}개 "
      f"(겹치면 LOT_ID=BLK_NO 조회 가능)")

# BLK_NO를 FDC LOT_ID로 직접 조회 시도 (실제 매칭 테스트)
if blk_nos:
    blk_test = "', '".join(blk_nos[:20])
    cur.execute(f"""
        SELECT COUNT(DISTINCT "LOT_ID") AS n
        FROM iceberg.ibg_lake.DW_FA_EQ_FDCXWSNEW_S
        WHERE "LOT_ID" IN ('{blk_test}')
    """)
    n_match = cur.fetchone()[0]
    print(f"  BLK_NO로 FDC 직접 조회 매칭: {n_match}개 / {len(blk_nos[:20])}개 시도")

# EQP 형태 비교
print(f"\n  3200.EQP_ID 형태: {eqp_ids[:3]} (예: TSW505)")
print(f"  FDC.EQP_NM 형태: {fdc_eqps[:3]} (예: BSWS30)")
print("  → 두 EQP 형태가 다르면, FDC 조회는 EQP 빼고 LOT_ID(=BLK_NO)만으로")

cur.close(); conn.close()
print("\n완료. 위 4번 결과가 핵심:")
print("  · BLK_NO ∩ FDC.LOT_ID > 0 이면 → LOT_ID=BLK_NO 조회 가능 (해결)")
print("  · 0 이면 → 다른 키 필요 (NEW_WIRE_ID 등 추가 확인)")
