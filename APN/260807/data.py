# -*- coding: utf-8 -*-
"""
pilot_wiresaw.py  (기존 FDC 테이블 + BLK_NO 직접 조회 · 10대 필터)
════════════════════════════════════════════════════════════
[확정된 해결책]
  · FDC 테이블: DW_FA_EQ_FDCXWSNEW_S (권한 있음, 인자가 이미 컬럼)
  · 조인: 3200.BLK_NO → FDC.LOT_ID 직접 조회 (진단 결과 20/20 = 100% 매칭)
  · EQP 조건 제거: 3200.EQP_ID(TSW체계) ≠ FDC.EQP_NM(BSWS체계)라 BLK_NO만으로
  · PARAM 매핑 불필요: SET_FRAME_TEMP 등이 이미 컬럼

  흐름: Y(3305) → LOT_ID → 3200에서 BLK_NO → FDC를 LOT_ID=BLK_NO로 조회 → 병합

[기존 문제였던 것]
  날짜(BASE_DT)로 FDC 긁어 병합 → common 2개 (실패)
  → BLK_NO 직접 조회로 100% 매칭 해결

대상 장비: BSWS30,31,35,42,48 (15-19연식) + BSWS51,53,55,57,58 (21연식)
"""
import trino
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import time
import traceback

warnings.filterwarnings("ignore")

# ============================================================
# [Settings]
# ============================================================

if not DB_CONFIG.get('password'):
    raise EnvironmentError("TRINO_PASSWORD missing.")

SEARCH_HOURS_AGO = 3            # 테스트=3, 하루=24, 일주일=168
TARGET_OPE_ID_Y = '3305'
TARGET_OPE_ID_X = '3200'

TARGET_EQPS = ['BSWS30', 'BSWS31', 'BSWS35', 'BSWS42', 'BSWS48',
               'BSWS51', 'BSWS53', 'BSWS55', 'BSWS57', 'BSWS58']

Y_MEAS_ITEMS = [
    'SL-WARP-BF', 'SL-BOW-BF', 'SL-TTV-ALL', 'SL-THK-AV',
    'PSE1-0005', 'PSE1-0006', 'PSE1-0007', 'PSE1-0008',
]

# 기존 FDC 테이블에서 가져올 컬럼 (인자가 이미 컬럼으로 존재)
FDC_COLS = [
    'LOT_ID', 'EQP_NM', 'RECIPE_ID', 'HST_REG_DTTM', 'NEW_WIRE_ID',
    'WIRE_RUN', 'RUNTIME_MINUTES', 'INGOT_LEN', 'WAIT_TIME',
    'FRAME_IN_TEMP', 'SET_FRAME_TEMP', 'SLURRY_IN_TEMP', 'SET_SLURRY_TEMP',
    'SHIFT_AMOUNT_WIREGUIDE_L', 'SHIFT_AMOUNT_WIREGUIDE_R',
    'SLURRY_LIFE_TIME', 'WIREGUIDE_LIFE_TIME', 'SET_TENSION', 'ELONGATION',
]


def match_target(name):
    s = str(name).upper()
    return any(t in s for t in TARGET_EQPS)


now = datetime.now()
start_time_range = now - timedelta(hours=SEARCH_HOURS_AGO)
date_str_start_full = start_time_range.strftime('%Y%m%d%H%M%S')
date_str_end_full = now.strftime('%Y%m%d%H%M%S')

print("=" * 80)
print(f"[WireSaw] 기존 FDC + BLK_NO 직접 조회 방식")
print(f"  Period: {date_str_start_full} ~ {date_str_end_full}")
print(f"  대상 장비: {len(TARGET_EQPS)}대")

conn = None
cur = None
df_final = None

try:
    conn = trino.dbapi.connect(
        host=DB_CONFIG['host'], port=DB_CONFIG['port'], user=DB_CONFIG['user'],
        http_scheme='https',
        auth=trino.auth.BasicAuthentication(DB_CONFIG['user'], DB_CONFIG['password']),
        catalog='iceberg', schema='ibg_lake', verify=False,
    )
    cur = conn.cursor()
    print("DB connected")

    # ========================================================
    # [Part 1] Y Quality (OPE_ID 3305) — 장비 필터 없음(기간만)
    # ========================================================
    print(f"\n[Part 1] Y Quality (OPE_ID={TARGET_OPE_ID_Y})...")
    t0 = time.time()
    query_y = f"""
        WITH V_CS_TYPE AS
        (SELECT C.MS_CODE, D.CODE_NAME AS CUST_SITE_NM, C.GRADE
            FROM ORACLE.OGGZMGR.PIMS_PROD C,
                 ORACLE.OGGZMGR.PIMS_CODE D,
                 ORACLE.OGGZMGR.PIMS_CODE E,
                 ORACLE.OGGZMGR.PIMS_CODE F
          WHERE C.SPEC_TYPE = 'CS'
            AND D.CODE_CAT = 'CUSTOMER' AND D.CODE_VALUE = C.CUSTOMER
            AND E.CODE_CAT = 'GRADE' AND E.CODE_VALUE = C.GRADE
            AND F.CODE_CAT = 'INGOT_GRADE' AND F.CODE_VALUE = C.INGOT_GRADE)
        SELECT T1.LOT_ID, T1.HIS_REGIST_DTTM, T1.FAB_ID, T1.RECIP_ID,
               T2.WAF_SEQ_NO, T3.EQP_NAME, T1.PROD_ID AS MS_CODE,
               CASE WHEN T5.EPI = 'Y' THEN 'EPI' ELSE 'PW' END AS PW_EP_GROUP,
               T6.CUST_SITE_NM AS CUSTOMER, T7.CHAR_VALUE,
               T1.SUBLOT_ID, T1.CAR_ID AS CST_ID, T2.SLOT_NO,
               T4.MEAS_ID, T4.MEAS_ITEM, T4.MEAS_DATA
        FROM ORACLE.OGGZMGR.ODB_DOPE_HIS T1
        INNER JOIN ORACLE.OGGZMGR.ODB_DWAF_OPE_HIS T2
            ON T1.HIS_REGIST_DTTM = T2.HIS_REGIST_DTTM AND T1.SUBLOT_ID = T2.SUBLOT_ID
        INNER JOIN ORACLE.OGGZMGR.ODB_DEQP T3
            ON T1.EQP_ID = T3.EQP_ID
        INNER JOIN ORACLE.OGGZMGR.ODB_DMS_C_HIS T4
            ON T1.HIS_REGIST_DTTM = T4.HIS_REGIST_DTTM
            AND T2.SINGLE_NO = T4.SINGLE_NO AND T2.WAF_SEQ_NO = T4.WAF_SEQ_NO
            AND T4.MEAS_ITEM IN ({', '.join([f"'{i}'" for i in Y_MEAS_ITEMS])})
        INNER JOIN ORACLE.OGGZMGR.PIMS_PROD T5 ON T1.PROD_ID = T5.MS_CODE
        INNER JOIN V_CS_TYPE T6 ON T1.PROD_ID = T6.MS_CODE
        INNER JOIN ORACLE.OGGZMGR.PIMS_SPEC T7
            ON T1.PROD_ID = T7.MS_CODE AND T7.SPEC_TYPE = 'CS'
            AND T7.PARAMETER = 'SILTRON_PART#'
        WHERE T1.HIS_REGIST_DTTM BETWEEN '{date_str_start_full}' AND '{date_str_end_full}'
            AND T1.OPE_ID = '{TARGET_OPE_ID_Y}'
            AND T1.HIS_CAT = 'OC'
    """
    cur.execute(query_y)
    df_y = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    print(f"  Done: {len(df_y):,} rows ({time.time()-t0:.1f}s)")

    if df_y.empty:
        print("  No Y data. Exiting.")
    else:
        df_y.columns = df_y.columns.str.upper()
        if 'MEAS_ITEM' in df_y.columns and 'MEAS_DATA' in df_y.columns:
            idx_cols = [c for c in df_y.columns if c not in ['MEAS_ITEM', 'MEAS_DATA', 'MEAS_ID']]
            df_y = df_y.pivot_table(index=idx_cols, columns='MEAS_ITEM',
                                    values='MEAS_DATA', aggfunc='first').reset_index()
        lot_ids = df_y['LOT_ID'].dropna().unique().tolist()
        print(f"  Unique LOT_IDs: {len(lot_ids)}")

        # ====================================================
        # [Part 2] 3200 → BLK_NO (LOT_ID로 조회)
        # ====================================================
        print(f"\n[Part 2] 3200 → BLK_NO...")
        t0 = time.time()
        lot_in = "', '".join(str(l) for l in lot_ids)
        cur.execute(f"""
            SELECT A.LOT_ID, A.BLK_NO, A.EQP_ID
            FROM ORACLE.OGGZMGR.ODB_DOPE_HIS A
            WHERE A.OPE_ID = '{TARGET_OPE_ID_X}'
              AND A.LOT_ID IN ('{lot_in}')
        """)
        df_3200 = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        df_3200.columns = df_3200.columns.str.upper()
        blk_nos = df_3200['BLK_NO'].dropna().astype(str).unique().tolist()
        blk_nos = [b for b in blk_nos if b.strip()]
        print(f"  3200: {len(df_3200)}행 → BLK_NO {len(blk_nos)}개 ({time.time()-t0:.1f}s)")

        # ====================================================
        # [Part 3] FDC — BLK_NO로 직접 조회 (기존 테이블, EQP 조건 없음)
        # ====================================================
        print(f"\n[Part 3] FDC (DW_FA_EQ_FDCXWSNEW_S) BLK_NO 직접 조회...")
        t0 = time.time()
        df_fdc = pd.DataFrame()
        if blk_nos:
            col_sel = ', '.join(f'"{c}"' for c in FDC_COLS)
            # BLK_NO를 배치로 나눠 조회 (IN 절 길이 제한 대비, 500개씩)
            frames = []
            for i in range(0, len(blk_nos), 500):
                batch = blk_nos[i:i+500]
                blk_in = "', '".join(batch)
                try:
                    cur.execute(f"""
                        SELECT {col_sel}
                        FROM iceberg.ibg_lake.DW_FA_EQ_FDCXWSNEW_S
                        WHERE "LOT_ID" IN ('{blk_in}')
                    """)
                    tmp = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
                    if not tmp.empty:
                        frames.append(tmp)
                except Exception as e:
                    print(f"  배치 {i}: 실패 — {e}")
            if frames:
                df_fdc = pd.concat(frames, ignore_index=True)
                df_fdc.columns = df_fdc.columns.str.upper()
        print(f"  FDC: {len(df_fdc)}행 ({time.time()-t0:.1f}s)")
        if not df_fdc.empty:
            # 시계열 확인 (pct 프로파일 가능 여부)
            pts = df_fdc.groupby('LOT_ID').size()
            print(f"  BLK_NO당 시계열 평균 {pts.mean():.1f}포인트 "
                  f"(2이상이면 pct 프로파일 생성 가능)")
            print(f"  FDC EQP_NM 샘플: {df_fdc['EQP_NM'].dropna().unique()[:5].tolist()}")

        # ====================================================
        # [Part 5] 병합: Y + 3200(BLK_NO) + FDC(LOT_ID=BLK_NO)
        # ====================================================
        print(f"\n[Part 5] Merge...")
        lot_to_blk = df_3200[['LOT_ID', 'BLK_NO']].drop_duplicates('LOT_ID')
        df_merged = pd.merge(df_y, lot_to_blk, on='LOT_ID', how='left')

        if not df_fdc.empty:
            # FDC의 LOT_ID = BLK_NO. rename 후 병합
            df_fdc_m = df_fdc.rename(columns={'LOT_ID': 'BLK_NO'})
            common = len(set(df_merged['BLK_NO'].dropna()) & set(df_fdc_m['BLK_NO'].dropna()))
            print(f"  BLK_NO 매칭: {common}개 (기존 날짜방식 2개 → 이만큼)")
            # FDC는 BLK_NO(LOT)당 여러 행(시계열) → 병합 시 확장됨 (preprocess가 pct 생성)
            df_merged = pd.merge(df_merged, df_fdc_m, on='BLK_NO', how='left')
        print(f"  merged: {len(df_merged)}행")

        # 최종 장비 필터 (FDC의 EQP_NM = BSWS 형태)
        eqp_col = None
        for c in df_merged.columns:
            if 'EQP_NM' in c.upper() or c.upper() == 'EQP_NM':
                if any(match_target(v) for v in df_merged[c].dropna().astype(str)):
                    eqp_col = c
                    break
        if eqp_col:
            before = len(df_merged)
            df_merged = df_merged[df_merged[eqp_col].apply(match_target)].copy()
            print(f"  [최종 장비필터] '{eqp_col}' 기준 {before} → {len(df_merged)}행")
        else:
            print("  ⚠ BSWS 형태 EQP 컬럼 없음 — 필터 스킵")
        df_final = df_merged

    if cur: cur.close()
    if conn: conn.close()
    print("\nDB closed.")

    # ========================================================
    # Save + 커버리지
    # ========================================================
    print("\n" + "=" * 80)
    if df_final is not None and not df_final.empty:
        print(f"[WireSaw] Done! Rows: {len(df_final):,}, Cols: {len(df_final.columns)}")
        eqp_col = None
        for c in df_final.columns:
            if 'EQP_NM' in c.upper():
                if any(match_target(v) for v in df_final[c].dropna().astype(str)):
                    eqp_col = c
                    break
        if eqp_col:
            got = set()
            for name in df_final[eqp_col].astype(str).unique():
                for t in TARGET_EQPS:
                    if t in name.upper():
                        got.add(t)
            missing = sorted(set(TARGET_EQPS) - got)
            print(f"  [10대 커버리지] {len(got)}/10 ('{eqp_col}'): {sorted(got)}")
            if missing:
                print(f"  ⚠ 데이터 없는 장비: {missing}")
            print(f"  [장비별 행수]")
            for k, v in df_final[eqp_col].value_counts().items():
                print(f"     {k}: {v}")

        out = './data/WireSaw_Field_Test.csv'
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        df_final.to_csv(out, encoding='cp949', index=False)
        print(f"  CSV saved: {out}")
    else:
        print("  No data to save.")
    print("=" * 80)

except Exception as e:
    print(f"\nError: {e}")
    traceback.print_exc()
finally:
    try:
        if cur: cur.close()
        if conn: conn.close()
    except:
        pass
    print("\nAll connections closed.")
