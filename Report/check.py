
# -*- coding: utf-8 -*-
"""
pilot_wiresaw.py  (DBA SQL 방식 · 10대 장비 필터)
════════════════════════════════════════════════════════════
DBA가 제공한 SQL 흐름을 그대로 반영:
  Y(3305 검사) → LOT_ID
    → 3200 조회로 (BLK_NO, EQP_ID) 취득
    → FDC(EQP_TRACE_*)를 LOT_ID=BLK_NO AND EQP_ID 로 직접 조회   ★핵심 변경
    → 병합 (BLK_NO 기준)

[기존 문제] FDC를 BASE_DT(날짜)로 긁어 BLK_NO 병합 → common 2개 (거의 실패)
[해결] BLK_NO+EQP_ID로 직접 조회 → 정확 매칭

★ 이미지로만 확인해 불확실한 곳은 '# ★확인' 표시.
  특히: SET_FRAME_TEMP / SET_SLURRY_TEMP 의 PARAM 코드가 이미지 목록에 없음.
        (이미지엔 FRAME_IN_TEMP=PARAM_036 만 있음)
        preprocess가 SET_FRAME_TEMP로 pct를 만드므로, 이게 없으면 추천 불가.
        → 실행 후 pct가 비면 PARAM_MAP에 SET_FRAME_TEMP 코드를 추가할 것.

대상 장비: 15-19연식 5대(BSWS30,31,35,42,48) + 21연식 5대(BSWS51,53,55,57,58)
"""
import trino
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import os
import warnings
import time
import traceback

warnings.filterwarnings("ignore")

# ============================================================
# [Settings]
# ============================================================


SEARCH_HOURS_AGO = 3            # ★ 테스트=3, 하루=24, 일주일=168
TARGET_OPE_ID_Y = '3305'
TARGET_OPE_ID_X = '3200'

# 대상 장비 10대
TARGET_EQPS = ['BSWS30', 'BSWS31', 'BSWS35', 'BSWS42', 'BSWS48',
               'BSWS51', 'BSWS53', 'BSWS55', 'BSWS57', 'BSWS58']

# 장비 필터 SQL 조각
_LIKE_CONDS = ' OR '.join([f"UPPER(T3.EQP_NAME) LIKE '%{e}%'" for e in TARGET_EQPS])
EQP_FILTER_Y = f"AND ({_LIKE_CONDS})"

Y_MEAS_ITEMS = [
    'SL-WARP-BF', 'SL-BOW-BF', 'SL-TTV-ALL', 'SL-THK-AV',
    'PSE1-0005', 'PSE1-0006', 'PSE1-0007', 'PSE1-0008',
]

# ── DBA SQL 이미지4 PARAM 매핑 (RAWDATA_MAPPING → 인자명) ──
#   ★확인: SET_FRAME_TEMP / SET_SLURRY_TEMP 코드가 이미지에 없음.
#   아래는 이미지에서 읽은 12개. preprocess가 필요로 하는 인자가 빠졌으면 추가.
PARAM_MAP = {
    'PARAM_003': 'AVG_MAIN_DRIVE_POWER',
    'PARAM_036': 'FRAME_IN_TEMP',
    'PARAM_043': 'SHIFT_AMOUNT_WIREGUIDE_L',
    'PARAM_018': 'SLURRY_DENSITY',
    'PARAM_024': 'SLURRY_IN_TEMP',
    'PARAM_015': 'SLURRY_MASS_FLOW_KG_MIN',
    'PARAM_075': 'WIREGUIDE_LIFE_TIME',
    'PARAM_060': 'SET_TABLE_SPEED',
    'PARAM_006': 'SET_TENSION',
    'PARAM_012': 'ACT_REVOLUTION_OF_SLURRY_PUMP_1',
    'PARAM_013': 'ACT_REVOLUTION_OF_SLURRY_PUMP_2',
    'PARAM_078': 'NEW_WIRE_ID',
    # ★확인: 아래는 이미지에 없지만 preprocess가 쓸 가능성이 큰 인자.
    #   실제 PARAM 코드를 DBA SQL 전문/DBA 확인 후 주석 해제·수정.
    # 'PARAM_0??': 'SET_FRAME_TEMP',
    # 'PARAM_0??': 'SET_SLURRY_TEMP',
    # 'PARAM_043_R?': 'SHIFT_AMOUNT_WIREGUIDE_R',   # L만 이미지에 있음
    # 'PARAM_0??': 'RUNTIME_MINUTES',               # pct x축용 (있어야 프로파일 가능)
}
FDC_PARAMS = list(PARAM_MAP.keys())


def match_target(name):
    s = str(name).upper()
    return any(t in s for t in TARGET_EQPS)


now = datetime.now()
start_time_range = now - timedelta(hours=SEARCH_HOURS_AGO)
date_str_start_full = start_time_range.strftime('%Y%m%d%H%M%S')
date_str_end_full = now.strftime('%Y%m%d%H%M%S')

print("=" * 80)
print(f"[WireSaw] DBA SQL 방식 수집")
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
    # [Part 1] Y Quality (OPE_ID 3305)  — 장비 LIKE 필터
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
    # ★ Part 1 장비 필터 제거: T3.EQP_NAME이 BSAFS/P3SAFS 형태라
    #   LIKE '%BSWS30%'가 0건. 정확 필터는 Part 5 최종 안전망(match_target)이 담당.
    #   (EQP_FILTER_Y는 정의만 남겨두고 쿼리엔 미적용)
    cur.execute(query_y)
    df_y = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    print(f"  Done: {len(df_y):,} rows ({time.time()-t0:.1f}s)")

    if df_y.empty:
        print("  No Y data. Exiting.")
    else:
        df_y.columns = df_y.columns.str.upper()
        # pivot: MEAS_ITEM → 컬럼
        if 'MEAS_ITEM' in df_y.columns and 'MEAS_DATA' in df_y.columns:
            idx_cols = [c for c in df_y.columns if c not in ['MEAS_ITEM', 'MEAS_DATA', 'MEAS_ID']]
            df_y = df_y.pivot_table(index=idx_cols, columns='MEAS_ITEM',
                                    values='MEAS_DATA', aggfunc='first').reset_index()
            df_y.columns = [c if isinstance(c, str) else c for c in df_y.columns]
        lot_ids = df_y['LOT_ID'].dropna().unique().tolist()
        print(f"  Unique LOT_IDs: {len(lot_ids)}")

        # ====================================================
        # [Part 2] 3200 → (BLK_NO, EQP_ID) 취득  (DBA 이미지3)
        # ====================================================
        print(f"\n[Part 2] 3200 → (BLK_NO, EQP_ID)...")
        t0 = time.time()
        lot_in = "', '".join(str(l) for l in lot_ids)
        q2 = f"""
            SELECT A.LOT_ID, A.BLK_NO, A.EQP_ID
            FROM ORACLE.OGGZMGR.ODB_DOPE_HIS A
            WHERE A.OPE_ID = '{TARGET_OPE_ID_X}'
              AND A.LOT_ID IN ('{lot_in}')
        """
        cur.execute(q2)
        df_3200 = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        df_3200.columns = df_3200.columns.str.upper()

        pairs = []
        if not df_3200.empty:
            pdf = df_3200[['BLK_NO', 'EQP_ID']].dropna()
            pdf = pdf[(pdf['BLK_NO'].astype(str).str.strip() != '') &
                      (pdf['EQP_ID'].astype(str).str.strip() != '')]
            pairs = list(pdf.drop_duplicates().itertuples(index=False, name=None))
        print(f"  3200: {len(df_3200)}행 → (BLK_NO,EQP_ID) {len(pairs)}쌍 ({time.time()-t0:.1f}s)")

        # ====================================================
        # [Part 3] FDC(EQP_TRACE) — BLK_NO+EQP_ID 직접 조회  (DBA 이미지4)
        # ====================================================
        print(f"\n[Part 3] FDC(EQP_TRACE) 조회...")
        t0 = time.time()
        df_fdc_long = pd.DataFrame()
        if pairs:
            case_expr = 'CASE ' + ' '.join(
                f"WHEN B.RAWDATA_MAPPING = '{p}' THEN '{PARAM_MAP[p]}'"
                for p in FDC_PARAMS) + ' ELSE NULL END AS PARA_NAME'
            param_in = "', '".join(FDC_PARAMS)

            eqp_to_blks = defaultdict(list)
            for blk, eqp in pairs:
                eqp_to_blks[str(eqp)].append(str(blk))

            frames = []
            for eqp, blks in eqp_to_blks.items():
                blk_in = "', '".join(blks)
                # ★확인: 컬럼명 A.RAWID/B.EQP_TRACE_RAWID/A.TRACE_DTTS/B.TRACE_VALUE
                #   TRACE_DTTS를 SELECT에 포함해 시계열 유지 (preprocess pct용)
                q3 = f"""
                    SELECT A.LOT_ID, A.EQP_ID, A.TRACE_DTTS,
                           {case_expr},
                           B.TRACE_VALUE
                    FROM ICEBERG.IBG_LAKE.EQP_TRACE_PH1_WS_RAWDATA_TRX_NEW A
                    INNER JOIN ICEBERG.IBG_LAKE.EQP_TRACE_PH1_WS_RAWDATA_VAL_NEW B
                        ON A.RAWID = B.EQP_TRACE_RAWID
                        AND A.TRACE_DTTS = B.TRACE_DTTS
                        AND B.RAWDATA_MAPPING IN ('{param_in}')
                    WHERE A.LOT_ID IN ('{blk_in}')
                      AND A.EQP_ID = '{eqp}'
                """
                try:
                    cur.execute(q3)
                    tmp = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
                    if not tmp.empty:
                        frames.append(tmp)
                        print(f"  EQP {eqp}: BLK {len(blks)}개 → {len(tmp)}행")
                except Exception as e:
                    print(f"  EQP {eqp}: 실패 — {e}")

            if frames:
                df_fdc_long = pd.concat(frames, ignore_index=True)
                df_fdc_long.columns = df_fdc_long.columns.str.upper()
        print(f"  FDC long: {len(df_fdc_long)}행 ({time.time()-t0:.1f}s)")

        # long → wide (시계열 유지: LOT_ID×EQP_ID×TRACE_DTTS 행, 인자 컬럼)
        df_fdc_wide = pd.DataFrame()
        if not df_fdc_long.empty:
            df_fdc_long['TRACE_VALUE'] = pd.to_numeric(df_fdc_long['TRACE_VALUE'], errors='coerce')
            df_fdc_wide = df_fdc_long.pivot_table(
                index=['LOT_ID', 'EQP_ID', 'TRACE_DTTS'],
                columns='PARA_NAME', values='TRACE_VALUE', aggfunc='mean'
            ).reset_index()
            df_fdc_wide.columns.name = None
            df_fdc_wide = df_fdc_wide.rename(columns={'LOT_ID': 'BLK_NO'})
            # 시계열 포인트 수 확인 (pct 프로파일 가능 여부)
            pts_per = df_fdc_wide.groupby('BLK_NO').size()
            print(f"  FDC wide: {len(df_fdc_wide)}행, BLK당 시계열 "
                  f"평균 {pts_per.mean():.1f}포인트")
            if pts_per.mean() < 2:
                print("  ⚠ 시계열이 BLK당 1포인트뿐 → preprocess가 pct 프로파일을 "
                      "못 만들 수 있음. DBA SQL에 TRACE_DTTS/시계열 확인 필요.")

        # ====================================================
        # [Part 5] 병합: Y + 3200(BLK_NO) + FDC(BLK_NO)
        # ====================================================
        print(f"\n[Part 5] Merge...")
        lot_to_blk = df_3200[['LOT_ID', 'BLK_NO', 'EQP_ID']].drop_duplicates('LOT_ID') \
                     if not df_3200.empty else pd.DataFrame(columns=['LOT_ID', 'BLK_NO', 'EQP_ID'])
        df_merged = pd.merge(df_y, lot_to_blk, on='LOT_ID', how='left')

        if not df_fdc_wide.empty:
            common = len(set(df_merged['BLK_NO'].dropna()) & set(df_fdc_wide['BLK_NO'].dropna()))
            print(f"  BLK_NO 매칭: {common}개 (기존 2개 → 늘어야 정상)")
            df_merged = pd.merge(df_merged, df_fdc_wide, on='BLK_NO', how='left')
        print(f"  merged: {len(df_merged)}행")

        # ── 진단: 장비 관련 컬럼에 뭐가 들었는지 (BSWS30 형태 찾기) ──
        print("  [진단] 장비 관련 컬럼 샘플:")
        eqp_cols = [c for c in df_merged.columns
                    if 'EQP' in c.upper() or c.upper() in ('EQP_ID', 'EQP_NAME', 'EQP_NM')]
        target_col = None
        for c in eqp_cols:
            vals = df_merged[c].dropna().astype(str).unique()[:5]
            has_bsws = any(match_target(v) for v in df_merged[c].dropna().astype(str))
            mark = ' ← BSWS 형태!' if has_bsws else ''
            print(f"    {c}: {list(vals)}{mark}")
            if has_bsws and target_col is None:
                target_col = c

        # ── 최종 장비 안전망: BSWS30 형태가 있는 컬럼으로 필터 ──
        if target_col:
            before = len(df_merged)
            df_merged = df_merged[df_merged[target_col].apply(match_target)].copy()
            print(f"  [최종 장비필터] '{target_col}' 기준 {before} → {len(df_merged)}행")
        else:
            print("  ⚠ BSWS30 형태 컬럼 없음 — 장비 필터 스킵 (전체 유지)")
            print("     → 위 진단에서 대상 장비가 어느 컬럼에 어떤 형태로 있는지 확인 필요")
        df_final = df_merged

    if cur: cur.close()
    if conn: conn.close()
    print("\nDB closed.")

    # ========================================================
    # Save + 커버리지 리포트
    # ========================================================
    print("\n" + "=" * 80)
    if df_final is not None and not df_final.empty:
        print(f"[WireSaw] Done! Rows: {len(df_final):,}, Cols: {len(df_final.columns)}")
        # BSWS 형태가 있는 컬럼 자동 탐색
        eqp_col = None
        for c in df_final.columns:
            if 'EQP' in c.upper():
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
            print(f"  [10대 커버리지] {len(got)}/10 ('{eqp_col}' 기준): {sorted(got)}")
            if missing:
                print(f"  ⚠ 데이터 없는 장비: {missing}")

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
