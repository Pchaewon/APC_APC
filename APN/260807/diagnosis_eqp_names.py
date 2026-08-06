# -*- coding: utf-8 -*-
"""
장비명(EQP) 형태 진단
─────────────────────────────────────────
목적: pilot_wiresaw.py에 장비 필터(IN)를 넣기 전에,
      각 테이블의 장비 식별자가 실제로 어떤 문자열인지 확인.
      (BSWS30 형태인지, 풀네임/코드인지 → 필터 위치·값 결정)

돌리는 법:
  python diagnose_eqp_names.py
"""
import os
import trino
import pandas as pd



TARGET_EQPS = ['BSWS30','BSWS31','BSWS35','BSWS42','BSWS48',
               'BSWS51','BSWS53','BSWS55','BSWS57','BSWS58']


cur = conn.cursor()
print("DB connected\n")

def run(label, q):
    print(f"=== {label} ===")
    try:
        cur.execute(q)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  실패: {e}")
    print()

# ── 1) FDC 테이블의 EQP_NM 실제 값 (BSWS30 형태일 가능성 가장 높음) ──
#     최근 데이터가 있는 아무 날짜나 하나 잡아서 DISTINCT
run("FDC EQP_NM 목록 (최근 BASE_DT 기준 상위 50)", """
    SELECT DISTINCT "EQP_NM"
    FROM iceberg.ibg_lake.DW_FA_EQ_FDCXWSNEW_S
    WHERE BASE_DT >= date_format(current_date - interval '30' day, '%Y%m%d')
    ORDER BY "EQP_NM"
    LIMIT 50
""")

# ── 2) 우리가 원하는 10대가 FDC에 그대로 있는지 직접 매칭 ──
in_list = ", ".join([f"'{e}'" for e in TARGET_EQPS])
run("FDC에서 우리 10대 직접 매칭 (있으면 그 이름·행수)", f"""
    SELECT "EQP_NM", COUNT(*) AS n_rows
    FROM iceberg.ibg_lake.DW_FA_EQ_FDCXWSNEW_S
    WHERE BASE_DT >= date_format(current_date - interval '30' day, '%Y%m%d')
      AND "EQP_NM" IN ({in_list})
    GROUP BY "EQP_NM"
    ORDER BY "EQP_NM"
""")

# ── 3) Part1 Y쿼리의 장비 마스터(ODB_DEQP) EQP_NAME 형태 ──
#     BSWS 로 시작하는 것만 (형태 확인용)
run("장비 마스터 ODB_DEQP.EQP_NAME (BSWS 포함)", """
    SELECT EQP_NAME, COUNT(*) AS n
    FROM ORACLE.OGGZMGR.ODB_DEQP
    WHERE UPPER(EQP_NAME) LIKE '%BSWS%'
    GROUP BY EQP_NAME
    ORDER BY EQP_NAME
    LIMIT 50
""")

cur.close(); conn.close()
print("완료. 위 세 결과를 붙여주세요.")
print("특히: (2)에서 10대가 그대로 잡히면 FDC 필터가 정답,")
print("      (3)에서 EQP_NAME이 BSWS30 형태와 다르면 Y쿼리 필터는 그 형태에 맞춰야 함.")
