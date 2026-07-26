## tttm_web의 초기 frame_temp 돌리는 코드 : 기간 길게
## 병성 프로님이 시키신 일


# ==========================================================
# WireSaw APC 전체 파이프라인 - 메모리 최적화 통합본
# (기존 스크립트 전면 대체용)
#
# 기존 대비 핵심 변경점
#  [1] FDC: 전 기간 concat 제거 → 일별 수집 + 7일 배치 전처리 + parquet append
#  [2] 자정 넘김 LOT carry-over 처리 (배치 경계에서 LOT 단절 방지)
#  [3] dtype 다운캐스트(float32/category)로 행당 메모리 50~70% 절감
#  [4] merge → map/transform 교체 (중간 복사본 제거)
#  [5] 다운스트림 merge는 'LOT당 1행 메타'로만 수행 (시계열 merge 폭발 제거)
#  [6] 모델 학습용 LOT별 X(csv)는 처리된 parquet에서 필요한 LOT만 추출
#  [7] Trino fetchall → fetchmany 스트리밍, 일별 resumable (중단 후 재실행 안전)
#  [8] DB 계정정보 평문 제거 → .env 환경변수 사용
# ==========================================================

# ==========================================================
# 0. Import
# ==========================================================
import os
import gc
import glob
import json
import warnings
import os.path as pt
from datetime import datetime, timedelta
from itertools import product as iterproduct

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trino
import xgboost as xgb

from dotenv import load_dotenv
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
load_dotenv()


# ==========================================================
# 1. 설정값 (Config)
# ==========================================================
class Config:
    # Trino 연결 - ⚠️ 계정정보는 .env 파일에 보관 (코드 평문 금지)
    #   .env 예시:
    #     TRINO_USER=257283
    #     TRINO_PASSWORD=********
    HOST = "aidp-trino-analysis.sksiltron.co.kr"
    PORT = 31085
    CATALOG = "iceberg"
    SCHEMA = "ibg_lake"
    USER = os.getenv("TRINO_USER", "")
    PASSWORD = os.getenv("TRINO_PASSWORD", "")

    # 날짜 범위
    START_DATE = "20240101"
    END_DATE = "20260726"

    # 출력 경로
    BASE_OUTPUT = r"D:\chaewon\APC\wire_saw\apc_code\add_web_data\result"
    X_DIR = pt.join(BASE_OUTPUT, "x_new")
    OUTPUT_FDC = pt.join(BASE_OUTPUT, "WireSaw_FDC")
    OUTPUT_3305 = pt.join(BASE_OUTPUT, "3305_flatness")
    OUTPUT_PIMS = pt.join(BASE_OUTPUT, "PIMS")
    OUTPUT_APC = pt.join(BASE_OUTPUT, "APC_model")
    OUTPUT_FRAME_BOW = pt.join(BASE_OUTPUT, "frame_to_bow")
    OUTPUT_APC_TEST = pt.join(BASE_OUTPUT, "Frame온도변경 Test")

    PARQUET_PATH = r'D:\chaewon\APC\wire_saw\apc_code\data\FDC_3200_data'

    # 메모리 최적화용 신규 경로
    FDC_CHUNK_DIR = pt.join(BASE_OUTPUT, "fdc_chunks_raw")       # 일별 원본 청크
    FDC_PROC_DIR = pt.join(BASE_OUTPUT, "fdc_processed")         # 전처리 완료 파티션
    CHUNK_3305_DIR = pt.join(BASE_OUTPUT, "chunks_3305")         # 3305 일별 청크

    # 배치 크기 (메모리 부족 시 3으로 축소, 여유 시 14로 확대)
    FDC_BATCH_DAYS = 7
    FETCH_BATCH_ROWS = 200_000

    # 모델 파라미터
    TARGET_RECIPE_TIME = ["133", "180"]
    N_SEGMENTS = 8
    SEGMENT_BOUNDS = np.array([0, 10, 20, 30, 50, 70, 90, 100])
    INTERP_LEN = 100
    TARGET_MAX_DEVIATION = 0.25
    WARP_TOLERANCE = 0.1
    GB_WARN_THRESHOLD = 1.0

    # Clustering 파라미터
    CLUSTERING_PARAMS = r"D:\chaewon\APC\wire_saw\apc_code\apc_model\final_clustering_model_with_merge.pkl"

    # XGBoost 파라미터 (BOW/WARP 공통)
    XGB_PARAMS = dict(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42
    )

    FRAME_TEMP_PARQUET = pt.join(
        r"D:\chaewon\APC\wire_saw\apc_code\data\datasets",
        "wiresaw", "wiresaw_frame_temp.parquet")


CFG = Config()
for d in [CFG.OUTPUT_FDC, CFG.OUTPUT_3305, CFG.OUTPUT_PIMS,
          CFG.OUTPUT_FRAME_BOW, CFG.OUTPUT_APC_TEST,
          CFG.FDC_CHUNK_DIR, CFG.FDC_PROC_DIR, CFG.CHUNK_3305_DIR]:
    os.makedirs(d, exist_ok=True)


# ==========================================================
# 2. DB 연결 헬퍼
# ==========================================================
def get_trino_conn() -> trino.dbapi.Connection:
    """환경변수 기반 Trino 연결 반환"""
    if not CFG.USER or not CFG.PASSWORD:
        raise RuntimeError(".env에 TRINO_USER / TRINO_PASSWORD를 설정하세요.")
    return trino.dbapi.connect(
        host=CFG.HOST,
        port=CFG.PORT,
        user=CFG.USER,
        http_scheme="https",
        auth=trino.auth.BasicAuthentication(CFG.USER, CFG.PASSWORD),
        catalog=CFG.CATALOG,
        schema=CFG.SCHEMA,
        verify=False,
    )


def save_csv(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ==========================================================
# 3. dtype 최적화 (메모리 절감의 기본기)
# ==========================================================
FDC_NEEDED_COLS = [
    "BASE_DT", "FAB_ID", "EQP_NM", "LOT_ID", "SUBLOT_ID",
    "RECIPE_ID", "PROD_ID", "HST_REG_DTTM", "RUNTIME_MINUTES",
    "INGOT_LEN", "BREAKING_WIRE_FLAG", "FRAME_IN_TEMP",
    "TABLE_SPEED", "NEW_WIRE_ID", "ELONGATION",
]
FLOAT32_COLS = ["RUNTIME_MINUTES", "INGOT_LEN", "FRAME_IN_TEMP",
                "TABLE_SPEED", "ELONGATION",
                "WARP_BF", "BOW_BF", "AVE_THK", "TTV", "TAPER",
                "CUT_POSITION", "BLK_LEN"]
CATEGORY_COLS = ["FAB_ID", "EQP_NM", "RECIPE_ID", "PROD_ID",
                 "BREAKING_WIRE_FLAG", "BASE_DT",
                 "EQP_NM_3200", "CREATE_CODE", "MEAS_EQP", "MS_CODE"]
HIGH_CARD_CATEGORY_COLS = ["LOT_ID", "SUBLOT_ID", "NEW_WIRE_ID"]


def optimize_dtypes(df: pd.DataFrame, use_high_card_category: bool = True) -> pd.DataFrame:
    """float64→float32, 반복 문자열→category. 통상 50~70% 메모리 절감."""
    for c in FLOAT32_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in CATEGORY_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if use_high_card_category:
        for c in HIGH_CARD_CATEGORY_COLS:
            if c in df.columns:
                df[c] = df[c].astype("category")
    return df


# ==========================================================
# 4. 일별 스트리밍 수집기 (공통, resumable)
# ==========================================================
def fetch_daily_chunks(conn, query_template, start_date: str, end_date: str,
                       chunk_dir: str,
                       transform_fn=None,
                       keep_cols: list = None,
                       data_root: str = None,
                       parquet_suffix: str = "_3200_WIRE_SAW.parquet",
                       overwrite: bool = False) -> list:
    """
    일별 Trino 조회(+로컬 parquet fallback) → (선택)변환 → dtype 최적화
    → 일별 parquet 저장 후 즉시 메모리 해제.
    - fetchmany 스트리밍: DB 응답도 배치로 수신
    - 이미 저장된 날짜는 스킵 → 중단 후 재실행해도 이어서 진행 (resumable)
    - 반환: 저장된 청크 파일 경로 리스트
    """
    os.makedirs(chunk_dir, exist_ok=True)
    current_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    saved_files = []

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        chunk_path = pt.join(chunk_dir, f"chunk_{date_str}.parquet")

        if pt.exists(chunk_path) and not overwrite:
            saved_files.append(chunk_path)
            current_dt += timedelta(days=1)
            continue

        df_daily = None

        # ── 1) Trino 스트리밍 조회
        cur = conn.cursor()
        try:
            cur.execute(query_template(date_str))
            cols = [d[0] for d in cur.description]
            parts = []
            while True:
                rows = cur.fetchmany(CFG.FETCH_BATCH_ROWS)
                if not rows:
                    break
                parts.append(pd.DataFrame(rows, columns=cols))
            if parts:
                df_daily = pd.concat(parts, ignore_index=True)
                del parts
                print(f"  [OK] [{date_str}] Trino {len(df_daily):,}건")
        except Exception as e:
            print(f"  [ERR] [{date_str}] Trino 오류: {e}")
        finally:
            cur.close()

        # ── 2) 로컬 parquet fallback (FDC 전용)
        if (df_daily is None or df_daily.empty) and data_root:
            folder = f"{str(current_dt.year)[-2:]}_{current_dt.month:02d}"
            fpath = pt.join(data_root, folder,
                            f"{folder}_{current_dt.day:02d}{parquet_suffix}")
            if pt.exists(fpath):
                try:
                    df_daily = pd.read_parquet(fpath)
                    print(f"  [LOCAL] [{date_str}] parquet {len(df_daily):,}건 복구")
                except Exception as e:
                    print(f"  [ERR] [{date_str}] parquet 읽기 오류: {e}")

        # ── 3) 정리 후 저장 → 즉시 메모리 해제
        if df_daily is not None and not df_daily.empty:
            df_daily.columns = df_daily.columns.str.upper()
            if keep_cols:
                keep = [c for c in keep_cols if c in df_daily.columns]
                df_daily = df_daily[keep]
            if transform_fn is not None:
                df_daily = transform_fn(df_daily)
            if df_daily is not None and not df_daily.empty:
                df_daily = df_daily.drop_duplicates(ignore_index=True)
                df_daily = optimize_dtypes(df_daily)
                df_daily.to_parquet(chunk_path, index=False,
                                    engine="pyarrow", compression="zstd")
                saved_files.append(chunk_path)
                print(f"  [SAVE] [{date_str}] {len(df_daily):,}행 저장")
            del df_daily
            gc.collect()
        else:
            print(f"  [SKIP] [{date_str}] 데이터 없음")

        current_dt += timedelta(days=1)

    return saved_files


def load_chunks(chunk_dir: str, columns: list = None) -> pd.DataFrame:
    """청크 디렉토리를 컬럼 프루닝하여 로드 (소규모 데이터 전용: 3305/PIMS 등)"""
    files = sorted(glob.glob(pt.join(chunk_dir, "chunk_*.parquet")))
    if not files:
        return pd.DataFrame()
    parts = []
    for i, f in enumerate(files):
        parts.append(pd.read_parquet(f, columns=columns))
        if (i + 1) % 50 == 0:
            gc.collect()
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return df.drop_duplicates(ignore_index=True)


# ==========================================================
# 5. FDC 전처리 (LOT 유효구간, merge 제거 버전)
# ==========================================================
def _roll10_sum(s: pd.Series) -> pd.Series:
    return s.rolling(window=10, min_periods=1).sum()


def get_valid_runtime_range_lean(df: pd.DataFrame) -> pd.DataFrame:
    """기존 get_valid_runtime_range와 동일 로직. merge → map/transform 교체."""
    df["TABLE_SPEED"] = pd.to_numeric(df["TABLE_SPEED"], errors="coerce")
    df = df[df["TABLE_SPEED"].notna()]
    if df.empty:
        return df

    # category LOT_ID는 groupby 폭발 방지를 위해 str로 환원
    if df["LOT_ID"].dtype.name == "category":
        df = df.assign(LOT_ID=df["LOT_ID"].astype(str))

    # ── START 탐지 (오름차순)
    df = df.sort_values(["LOT_ID", "RUNTIME_MINUTES"], ignore_index=True)
    cs = df.groupby("LOT_ID", sort=False)["TABLE_SPEED"].transform(_roll10_sum)
    cs_next = cs.groupby(df["LOT_ID"], sort=False).shift(-1)
    start_time = (df.loc[cs != cs_next]
                    .groupby("LOT_ID")["RUNTIME_MINUTES"].min())
    del cs, cs_next

    # ── END 탐지 (내림차순 rolling)
    df_desc = df.sort_values(["LOT_ID", "RUNTIME_MINUTES"],
                             ascending=[True, False], ignore_index=True)
    cs_d = df_desc.groupby("LOT_ID", sort=False)["TABLE_SPEED"].transform(_roll10_sum)
    cs_d_prev = cs_d.groupby(df_desc["LOT_ID"], sort=False).shift(1).fillna(0)
    end_time = (df_desc.loc[cs_d != cs_d_prev]
                      .groupby("LOT_ID")["RUNTIME_MINUTES"].max())
    del df_desc, cs_d, cs_d_prev
    gc.collect()

    # ── 구간 필터링 (merge 대신 map: 중간 복사본 없음)
    st = df["LOT_ID"].map(start_time)
    en = df["LOT_ID"].map(end_time)
    df = df[(df["RUNTIME_MINUTES"] > st) & (df["RUNTIME_MINUTES"] <= en)].copy()
    del st, en
    if df.empty:
        return df

    # ── RUNTIME 0-기준 재산출
    df["RUNTIME_MINUTES"] = (
        df["RUNTIME_MINUTES"]
        - df.groupby("LOT_ID", sort=False)["RUNTIME_MINUTES"].transform("min")
    ).astype("float32")

    return (df.sort_values(["EQP_NM", "BASE_DT", "RUNTIME_MINUTES"])
              .reset_index(drop=True))


# ==========================================================
# 6. FDC 배치 전처리 + carry-over (자정 넘김 LOT 이월)
# ==========================================================
def process_fdc_chunks_in_batches(chunk_dir: str, out_dir: str,
                                  batch_days: int = None,
                                  max_carry_days: int = 2) -> list:
    """
    일별 청크를 batch_days개씩 묶어 처리 후 즉시 parquet append.
    배치 마지막 날짜에 존재하는 LOT은 자정을 넘겨 진행 중일 수 있으므로
    해당 LOT의 전체 행을 다음 배치로 이월(carry-over) →
    START/END 탐지가 LOT 단절 없이 정확하게 수행됨.
    (공정 시간 133~180분이므로 하루 이월이면 충분, 설비 정지 대비 상한 설정)
    """
    batch_days = batch_days or CFG.FDC_BATCH_DAYS
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(pt.join(chunk_dir, "chunk_*.parquet")))
    if not files:
        print("처리할 청크가 없습니다.")
        return []

    batches = [files[i:i + batch_days] for i in range(0, len(files), batch_days)]
    carry = None
    out_files = []

    for bi, batch in enumerate(batches):
        is_last = (bi == len(batches) - 1)
        tag = pt.splitext(pt.basename(batch[0]))[0].replace("chunk_", "")
        out_path = pt.join(out_dir, f"fdc_processed_{tag}.parquet")

        # resumable: 이미 처리된 배치는 스킵 (단, carry 연속성을 위해
        #            중간부터 재개하려면 carry가 없는 상태여야 안전)
        if pt.exists(out_path) and carry is None:
            out_files.append(out_path)
            print(f"[Batch {bi+1}/{len(batches)}] 기존 결과 재사용: {out_path}")
            continue

        dfs = ([carry] if carry is not None else []) + \
              [pd.read_parquet(f) for f in batch]
        df = pd.concat(dfs, ignore_index=True)
        del dfs
        carry = None
        gc.collect()

        df = optimize_dtypes(df, use_high_card_category=False)
        df["BASE_DT"] = df["BASE_DT"].astype(str)

        # ── carry-over 분리
        if not is_last:
            last_day = df["BASE_DT"].max()
            carry_lots = df.loc[df["BASE_DT"] == last_day, "LOT_ID"].unique()
            mask_carry = df["LOT_ID"].isin(carry_lots)
            cutoff = (datetime.strptime(last_day, "%Y%m%d")
                      - timedelta(days=max_carry_days)).strftime("%Y%m%d")
            mask_carry &= (df["BASE_DT"] >= cutoff)
            carry = df[mask_carry].copy()
            df = df[~mask_carry]
            print(f"[Batch {bi+1}/{len(batches)}] 이월 LOT {len(carry_lots)}개 "
                  f"({len(carry):,}행)")

        if df.empty:
            continue

        # ── 배치 내 중복 제거
        #    (carry-over 구조상 같은 LOT은 항상 같은 배치에서만 처리되므로
        #     전 기간 글로벌 dedup 없이도 동등한 결과 보장)
        df = df.drop_duplicates(ignore_index=True)

        # ── LOT 유효구간 전처리
        df = get_valid_runtime_range_lean(df)
        if df.empty:
            continue

        # ── 즉시 디스크 append
        df = optimize_dtypes(df)
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
        out_files.append(out_path)
        print(f"[Batch {bi+1}/{len(batches)}] {len(df):,}행 → {out_path}")

        del df
        gc.collect()

    return out_files


# ==========================================================
# 7. FDC 결과 lazy 로더 + LOT 메타 + LOT별 시계열 추출
# ==========================================================
def load_fdc_processed(out_dir: str, columns: list = None,
                       eqp: str = None, start: str = None,
                       end: str = None) -> pd.DataFrame:
    """pyarrow dataset: 컬럼 프루닝 + 필터 푸시다운 (전체 로드 금지)"""
    import pyarrow.dataset as ds
    dataset = ds.dataset(out_dir, format="parquet")
    flt = None
    if eqp:
        flt = ds.field("EQP_NM") == eqp
    if start:
        f2 = ds.field("BASE_DT") >= start
        flt = f2 if flt is None else (flt & f2)
    if end:
        f3 = ds.field("BASE_DT") <= end
        flt = f3 if flt is None else (flt & f3)
    return dataset.to_table(columns=columns, filter=flt).to_pandas()


def make_fdc_lot_meta(out_dir: str) -> pd.DataFrame:
    """
    다운스트림 merge용 LOT 단위 메타 테이블 (LOT당 1행, 수 MB).
    ⚠️ 기존 코드는 FDC '시계열 전체'를 wafer pivot과 merge하여
       행이 시점 수만큼 폭발 → 메모리 오류의 2차 원인이었음.
    """
    import pyarrow.dataset as ds
    dataset = ds.dataset(out_dir, format="parquet")
    meta_cols = ["BASE_DT", "FAB_ID", "EQP_NM", "LOT_ID", "SUBLOT_ID",
                 "RECIPE_ID", "PROD_ID", "INGOT_LEN", "BREAKING_WIRE_FLAG",
                 "NEW_WIRE_ID", "ELONGATION"]
    parts = []
    for frag in dataset.get_fragments():
        t = frag.to_table(columns=meta_cols).to_pandas()
        for c in t.columns:
            if t[c].dtype.name == "category":
                t[c] = t[c].astype(str)
        parts.append(t.drop_duplicates(subset=["LOT_ID", "SUBLOT_ID", "EQP_NM"]))
        del t
    meta = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return meta.drop_duplicates(subset=["LOT_ID", "SUBLOT_ID", "EQP_NM"],
                                ignore_index=True)


def export_lot_timeseries(proc_dir: str, lot_ids, x_dir: str,
                          overwrite: bool = False) -> int:
    """
    처리된 FDC parquet에서 지정 LOT들의 FRAME_IN_TEMP 시계열만
    LOT별 CSV로 추출 (process_x 입력용).
    파티션 단위 순회 → 메모리 상주는 파티션 1개 분량으로 상한.
    carry-over 구조 덕분에 한 LOT은 반드시 한 파티션에만 존재.
    """
    import pyarrow.dataset as ds
    os.makedirs(x_dir, exist_ok=True)
    lot_set = set(map(str, lot_ids))
    if not overwrite:
        done = {pt.splitext(f)[0] for f in os.listdir(x_dir) if f.endswith(".csv")}
        lot_set -= done
    if not lot_set:
        print("  추출할 신규 LOT 없음")
        return 0

    dataset = ds.dataset(proc_dir, format="parquet")
    n_written = 0
    cols = ["LOT_ID", "RUNTIME_MINUTES", "FRAME_IN_TEMP"]
    for frag in dataset.get_fragments():
        t = frag.to_table(columns=cols).to_pandas()
        if t["LOT_ID"].dtype.name == "category":
            t["LOT_ID"] = t["LOT_ID"].astype(str)
        t = t[t["LOT_ID"].isin(lot_set)]
        if t.empty:
            del t
            continue
        for lot, g in t.groupby("LOT_ID", sort=False):
            g.sort_values("RUNTIME_MINUTES").to_csv(
                pt.join(x_dir, f"{lot}.csv"), index=False, encoding="utf-8")
            n_written += 1
        del t
        gc.collect()
    print(f"  LOT 시계열 CSV {n_written}개 추출 완료 → {x_dir}")
    return n_written


# ==========================================================
# 8. FDC 파이프라인 진입점 (기존 run_fdc_pipeline 대체)
# ==========================================================
def run_fdc_pipeline_streaming(conn):
    def _q(date_str):
        return f"""
            SELECT BASE_DT, FAB_ID, EQP_NM, LOT_ID, SUBLOT_ID, RECIPE_ID,
                   PROD_ID, HST_REG_DTTM, RUNTIME_MINUTES, INGOT_LEN,
                   BREAKING_WIRE_FLAG, FRAME_IN_TEMP, TABLE_SPEED,
                   NEW_WIRE_ID, ELONGATION
            FROM iceberg.ibg_lake.DW_FA_EQ_FDCXWSNEW_S a
            WHERE a.BASE_DT = '{date_str}'
        """

    print("=== [FDC] Phase 1: 일별 수집 (resumable) ===")
    fetch_daily_chunks(conn, _q, CFG.START_DATE, CFG.END_DATE,
                       chunk_dir=CFG.FDC_CHUNK_DIR,
                       keep_cols=FDC_NEEDED_COLS,
                       data_root=CFG.PARQUET_PATH)

    print("=== [FDC] Phase 2: 배치 전처리 + parquet append ===")
    process_fdc_chunks_in_batches(CFG.FDC_CHUNK_DIR, CFG.FDC_PROC_DIR)

    print("=== [FDC] Phase 3: LOT 메타 생성 ===")
    meta = make_fdc_lot_meta(CFG.FDC_PROC_DIR)
    save_csv(meta, CFG.OUTPUT_FDC,
             f"WireSaw_FDC_LOT_META_{CFG.START_DATE}_to_{CFG.END_DATE}.csv")
    print(f"  LOT 메타 {len(meta):,}행 저장")
    return meta


# ==========================================================
# 9. Flatness (3305) 전처리 (wafer 단위 → 상대적으로 소규모)
# ==========================================================
def add_blk_pos(tp: pd.DataFrame) -> pd.DataFrame:
    """CUT_POSITION 기반 BLK_POS / POS_TYPE 컬럼 추가 (merge → transform)"""
    for col in ["CUT_POSITION", "BLK_LEN"]:
        if col in tp.columns:
            tp[col] = pd.to_numeric(tp[col], errors="coerce").astype("float32")

    g = tp.groupby("USER_LOT_ID", sort=False)["CUT_POSITION"]
    tp["CUT_POSITION_MIN"] = g.transform("min")
    tp["CUT_POSITION_MAX"] = g.transform("max")
    tp["CUT_POSITION_MAX_CAL"] = tp["CUT_POSITION_MIN"] + 10 * tp["BLK_LEN"]
    tp["CUT_POSITION_MAX_APP"] = tp[["CUT_POSITION_MAX", "CUT_POSITION_MAX_CAL"]].max(axis=1)
    tp["BLK_POS"] = (tp["CUT_POSITION"] - tp["CUT_POSITION_MIN"]) / (
        tp["CUT_POSITION_MAX_APP"] - tp["CUT_POSITION_MIN"]
    )
    tp["BLK_POS"] = tp["BLK_POS"].fillna(0.5)
    tp["POS_TYPE"] = pd.cut(
        tp["BLK_POS"],
        bins=[-np.inf, 0.2, 0.8, np.inf],
        labels=["SEED", "MID", "TAIL"],
    )
    return tp


def make_flatness_pivot(tp: pd.DataFrame) -> pd.DataFrame:
    """WARP/BOW 전체 평균 + POS_TYPE별 피벗 병합"""
    grp_keys = ["MS_CODE", "DATE_3200", "USER_LOT_ID", "SUBLOT_ID_3200",
                "BLK_LEN", "EQP_NM_3200"]

    df_total = (
        tp.groupby(grp_keys, as_index=False, observed=True)[["WARP_BF", "BOW_BF"]]
        .mean()
        .rename(columns={"WARP_BF": "WARP_BF_MEAN_TOTAL",
                         "BOW_BF": "BOW_BF_MEAN_TOTAL"})
    )

    df_by_pos = tp.groupby(grp_keys + ["POS_TYPE"], as_index=False,
                           observed=True)[["WARP_BF", "BOW_BF"]].mean()
    df_pivot = df_by_pos.pivot_table(
        index=grp_keys, columns="POS_TYPE", values=["WARP_BF", "BOW_BF"],
        aggfunc="mean", fill_value=None, observed=True
    )
    df_pivot.columns = [f"{v}_{p}" for v, p in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    return (pd.merge(df_total, df_pivot, on=grp_keys, how="outer")
              .sort_values(grp_keys).reset_index(drop=True))


def run_3305_flatness_pipeline(conn, start=None, end=None, eqp_filter=None):
    """3305 Flatness 수집(일별 청크) → BLK_POS → Pivot → CSV 저장"""
    start = start or CFG.START_DATE
    end = end or CFG.END_DATE
    where_extra = f"AND a.EQP_NM_3200 = '{eqp_filter}'" if eqp_filter else ""

    def _q(date_str):
        return f"""
            SELECT "MS_CODE", "MEAS_OPER_ID", "MEAS_TIME", "MEAS_DATE_TIME",
                   "MEAS_EQP", "MEAS_MODEL", "USER_LOT_ID", "CREATE_CODE",
                   "SUBLOT_ID_3200", "WAF_SEQ", "BLK_LEN", "SLOT_NO",
                   "CUT_POSITION", "EQP_NM_3200", "EQP_MODEL_3200",
                   "HST_REG_DTTM_3200", "DATE_3200",
                   "WARP_BF", "BOW_BF", "AVE_THK", "TTV", TAPER
            FROM iceberg.ibg_lake.DM_QM_PW_LSPWXY3305_S a
            WHERE a.meas_date = '{date_str}' {where_extra}
        """

    # 일별 수집 시점에 FS 필터를 먼저 적용 → 저장/메모리 모두 절감
    def _filter_fs(df):
        if "CREATE_CODE" in df.columns:
            df = df[df["CREATE_CODE"] == "FS"]
        return df

    fetch_daily_chunks(conn, _q, start, end,
                       chunk_dir=CFG.CHUNK_3305_DIR,
                       transform_fn=_filter_fs)

    tp = load_chunks(CFG.CHUNK_3305_DIR)
    if tp.empty:
        return tp, pd.DataFrame()

    for c in tp.columns:
        if tp[c].dtype.name == "category":
            tp[c] = tp[c].astype(str)

    tp["key"] = tp["USER_LOT_ID"].str.slice(0, 5) + "_" + \
        tp["WAF_SEQ"].astype(float).astype(int).astype(str)
    tp = add_blk_pos(tp)
    tp = tp.sort_values(["SUBLOT_ID_3200", "BLK_POS"]).reset_index(drop=True)

    out_dir = CFG.OUTPUT_APC_TEST if eqp_filter else CFG.OUTPUT_3305
    fname = "3305_Flatness.csv" if eqp_filter else f"3305_Flatness_{start}_to_{end}.csv"
    save_csv(tp, out_dir, fname)

    df_pivot = make_flatness_pivot(tp)
    save_csv(df_pivot, out_dir, f"pivot_{start}_to_{end}.csv")
    return tp, df_pivot


# ==========================================================
# 10. PIMS 전처리 (소규모 - 1회 조회)
# ==========================================================
def fill_target(row: pd.Series):
    target = row["TARGET"]
    if pd.notna(target):
        try:
            return float(target)
        except (ValueError, TypeError):
            pass
    min_v = pd.to_numeric(row["MIN"], errors="coerce")
    max_v = pd.to_numeric(row["MAX"], errors="coerce")
    if pd.notna(min_v) and pd.notna(max_v):
        return (min_v + max_v) / 2
    return target


def run_pims_pipeline(conn):
    """PIMS SPEC 수집 → pivot → CSV 저장 (날짜 무관 스펙 테이블 - 1회 조회로 충분)"""
    today_str = datetime.now().strftime("%Y%m%d")
    query = """
        SELECT "ms_code","spec_type","parameter","target","min","max"
        FROM iceberg.ibg_lake.PIMS_SPEC
        WHERE parameter IN ('CRYSTAL_ORIENTATION','NOTCH_ORIENTATION','GR_RESISTIVITY')
          AND spec_type = 'PS'
    """
    cur = conn.cursor()
    try:
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    finally:
        cur.close()
    if not rows:
        return pd.DataFrame()

    tp = pd.DataFrame(rows, columns=cols)
    tp.columns = tp.columns.str.upper()
    tp = tp.drop_duplicates(ignore_index=True)
    tp["TARGET"] = tp.apply(fill_target, axis=1)
    save_csv(tp, CFG.OUTPUT_PIMS, f"PIMS_RAW_{today_str}.csv")
    pivot = tp.pivot_table(index="MS_CODE", columns="PARAMETER", values="TARGET",
                           aggfunc="first").round(4).reset_index()
    save_csv(pivot, CFG.OUTPUT_PIMS, f"PIMS_{today_str}.csv")
    return pivot


# ==========================================================
# 11. 데이터 병합 (LOT 메타 기반 - 시계열 merge 폭발 제거)
# ==========================================================
def convert_data_cols_naming(df_pims: pd.DataFrame,
                             pivot_3305: pd.DataFrame,
                             df_fdc_meta: pd.DataFrame,
                             output_dir):
    """
    기존 함수와 동일한 산출물(LOT×wafer집계 1행)을 만들되,
    FDC는 '시계열 전체' 대신 'LOT당 1행 메타'로 merge.
    → 기존: (FDC 시점 수 × pivot) 폭발 / 개선: (LOT 수 × pivot) 수준
    """
    # PIMS
    df_pims = df_pims.rename(columns={'MS_CODE': 'PROD_ID'})
    columns = ["PROD_ID", "CRYSTAL_ORIENTATION",
               "NOTCH_ORIENTATION", "GR_RESISTIVITY"]
    df_pims_sort = df_pims[columns].drop_duplicates().reset_index(drop=True)

    # 3305 Pivot
    pivot_3305 = pivot_3305.rename(columns={'MS_CODE': 'PROD_ID'})

    # FDC 메타 (LOT당 1행)
    df_fdc_meta = df_fdc_meta.rename(columns={
        'EQP_NM': 'EQP_NM_3200',
        'LOT_ID': 'USER_LOT_ID',
        'SUBLOT_ID': 'SUBLOT_ID_3200'})

    df_Y_pims = pd.merge(pivot_3305, df_pims_sort, on=['PROD_ID'], how='inner')
    df_Y_pims_FDC = pd.merge(
        df_Y_pims, df_fdc_meta,
        on=['USER_LOT_ID', 'SUBLOT_ID_3200', 'EQP_NM_3200'],
        how='inner')

    # INGOT_LEN 결측 → BLK_LEN 대체
    df_Y_pims_FDC['INGOT_LEN'] = np.where(
        df_Y_pims_FDC['INGOT_LEN'].isna(),
        df_Y_pims_FDC['BLK_LEN'],
        df_Y_pims_FDC['INGOT_LEN'])

    df_cleaned = df_Y_pims_FDC.dropna(subset=['INGOT_LEN', 'ELONGATION'], how='any')
    df_cleaned['CRYSTAL_ORIENTATION'] = (
        df_cleaned['CRYSTAL_ORIENTATION'].astype(str).str.strip())
    df_cleaned = df_cleaned[df_cleaned['CRYSTAL_ORIENTATION'] == '100.0']

    ts = datetime.now().strftime("%Y%m%d")
    save_csv(df_cleaned, output_dir, f'df_Y_pims_FDC_{ts}.csv')
    return df_cleaned


# ==========================================================
# 12. 클러스터링 (군집 분류)
# ==========================================================
def predict_group(df_input: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """학습된 클러스터링 모델로 Group 라벨 반환"""
    loaded = joblib.load(model_path)
    scaler = loaded["scaler"]
    weights = loaded["weights_l2"]
    centroids = loaded["centroids_final"]
    feature_cols = loaded["feature_cols"]

    df_input['INGOT_LEN'] = np.where(
        df_input['INGOT_LEN'].isna(),
        df_input['BLK_LEN'],
        df_input['INGOT_LEN'])

    if not all(col in df_input.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in df_input.columns]
        raise KeyError(f"다음 컬럼이 새로운 데이터에 없습니다 : {missing}")

    X_new = df_input[feature_cols].copy().dropna()
    print(f"전처리 후 유효 데이터 수: {len(X_new)} / {len(df_input)}")

    X_scaled = scaler.transform(X_new)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X_new.index)

    X_weighted = X_scaled.copy()
    for col, w in weights.items():
        X_weighted[col] *= w

    X_weighted.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_clean = X_weighted.dropna()
    valid_new_indices = X_clean.index
    print(f"가중치 적용 후 유효 데이터: {len(X_clean)}개")

    distances = cdist(X_clean.values, centroids.values, metric='euclidean')
    predicted_labels = np.argmin(distances, axis=1) + 1
    group_names = np.char.add('0', predicted_labels.astype(str))

    df_result = df_input.copy()
    df_result['Group'] = pd.NA
    df_result.loc[valid_new_indices, 'Group'] = group_names
    print(f"할당된 그룹 수: {df_result['Group'].nunique(dropna=True)}")
    return df_result


def predict_single_lot_group(df_cleaned: pd.DataFrame, x_dir: str,
                             output_path: str, model_path: str,
                             fdc_proc_dir: str) -> pd.DataFrame:
    """
    클러스터링 예측 + 모델 학습용 LOT별 시계열 CSV 추출.
    ⚠️ 기존: merge된 거대 프레임을 LOT별로 쪼개 저장 (메모리 부담)
       개선: 처리된 FDC parquet에서 필요한 LOT 시계열만 스트리밍 추출
    """
    df_result = predict_group(df_cleaned, model_path)

    # 모델 학습(process_x)에 필요한 LOT 시계열 CSV 추출
    lots = df_result['USER_LOT_ID'].dropna().unique()
    export_lot_timeseries(fdc_proc_dir, lots, x_dir)

    # 컬럼 정리 (존재하는 컬럼만 선택 → suffix 차이에 안전)
    desired = [
        'PROD_ID_x', 'PROD_ID_X', 'DATE_3200', 'USER_LOT_ID', 'SUBLOT_ID_3200',
        'BLK_LEN', 'EQP_NM_3200', 'WARP_BF_MEAN_TOTAL', 'BOW_BF_MEAN_TOTAL',
        'BOW_BF_MID', 'BOW_BF_SEED', 'BOW_BF_TAIL', 'WARP_BF_MID',
        'WARP_BF_SEED', 'WARP_BF_TAIL', 'CRYSTAL_ORIENTATION',
        'NOTCH_ORIENTATION', 'GR_RESISTIVITY', 'BASE_DT', 'FAB_ID',
        'RECIPE_ID', 'PROD_ID_y', 'PROD_ID_Y', 'INGOT_LEN',
        'BREAKING_WIRE_FLAG', 'NEW_WIRE_ID', 'ELONGATION', 'Group'
    ]
    keep = [c for c in desired if c in df_result.columns]
    df_clean = df_result[keep].drop_duplicates().reset_index(drop=True)
    df_clean['RECIPE_time'] = df_clean['RECIPE_ID'].str.extract(r'(\d\d\d)').astype(float)

    ts = datetime.now().strftime("%Y%m%d")
    save_csv(df_clean, output_path, f'df_wiresaw_meta_{ts}.csv')
    return df_clean


# ==========================================================
# 13. 신호 처리 (FRAME_IN_TEMP 보간·스무딩)
# ==========================================================
def interp_x(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 1:
        return np.repeat(x[0], CFG.INTERP_LEN)
    orig = np.linspace(0, 100, n)
    tgt = np.linspace(0, 100, CFG.INTERP_LEN)
    return interp1d(orig, x, kind="linear", fill_value="extrapolate")(tgt)


def process_x(x_path: str) -> np.ndarray:
    """LOT별 CSV에서 FRAME_IN_TEMP 추출 → 보간 → 가우시안 스무딩
    (export_lot_timeseries가 만든 표준 CSV만 읽으므로 xlwings 의존 제거)"""
    df_x = None
    for enc in ["utf-8", "cp949"]:
        try:
            df_x = pd.read_csv(x_path, encoding=enc,
                               usecols=lambda c: c.upper() in
                               ("RUNTIME_MINUTES", "FRAME_IN_TEMP"))
            break
        except Exception:
            pass
    if df_x is None:
        return None

    df_x.columns = df_x.columns.str.upper()
    df_x = df_x.sort_values("RUNTIME_MINUTES").reset_index(drop=True)
    x = pd.to_numeric(df_x["FRAME_IN_TEMP"], errors="coerce").values
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None
    return gaussian_filter1d(interp_x(x), sigma=1.5)


def extract_boundary_temps(signal: np.ndarray) -> np.ndarray:
    idx = np.linspace(0, 100, len(signal))
    return interp1d(idx, signal, kind="linear",
                    fill_value="extrapolate")(CFG.SEGMENT_BOUNDS)


# ==========================================================
# 14. 역설계 최적화
# ==========================================================
def calculate_smoothness_weight(f_temp: np.ndarray, y_target: np.ndarray) -> float:
    changes = np.array([np.sum(np.diff(x) ** 2) for x in f_temp])
    if np.std(changes) == 0:
        return 0.1
    corr, _ = pearsonr(changes, y_target)
    return round(0.05 + 0.45 * max(0, corr), 3)


def calculate_proximity_weight(y_target: np.ndarray, base_mae: float = 0.3) -> float:
    return round(float(np.clip(base_mae / max(np.std(y_target), 1e-8), 0.1, 2.0)), 3)


def estimate_minimum_deviation(model, scaler, n_samples: int = 1000,
                               temp_range=(20, 30)):
    np.random.seed(42)
    X_r = np.random.uniform(*temp_range, (n_samples, CFG.N_SEGMENTS))
    preds = model.predict(scaler.transform(X_r))
    return preds.min(), preds.mean()


def generate_initial_profiles(f_temp: np.ndarray, base: np.ndarray,
                              n_random: int = 5) -> list:
    starts = [
        base.copy(),
        base + np.linspace(-0.5, 0.5, CFG.N_SEGMENTS),
        base + np.linspace(0.5, -0.5, CFG.N_SEGMENTS),
        base + 0.4 * np.array([0.0, 0.2, 0.4, 0.6, 0.6, 0.4, 0.2, 0.0]),
    ]
    if len(f_temp) > 1:
        std = f_temp.std(axis=0)
        starts += [base + 0.5 * std, base - 0.5 * std]
    np.random.seed(42)
    for _ in range(n_random):
        starts.append(base + np.random.uniform(-0.3, 0.3, CFG.N_SEGMENTS))
    return [np.clip(s, 23.0, 33.0) for s in starts]


def inverse_design_robust(model_bow, model_warp, scaler,
                          base_profile, base_warp,
                          smooth_weight, prox_weight):
    """역설계 SLSQP 최적화 - (최적 온도 프로파일, BOW편차, WARP) 반환"""
    min_pred, avg_pred = estimate_minimum_deviation(model_bow, scaler)
    print(f"  Min |BOW-1.75| est.: {min_pred:.3f} | Avg: {avg_pred:.3f}")
    warp_w = 0.5

    def objective(x):
        xs = scaler.transform(x.reshape(1, -1))
        bow = model_bow.predict(xs)[0]
        warp = model_warp.predict(xs)[0]
        return (bow
                + warp_w * max(0, warp - base_warp)
                + prox_weight * np.sum((x - base_profile) ** 2)
                + smooth_weight * np.sum(np.diff(x) ** 2))

    constraints = [
        {"type": "ineq", "fun": lambda x: -np.diff(x, n=2) - 1e-6},
        {"type": "ineq", "fun": lambda x: 1.0 - (
            np.sum(np.diff(np.sign(np.diff(x)[np.diff(x) != 0])) != 0)
            if len(np.diff(x)[np.diff(x) != 0]) >= 2 else 0)},
        {"type": "ineq", "fun": lambda x: np.mean(x[3:6]) - np.mean([x[1], x[6], x[7]])},
    ]
    bounds = [(20.0, 30.0)] * CFG.N_SEGMENTS
    starts = generate_initial_profiles(base_profile, base_profile)

    best_x, best_obj = None, np.inf
    for x0 in starts:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 300, "ftol": 1e-4})
        if res.success:
            val = objective(res.x)
            if val < best_obj:
                best_x, best_obj = res.x, val

    if best_x is not None:
        xs = scaler.transform(best_x.reshape(1, -1))
        return best_x, model_bow.predict(xs)[0], model_warp.predict(xs)[0]

    # 풀백: 랜덤 탐색
    valid = []
    np.random.seed(42)
    for _ in range(1000):
        x = np.random.uniform(20, 30, CFG.N_SEGMENTS)
        dx = np.diff(x)
        sc = np.sum(np.diff(np.sign(dx[dx != 0])) != 0) if len(dx[dx != 0]) >= 2 else 0
        if (np.all(np.diff(x, n=2) <= 1e-6) and sc <= 1
                and np.mean(x[3:6]) >= np.mean([x[0], x[1], x[6], x[7]])):
            valid.append(x)

    if valid:
        valid = np.array(valid)
        scores = []
        for x in valid:
            xs = scaler.transform(x.reshape(1, -1))
            scores.append(model_bow.predict(xs)[0]
                          + warp_w * max(0, model_warp.predict(xs)[0] - base_warp)
                          + prox_weight * np.sum((x - base_profile) ** 2)
                          + smooth_weight * np.sum(np.diff(x) ** 2))
        xb = valid[np.argmin(scores)]
        xs = scaler.transform(xb.reshape(1, -1))
        return xb, model_bow.predict(xs)[0], model_warp.predict(xs)[0]

    print(" 유효한 오목 프로파일 없음 → 기본 프로파일 반환")
    xs = scaler.transform(base_profile.reshape(1, -1))
    return base_profile, model_bow.predict(xs)[0], base_warp


# ==========================================================
# 15. 시각화 헬퍼
# ==========================================================
def plot_profiles_by_eqp(results_df: pd.DataFrame, resdir: str):
    rows = [(r["EQP"], r["Group"],
             np.array([float(v.strip()) for v in r["Base_Profile"].split("→")]),
             np.array([float(v.strip()) for v in r["Recommended_Profile"].split("→")]))
            for _, r in results_df.iterrows()
            if r["Recommended_Profile"] not in ("-", "None")]

    if not rows:
        print("시각화할 추천 프로파일이 없습니다.")
        return

    n = len(rows)
    ncols, nrows = 2, (n + 1) // 2
    seg_labels = [f"Seg{i+1}\n({b}%)" for i, b in enumerate(CFG.SEGMENT_BOUNDS)]
    cmap = plt.cm.tab10
    x_pos = np.arange(CFG.N_SEGMENTS)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 + 3 * nrows))
    axes = np.array(axes).reshape(-1)
    plt.suptitle("Equipment-wise Recommended FRAME_IN_TEMP Profiles\n"
                 "(Base vs. Optimized for |BOW-1.75| with WARP Constraint)",
                 fontsize=13, fontweight="bold", y=0.98)

    idx = 0
    for idx, (eqp, grp, base, opt) in enumerate(rows):
        ax = axes[idx]
        ax.plot(x_pos, base, "o--", color=cmap(0), label="Base (Mean)", alpha=0.8)
        ax.plot(x_pos, opt, "s-", color=cmap(1), label="Recommended", linewidth=2)
        ax.fill_between(x_pos, opt, color=cmap(1), alpha=0.1)
        ax.set_title(f"{eqp}_{grp}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Temp (°C)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(seg_labels, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune="both", nbins=6))
        ax.set_ylim(27, 32)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = pt.join(resdir, "recommended_temp_profiles_visualization.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)   # 메모리: show 대신 close로 figure 즉시 해제
    print(f"  저장: {path}")


def plot_all_profiles_overlaid(results_df: pd.DataFrame, resdir: str):
    profiles, labels = [], []
    for _, r in results_df.iterrows():
        if r["Recommended_Profile"] in ("-", "None"):
            continue
        try:
            profiles.append(np.array(
                [float(v.strip()) for v in r["Recommended_Profile"].split("→")]))
            labels.append(f"{r['EQP']}_{r['Group']}")
        except Exception as e:
            print(f"  파싱 실패 {r['EQP']} G{r['Group']}: {e}")

    if not profiles:
        print("유효한 최적 프로파일이 없습니다.")
        return

    x_pos = np.arange(CFG.N_SEGMENTS)
    seg_labels = [f"{b}%" for b in CFG.SEGMENT_BOUNDS]
    fig = plt.figure(figsize=(12, 6))
    for p, lbl in zip(profiles, labels):
        plt.plot(x_pos, p, "s-", linewidth=1.2, alpha=0.7, label=lbl)
    plt.xlabel("Position (%)")
    plt.ylabel("Recommended FRAME_IN_TEMP (°C)")
    plt.title("All Equipment & Group: Recommended Temperature Profiles\n"
              "(Optimal Profiles Only, Overlaid)",
              fontsize=13, fontweight="bold")
    plt.xticks(x_pos, seg_labels)
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.ylim(27, 32)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
               ncol=1, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    path = pt.join(resdir, "all_optimal_profiles_overlaid.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ==========================================================
# 16. 모델 저장 / 로드 헬퍼
# ==========================================================
def _model_key(eqp: str, grp: str, recipe: str = "") -> str:
    safe_eqp = eqp.replace(" ", "_").replace("/", "_")
    if recipe:
        return f"{safe_eqp}_G{grp}_R{recipe}"
    return f"{safe_eqp}_G{grp}"


def save_bow_models(eqp: str, grp: str, lot_ids: list,
                    scaler: StandardScaler, model_bow, model_warp,
                    all_train_lots: dict, save_dir: str = None,
                    recipe: str = ""):
    save_dir = save_dir or CFG.OUTPUT_APC
    os.makedirs(save_dir, exist_ok=True)
    key = _model_key(eqp, grp, recipe)
    joblib.dump(scaler, pt.join(save_dir, f"bow_scaler_{key}.pkl"))
    joblib.dump(model_bow, pt.join(save_dir, f"bow_model_{key}.pkl"))
    joblib.dump(model_warp, pt.join(save_dir, f"warp_model_{key}.pkl"))
    all_train_lots[key] = lot_ids
    with open(pt.join(save_dir, "train_lots.json"), "w", encoding="utf-8") as f:
        json.dump(all_train_lots, f, indent=2, ensure_ascii=False)


def load_bow_models(eqp: str, grp: str, save_dir: str = None, recipe: str = ""):
    save_dir = save_dir or CFG.OUTPUT_APC
    key = _model_key(eqp, grp, recipe)
    p_bow = pt.join(save_dir, f"bow_model_{key}.pkl")
    p_warp = pt.join(save_dir, f"warp_model_{key}.pkl")
    p_scaler = pt.join(save_dir, f"bow_scaler_{key}.pkl")
    if pt.exists(p_bow) and pt.exists(p_warp) and pt.exists(p_scaler):
        return (joblib.load(p_scaler), joblib.load(p_bow), joblib.load(p_warp))
    return (None, None, None)


def load_train_lots(save_dir: str = None) -> dict:
    save_dir = save_dir or CFG.OUTPUT_APC
    p = pt.join(save_dir, "train_lots.json")
    if pt.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {}


# ==========================================================
# 17. BOW 역설계 파이프라인 (기존 로직 유지)
# ==========================================================
def run_bow_optimization_pipeline(df):
    """그룹×장비×레시피별 XGBoost BOW/WARP 학습 + 역설계 최적화"""
    df = df.dropna(subset=["Group", "BOW_BF_MEAN_TOTAL"]).reset_index(drop=True)
    df["RECIPE_time"] = df["RECIPE_ID"].str.extract(r'(\d\d\d)').astype(float)

    recipe_map = {"133": [133, 150], "180": [185, 180]}
    reverse_recipe_map = {133: "133", 150: "133", 185: "180", 180: "180"}

    target_recipes = CFG.TARGET_RECIPE_TIME
    allowed_times = []
    for r in target_recipes:
        if r in recipe_map:
            allowed_times.extend(recipe_map[r])
        else:
            print(f"정의되지 않은 TARGET_RECIPE_TIME: {r}")
    if not allowed_times:
        print("사용할 레시피가 없습니다.")
        return pd.DataFrame()

    df = df[df["RECIPE_time"].isin(allowed_times)]
    print(f"RECIPE 필터 적용 ({target_recipes}): {len(df)} LOT 유지")

    xdir = CFG.X_DIR
    resdir = CFG.OUTPUT_FRAME_BOW
    os.makedirs(resdir, exist_ok=True)

    EQP_NAMES = sorted(df["EQP_NM_3200"].unique())
    GROUPS = sorted(df["Group"].astype(str).unique())
    results = []
    all_train_lots = load_train_lots()

    print(f"Total: {len(EQP_NAMES)} EQPs x {len(GROUPS)} Groups "
          f"(Target Recipes: {target_recipes})")

    df["TARGET_RECIPE_TIME"] = df["RECIPE_time"].apply(
        lambda rt: reverse_recipe_map.get(int(rt), "133") if pd.notna(rt) else "133")

    for eqp, grp in iterproduct(EQP_NAMES, GROUPS):
        eqp_grp_sub = df[(df["EQP_NM_3200"] == eqp) &
                         (df["Group"].astype(str) == str(grp))].copy()

        available_recipes = sorted(
            eqp_grp_sub["TARGET_RECIPE_TIME"].dropna().unique()) \
            if "TARGET_RECIPE_TIME" in eqp_grp_sub.columns else ["133"]

        for recipe in available_recipes:
            sub = eqp_grp_sub[eqp_grp_sub["TARGET_RECIPE_TIME"] == recipe].copy()
            if len(sub) < 5:
                continue

            f_temp_list, y_bow_list, y_warp_list, valid_lot_ids = [], [], [], []

            for i, lot in enumerate(sub["USER_LOT_ID"]):
                file_path = pt.join(xdir, f"{lot}.csv")
                if not os.path.exists(file_path):
                    continue
                x = process_x(file_path)
                if x is not None:
                    bt = extract_boundary_temps(x)
                    if not np.isnan(bt).any():
                        f_temp_list.append(bt)
                        y_bow_list.append(abs(sub["BOW_BF_MEAN_TOTAL"].iloc[i] - 1.75))
                        y_warp_list.append(sub["WARP_BF_MEAN_TOTAL"].iloc[i])
                        valid_lot_ids.append(lot)

            if len(f_temp_list) < 2:
                continue

            f_temp = np.array(f_temp_list)
            y_bow = np.array(y_bow_list)
            y_warp = np.array(y_warp_list)
            total_n_lots = len(y_bow)

            scaler = StandardScaler()
            Xs = scaler.fit_transform(f_temp)
            model_bow = xgb.XGBRegressor(**CFG.XGB_PARAMS).fit(Xs, y_bow, verbose=False)
            model_warp = xgb.XGBRegressor(**CFG.XGB_PARAMS).fit(Xs, y_warp, verbose=False)

            r2b = r2_score(y_bow, model_bow.predict(Xs))
            maeb = mean_absolute_error(y_bow, model_bow.predict(Xs))
            r2w = r2_score(y_warp, model_warp.predict(Xs))
            print(f"  {eqp} G{grp} R{recipe} | BOW R2={r2b:.3f} MAE={maeb:.3f} "
                  f"| WARP R2={r2w:.3f} | n={total_n_lots}")

            save_bow_models(eqp, grp, valid_lot_ids, scaler,
                            model_bow, model_warp, all_train_lots, recipe=recipe)

            base_p = f_temp.mean(axis=0)
            base_w = y_warp.mean()
            sw = calculate_smoothness_weight(f_temp, y_bow)
            pw = calculate_proximity_weight(y_bow)

            opt_p, pred_bow_opt, pred_warp_opt = inverse_design_robust(
                model_bow, model_warp, scaler, base_p, base_w, sw, pw)

            base_bow_pred = model_bow.predict(
                scaler.transform(base_p.reshape(1, -1)))[0]
            prof_str = " → ".join(f"{t:.1f}" for t in opt_p) \
                if opt_p is not None else "None"
            base_str = " → ".join(f"{t:.1f}" for t in base_p)

            results.append({
                "EQP": eqp, "Group": grp, "TARGET_RECIPE_TIME": recipe,
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "R2_BOW": round(r2b, 3), "MAE_BOW": round(maeb, 3),
                "R2_WARP": round(r2w, 3), "N": total_n_lots,
                "Base_BOW_Dev": round(base_bow_pred, 3),
                "Optimal_BOW_Dev": round(pred_bow_opt, 3)
                    if pred_bow_opt is not None else None,
                "Improvement": round(base_bow_pred - pred_bow_opt, 3)
                    if pred_bow_opt is not None else None,
                "Base_WARP": round(base_w, 3),
                "Predicted_WARP": round(pred_warp_opt, 3)
                    if pred_warp_opt is not None else None,
                "Recommended_Profile": prof_str,
                "Avg_Recommended_Temp": round(opt_p.mean(), 2)
                    if opt_p is not None else None,
                "Base_Profile": base_str
            })

            # 그룹 처리 후 즉시 해제
            del f_temp, y_bow, y_warp, Xs
            gc.collect()

    if not results:
        print("생성된 결과가 없습니다.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    for col, sign in [("Optimal", -1), ("Base", -1)]:
        dev_col = f"{col}_BOW_Dev"
        if dev_col in results_df.columns:
            results_df[f"{col}_BOW_Lower"] = (1.75 + sign * results_df[dev_col]).round(3)
            results_df[f"{col}_BOW_Upper"] = (1.75 - sign * results_df[dev_col]).round(3)

    results_df = results_df.fillna("-")
    ts = datetime.now().strftime("%Y%m%d")
    save_csv(results_df, resdir, f"bow_range_8point_by_EQP_Group_Recipe_{ts}.csv")

    plot_profiles_by_eqp(results_df, resdir)
    plot_all_profiles_overlaid(results_df, resdir)
    print(f"결과 저장 완료: {len(results_df)}행 (EQP×Group×Recipe)")
    return results_df


# ==========================================================
# 18. 실행 진입점
# ==========================================================
if __name__ == "__main__":
    conn = get_trino_conn()

    # ─ Step 1: FDC 수집 & 전처리 (스트리밍/배치, resumable)
    #   → 시계열은 CFG.FDC_PROC_DIR parquet에, 메타는 DataFrame으로 반환
    df_fdc_meta = run_fdc_pipeline_streaming(conn)

    # ─ Step 2: 3305 Flatness 수집 & 전처리 (일별 청크, resumable)
    df_3305, pivot_3305 = run_3305_flatness_pipeline(conn)
    del df_3305
    gc.collect()

    # ─ Step 3: PIMS 수집 & 전처리 (스펙 테이블 1회 조회)
    pivot_pims = run_pims_pipeline(conn)

    # ─ Step 4: 데이터 병합 (LOT 메타 기반 → 행 폭발 없음)
    df_cleaned = convert_data_cols_naming(
        pivot_pims, pivot_3305, df_fdc_meta, CFG.OUTPUT_FDC)
    del pivot_3305, pivot_pims, df_fdc_meta
    gc.collect()

    # ─ Step 5: 클러스터링 + LOT별 시계열 CSV 추출
    df_clean = predict_single_lot_group(
        df_cleaned, CFG.X_DIR, CFG.OUTPUT_FDC,
        CFG.CLUSTERING_PARAMS, CFG.FDC_PROC_DIR)
    del df_cleaned
    gc.collect()

    # ─ Step 6: 역모델 학습 + 역설계 최적화
    run_bow_optimization_pipeline(df_clean)

    conn.close()
    print("\n전체 파이프라인 완료")
