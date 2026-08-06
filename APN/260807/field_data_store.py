# -*- coding: utf-8 -*-
"""
실시간 데이터 저장·누적 + total 컬럼 생성
─────────────────────────────────────────
흐름:
  · 초기: DB에서 일주일치 → 저장 (field_store.csv)
  · 이후: 1시간마다 1시간 분량 → append (중복 제거)
  · 저장 데이터를 장비 > wire id > WAF_SEQ_NO 순 정렬
  · total 컬럼 생성 (wire 내 WAF_SEQ_NO 전체 평균)
    - avg_bow_bf_total, avg_warp_bf_total
    - (seed/mid/tail은 추후 기준 받아서)

핵심:
  · append 누적 (중복은 장비+wire+WAF_SEQ_NO+시간 기준 제거)
  · total = 같은 wire 안 모든 wafer 평균
"""
import os
import os.path as pt
import pandas as pd
import numpy as np
from datetime import datetime
from preprocess_adapter import adapt_columns

STORE_CONFIG = {
    'store_csv':  r'./data/field_store.csv',   # 누적 저장 파일
    'eqp_col':    'eqp_nm_3200',
    'wire_col':   'fdc_new_wire_id',
    'seq_col':    'waf_seq_no',
    'date_col':   'date_3200',
    # total 낼 대상 (wire 내 WAF_SEQ_NO 평균)
    'total_targets': {
        'avg_bow_bf_total':  ['avg_bow_bf_total', 'bow_bf'],
        'avg_warp_bf_total': ['avg_warp_bf_total', 'warp_bf'],
    },
    # 중복 판정 키
    'dedup_keys': ['eqp_nm_3200', 'fdc_new_wire_id', 'waf_seq_no', 'date_3200'],
    'collapse_to_wire': True,   # WAF 평균 후 wire당 1행으로
    'encoding':   'utf-8',
}


def append_to_store(new_df, cfg, is_preprocessed=True):
    """
    새 데이터를 저장소에 누적 (append + 중복 제거).
    new_df: preprocess 출력 (대문자) 또는 이미 변환된 것
    """
    os.makedirs(pt.dirname(cfg['store_csv']) or '.', exist_ok=True)

    # 컬럼 변환 (preprocess 출력이면)
    if is_preprocessed:
        new_df = adapt_columns(new_df, verbose=False)

    # 기존 저장소 로드
    if os.path.exists(cfg['store_csv']):
        store = pd.read_csv(cfg['store_csv'], encoding=cfg['encoding'],
                            encoding_errors='replace')
        combined = pd.concat([store, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    # 중복 제거 (키 기준, 최신 유지)
    keys = [k for k in cfg['dedup_keys'] if k in combined.columns]
    if keys:
        before = len(combined)
        combined = combined.drop_duplicates(subset=keys, keep='last')
        removed = before - len(combined)
    else:
        removed = 0

    # 정렬: 장비 > wire > WAF_SEQ_NO
    sort_cols = [c for c in [cfg['eqp_col'], cfg['wire_col'], cfg['seq_col']]
                 if c in combined.columns]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    combined.to_csv(cfg['store_csv'], index=False, encoding='utf-8-sig')
    print(f"[저장소] {len(new_df)}행 추가 → 총 {len(combined)}행 "
          f"(중복 {removed}개 제거)")
    return combined


def build_total_columns(df, cfg):
    """
    WAF 단위 → wire 단위 변환 (wire 내 WAF_SEQ_NO 전체 평균).
    학습 데이터가 이미 wire 평균 형태이므로, 실시간 WAF 데이터를
    같은 형태로 맞추기 위해 모든 수치 feature를 wire 평균.

    ★ total = wire 내 모든 WAF_SEQ_NO 평균
    · BOW/WARP total
    · 온도 프로파일 (set_frame_temp_*pct 등) — inverse feature
    · 조건 (fdc_*, range_*) — roll 조건 원본
    (seed/mid/tail은 추후 기준 받아 추가)

    반환: wire 단위로 집계된 DataFrame (wire당 1행)
    """
    df = df.copy()
    EQP, WIRE = cfg['eqp_col'], cfg['wire_col']
    group = [EQP, WIRE]

    # 1) total 타깃 (BOW/WARP): 후보명에서 찾아 target명으로
    for target, cands in cfg['total_targets'].items():
        src = None
        for cand in cands:
            if cand in df.columns:
                src = cand; break
        if src is None:
            print(f"  ⚠ {target}: 원본 컬럼 없음 (후보 {cands})")
            continue
        df[target] = df.groupby(group)[src].transform('mean')

    # 2) 나머지 수치형 feature도 wire 평균 (온도 프로파일·조건)
    #    inverse에 쓰는 feature가 WAF마다 다르면 wire 평균해야 학습 형태와 일치
    id_like = set([EQP, WIRE, cfg.get('seq_col'), cfg.get('date_col'),
                   'lot_id', 'process_time'])
    numeric_cols = [c for c in df.columns
                    if c not in id_like
                    and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        df[numeric_cols] = df.groupby(group)[numeric_cols].transform('mean')

    # 3) wire당 1행으로 축약 (중복 제거) — 옵션
    if cfg.get('collapse_to_wire', True):
        # 대표 행: wire별 첫 행 (평균이 이미 transform으로 들어감)
        df = df.drop_duplicates(subset=group, keep='first').reset_index(drop=True)

    print(f"[wire 평균] WAF 단위 → wire 단위 "
          f"({len(numeric_cols)}개 수치 컬럼 평균, {len(df)}행)")
    return df


def load_and_prepare(cfg, eqp=None):
    """
    저장소 로드 → total 생성 → (선택) 장비 필터.
    inverse에 넣을 준비된 데이터 반환.
    """
    if not os.path.exists(cfg['store_csv']):
        raise FileNotFoundError(f"저장소 없음: {cfg['store_csv']}")
    df = pd.read_csv(cfg['store_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    df = build_total_columns(df, cfg)
    if eqp:
        df = df[df[cfg['eqp_col']] == eqp].copy()
    # 정렬 재확인
    sort_cols = [c for c in [cfg['eqp_col'], cfg['wire_col'], cfg['seq_col']]
                 if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


# ═══════════════════════════════════════
# 초기 적재 / 시간별 append 진입점
# ═══════════════════════════════════════
def initial_load(preprocessed_csv, cfg):
    """초기: 일주일치 preprocess CSV → 저장소 생성."""
    print(f"[초기 적재] {preprocessed_csv}")
    df = pd.read_csv(preprocessed_csv, encoding='cp949', encoding_errors='replace')
    return append_to_store(df, cfg, is_preprocessed=True)


def hourly_append(preprocessed_csv, cfg):
    """1시간마다: 1시간 분량 preprocess CSV → append."""
    print(f"[시간별 append] {preprocessed_csv}")
    df = pd.read_csv(preprocessed_csv, encoding='cp949', encoding_errors='replace')
    return append_to_store(df, cfg, is_preprocessed=True)


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'prepare'
    csv = sys.argv[2] if len(sys.argv) > 2 else None
    if mode == 'init' and csv:
        initial_load(csv, STORE_CONFIG)
    elif mode == 'append' and csv:
        hourly_append(csv, STORE_CONFIG)
    elif mode == 'prepare':
        df = load_and_prepare(STORE_CONFIG)
        print(f"준비 완료: {len(df)}행")
        print(f"total 컬럼: {[c for c in df.columns if 'total' in c]}")
    else:
        print("사용법: python field_data_store.py [init|append|prepare] [csv]")
