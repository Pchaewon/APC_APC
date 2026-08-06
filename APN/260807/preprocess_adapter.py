# -*- coding: utf-8 -*-
"""
preprocess 출력 → 리포트 입력 컬럼 어댑터
─────────────────────────────────────────
pilot_wiresaw_preprocess.py 출력(대문자, SL-BOW-BF 등)을
generate_report.py가 기대하는 컬럼명(avg_bow_bf_total 등)으로 변환.

현재 단계: Total만. seed/mid/tail은 컬럼 없으면 리포트가 자동 "데이터 없음" 처리.

사용:
  from preprocess_adapter import adapt_columns
  df_report = adapt_columns(df_preprocessed)
"""
import pandas as pd
import numpy as np

PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]


def _find(df, candidates):
    """후보 컬럼명 중 실제 존재하는 첫 번째 반환 (대소문자 무시)."""
    upper_map = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in upper_map:
            return upper_map[cand.upper()]
    return None


def adapt_columns(df, verbose=True):
    """
    preprocess DataFrame → 리포트용 DataFrame (컬럼명 변환).
    없는 컬럼은 건너뜀 (리포트가 "데이터 없음" 처리).
    """
    df = df.copy()
    # 컬럼명 공백 제거
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    missing = []

    # ── 1. 식별자/메타 ──
    id_map = {
        'eqp_nm_3200':    ['EQP_NM_3200', 'EQP_NM'],
        'fdc_new_wire_id':['NEW_WIRE_ID', 'NEW_FDC_WIRE_ID', 'FDC_NEW_WIRE_ID'],
        'date_3200':      ['HIS_REGIST_DTTM', 'HST_REG_DTTM', 'DATE_3200'],
        'lot_id':         ['LOT_ID', 'USER_LOT_ID'],
        'wire_seq_no':    ['WAF_SEQ_NO', 'WIRE_SEQ_NO'],
    }
    for target, cands in id_map.items():
        src = _find(df, cands)
        if src:
            rename[src] = target
        else:
            missing.append(target)

    # ── 2. 온도 프로파일 (대소문자만) ──
    for p in PCTS:
        for base_src, base_tgt in [('SET_FRAME_TEMP', 'set_frame_temp'),
                                    ('SET_SLURRY_TEMP', 'set_slurry_temp')]:
            src = _find(df, [f'{base_src}_{p}pct'])
            if src:
                rename[src] = f'{base_tgt}_{p}pct'

    # ── 3. Wire Guide 프로파일 (명명 변환) ──
    for p in PCTS:
        for side in ['L', 'R']:
            src = _find(df, [f'SHIFT_AMOUNT_WIREGUIDE_{side}_{p}pct'])
            if src:
                rename[src] = f'shift_amount_wireguide_{side.lower()}_{p}pct'

    # ── 4. BOW / WARP (Total만) ──
    bow_map = {
        'avg_bow_bf_total':  ['BOW_BF_MEAN_TOTAL', 'SL-BOW-BF', 'SL_BOW_BF'],
        'avg_warp_bf_total': ['WARP_BF_MEAN_TOTAL', 'SL-WARP-BF', 'SL_WARP_BF'],
        # seed/mid/tail (있으면 매핑, 없으면 스킵)
        'avg_bow_bf_seed':   ['BOW_BF_SEED'],
        'avg_bow_bf_mid':    ['BOW_BF_MID'],
        'avg_bow_bf_tail':   ['BOW_BF_TAIL'],
        'avg_warp_bf_seed':  ['WARP_BF_SEED'],
        'avg_warp_bf_mid':   ['WARP_BF_MID'],
        'avg_warp_bf_tail':  ['WARP_BF_TAIL'],
    }
    for target, cands in bow_map.items():
        src = _find(df, cands)
        if src:
            rename[src] = target
        elif 'total' in target:
            missing.append(target)   # total은 필수라 경고

    # ── 5. 스칼라 조건 ──
    scalar_map = {
        'fdc_ingot_len':        ['INGOT_LEN', 'FDC_INGOT_LEN'],
        'fdc_wait_time':        ['WAIT_TIME', 'FDC_WAIT_TIME'],
        'fdc_set_tension':      ['SET_TENSION', 'FDC_SET_TENSION'],
        'fdc_warm_up_time':     ['WARM_UP_TIME', 'WARMUP_TIME', 'FDC_WARM_UP_TIME'],
        'range_slurry_temp_10_0':['RANGE_SLURRY_TEMP_10_0'],
        'range_wire_guide_10_99':['RANGE_WIRE_GUIDE_10_99'],
        'process_time':         ['PROCESS_TIME', 'RUNTIME_MINUTES'],
    }
    for target, cands in scalar_map.items():
        src = _find(df, cands)
        if src:
            rename[src] = target
        else:
            missing.append(target)

    # 적용
    df = df.rename(columns=rename)

    if verbose:
        print(f"[어댑터] {len(rename)}개 컬럼 변환")
        # 리포트 필수 컬럼 존재 확인
        need_total = ['eqp_nm_3200', 'new_fdc_wire_id', 'date_3200',
                      'avg_bow_bf_total']
        for c in need_total:
            mark = '✓' if c in df.columns else '✗ 없음'
            print(f"  {mark} {c}")
        if missing:
            print(f"  ⚠ 미확인(스킵됨): {[m for m in missing if 'total' in m or m in need_total]}")

    return df


def adapt_from_csv(csv_path, encoding='cp949'):
    """preprocess CSV 파일을 읽어 어댑터 적용."""
    df = pd.read_csv(csv_path, encoding=encoding, encoding_errors='replace')
    return adapt_columns(df)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        out = adapt_from_csv(sys.argv[1])
        print(f"\n최종 {len(out)}행 x {len(out.columns)}컬럼")
        # 리포트 관련 컬럼만 출력
        report_cols = [c for c in out.columns if c.startswith(('set_', 'avg_',
                       'shift_of_', 'fdc_', 'eqp_', 'new_', 'date_', 'range_',
                       'process_', 'warm_'))]
        print(f"리포트 컬럼 {len(report_cols)}개:")
        for c in sorted(report_cols):
            print(f"  {c}")
    else:
        print("사용법: python preprocess_adapter.py {preprocessed.csv}")
