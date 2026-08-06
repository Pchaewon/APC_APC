# -*- coding: utf-8 -*-
"""
preprocess 출력 → 리포트 입력 컬럼 어댑터  (process_time 재파생 수정판)
─────────────────────────────────────────
pilot_wiresaw_preprocess.py 출력(대문자, SL-BOW-BF 등)을
generate_report.py가 기대하는 컬럼명(avg_bow_bf_total 등)으로 변환.

★ 이번 수정 핵심 (process_time=0 버그)
  기존: `if 'process_time' not in df.columns:` → PROCESS_TIME(=0) placeholder가
        rename으로 들어오면 "이미 존재"로 판정, 파생 스킵 → 0이 그대로 남음
  수정: RECIPE_ID를 권위 소스로, process_time 값이 무효(0/nan/'' 등)면 재파생.
        파생 실패는 0이 아닌 NaN + 경고 로깅 (하류 미검출 방지).

사용:
  from preprocess_adapter import adapt_columns
  df_report = adapt_columns(df_preprocessed)
"""
import re
import pandas as pd
import numpy as np

PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]

# process_time 유효 라벨 / RECIPE_ID recipe_time(3자리) → 라벨 매핑
#   ★ 실제 RECIPE_ID 분포 확인 후 아래 매핑을 잠글 것
VALID_PT = {'13.3Hr', '18.5Hr'}
RECIPE_TIME_MAP = {
    '133': '13.3Hr', '150': '13.3Hr',
    '185': '18.5Hr', '180': '18.5Hr',
}


def _find(df, candidates):
    """후보 컬럼명 중 실제 존재하는 첫 번째 반환 (대소문자 무시, 완전일치)."""
    upper_map = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in upper_map:
            return upper_map[cand.upper()]
    return None


def _derive_pt(recipe_id):
    """RECIPE_ID('2-133', '10-150' 등) → process_time 라벨.
    파생 불가 시 np.nan (0이나 임의값으로 채우지 않음)."""
    if pd.isna(recipe_id):
        return np.nan
    s = str(recipe_id).strip()
    # 하이픈 뒤 3자리 숫자, 끝에 '.0' 붙은 float화 방어 ('2-133.0')
    m = re.search(r'-\s*(\d{3})(?:\.0+)?\s*$', s)
    if not m:
        return np.nan
    return RECIPE_TIME_MAP.get(m.group(1), np.nan)


def adapt_columns(df, verbose=True):
    """
    preprocess DataFrame → 리포트용 DataFrame (컬럼명 변환 + process_time 재파생).
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

    # ════════════════════════════════════════════════════════════
    # ★ process_time 파생/재파생 (RECIPE_ID = 권위 소스)
    #   정책: PROCESS_TIME/RUNTIME_MINUTES 컬럼이 rename으로 들어와도(0 등),
    #         값이 무효면 신뢰하지 않고 RECIPE_ID에서 재파생.
    #         파생 실패는 NaN + 경고 (0으로 채워 조용히 오추천 내는 것 방지).
    # ════════════════════════════════════════════════════════════
    rid_col = _find(df, ['RECIPE_ID'])   # 정확 매칭: RECIP_ID / RECIPE_ID_3200 제외
    if rid_col is None:
        if verbose:
            print("  ⚠ RECIPE_ID 컬럼 없음 → process_time 재파생 불가")
        if 'process_time' not in df.columns:
            df['process_time'] = np.nan
    else:
        derived = df[rid_col].map(_derive_pt)
        if 'process_time' not in df.columns:
            df['process_time'] = derived
            n_redrv = int(df['process_time'].notna().sum())
        else:
            existing = df['process_time'].astype(str).str.strip()
            invalid = ~existing.isin(VALID_PT)   # 0, 0.0, '', nan 전부 무효 취급
            n_redrv = int(invalid.sum())
            # process_time이 숫자(int/float 0)로 들어와 있으면 문자열 라벨 대입 시
            # dtype 충돌 → 먼저 object로 캐스팅
            df['process_time'] = df['process_time'].astype(object)
            df.loc[invalid, 'process_time'] = derived[invalid].values

        n_ok = int(df['process_time'].isin(VALID_PT).sum())
        if verbose:
            print(f"  process_time: 유효 {n_ok}/{len(df)}행 "
                  f"(무효→재파생 시도 {n_redrv}행)")
            if n_ok < len(df):
                bad = (df.loc[~df['process_time'].isin(VALID_PT), rid_col]
                       .astype(str).unique()[:5])
                print(f"    ⚠ 여전히 무효 {len(df)-n_ok}행 — "
                      f"RECIPE_ID 샘플: {list(bad)}")

    if verbose:
        print(f"[어댑터] {len(rename)}개 컬럼 변환")
        need_total = ['eqp_nm_3200', 'fdc_new_wire_id', 'date_3200',
                      'avg_bow_bf_total']
        for c in need_total:
            mark = '✓' if c in df.columns else '✗ 없음'
            print(f"  {mark} {c}")
        if missing:
            print(f"  ⚠ 미확인(스킵됨): "
                  f"{[m for m in missing if 'total' in m or m in need_total]}")

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
        report_cols = [c for c in out.columns if c.startswith(('set_', 'avg_',
                       'shift_of_', 'fdc_', 'eqp_', 'new_', 'date_', 'range_',
                       'process_', 'warm_'))]
        print(f"리포트 컬럼 {len(report_cols)}개:")
        for c in sorted(report_cols):
            print(f"  {c}")
    else:
        print("사용법: python preprocess_adapter.py {preprocessed.csv}")
