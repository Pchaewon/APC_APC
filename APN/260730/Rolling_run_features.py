# -*- coding: utf-8 -*-
"""
Wire 내 직전 run 평균 feature 생성 (배포 현실 반영)
─────────────────────────────────────────
배포 구조:
  1 reel = 5 block, 1 block = 15 run
  현재 run(예: 15번째) 가공 시 데이터센터엔 13번째까지 올라옴 (2run 지연)
  → 같은 wire 내 직전 최대 10run의 BOW·feature 평균 사용

이 모듈:
  · 같은 new_fdc_wire_id 안에서 날짜순 정렬
  · 각 run에 대해, "직전 (lag)run 제외 후 최대 W run" 평균 계산
  · BOW와 feature 둘 다 평균 → prefix 'roll_' 컬럼으로 추가

파라미터:
  lag: 데이터 지연 (현재 15번째, 13번째까지 → lag=2)
  window: 평균낼 최대 run 수 (10)

배포 가능성:
  · 평균은 '직전 run들'로만 계산 (현재 run 값 미포함) → 사전값, 배포 OK
"""
import numpy as np
import pandas as pd


def add_rolling_run_features(df, cfg):
    """
    같은 wire 내 직전 run 평균 feature 추가.

    cfg 필요 키:
      wire_col, date_col, feature_cols, target_col, lag, window
    반환: roll_{feature}, roll_{target} 컬럼이 추가된 df
          (직전 run이 부족하면 NaN)
    """
    WIRE = cfg['wire_col']
    DATE = cfg['date_col']
    FEATS = cfg['feature_cols']
    TARGET = cfg['target_col']
    lag = cfg.get('lag', 2)
    window = cfg.get('window', 10)
    min_runs = cfg.get('min_runs', 3)   # 최소 이만큼 직전 run 있어야 평균

    df = df.copy()
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    # wire 내 시간순 정렬
    df = df.sort_values([WIRE, DATE]).reset_index(drop=True)

    cols_to_roll = [c for c in FEATS + [TARGET] if c in df.columns]

    # 각 wire 그룹 안에서 shift(lag) 후 rolling(window) 평균
    def roll_group(g):
        # 직전 run만: shift(lag)로 현재+최근(lag-1)run 제외
        shifted = g[cols_to_roll].shift(lag)
        # 최대 window run 평균 (min_periods로 최소 개수 보장)
        rolled = shifted.rolling(window, min_periods=min_runs).mean()
        return rolled

    rolled = df.groupby(WIRE, group_keys=False).apply(roll_group)
    rolled.columns = [f'roll_{c}' for c in cols_to_roll]
    df = pd.concat([df, rolled], axis=1)

    # 직전 run 개수도 기록 (신뢰도 참고용)
    def count_prev(g):
        return g[TARGET].shift(lag).rolling(window, min_periods=1).count()
    df['roll_n_runs'] = df.groupby(WIRE, group_keys=False).apply(count_prev)

    return df


# ─────────────────────────────────────────
# 단독 실행: 예시 + 검증
# ─────────────────────────────────────────
if __name__ == '__main__':
    CONFIG = {
        'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
        'out_csv':    r'./data_with_rolling.csv',
        'wire_col':   'new_fdc_wire_id',
        'date_col':   'date_3200',
        'target_col': 'avg_bow_bf_total',
        'feature_cols': [
            'set_frame_temp_0pct','set_frame_temp_10pct','set_frame_temp_20pct',
            'set_frame_temp_30pct','set_frame_temp_40pct','set_frame_temp_50pct',
            'set_frame_temp_60pct','set_frame_temp_70pct','set_frame_temp_80pct',
            'set_frame_temp_90pct','set_frame_temp_99pct','set_frame_temp_100pct',
            'fdc_set_tension','fdc_wait_time','fdc_ingot_len',
            'range_slurry_temp_10_0',
        ],
        'lag': 2,        # 현재 15번째, 13번째까지 → 2run 지연
        'window': 10,    # 최대 10run 평균
        'min_runs': 3,
        'encoding': 'utf-8',
    }
    import os.path as pt
    df = pd.read_csv(CONFIG['input_csv'], encoding=CONFIG['encoding'],
                     encoding_errors='replace')
    print(f"[로드] {len(df)}행")
    out = add_rolling_run_features(df, CONFIG)

    roll_cols = [c for c in out.columns if c.startswith('roll_')]
    print(f"[추가] roll_ 컬럼 {len(roll_cols)}개")
    print(f"  직전 run 평균 있는 행: {out['roll_avg_bow_bf_total'].notna().sum()} "
          f"/ {len(out)}")
    print(f"  (직전 run 부족으로 NaN: {out['roll_avg_bow_bf_total'].isna().sum()})")

    # wire별 run 수 분포
    wire_run_counts = df.groupby(CONFIG['wire_col']).size()
    print(f"\n[wire별 run 수] 평균 {wire_run_counts.mean():.1f}, "
          f"중앙값 {wire_run_counts.median():.0f}, "
          f"최대 {wire_run_counts.max()}")

    out.to_csv(CONFIG['out_csv'], index=False, encoding='utf-8-sig')
    print(f"\n💾 저장: {CONFIG['out_csv']}")
