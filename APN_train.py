# -*- coding: utf-8 -*-
"""
Forward 모델 학습 + 저장 (train/test) — Inverse/검증 재사용
─────────────────────────────────────────
Leakage 제거 (정밀):
  · test 장비 8대의 test 기간(2026-03-01 이후) 데이터만 제외
  · 나머지 34대는 전 기간 유지, test 8대의 과거(3월 이전)도 유지
Feature:
  · recipe(temp 12 + tension + wait + ingot) + condition 분리
  · tension을 recipe에 포함 (21년식 역산 대상)
출력:
  · model.pkl + scaler.pkl + feature_meta.json + train_log.json
  · 계수 부호 자동 출력 (공정지식 대조)
"""
import os
import os.path as pt
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'paths': {
        'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
        'output_dir': r'./apc_model',
    },
    'filter': {
        'process_time': '13.3Hr',
        'meas_eqp':     None,          # None이면 전체 (검증은 42대 학습)
    },
    'leakage': {
        'enabled':    True,
        'test_eqps':  ['BSWS38','BSWS42','BSWS44','BSWS52',
                       'BSWS54','BSWS55','BSWS56','BSWS61'],
        'test_start': '2026-03-01',    # 이 8대의 이 날짜 이후만 제외
        'date_col':   'date_3200',
        'eqp_col':    'eqp_nm_3200',
    },
    'model': {
        'target':       'avg_bow_bf_total',
        'model_type':   'linear',      # 'linear' | 'ridge'
        'ridge_alpha':  5.0,
        'split_ratio':  0.8,
        'random_state': 42,
        'use_scaler':   True,
        'use_eqp_dummy': True,   # ★ 장비 one-hot (temp 부호 복원)
    },
    'features': {
        'recipe_cols': [
            'set_frame_temp_0pct',  'set_frame_temp_10pct', 'set_frame_temp_20pct',
            'set_frame_temp_30pct', 'set_frame_temp_40pct', 'set_frame_temp_50pct',
            'set_frame_temp_60pct', 'set_frame_temp_70pct', 'set_frame_temp_80pct',
            'set_frame_temp_90pct', 'set_frame_temp_99pct', 'set_frame_temp_100pct',
            'fdc_set_tension',        # ← 21년식 역산 대상
            'fdc_wait_time',
            'fdc_ingot_len',
        ],
        'condition_cols': [
            'range_slurry_temp_10_0',
            # 'VP_C',
        ],
    },
    'meta_cols': {'date': 'date_3200', 'eqp': 'eqp_nm_3200'},
    'min_samples': 50,
    'encoding': 'utf-8',
}


def apply_leakage_removal(df, cfg):
    """test 장비의 test 기간 데이터만 제외 (나머지 전부 유지)."""
    lk = cfg['leakage']
    if not lk['enabled']:
        return df
    DATE, EQP = lk['date_col'], lk['eqp_col']
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    cutoff = pd.to_datetime(lk['test_start'])
    mask_leak = (df[EQP].isin(lk['test_eqps'])) & (df[DATE] >= cutoff)
    n0, n_removed = len(df), int(mask_leak.sum())
    df = df[~mask_leak].copy()
    print(f"[Leakage 제거] {n0} → {len(df)} "
          f"(test 8대의 {cutoff.date()} 이후 {n_removed}행 제외)")
    print(f"  · 나머지 34대: 전 기간 유지")
    print(f"  · test 8대: {cutoff.date()} 이전 데이터는 유지")
    return df


def train_and_save(d, tag, cfg, out_dir):
    RECIPE = cfg['features']['recipe_cols']
    COND = [c for c in cfg['features']['condition_cols'] if c in d.columns]
    EQP = cfg['meta_cols']['eqp']
    TARGET = cfg['model']['target']
    DATE = cfg['meta_cols']['date']
    use_eqp = cfg['model'].get('use_eqp_dummy', True)

    base_cols = RECIPE + COND
    sub = d[base_cols + [TARGET, DATE, EQP]].dropna().copy()
    if len(sub) < cfg['min_samples']:
        print(f"  [{tag}] N={len(sub)} < {cfg['min_samples']} → 스킵")
        return None

    # 장비 one-hot (Simpson's Paradox 방지 — 장비 개체 효과 흡수)
    if use_eqp:
        eqp_dummies = pd.get_dummies(sub[EQP], prefix='eqp')
        eqp_cols = list(eqp_dummies.columns)
        sub = pd.concat([sub, eqp_dummies], axis=1)
        FEATURES = base_cols + eqp_cols
    else:
        eqp_cols = []
        FEATURES = base_cols

    X_all = sub[FEATURES].values.astype(float)
    y_all = sub[TARGET].values

    # 랜덤 분할 (참고)
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X_all, y_all, test_size=1-cfg['model']['split_ratio'],
        random_state=cfg['model']['random_state'])

    # 시간 분할 (배포 근사) — 최종 모델은 이걸로
    sub[DATE] = pd.to_datetime(sub[DATE], errors='coerce')
    sub_t = sub.sort_values(DATE).reset_index(drop=True)
    si = int(len(sub_t) * cfg['model']['split_ratio'])
    Xtr_t = sub_t[FEATURES].iloc[:si].values.astype(float)
    Xte_t = sub_t[FEATURES].iloc[si:].values.astype(float)
    ytr_t = sub_t[TARGET].iloc[:si].values
    yte_t = sub_t[TARGET].iloc[si:].values

    def make_model():
        return (Ridge(alpha=cfg['model']['ridge_alpha'])
                if cfg['model']['model_type'] == 'ridge'
                else LinearRegression())

    def fit_eval(Xtr, Xte, ytr, yte):
        if cfg['model']['use_scaler']:
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        else:
            sc, Xtr_s, Xte_s = None, Xtr, Xte
        m = make_model().fit(Xtr_s, ytr)
        return (m, sc, r2_score(ytr, m.predict(Xtr_s)),
                r2_score(yte, m.predict(Xte_s)),
                float(np.sqrt(mean_squared_error(yte, m.predict(Xte_s)))),
                float(mean_absolute_error(yte, m.predict(Xte_s))))

    _, _, r2tr_rand, r2te_rand, _, _ = fit_eval(Xtr_r, Xte_r, ytr_r, yte_r)
    model, scaler, r2tr_t, r2te_t, rmse, mae = fit_eval(Xtr_t, Xte_t, ytr_t, yte_t)

    print(f"  [{tag}] N={len(sub)} | 랜덤 Test R²={r2te_rand:.3f} | "
          f"시간 Test R²={r2te_t:.3f}")

    # 계수 부호 (공정지식 대조)
    print(f"  [{tag}] 주요 계수 부호:")
    coef_map = dict(zip(FEATURES, model.coef_))
    for key in ['set_frame_temp_60pct', 'fdc_wait_time',
                'fdc_set_tension', 'fdc_ingot_len']:
        if key in coef_map:
            sign = '↓BOW' if coef_map[key] < 0 else '↑BOW'
            print(f"      {key:28s}: {coef_map[key]:+.4f} ({sign})")

    # 저장
    cdir = pt.join(out_dir, tag)
    os.makedirs(cdir, exist_ok=True)
    with open(pt.join(cdir, 'forward_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    if scaler is not None:
        with open(pt.join(cdir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    def stats_of(a):
        return {'mean': float(np.mean(a)), 'std': float(np.std(a)),
                'min': float(np.min(a)), 'max': float(np.max(a)),
                'q01': float(np.quantile(a, 0.01)),
                'q99': float(np.quantile(a, 0.99))}
    # x_stats는 recipe/condition만 (더미는 0/1이라 분위수 의미 없음)
    stat_cols = base_cols
    meta = {
        'tag': tag, 'target': TARGET, 'feature_cols': FEATURES,
        'recipe_cols': RECIPE, 'condition_cols': COND,
        'eqp_cols': eqp_cols,
        'temp_cols': [c for c in RECIPE if 'set_frame_temp' in c],
        'tension_col': 'fdc_set_tension',
        'use_scaler': cfg['model']['use_scaler'],
        'use_eqp_dummy': use_eqp,
        'eqp_prefix': 'eqp_',
        'model_type': cfg['model']['model_type'],
        'x_stats': {c: stats_of(sub_t[c].values) for c in stat_cols},
        'y_stats': stats_of(y_all),
    }
    with open(pt.join(cdir, 'feature_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(sub),
        'metrics': {
            'random_split': {'r2_train': round(r2tr_rand, 4),
                             'r2_test': round(r2te_rand, 4)},
            'time_split': {'r2_train': round(r2tr_t, 4), 'r2_test': round(r2te_t, 4),
                           'rmse': round(rmse, 4), 'mae': round(mae, 4)},
        },
        'coefficients': {c: float(v) for c, v in zip(FEATURES, model.coef_)},
        'intercept': float(model.intercept_),
    }
    with open(pt.join(cdir, 'train_log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return {'tag': tag, 'n': len(sub),
            'r2_random': round(r2te_rand, 4), 'r2_time': round(r2te_t, 4)}


def main():
    cfg = CONFIG
    out_dir = cfg['paths']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(cfg['paths']['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    print(f"Loaded: {len(df)} rows")

    if cfg['filter']['process_time']:
        df = df[df['process_time'] == cfg['filter']['process_time']]
    if cfg['filter']['meas_eqp']:
        df = df[df['meas_eqp'] == cfg['filter']['meas_eqp']]
    print(f"필터 후: {len(df)} rows")

    df = apply_leakage_removal(df, cfg)

    if 'VP_C' in cfg['features']['condition_cols'] and 'VP_C' not in df.columns:
        cfg['features']['condition_cols'] = [
            c for c in cfg['features']['condition_cols'] if c != 'VP_C']
        print("  ⚠ VP_C 없음 → condition에서 제거")

    print(f"\n{'='*60}\n전체 단일 모델 학습 (검증용)\n{'='*60}")
    res = train_and_save(df, 'full', cfg, out_dir)

    if res:
        sm = pd.DataFrame([res])
        sm.to_csv(pt.join(out_dir, 'training_summary.csv'),
                  index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {out_dir}/full/")


if __name__ == '__main__':
    main()
