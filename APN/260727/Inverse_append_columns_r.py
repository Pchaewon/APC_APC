# -*- coding: utf-8 -*-
"""
Inverse 결과를 원본 데이터에 컬럼으로 붙여서 새 CSV 생성
─────────────────────────────────────────
각 행의 실제 BOW를 target으로 inverse 실행 →
  pred_set_frame_temp_0pct ~ 100pct  (온도 12개)
  pred_set_table_speed_0pct ~ 100pct (table_speed 12개)
  pred_fdc_set_tension               (tension)
  pred_bow                           (역산 recipe로 예측한 BOW, 검증용)
을 원본 컬럼 옆에 붙여서 저장.

change_group 등 방향검증 관련 로직 없음. 순수하게 recipe 역산 → 컬럼 추가만.
"""
import os
import os.path as pt
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize

CONFIG = {
    'model_dir': r'./apc_model/full',
    'test_csv':  r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',
    'out_csv':   r'./test_df_with_pred.csv',
    'target':    'avg_bow_bf_total',
    'eqp_col':   'eqp_nm_3200',
    # 어떤 recipe를 역산할지: 'temp' | 'table_speed' | 'tension' | 'both' | 'all'
    'optimize':  'all',
    'lambda_smooth': 0.0,     # 진단 결과 λ=0이 실측 재현 최선
    'encoding':  'utf-8',
}


def load_model(model_dir):
    with open(pt.join(model_dir, 'forward_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(pt.join(model_dir, 'feature_meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    scaler = None
    if meta.get('use_scaler') and os.path.exists(pt.join(model_dir, 'scaler.pkl')):
        with open(pt.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler, meta


def build_predictor(model, scaler, meta, eqp_name):
    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    eqp_cols = meta.get('eqp_cols', []); pfx = meta.get('eqp_prefix', 'eqp_')
    def predict(base_row, override=None):
        def gv(c):
            if c in eqp_cols:
                return 1.0 if c == f'{pfx}{eqp_name}' else 0.0
            if override and c in override:
                return float(override[c])
            v = base_row.get(c, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return float(X_STATS.get(c, {}).get('mean', 0.0))
            return float(v)
        x = np.array([gv(c) for c in FEATURES]).reshape(1, -1)
        if scaler is not None:
            x = scaler.transform(x)
        return float(model.predict(x)[0])
    return predict


def inverse(model, scaler, meta, target_y, base_row, eqp_name,
            optimize, lam):
    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    temp_cols = meta['temp_cols']
    table_speed_cols = meta.get('table_speed_cols', [])
    tension_col = meta['tension_col']

    if optimize == 'temp':
        opt_cols = temp_cols
    elif optimize == 'table_speed':
        opt_cols = table_speed_cols
    elif optimize == 'tension':
        opt_cols = [tension_col]
    elif optimize == 'both':
        opt_cols = temp_cols + [tension_col]
    elif optimize == 'all':
        opt_cols = temp_cols + table_speed_cols + [tension_col]
    else:
        raise ValueError(optimize)

    predict = build_predictor(model, scaler, meta, eqp_name)
    n_temp = len(temp_cols); n_ts = len(table_speed_cols)

    def objective(opt_vec):
        override = dict(zip(opt_cols, opt_vec))
        loss = (predict(base_row, override) - target_y) ** 2
        if lam > 0:
            if optimize in ('temp', 'table_speed'):
                loss += lam * np.sum(np.diff(opt_vec) ** 2)
            elif optimize == 'both':
                loss += lam * np.sum(np.diff(opt_vec[:n_temp]) ** 2)
            elif optimize == 'all':
                loss += lam * np.sum(np.diff(opt_vec[:n_temp]) ** 2)
                loss += lam * np.sum(np.diff(opt_vec[n_temp:n_temp+n_ts]) ** 2)
        return loss

    x0 = np.array([float(base_row.get(c, X_STATS[c]['mean'])) for c in opt_cols])
    bounds = [(X_STATS[c]['q01'], X_STATS[c]['q99']) for c in opt_cols]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})
    rec = dict(zip(opt_cols, res.x))
    y_pred = predict(base_row, rec)
    return rec, y_pred


def main(cfg):
    model, scaler, meta = load_model(cfg['model_dir'])
    df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    EQP, TARGET = cfg['eqp_col'], cfg['target']
    optimize = cfg['optimize']

    temp_cols = meta['temp_cols']
    table_speed_cols = meta.get('table_speed_cols', [])
    tension_col = meta['tension_col']

    print(f"[로드] {len(df)}행, feature {len(meta['feature_cols'])}개")
    print(f"[optimize] {optimize}")

    # 역산 대상 컬럼 결정
    pred_temp = optimize in ('temp', 'both', 'all')
    pred_ts   = optimize in ('table_speed', 'all')
    pred_ten  = optimize in ('tension', 'both', 'all')

    # pred 컬럼 미리 생성
    if pred_temp:
        for c in temp_cols:
            df[f'pred_{c}'] = np.nan
    if pred_ts:
        for c in table_speed_cols:
            df[f'pred_{c}'] = np.nan
    if pred_ten:
        df[f'pred_{tension_col}'] = np.nan
    df['pred_bow'] = np.nan

    # 학습에 쓰인 장비만 역산 가능 (더미에 있는 장비)
    eqp_cols = meta.get('eqp_cols', [])
    known_eqps = {c.replace(meta.get('eqp_prefix', 'eqp_'), '') for c in eqp_cols}

    n_done, n_skip = 0, 0
    for idx, row in df.iterrows():
        eqp = row[EQP]
        if pd.isna(row[TARGET]):
            continue
        if known_eqps and eqp not in known_eqps:
            n_skip += 1
            continue

        rec, y_pred = inverse(model, scaler, meta, float(row[TARGET]),
                              row.to_dict(), eqp, optimize, cfg['lambda_smooth'])

        if pred_temp:
            for c in temp_cols:
                df.at[idx, f'pred_{c}'] = round(rec.get(c, np.nan), 4)
        if pred_ts:
            for c in table_speed_cols:
                df.at[idx, f'pred_{c}'] = round(rec.get(c, np.nan), 4)
        if pred_ten:
            df.at[idx, f'pred_{tension_col}'] = round(rec.get(tension_col, np.nan), 4)
        df.at[idx, 'pred_bow'] = round(y_pred, 4)

        n_done += 1
        if n_done % 200 == 0:
            print(f"  진행: {n_done}행")

    df.to_csv(cfg['out_csv'], index=False, encoding='utf-8-sig')

    print(f"\n✅ 저장: {cfg['out_csv']}")
    print(f"   역산 완료: {n_done}행" + (f" (학습외 장비 {n_skip}행 스킵)" if n_skip else ""))
    added = []
    if pred_temp: added.append(f"pred_set_frame_temp_*({len(temp_cols)}개)")
    if pred_ts:   added.append(f"pred_set_table_speed_*({len(table_speed_cols)}개)")
    if pred_ten:  added.append("pred_fdc_set_tension")
    added.append("pred_bow")
    print(f"   추가 컬럼: {', '.join(added)}")
    return df


if __name__ == '__main__':
    main(CONFIG)
