# -*- coding: utf-8 -*-
"""
Inverse 결과를 원본 데이터에 컬럼으로 추가
─────────────────────────────────────────
각 행의 실제 BOW를 target으로 inverse 실행 →
  · 15-19년식: pred_set_frame_temp_0pct ~ 100pct (12개)
  · 21년식: pred_fdc_set_tension
원본 데이터에 pred_ 컬럼으로 붙여서 저장.

출력: test_df_with_pred.csv (원본 전체 컬럼 + pred_ 컬럼)
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
    'eqp_groups': {
        '15-19': {'eqps': ['BSWS38','BSWS42','BSWS44'], 'optimize': 'temp'},
        '21':    {'eqps': ['BSWS52','BSWS54','BSWS55','BSWS56','BSWS61'],
                  'optimize': 'tension'},
    },
    'lambda_smooth': 0.0,   # 진단 결과 λ=0이 실측 재현 최선 (r=0.98)
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


def inverse_for_target(model, scaler, meta, target_y, base_row,
                       optimize, eqp_name, lam):
    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    temp_cols = meta['temp_cols']; tension_col = meta['tension_col']

    if optimize == 'temp':
        opt_cols = temp_cols
    else:
        opt_cols = [tension_col]

    predict = build_predictor(model, scaler, meta, eqp_name)

    def objective(opt_vec):
        override = dict(zip(opt_cols, opt_vec))
        loss = (predict(base_row, override) - target_y) ** 2
        if optimize == 'temp' and lam > 0:
            loss += lam * np.sum(np.diff(opt_vec) ** 2)
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
    temp_cols = meta['temp_cols']
    tension_col = meta['tension_col']

    print(f"[로드] {len(df)}행, feature {len(meta['feature_cols'])}개")

    # 장비 → optimize 매핑
    eqp2opt = {}
    for grp, gc in cfg['eqp_groups'].items():
        for e in gc['eqps']:
            eqp2opt[e] = gc['optimize']

    # pred 컬럼 미리 생성 (NaN 초기화)
    for c in temp_cols:
        df[f'pred_{c}'] = np.nan
    df['pred_fdc_set_tension'] = np.nan
    df['pred_bow'] = np.nan      # 역산 recipe로 예측한 BOW (검증용)

    n_done = 0
    for idx, row in df.iterrows():
        eqp = row[EQP]
        if eqp not in eqp2opt:
            continue
        if pd.isna(row[TARGET]):
            continue

        optimize = eqp2opt[eqp]
        target_y = float(row[TARGET])
        rec, y_pred = inverse_for_target(model, scaler, meta, target_y,
                                         row.to_dict(), optimize, eqp,
                                         cfg['lambda_smooth'])

        if optimize == 'temp':
            for c in temp_cols:
                df.at[idx, f'pred_{c}'] = round(rec[c], 4)
        else:  # tension
            df.at[idx, 'pred_fdc_set_tension'] = round(rec[tension_col], 4)

        df.at[idx, 'pred_bow'] = round(y_pred, 4)
        n_done += 1
        if n_done % 500 == 0:
            print(f"  진행: {n_done}행")

    df.to_csv(cfg['out_csv'], index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {cfg['out_csv']}")
    print(f"   총 {n_done}행 역산 완료")
    print(f"   추가 컬럼: pred_set_frame_temp_0pct~100pct (12개), "
          f"pred_fdc_set_tension, pred_bow")

    # 요약
    temp_done = df['pred_set_frame_temp_60pct'].notna().sum()
    tension_done = df['pred_fdc_set_tension'].notna().sum()
    print(f"\n   temp 역산: {temp_done}행 (15-19년식)")
    print(f"   tension 역산: {tension_done}행 (21년식)")
    return df


if __name__ == '__main__':
    main(CONFIG)
