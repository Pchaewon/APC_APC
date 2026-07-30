# -*- coding: utf-8 -*-
"""
Inverse + 신뢰도 태그 (방식 A: 항상 추천 + 신뢰도 표시)
─────────────────────────────────────────
모든 lot에 recipe 역산 → pred 컬럼 추가.
추가로 직전 WG 상태로 신뢰도 태그:
  · HIGH_VAR (직전 WG 고변동) → 🟢 신뢰도 높음 (R²≈0.44)
  · LOW_VAR  (직전 WG 저변동) → 🟡 신뢰도 낮음 (R²≈0.09, 참고용)

추가 컬럼:
  pred_set_frame_temp_*, pred_fdc_set_tension, pred_bow (기존)
  + prev_wg_state       (HIGH_VAR / LOW_VAR)
  + confidence          (high / low)
  + confidence_note     (엔지니어용 안내 문구)
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
    'out_csv':   r'./test_df_with_pred_confidence.csv',
    'target':    'avg_bow_bf_total',
    'eqp_col':   'eqp_nm_3200',
    'date_col':  'date_3200',
    'wg_col':    'range_wire_guide_10_99',
    'optimize':  'temp',       # 'temp' | 'both' | 'all' (속도 고려 temp 권장)
    'lambda_smooth': 0.0,
    # 신뢰도 판정 (직전 WG 상태)
    'wg_var_threshold': 11.6,
    'wg_roll_k': 3,
    'conf_r2': {'high': 0.44, 'low': 0.09},   # 안내 문구용
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


def inverse(model, scaler, meta, target_y, base_row, eqp_name, optimize, lam):
    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    temp_cols = meta['temp_cols']
    table_speed_cols = meta.get('table_speed_cols', [])
    tension_col = meta['tension_col']

    if optimize == 'temp':
        opt_cols = temp_cols
    elif optimize == 'both':
        opt_cols = temp_cols + [tension_col]
    elif optimize == 'all':
        opt_cols = temp_cols + table_speed_cols + [tension_col]
    else:
        opt_cols = temp_cols
    if len(opt_cols) == 0:
        return {}, np.nan

    predict = build_predictor(model, scaler, meta, eqp_name)
    n_temp = len(temp_cols); n_ts = len(table_speed_cols)

    def objective(v):
        loss = (predict(base_row, dict(zip(opt_cols, v))) - target_y) ** 2
        if lam > 0:
            if optimize == 'temp':
                loss += lam * np.sum(np.diff(v) ** 2)
            elif optimize == 'both':
                loss += lam * np.sum(np.diff(v[:n_temp]) ** 2)
            elif optimize == 'all':
                loss += lam * np.sum(np.diff(v[:n_temp]) ** 2)
                loss += lam * np.sum(np.diff(v[n_temp:n_temp+n_ts]) ** 2)
        return loss

    x0 = np.array([float(base_row.get(c, X_STATS[c]['mean'])) for c in opt_cols])
    bounds = [(X_STATS[c]['q01'], X_STATS[c]['q99']) for c in opt_cols]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})
    rec = dict(zip(opt_cols, res.x))
    return rec, predict(base_row, rec)


def add_prev_wg_state(df, cfg):
    """직전 WG 상태 판정 (사전값, 배포 가능)."""
    EQP, DATE, WG = cfg['eqp_col'], cfg['date_col'], cfg['wg_col']
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    df = df.sort_values([EQP, DATE]).reset_index(drop=True)
    df['prev_wg'] = (df.groupby(EQP)[WG]
                     .transform(lambda s: s.shift(1)
                                .rolling(cfg['wg_roll_k'], min_periods=1).median()))
    df['prev_wg_state'] = np.where(
        df['prev_wg'].isna(), 'UNKNOWN',
        np.where(df['prev_wg'] >= cfg['wg_var_threshold'], 'HIGH_VAR', 'LOW_VAR'))
    return df


def main(cfg):
    model, scaler, meta = load_model(cfg['model_dir'])
    df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    EQP, TARGET = cfg['eqp_col'], cfg['target']
    optimize = cfg['optimize']
    temp_cols = meta['temp_cols']
    tension_col = meta['tension_col']
    table_speed_cols = meta.get('table_speed_cols', [])

    print(f"[로드] {len(df)}행, feature {len(meta['feature_cols'])}개")
    print(f"[optimize] {optimize}")

    # 직전 WG 상태 추가
    df = add_prev_wg_state(df, cfg)
    print(f"[신뢰도] HIGH_VAR={sum(df['prev_wg_state']=='HIGH_VAR')}, "
          f"LOW_VAR={sum(df['prev_wg_state']=='LOW_VAR')}, "
          f"UNKNOWN={sum(df['prev_wg_state']=='UNKNOWN')}")

    # pred 컬럼 준비
    pred_temp = optimize in ('temp', 'both', 'all')
    pred_ts = optimize in ('all',)
    pred_ten = optimize in ('both', 'all')
    if pred_temp:
        for c in temp_cols: df[f'pred_{c}'] = np.nan
    if pred_ts:
        for c in table_speed_cols: df[f'pred_{c}'] = np.nan
    if pred_ten:
        df[f'pred_{tension_col}'] = np.nan
    df['pred_bow'] = np.nan
    df['confidence'] = ''
    df['confidence_note'] = ''

    eqp_cols = meta.get('eqp_cols', [])
    known = {c.replace(meta.get('eqp_prefix','eqp_'),'') for c in eqp_cols}

    hi_r2 = cfg['conf_r2']['high']; lo_r2 = cfg['conf_r2']['low']
    n_done = 0
    for idx, row in df.iterrows():
        eqp = row[EQP]
        if pd.isna(row[TARGET]) or (known and eqp not in known):
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

        # 신뢰도 태그
        state = row['prev_wg_state']
        if state == 'HIGH_VAR':
            df.at[idx, 'confidence'] = 'high'
            df.at[idx, 'confidence_note'] = (
                f'신뢰도 높음 (직전 WG 고변동, 예측 R²≈{hi_r2}). '
                f'온도-BOW 관계 뚜렷 → 적극 반영 권장')
        elif state == 'LOW_VAR':
            df.at[idx, 'confidence'] = 'low'
            df.at[idx, 'confidence_note'] = (
                f'신뢰도 낮음 (직전 WG 저변동, 예측 R²≈{lo_r2}). '
                f'관계 약함 → 참고용, 현재 recipe 유지 고려')
        else:
            df.at[idx, 'confidence'] = 'unknown'
            df.at[idx, 'confidence_note'] = '직전 WG 정보 부족 → 신뢰도 판정 불가'
        n_done += 1
        if n_done % 500 == 0:
            print(f"  진행: {n_done}행")

    df.to_csv(cfg['out_csv'], index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {cfg['out_csv']}")
    print(f"   역산 {n_done}행")

    # 신뢰도 분포
    done = df[df['pred_bow'].notna()]
    print(f"\n[신뢰도 분포]")
    for conf in ['high', 'low', 'unknown']:
        n = sum(done['confidence'] == conf)
        if len(done) > 0:
            print(f"  {conf}: {n}행 ({n/len(done)*100:.0f}%)")
    return df


if __name__ == '__main__':
    main(CONFIG)
