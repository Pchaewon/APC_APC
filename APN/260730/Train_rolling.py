# -*- coding: utf-8 -*-
"""
직전 run 평균 기반 학습 + 역산 (배포 현실 반영)
─────────────────────────────────────────
구조:
  · 같은 wire 내 직전 최대 10run(2run 지연) 평균을 조건으로 사용
  · 학습: roll_feature → 현재 run BOW 예측
  · 역산: 현재 run의 목표 BOW → recipe(온도) 역산
          단, roll_ 조건(직전 평균)은 고정, 온도만 최적화

핵심 아이디어:
  현재 run의 recipe(온도)를 정하려는데, 아직 안 가공했으니 현재 값이 없음.
  → 직전 run들의 평균 상태(roll_)를 조건으로,
     "이 상태에서 목표 BOW를 내려면 온도를 어떻게?"를 역산.

두 모드:
  train : Ridge 학습 + 저장
  inverse : 저장 모델로 역산
"""
import os
import os.path as pt
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from rolling_run_features import add_rolling_run_features

CONFIG = {
    'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
    'model_dir':  r'./apc_model_rolling',
    'process_time': '13.3Hr',
    'target':     'avg_bow_bf_total',
    'wire_col':   'new_fdc_wire_id',
    'eqp_col':    'eqp_nm_3200',
    'date_col':   'date_3200',
    # 현재 run이 최적화할 recipe (온도)
    'temp_cols':  [f'set_frame_temp_{p}pct' for p in
                   [0,10,20,30,40,50,60,70,80,90,99,100]],
    # 직전 평균으로 쓸 원본 컬럼 (roll_ 접두어로 변환됨)
    'roll_source_cols': [
        'fdc_set_tension','fdc_wait_time','fdc_ingot_len','range_slurry_temp_10_0',
    ],
    'lag': 2,
    'window': 10,
    'min_runs': 3,
    'use_eqp_dummy': True,
    'ridge_alpha': 5.0,
    'split_ratio': 0.8,
    'lambda_smooth': 0.0,
    'encoding':   'utf-8',
}


def build_dataset(cfg):
    """직전 run 평균 feature가 추가된 데이터 구성."""
    df = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    if cfg['process_time']:
        df = df[df['process_time'] == cfg['process_time']]

    # rolling feature 생성 (BOW + 조건 컬럼 평균)
    roll_cfg = {
        'wire_col': cfg['wire_col'], 'date_col': cfg['date_col'],
        'target_col': cfg['target'],
        'feature_cols': cfg['temp_cols'] + cfg['roll_source_cols'],
        'lag': cfg['lag'], 'window': cfg['window'], 'min_runs': cfg['min_runs'],
    }
    df = add_rolling_run_features(df, roll_cfg)
    return df


def get_feature_sets(cfg):
    """
    X feature 구성:
      · 현재 run의 온도 12개 (recipe, 최적화 대상)
      · 직전 평균: roll_avg_bow, roll_{조건}   (고정 조건)
      · 장비 더미
    """
    temp_cols = cfg['temp_cols']
    roll_cols = [f'roll_{cfg["target"]}'] + \
                [f'roll_{c}' for c in cfg['roll_source_cols']]
    return temp_cols, roll_cols


def train(cfg):
    os.makedirs(cfg['model_dir'], exist_ok=True)
    df = build_dataset(cfg)
    temp_cols, roll_cols = get_feature_sets(cfg)
    EQP = cfg['eqp_col']; TARGET = cfg['target']; DATE = cfg['date_col']

    base_feats = temp_cols + roll_cols
    sub = df[base_feats + [TARGET, DATE, EQP]].dropna().copy()
    sub = sub.sort_values(DATE).reset_index(drop=True)
    print(f"[학습 데이터] {len(sub)}행 (직전 run 평균 있는 것만)")

    # 장비 더미
    if cfg['use_eqp_dummy']:
        dummies = pd.get_dummies(sub[EQP], prefix='eqp')
        FEATURES = base_feats + list(dummies.columns)
        sub = pd.concat([sub, dummies], axis=1)
    else:
        FEATURES = base_feats

    X = sub[FEATURES].values.astype(float)
    y = sub[TARGET].values

    # 시간 분할
    si = int(len(sub) * cfg['split_ratio'])
    scaler = StandardScaler().fit(X[:si])
    model = Ridge(alpha=cfg['ridge_alpha']).fit(scaler.transform(X[:si]), y[:si])
    pred = model.predict(scaler.transform(X[si:]))
    r2 = r2_score(y[si:], pred); mae = mean_absolute_error(y[si:], pred)
    print(f"  시간 분할 Test R²={r2:.4f}, MAE={mae:.4f}")

    # 전체로 재학습 (배포용)
    scaler_full = StandardScaler().fit(X)
    model_full = Ridge(alpha=cfg['ridge_alpha']).fit(scaler_full.transform(X), y)

    # 저장
    with open(pt.join(cfg['model_dir'], 'model.pkl'), 'wb') as f:
        pickle.dump(model_full, f)
    with open(pt.join(cfg['model_dir'], 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_full, f)

    def stats(a):
        return {'mean': float(np.mean(a)), 'std': float(np.std(a)),
                'q01': float(np.quantile(a, 0.01)), 'q99': float(np.quantile(a, 0.99))}
    meta = {
        'target': TARGET, 'feature_cols': FEATURES,
        'temp_cols': temp_cols, 'roll_cols': roll_cols,
        'eqp_cols': list(dummies.columns) if cfg['use_eqp_dummy'] else [],
        'eqp_prefix': 'eqp_', 'use_scaler': True,
        'x_stats': {c: stats(sub[c].values) for c in base_feats},
        'metrics': {'r2_time': round(r2, 4), 'mae': round(mae, 4)},
    }
    with open(pt.join(cfg['model_dir'], 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"💾 저장: {cfg['model_dir']}/")
    return model_full, scaler_full, meta


def inverse_current_run(cfg, target_bow, roll_values, eqp_name,
                        current_temp=None):
    """
    현재 run recipe(온도) 역산.
    roll_values: {roll_avg_bow_bf_total: .., roll_fdc_set_tension: .., ...}
    target_bow: 현재 run에서 달성하려는 BOW
    """
    with open(pt.join(cfg['model_dir'], 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(pt.join(cfg['model_dir'], 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(pt.join(cfg['model_dir'], 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)

    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    temp_cols = meta['temp_cols']; roll_cols = meta['roll_cols']
    eqp_cols = meta['eqp_cols']; pfx = meta['eqp_prefix']

    def gv(c, temp_override):
        if c in eqp_cols:
            return 1.0 if c == f'{pfx}{eqp_name}' else 0.0
        if c in temp_cols and temp_override is not None:
            return float(temp_override[temp_cols.index(c)])
        if c in roll_cols:
            return float(roll_values.get(c, X_STATS.get(c, {}).get('mean', 0.0)))
        return float(X_STATS.get(c, {}).get('mean', 0.0))

    def predict(temp_vec):
        x = np.array([gv(c, temp_vec) for c in FEATURES]).reshape(1, -1)
        return float(model.predict(scaler.transform(x))[0])

    def objective(temp_vec):
        loss = (predict(temp_vec) - target_bow) ** 2
        if cfg['lambda_smooth'] > 0:
            loss += cfg['lambda_smooth'] * np.sum(np.diff(temp_vec) ** 2)
        return loss

    if current_temp is not None:
        x0 = np.array(current_temp)
    else:
        x0 = np.array([X_STATS[c]['mean'] for c in temp_cols])
    bounds = [(X_STATS[c]['q01'], X_STATS[c]['q99']) for c in temp_cols]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})
    rec = {c: round(float(v), 4) for c, v in zip(temp_cols, res.x)}
    return {'recommended_temp': rec, 'predicted_bow': round(predict(res.x), 4),
            'target_bow': target_bow}


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if mode == 'train':
        train(CONFIG)
    else:
        # 역산 예시
        roll_vals = {
            'roll_avg_bow_bf_total': 1.8,
            'roll_fdc_set_tension': 0.8,
            'roll_fdc_wait_time': 45,
            'roll_fdc_ingot_len': 38,
            'roll_range_slurry_temp_10_0': 2.3,
        }
        r = inverse_current_run(CONFIG, target_bow=1.75, roll_values=roll_vals,
                                eqp_name='BSWS38')
        print(json.dumps(r, indent=2, ensure_ascii=False))
