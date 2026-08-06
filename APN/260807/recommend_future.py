# -*- coding: utf-8 -*-
"""
미래 lot recipe 추천 (배포 inverse)
─────────────────────────────────────────
장비별 가장 최근 wire 기준으로, 다음다음 lot(현실적 lead)의
recipe를 dual 모델 inverse로 추천.

흐름:
  · field_store에서 장비 데이터 로드 (total 컬럼 포함)
  · 장비별 최근 wire의 직전 run들 평균(roll_조건) 계산
  · frame/slurry 각 모델로 온도 역산 → 목표 BOW 달성 recipe
  · 13.3Hr 기준

핵심:
  · '미래 lot' = 장비별 최근 wire의 다음다음 (lag 반영)
  · roll_조건 = 그 wire 최근 WAF_SEQ_NO들의 조건 평균
"""
import os
import os.path as pt
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from field_data_store import load_and_prepare, STORE_CONFIG

RECOMMEND_CONFIG = {
    'model_dir':  r'./apc_model_dual',
    'store_cfg':  STORE_CONFIG,
    'out_csv':    r'./recommend_future.csv',
    'process_time_filter': '13.3Hr',    # 13.3 기준
    'target_bow': 1.75,
    'eqp_col':    'eqp_nm_3200',
    'wire_col':   'fdc_new_wire_id',
    'seq_col':    'waf_seq_no',
    'date_col':   'date_3200',
    'frame_start_default': 28.0,
    # roll 조건 (모델 학습 feature와 일치)
    'roll_source_cols': ['fdc_set_tension','fdc_wait_time','fdc_ingot_len',
                         'range_slurry_temp_10_0'],
    # 대상 장비
    'target_eqps': ['BSWS30','BSWS31','BSWS35','BSWS42','BSWS48',
                    'BSWS51','BSWS53','BSWS55','BSWS57','BSWS58'],
    'roll_window': 10,     # 최근 몇 WAF 평균
    'encoding':   'utf-8',
}


def load_profile_model(model_dir, name):
    mdir = pt.join(model_dir, name)
    if not os.path.exists(pt.join(mdir, 'model.pkl')):
        return None
    with open(pt.join(mdir, 'model.pkl'), 'rb') as f: model = pickle.load(f)
    with open(pt.join(mdir, 'scaler.pkl'), 'rb') as f: scaler = pickle.load(f)
    with open(pt.join(mdir, 'meta.json'), encoding='utf-8') as f: meta = json.load(f)
    return model, scaler, meta


def inverse_profile(model_dir, name, target_bow, roll_values, eqp_name,
                    frame_start=None):
    loaded = load_profile_model(model_dir, name)
    if loaded is None:
        return None
    model, scaler, meta = loaded
    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    profile_cols = meta['profile_cols']; roll_cols = meta.get('roll_cols', [])
    eqp_cols = meta.get('eqp_cols', []); pfx = meta.get('eqp_prefix', 'eqp_')
    opt_cols = [c for c in profile_cols if c in FEATURES]
    if not opt_cols:
        return None

    def gv(c, override):
        if c in eqp_cols:
            return 1.0 if c == f'{pfx}{eqp_name}' else 0.0
        if override is not None and c in opt_cols:
            return float(override[opt_cols.index(c)])
        if c in roll_cols:
            return float(roll_values.get(c, X_STATS.get(c, {}).get('mean', 0.0)))
        return float(X_STATS.get(c, {}).get('mean', 0.0))

    def predict(vec):
        x = np.array([gv(c, vec) for c in FEATURES]).reshape(1, -1)
        return float(model.predict(scaler.transform(x))[0])

    x0 = np.array([X_STATS.get(c, {}).get('mean', 29.0) for c in opt_cols])
    if name == 'frame' and frame_start is not None:
        for i, c in enumerate(opt_cols):
            if c.endswith('_0pct'):
                x0[i] = frame_start
    bounds = [(X_STATS.get(c, {}).get('q01', x0[i]-1),
               X_STATS.get(c, {}).get('q99', x0[i]+1))
              for i, c in enumerate(opt_cols)]
    res = minimize(lambda v: (predict(v)-target_bow)**2, x0,
                   method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})
    rec = {c: round(float(v), 2) for c, v in zip(opt_cols, res.x)}
    return {'recipe': rec, 'predicted_bow': round(predict(res.x), 3)}


def compute_roll_for_latest_wire(edf, cfg):
    """
    장비 데이터(wire 단위)에서 최근 wire들의 조건 평균 → roll_값.

    데이터는 이미 wire 단위(build_total_columns로 WAF 평균됨).
    미래 lot 추천이므로, 최근 wire들의 조건 평균을 roll로 사용
    (모델의 roll feature = 직전 wire들 평균과 동일 구조).
    """
    WIRE = cfg['wire_col']; DATE = cfg['date_col']
    # 시간순 정렬 (wire 단위)
    if DATE in edf.columns:
        edf = edf.sort_values(DATE)
    latest_wire = edf[WIRE].iloc[-1]
    # 최근 window개 wire의 조건 평균
    tail = edf.tail(cfg['roll_window'])

    roll_values = {}
    for src in cfg['roll_source_cols']:
        if src in tail.columns:
            v = tail[src].mean()
            if pd.notna(v):
                roll_values[f'roll_{src}'] = float(v)
    return roll_values, latest_wire, len(tail)


def recommend_all(cfg):
    """대상 장비별 미래 lot recipe 추천."""
    scfg = cfg['store_cfg']
    rows = []
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for eqp in cfg['target_eqps']:
        try:
            edf = load_and_prepare(scfg, eqp=eqp)
        except FileNotFoundError as e:
            print(f"  ❌ {e}"); return
        if len(edf) == 0:
            print(f"  ⚠ {eqp}: 데이터 없음 — 스킵")
            continue
        # 13.3 필터
        if cfg['process_time_filter'] and 'process_time' in edf.columns:
            edf = edf[edf['process_time'] == cfg['process_time_filter']]
            if len(edf) == 0:
                print(f"  ⚠ {eqp}: 13.3Hr 데이터 없음 — 스킵")
                continue

        roll_values, latest_wire, n_waf = compute_roll_for_latest_wire(edf, cfg)
        if not roll_values:
            print(f"  ⚠ {eqp}: roll 조건 계산 불가 — 스킵")
            continue

        row = {'eqp': eqp, 'latest_wire': latest_wire, 'n_waf_used': n_waf,
               'target_bow': cfg['target_bow'], 'timestamp': stamp,
               'lead': '다음다음 lot'}
        for k, v in roll_values.items():
            row[k] = round(v, 4)

        # frame / slurry 역산
        for name in ['frame', 'slurry']:
            inv = inverse_profile(cfg['model_dir'], name, cfg['target_bow'],
                                  roll_values, eqp,
                                  frame_start=cfg.get('frame_start_default'))
            if inv is None:
                continue
            row[f'{name}_pred_bow'] = inv['predicted_bow']
            for c, val in inv['recipe'].items():
                row[f'rec_{c}'] = val
        rows.append(row)
        print(f"  ✅ {eqp}: 최근 wire={latest_wire} (WAF {n_waf}개) → 추천 완료")

    res = pd.DataFrame(rows)
    res.to_csv(cfg['out_csv'], index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {cfg['out_csv']} ({len(res)}개 장비)")
    return res


if __name__ == '__main__':
    recommend_all(RECOMMEND_CONFIG)
