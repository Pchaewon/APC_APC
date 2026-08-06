# -*- coding: utf-8 -*-
"""
배포용 Inverse (최근 10 lot 기반 recipe 추천 + 저장/재실행)
─────────────────────────────────────────
배포 흐름:
  [입력] 실시간 DB → 최근 10 lot을 CSV로 저장 (recent_10lot_{eqp}.csv)
  [로드] 그 CSV를 불러와 직전 평균(roll_조건) 계산
  [역산] frame + slurry 각각 추천 (dual 모델)
  [저장] 추천 recipe + 입력 최근10lot을 함께 저장 (재현/평가용)
  [재실행] 저장된 스냅샷을 불러와 동일 inverse 재현

모드:
  run   : 최근 10 lot CSV 로드 → inverse → 결과+입력 저장
  rerun : 저장된 스냅샷 로드 → inverse 재실행 (재현 검증)
"""
import os
import os.path as pt
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize

CONFIG = {
    'model_dir':   r'./apc_model_dual',        # dual (frame/, slurry/)
    # 최근 10 lot 입력 CSV (실시간 DB에서 저장해 둔 것)
    'recent_csv':  r'./recent_10lot.csv',
    # 추천/스냅샷 저장 폴더
    'save_dir':    r'./inverse_records',
    'target':      'avg_bow_bf_total',
    'target_bow':  1.75,
    'eqp_col':     'eqp_nm_3200',
    'date_col':    'date_3200',
    'wire_col':    'new_fdc_wire_id',
    'profiles':    ['frame', 'slurry'],
    'roll_source_cols': ['fdc_set_tension','fdc_wait_time','fdc_ingot_len',
                         'range_slurry_temp_10_0'],
    'window':      10,      # 최근 몇 lot 평균
    'encoding':    'utf-8',
}


# ═══════════════════════════════════════
# 모델 로드 + 역산
# ═══════════════════════════════════════
def load_profile_model(model_dir, name):
    mdir = pt.join(model_dir, name)
    if not os.path.exists(pt.join(mdir, 'model.pkl')):
        return None
    with open(pt.join(mdir, 'model.pkl'), 'rb') as f: model = pickle.load(f)
    with open(pt.join(mdir, 'scaler.pkl'), 'rb') as f: scaler = pickle.load(f)
    with open(pt.join(mdir, 'meta.json'), encoding='utf-8') as f: meta = json.load(f)
    return model, scaler, meta


def compute_roll_values(recent_lots, roll_source_cols, window):
    """
    최근 lot들의 조건 평균 → roll_ 값.
    recent_lots: 최근 10 lot DataFrame (시간순)
    반환: {roll_fdc_set_tension: .., ...}
    """
    roll_values = {}
    tail = recent_lots.tail(window)
    for src in roll_source_cols:
        if src in tail.columns:
            v = tail[src].mean()
            if pd.notna(v):
                roll_values[f'roll_{src}'] = float(v)
    return roll_values


def inverse_profile(model_dir, name, target_bow, roll_values, eqp_name):
    """해당 프로파일 전용 모델로 온도 역산."""
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
    bounds = [(X_STATS.get(c, {}).get('q01', x0[i]-1),
               X_STATS.get(c, {}).get('q99', x0[i]+1))
              for i, c in enumerate(opt_cols)]
    res = minimize(lambda v: (predict(v)-target_bow)**2, x0,
                   method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})
    rec = {c: round(float(v), 2) for c, v in zip(opt_cols, res.x)}
    return {'profile': name, 'recipe': rec,
            'predicted_bow': round(predict(res.x), 3),
            'roll_cols': roll_cols, 'mae': meta.get('metrics', {}).get('mae', 0.1)}


# ═══════════════════════════════════════
# 배포 실행 (run)
# ═══════════════════════════════════════
def run_inverse(cfg, recent_csv=None):
    """
    최근 10 lot CSV 로드 → 장비별 inverse → 결과+입력 스냅샷 저장.
    """
    os.makedirs(cfg['save_dir'], exist_ok=True)
    recent_csv = recent_csv or cfg['recent_csv']
    recent = pd.read_csv(recent_csv, encoding=cfg['encoding'],
                         encoding_errors='replace')
    recent[cfg['date_col']] = pd.to_datetime(recent[cfg['date_col']],
                                             errors='coerce')
    EQP = cfg['eqp_col']
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_recs = []
    for eqp in sorted(recent[EQP].dropna().unique()):
        esub = recent[recent[EQP] == eqp].sort_values(cfg['date_col'])
        if len(esub) == 0:
            continue
        # 최근 window lot 조건 평균
        roll_values = compute_roll_values(esub, cfg['roll_source_cols'],
                                          cfg['window'])
        if not roll_values:
            print(f"  {eqp}: 조건 평균 계산 불가 — 스킵")
            continue

        rec_row = {'eqp': eqp, 'target_bow': cfg['target_bow'],
                   'n_recent_lot': len(esub.tail(cfg['window'])),
                   'timestamp': stamp}
        # roll 조건도 기록 (재현용)
        for k, v in roll_values.items():
            rec_row[k] = round(v, 4)

        # frame / slurry 각각 역산
        for name in cfg['profiles']:
            inv = inverse_profile(cfg['model_dir'], name, cfg['target_bow'],
                                  roll_values, eqp)
            if inv is None:
                print(f"  {eqp}/{name}: 모델 없음 — 스킵")
                continue
            rec_row[f'{name}_pred_bow'] = inv['predicted_bow']
            for c, val in inv['recipe'].items():
                rec_row[f'rec_{c}'] = val
        all_recs.append(rec_row)
        print(f"  {eqp}: 추천 완료 (최근 {rec_row['n_recent_lot']} lot 기반)")

    rec_df = pd.DataFrame(all_recs)

    # ── 저장 ① 추천 결과 ──
    rec_path = pt.join(cfg['save_dir'], f'recommend_{stamp}.csv')
    rec_df.to_csv(rec_path, index=False, encoding='utf-8-sig')

    # ── 저장 ② 입력된 최근 10 lot 스냅샷 (재현/평가용) ──
    snap_path = pt.join(cfg['save_dir'], f'input_snapshot_{stamp}.csv')
    recent.to_csv(snap_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 추천 저장: {rec_path}")
    print(f"✅ 입력 스냅샷 저장: {snap_path}")
    print(f"   (재실행: python deploy_inverse.py rerun {stamp})")
    return rec_df, stamp


# ═══════════════════════════════════════
# 재실행 (rerun) — 저장된 스냅샷으로 재현
# ═══════════════════════════════════════
def rerun_inverse(cfg, stamp):
    """
    저장된 input_snapshot을 불러와 inverse 재실행 → 이전 추천과 비교.
    """
    snap_path = pt.join(cfg['save_dir'], f'input_snapshot_{stamp}.csv')
    prev_path = pt.join(cfg['save_dir'], f'recommend_{stamp}.csv')
    if not os.path.exists(snap_path):
        print(f"❌ 스냅샷 없음: {snap_path}")
        return
    print(f"[재실행] 스냅샷 {stamp} 로드 → inverse 재현\n")

    # 스냅샷으로 inverse 재실행
    new_df, _ = run_inverse(cfg, recent_csv=snap_path)

    # 이전 추천과 비교
    if os.path.exists(prev_path):
        prev = pd.read_csv(prev_path)
        rec_cols = [c for c in prev.columns if c.startswith('rec_')]
        print(f"\n[재현 검증] 이전 추천 vs 재실행")
        merged = prev[['eqp'] + rec_cols].merge(
            new_df[['eqp'] + rec_cols], on='eqp', suffixes=('_prev', '_new'))
        max_diff = 0
        for c in rec_cols:
            if f'{c}_prev' in merged and f'{c}_new' in merged:
                d = (merged[f'{c}_prev'] - merged[f'{c}_new']).abs().max()
                max_diff = max(max_diff, d)
        print(f"  최대 추천값 차이: {max_diff:.4f}")
        if max_diff < 0.01:
            print("  ✅ 완전 재현 (스냅샷 기반 재실행 일치)")
        else:
            print("  ⚠ 차이 발생 — 모델/데이터 변경 확인")
    return new_df


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if mode == 'run':
        run_inverse(CONFIG)
    elif mode == 'rerun':
        if len(sys.argv) < 3:
            print("사용법: python deploy_inverse.py rerun {timestamp}")
        else:
            rerun_inverse(CONFIG, sys.argv[2])
