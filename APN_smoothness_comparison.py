# -*- coding: utf-8 -*-
"""
Smoothness 영향 비교 (0 vs 0.01 vs 0.1)
─────────────────────────────────────────
목적: smoothness 페널티가 역산 온도를 얼마나 실측에서 밀어내는지 정량화.
      "매끄러움 ↔ 실측 근접" 트레이드오프를 발표에 정직하게 제시.

3개 패널:
  [좌] 실측 근접도: smoothness별 MAE·상관 (막대)
  [중] 역산 vs 실측 scatter (smoothness별 색)
  [우] 대표 wire 1개의 온도 프로파일 (실측 + 3개 smoothness 역산 곡선)
"""
import os
import os.path as pt
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

CONFIG = {
    'model_dir':  r'./apc_model/full',
    'test_csv':   r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',
    'out_dir':    r'./smoothness_comparison',
    'target':     'avg_bow_bf_total',
    'eqp_col':    'eqp_nm_3200',
    'temp_eqps':  ['BSWS38','BSWS42','BSWS44'],
    'temp_cols':  [f'set_frame_temp_{p}pct' for p in
                   [0,10,20,30,40,50,60,70,80,90,99,100]],
    'temp_pcts':  [0,10,20,30,40,50,60,70,80,90,99,100],
    'temp_rep':   'set_frame_temp_60pct',
    'smooth_levels': [0.0, 0.01, 0.1],   # 비교할 smoothness
    'sample_n':   300,
    'encoding':   'utf-8',
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


def inverse_temp(model, scaler, meta, target_y, base_row, eqp_name, lam):
    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    temp_cols = meta['temp_cols']
    predict = build_predictor(model, scaler, meta, eqp_name)
    def obj(tv):
        loss = (predict(base_row, dict(zip(temp_cols, tv))) - target_y) ** 2
        if lam > 0:
            loss += lam * np.sum(np.diff(tv) ** 2)
        return loss
    x0 = np.array([float(base_row.get(c, X_STATS[c]['mean'])) for c in temp_cols])
    bounds = [(X_STATS[c]['q01'], X_STATS[c]['q99']) for c in temp_cols]
    res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})
    return dict(zip(temp_cols, res.x)), predict(base_row, dict(zip(temp_cols, res.x)))


def compare(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    model, scaler, meta = load_model(cfg['model_dir'])
    test_df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                          encoding_errors='replace')
    EQP, TARGET = cfg['eqp_col'], cfg['target']
    TEMP, REP = cfg['temp_cols'], cfg['temp_rep']
    LEVELS = cfg['smooth_levels']

    # 샘플 수집
    sub_all = []
    for eqp in cfg['temp_eqps']:
        s = test_df[test_df[EQP] == eqp].dropna(subset=[TARGET] + TEMP)
        if len(s) > 0:
            s = s.copy(); s['_eqp'] = eqp
            sub_all.append(s)
    sub_all = pd.concat(sub_all, ignore_index=True)
    if len(sub_all) > cfg['sample_n']:
        sub_all = sub_all.sample(cfg['sample_n'], random_state=42).reset_index(drop=True)

    # 각 smoothness별 역산 → temp60 비교
    records = {lam: {'actual': [], 'rec': []} for lam in LEVELS}
    for _, row in sub_all.iterrows():
        rd = row.to_dict(); eqp = row['_eqp']; ty = float(row[TARGET])
        for lam in LEVELS:
            rec, _ = inverse_temp(model, scaler, meta, ty, rd, eqp, lam)
            records[lam]['actual'].append(float(row[REP]))
            records[lam]['rec'].append(rec[REP])

    # 지표 계산
    metrics = {}
    for lam in LEVELS:
        a = np.array(records[lam]['actual'])
        r = np.array(records[lam]['rec'])
        mae = np.mean(np.abs(r - a))
        corr = stats.pearsonr(a, r)[0] if len(a) > 2 and np.std(r) > 1e-9 else np.nan
        metrics[lam] = {'MAE': mae, 'r': corr}
        print(f"smoothness={lam}: MAE={mae:.3f}, r={corr:.3f}")

    # ── 대표 wire 1개의 전체 프로파일 (3개 smoothness) ──
    rep_row = sub_all.iloc[0]
    rep_rd = rep_row.to_dict(); rep_eqp = rep_row['_eqp']
    rep_ty = float(rep_row[TARGET])
    profiles = {}
    for lam in LEVELS:
        rec, _ = inverse_temp(model, scaler, meta, rep_ty, rep_rd, rep_eqp, lam)
        profiles[lam] = [rec[c] for c in TEMP]
    actual_profile = [rep_row[c] for c in TEMP]

    # ── 시각화 ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle('Smoothness 페널티 영향: 매끄러움 ↔ 실측 근접 트레이드오프',
                 fontsize=13, fontweight='bold')
    palette = {0.0: '#e74c3c', 0.01: '#f39c12', 0.1: '#3498db'}

    # [좌] MAE·r 막대
    x = np.arange(len(LEVELS))
    w = 0.35
    maes = [metrics[l]['MAE'] for l in LEVELS]
    rs = [metrics[l]['r'] for l in LEVELS]
    b1 = ax1.bar(x - w/2, maes, w, label='MAE (낮을수록 근접)',
                 color='#e74c3c', edgecolor='k', linewidth=0.5)
    ax1b = ax1.twinx()
    b2 = ax1b.bar(x + w/2, rs, w, label='상관 r (높을수록 근접)',
                  color='#3498db', edgecolor='k', linewidth=0.5)
    for bar, v in zip(b1, maes):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, v in zip(b2, rs):
        ax1b.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.2f}',
                  ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels([f'λ={l}' for l in LEVELS])
    ax1.set_ylabel('MAE (실측-역산 온도)', color='#e74c3c')
    ax1b.set_ylabel('상관 r', color='#3498db')
    ax1.set_title('① 실측 근접도\nλ 커질수록 멀어짐', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # [중] scatter (smoothness별)
    for lam in LEVELS:
        a = records[lam]['actual']; r = records[lam]['rec']
        ax2.scatter(a, r, s=18, alpha=0.4, color=palette[lam],
                    label=f'λ={lam} (MAE={metrics[lam]["MAE"]:.2f})')
    allv = [v for lam in LEVELS for v in records[lam]['actual']]
    lims = [min(allv)-0.1, max(allv)+0.1]
    ax2.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.5, label='y=x')
    ax2.set_xlabel('실측 온도 (temp60)'); ax2.set_ylabel('역산 온도 (temp60)')
    ax2.set_title('② 역산 vs 실측\nλ=0(빨강)이 y=x에 가장 근접', fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # [우] 대표 wire 프로파일
    POS = cfg['temp_pcts']
    ax3.plot(POS, actual_profile, 'ko-', linewidth=2.2, markersize=6,
             label='실측', zorder=5)
    for lam in LEVELS:
        ax3.plot(POS, profiles[lam], 's--', color=palette[lam], linewidth=1.5,
                 markersize=5, alpha=0.8, label=f'역산 λ={lam}')
    ax3.set_xlabel('Position (%)'); ax3.set_ylabel('set_frame_temp')
    ax3.set_title(f'③ 온도 프로파일 예시 ({rep_eqp})\nλ 커질수록 평평해짐',
                  fontweight='bold')
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    plt.tight_layout()
    fpath = pt.join(cfg['out_dir'], 'smoothness_comparison.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 저장: {fpath}")

    # 지표 CSV
    mdf = pd.DataFrame([{'smoothness': l, **metrics[l]} for l in LEVELS])
    mdf.to_csv(pt.join(cfg['out_dir'], 'smoothness_metrics.csv'),
               index=False, encoding='utf-8-sig')
    print(f"💾 지표: {cfg['out_dir']}/smoothness_metrics.csv")
    return metrics


if __name__ == '__main__':
    compare(CONFIG)
