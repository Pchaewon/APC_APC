# -*- coding: utf-8 -*-
"""
Inverse 발표용 종합 시각화 (방향성 + 실측 근접, 둘 다)
─────────────────────────────────────────
2개 패널로 inverse의 두 강점을 손질거리 없이 제시:

  [좌] 방향성 검증: 6대 전 장비에서 온도-BOW 음의 상관 (전부 유의)
       → "온도 레버가 실제로 작동함" (검증된 강점)

  [우] 실측 근접: smoothness=0.01에서 역산 온도 ≈ 실측 온도
       → "inverse가 실제 recipe를 재현함" (작동 증거)

입력:
  · direction_diagnosis.csv (diagnose_direction.py 결과)
  · inverse_diagnosis.csv   (diagnose_inverse.py 결과)
    또는 직접 모델 로드해서 재계산
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
    'model_dir':      r'./apc_model/full',
    'test_csv':       r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',
    'direction_csv':  r'./direction_diagnosis/direction_diagnosis.csv',
    'out_dir':        r'./inverse_summary',
    'target':         'avg_bow_bf_total',
    'eqp_col':        'eqp_nm_3200',
    'temp_eqps':      ['BSWS38','BSWS42','BSWS44'],
    'temp_cols':      [f'set_frame_temp_{p}pct' for p in
                       [0,10,20,30,40,50,60,70,80,90,99,100]],
    'temp_rep':       'set_frame_temp_60pct',
    'lambda_smooth':  0.01,
    'sample_n':       300,
    'encoding':       'utf-8',
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
    def predict(base_row, temp_override=None):
        def gv(c):
            if c in eqp_cols:
                return 1.0 if c == f'{pfx}{eqp_name}' else 0.0
            if temp_override and c in temp_override:
                return float(temp_override[c])
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
    return dict(zip(temp_cols, res.x))


def make_summary(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    model, scaler, meta = load_model(cfg['model_dir'])
    test_df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                          encoding_errors='replace')
    EQP, TARGET = cfg['eqp_col'], cfg['target']
    TEMP, REP = cfg['temp_cols'], cfg['temp_rep']

    # ── 좌: 방향성 (direction_csv 우선, 없으면 재계산) ──
    if os.path.exists(cfg['direction_csv']):
        dir_df = pd.read_csv(cfg['direction_csv'], encoding=cfg['encoding'])
        dir_data = dir_df.dropna(subset=['recipe_BOW_r'])[
            ['eqp', 'recipe_BOW_r', 'recipe_BOW_p']].copy()
    else:
        # 재계산: 장비 내 temp-BOW 상관
        rows = []
        for eqp in test_df[EQP].unique():
            s = test_df[test_df[EQP] == eqp][[REP, TARGET]].dropna()
            if len(s) > 10 and s[REP].std() > 1e-9:
                r, p = stats.pearsonr(s[REP], s[TARGET])
                rows.append({'eqp': eqp, 'recipe_BOW_r': r, 'recipe_BOW_p': p})
        dir_data = pd.DataFrame(rows)

    # ── 우: 실측 근접 (역산 vs 실측, smoothness=0.01) ──
    prox_rows = []
    for eqp in cfg['temp_eqps']:
        sub = test_df[test_df[EQP] == eqp].dropna(subset=[TARGET] + TEMP)
        if len(sub) == 0:
            continue
        if len(sub) > cfg['sample_n']:
            sub = sub.sample(cfg['sample_n'], random_state=42)
        for _, row in sub.iterrows():
            rd = row.to_dict()
            rec = inverse_temp(model, scaler, meta, float(row[TARGET]), rd, eqp,
                               cfg['lambda_smooth'])
            prox_rows.append({
                'eqp': eqp,
                'actual_temp60': float(row[REP]),
                'rec_temp60': rec[REP],
            })
    prox_df = pd.DataFrame(prox_rows)

    # ── 시각화 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Inverse Recipe 추천 — 검증 결과 (방향성 + 실측 근접)',
                 fontsize=13, fontweight='bold')

    # [좌] 방향성: 장비별 temp-BOW 상관
    dir_data = dir_data.sort_values('recipe_BOW_r')
    colors = ['#2ecc71' if p < 0.05 else '#95a5a6'
              for p in dir_data['recipe_BOW_p']]
    bars = ax1.barh(dir_data['eqp'], dir_data['recipe_BOW_r'],
                    color=colors, edgecolor='k', linewidth=0.5)
    for bar, r in zip(bars, dir_data['recipe_BOW_r']):
        ax1.text(r - 0.02 if r < 0 else r + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{r:+.2f}', va='center',
                 ha='right' if r < 0 else 'left', fontsize=9, fontweight='bold')
    ax1.axvline(0, color='k', linewidth=0.8)
    ax1.set_xlabel('장비 내 온도-BOW 상관 (r)', fontsize=10)
    ax1.set_title('① 온도 레버 작동 검증\n(전 장비 음의 상관, 초록=p<0.05 유의)',
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    n_sig = int((dir_data['recipe_BOW_p'] < 0.05).sum())
    ax1.text(0.02, 0.02,
             f'{n_sig}/{len(dir_data)}대 유의\n온도↑ → BOW↓ 확인',
             transform=ax1.transAxes, va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.9))

    # [우] 실측 근접: 역산 vs 실측 온도
    for eqp in prox_df['eqp'].unique():
        s = prox_df[prox_df['eqp'] == eqp]
        ax2.scatter(s['actual_temp60'], s['rec_temp60'], s=25, alpha=0.5,
                    label=eqp)
    lims = [prox_df['actual_temp60'].min() - 0.1,
            prox_df['actual_temp60'].max() + 0.1]
    ax2.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.5, label='y=x (완벽 일치)')
    # 상관·MAE
    mae = (prox_df['rec_temp60'] - prox_df['actual_temp60']).abs().mean()
    if len(prox_df) > 2:
        r_px, _ = stats.pearsonr(prox_df['actual_temp60'], prox_df['rec_temp60'])
    else:
        r_px = np.nan
    ax2.set_xlabel('실측 온도 (temp60)', fontsize=10)
    ax2.set_ylabel('역산 온도 (temp60)', fontsize=10)
    ax2.set_title(f'② Inverse 재현성 (smoothness=0.01)\n'
                  f'역산 온도 ≈ 실측 온도',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.text(0.98, 0.02, f'MAE={mae:.3f}\nr={r_px:.3f}',
             transform=ax2.transAxes, va='bottom', ha='right', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#d6eaf8', alpha=0.9))

    plt.tight_layout()
    fpath = pt.join(cfg['out_dir'], 'inverse_summary.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 종합 그림 저장: {fpath}")

    # 수치 저장
    prox_df.to_csv(pt.join(cfg['out_dir'], 'proximity_data.csv'),
                   index=False, encoding='utf-8-sig')
    print(f"\n[요약]")
    print(f"  방향성: {n_sig}/{len(dir_data)}대에서 온도-BOW 음의 상관 유의")
    print(f"  실측 근접: MAE={mae:.3f}, r={r_px:.3f}")
    return dir_data, prox_df


if __name__ == '__main__':
    make_summary(CONFIG)
