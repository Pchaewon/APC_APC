# -*- coding: utf-8 -*-
"""
Inverse 결과 진단 — 예측 온도가 실측보다 낮게 나오는 원인 규명
─────────────────────────────────────────
확인 사항:
  ① gap 분포: 목표 BOW를 실제로 맞췄나? (gap≈0이면 맞춘 것)
  ② 예측 온도 vs 실측 온도 차이 분포 (얼마나, 어느 방향으로 벗어났나)
  ③ 원인 분해: smoothness 영향인지 확인
     - smoothness=0으로 재역산 → 실측에 가까워지면 smoothness가 원인
  ④ 다중해 문제 확인: 온도를 실측/역산 넣었을 때 BOW 예측이 같은가?
     - 같다면 "온도가 달라도 같은 BOW" = 온도-BOW 관계 약함(다중해)
"""
import os
import os.path as pt
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8

CONFIG = {
    'model_dir': r'./apc_model/full',       # 또는 apc_model_r/full
    'test_csv':  r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',
    'out_dir':   r'./inverse_diagnosis',
    'target':    'avg_bow_bf_total',
    'eqp_col':   'eqp_nm_3200',
    'temp_eqps': ['BSWS38','BSWS42','BSWS44'],   # 15-19년식 temp
    'temp_cols': [f'set_frame_temp_{p}pct' for p in
                  [0,10,20,30,40,50,60,70,80,90,99,100]],
    'temp_rep':  'set_frame_temp_60pct',
    'encoding':  'utf-8',
    'sample_n':  200,   # 진단용 샘플 (너무 많으면 느림)
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
    """특정 장비의 예측 함수 (더미 세팅 포함)."""
    FEATURES = meta['feature_cols']
    X_STATS = meta['x_stats']
    eqp_cols = meta.get('eqp_cols', [])
    eqp_prefix = meta.get('eqp_prefix', 'eqp_')

    def predict(base_row, temp_override=None):
        def gv(c):
            if c in eqp_cols:
                return 1.0 if c == f'{eqp_prefix}{eqp_name}' else 0.0
            if temp_override is not None and c in temp_override:
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


def inverse_temp(model, scaler, meta, target_y, base_row, eqp_name,
                 lambda_smooth=0.1):
    """온도 12개 역산 (smoothness 조절 가능)."""
    FEATURES = meta['feature_cols']
    X_STATS = meta['x_stats']
    temp_cols = meta['temp_cols']
    predict = build_predictor(model, scaler, meta, eqp_name)

    def objective(tv):
        override = dict(zip(temp_cols, tv))
        loss = (predict(base_row, override) - target_y) ** 2
        if lambda_smooth > 0:
            loss += lambda_smooth * np.sum(np.diff(tv) ** 2)
        return loss

    x0 = np.array([float(base_row.get(c, X_STATS[c]['mean'])) for c in temp_cols])
    bounds = [(X_STATS[c]['q01'], X_STATS[c]['q99']) for c in temp_cols]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})
    rec = dict(zip(temp_cols, res.x))
    y_pred = predict(base_row, rec)
    return rec, y_pred


def diagnose(cfg):
    model, scaler, meta = load_model(cfg['model_dir'])
    test_df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                          encoding_errors='replace')
    os.makedirs(cfg['out_dir'], exist_ok=True)
    EQP, TARGET = cfg['eqp_col'], cfg['target']
    TEMP = cfg['temp_cols']; REP = cfg['temp_rep']

    rows = []
    for eqp in cfg['temp_eqps']:
        sub = test_df[test_df[EQP] == eqp].dropna(subset=[TARGET] + TEMP)
        if len(sub) == 0:
            continue
        if len(sub) > cfg['sample_n']:
            sub = sub.sample(cfg['sample_n'], random_state=42)

        predict = build_predictor(model, scaler, meta, eqp)

        for _, row in sub.iterrows():
            rd = row.to_dict()
            target_y = float(row[TARGET])

            # ── ④ 다중해 확인: 실측 온도로 예측한 BOW ──
            bow_at_actual = predict(rd)   # 실측 온도 그대로

            # ── 역산 (smoothness 0.1) ──
            rec_s, ypred_s = inverse_temp(model, scaler, meta, target_y, rd, eqp,
                                          lambda_smooth=0.1)
            # ── 역산 (smoothness 0) ──
            rec_0, ypred_0 = inverse_temp(model, scaler, meta, target_y, rd, eqp,
                                          lambda_smooth=0.0)

            rows.append({
                'eqp': eqp,
                'target_bow': round(target_y, 3),
                'bow_at_actual_temp': round(bow_at_actual, 3),   # 실측온도→BOW
                'bow_pred_smooth01': round(ypred_s, 3),
                'bow_pred_smooth0':  round(ypred_0, 3),
                'gap_smooth01': round(abs(ypred_s - target_y), 3),
                'gap_smooth0':  round(abs(ypred_0 - target_y), 3),
                # 대표 위치 온도 비교
                'actual_temp60': round(float(row[REP]), 3),
                'rec_temp60_smooth01': round(rec_s[REP], 3),
                'rec_temp60_smooth0':  round(rec_0[REP], 3),
                # 전체 온도 평균 차이
                'actual_temp_mean': round(np.mean([row[c] for c in TEMP]), 3),
                'rec_temp_mean_s01': round(np.mean(list(rec_s.values())), 3),
                'rec_temp_mean_s0':  round(np.mean(list(rec_0.values())), 3),
            })

    res = pd.DataFrame(rows)
    res.to_csv(pt.join(cfg['out_dir'], 'inverse_diagnosis.csv'),
               index=False, encoding='utf-8-sig')

    # ── 종합 판정 ──
    print(f"\n{'='*64}\nInverse 진단 종합\n{'='*64}")

    # ① gap
    print(f"\n① 목표 BOW 달성 (gap):")
    print(f"   smoothness=0.1 평균 gap: {res['gap_smooth01'].mean():.4f}")
    print(f"   smoothness=0   평균 gap: {res['gap_smooth0'].mean():.4f}")

    # ② 실측온도로 예측한 BOW vs 목표 BOW (모델 자체 재현력)
    diff_actual = (res['bow_at_actual_temp'] - res['target_bow']).abs()
    print(f"\n② 실측 온도로 예측한 BOW vs 실제 BOW:")
    print(f"   평균 차이: {diff_actual.mean():.4f}")
    print(f"   → 이게 크면 모델이 실측 자체를 재현 못 함 (R² 한계)")

    # ③ 온도 차이 (역산 - 실측)
    print(f"\n③ 역산 온도 vs 실측 온도 (temp60 기준):")
    d_s01 = res['rec_temp60_smooth01'] - res['actual_temp60']
    d_s0  = res['rec_temp60_smooth0'] - res['actual_temp60']
    print(f"   smoothness=0.1: mean={d_s01.mean():+.3f}, |mean|={d_s01.abs().mean():.3f}")
    print(f"   smoothness=0  : mean={d_s0.mean():+.3f}, |mean|={d_s0.abs().mean():.3f}")

    # ④ 다중해 판정
    print(f"\n④ 원인 판정:")
    if res['gap_smooth0'].mean() < 0.05:
        # gap이 작은데 온도가 다르다 = 다중해
        if d_s0.abs().mean() > 0.1:
            print("   ⚠ 다중해: 목표 BOW는 맞추지만(gap≈0) 온도는 실측과 다름")
            print("      → 온도-BOW 관계가 약해(R²낮음) 여러 온도가 같은 BOW를 냄")
            print("      → inverse가 유일한 정답을 못 찾음 (구조적 한계)")
    if abs(d_s01.mean() - d_s0.mean()) > 0.1:
        print("   ⚠ smoothness 영향: smoothness 유무로 온도가 크게 달라짐")
        print("      → lambda_smooth를 낮추면 실측에 가까워짐")
    else:
        print("   · smoothness 영향은 작음 (온도 차이는 다중해가 주원인)")

    # ── 시각화 ──
    _plot(res, cfg['out_dir'])
    print(f"\n💾 저장: {cfg['out_dir']}/")
    return res


def _plot(res, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ① gap 히스토그램
    ax = axes[0]
    ax.hist(res['gap_smooth01'], bins=30, alpha=0.6, color='#3498db',
            label='smooth=0.1', edgecolor='k', linewidth=0.3)
    ax.hist(res['gap_smooth0'], bins=30, alpha=0.6, color='#e74c3c',
            label='smooth=0', edgecolor='k', linewidth=0.3)
    ax.set_xlabel('gap (|예측BOW - 목표BOW|)'); ax.set_ylabel('Count')
    ax.set_title('① 목표 BOW 달성도\n(0에 가까울수록 좋음)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ② 실측온도→BOW vs 목표BOW
    ax = axes[1]
    ax.scatter(res['target_bow'], res['bow_at_actual_temp'], s=20, alpha=0.5,
               color='#9b59b6')
    lims = [res['target_bow'].min(), res['target_bow'].max()]
    ax.plot(lims, lims, 'k--', alpha=0.6, label='y=x (완벽 재현)')
    ax.set_xlabel('실제 BOW'); ax.set_ylabel('실측온도로 예측한 BOW')
    ax.set_title('② 모델의 실측 재현력\n(R² 한계 확인)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ③ 역산온도 vs 실측온도
    ax = axes[2]
    ax.scatter(res['actual_temp60'], res['rec_temp60_smooth0'], s=20, alpha=0.5,
               color='#e74c3c', label='smooth=0')
    ax.scatter(res['actual_temp60'], res['rec_temp60_smooth01'], s=20, alpha=0.5,
               color='#3498db', label='smooth=0.1')
    lims = [res['actual_temp60'].min(), res['actual_temp60'].max()]
    ax.plot(lims, lims, 'k--', alpha=0.6, label='y=x')
    ax.set_xlabel('실측 temp60'); ax.set_ylabel('역산 temp60')
    ax.set_title('③ 역산 vs 실측 온도\n(y=x에서 벗어난 정도)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(pt.join(out_dir, 'inverse_diagnosis.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 진단 그림 저장")


if __name__ == '__main__':
    diagnose(CONFIG)
