# -*- coding: utf-8 -*-
"""
[3/3] 요약 feature 탐색: 통계치가 position보다 나은가?
─────────────────────────────────────────
⚠️ inverse 안 함 (요약 feature는 프로파일 복원 불가).
   순수하게 "예측 정확도 상한이 더 있나" 확인용.

비교:
  [A] position 12개 (현재, inverse 가능)
  [B] 요약 통계 (평균/std/기울기/peak위치/범위, inverse 불가)
  [C] position + 요약 (둘 다)

결론 활용:
  · B가 A보다 높으면 → position이 정보 손실? (하지만 inverse 위해 A 유지 불가피)
  · B ≈ A → position 유지가 손해 아님 (정당성 확보)
  · 어느 쪽이든 recipe 추천은 position(A) 유지 필수
"""
import os
import os.path as pt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
    'out_dir':    r'./summary_feature_test',
    'process_time': '13.3Hr',
    'target':     'avg_bow_bf_total',
    'eqp_col':    'eqp_nm_3200',
    'date_col':   'date_3200',
    'leakage': {
        'test_eqps': ['BSWS38','BSWS42','BSWS44','BSWS52',
                      'BSWS54','BSWS55','BSWS56','BSWS61'],
        'test_start': '2026-03-01',
    },
    'temp_cols': [f'set_frame_temp_{p}pct' for p in
                  [0,10,20,30,40,50,60,70,80,90,99,100]],
    'temp_pos': [0,10,20,30,40,50,60,70,80,90,99,100],
    'other_recipe': ['fdc_set_tension','fdc_wait_time','fdc_ingot_len'],
    'condition_cols': ['range_slurry_temp_10_0'],
    'use_eqp_dummy': True,
    'ridge_alpha': 5.0,
    'split_ratio': 0.8,
    'encoding':   'utf-8',
}


def temp_summary(temp_matrix, positions):
    """온도 프로파일 → 통계 요약 feature."""
    pos = np.array(positions)
    feats = {}
    feats['temp_mean'] = temp_matrix.mean(axis=1)
    feats['temp_std'] = temp_matrix.std(axis=1)
    feats['temp_min'] = temp_matrix.min(axis=1)
    feats['temp_max'] = temp_matrix.max(axis=1)
    feats['temp_range'] = feats['temp_max'] - feats['temp_min']
    # 기울기 (선형 회귀 slope)
    slopes = []
    for row in temp_matrix:
        slope = np.polyfit(pos, row, 1)[0]
        slopes.append(slope)
    feats['temp_slope'] = np.array(slopes)
    # peak 위치
    feats['temp_peak_pos'] = pos[np.argmax(temp_matrix, axis=1)]
    # 앞/뒤 평균 차 (경향)
    feats['temp_front_back'] = temp_matrix[:, :6].mean(axis=1) - temp_matrix[:, 6:].mean(axis=1)
    return pd.DataFrame(feats)


def prepare(cfg):
    df = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    if cfg['process_time']:
        df = df[df['process_time'] == cfg['process_time']]
    DATE, EQP = cfg['date_col'], cfg['eqp_col']
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    cutoff = pd.to_datetime(cfg['leakage']['test_start'])
    mask = (df[EQP].isin(cfg['leakage']['test_eqps'])) & (df[DATE] >= cutoff)
    df = df[~mask].copy()

    COND = [c for c in cfg['condition_cols'] if c in df.columns]
    allcols = cfg['temp_cols'] + cfg['other_recipe'] + COND
    sub = df[allcols + [cfg['target'], DATE, EQP]].dropna().copy()
    sub = sub.sort_values(DATE).reset_index(drop=True)
    return sub, COND


def eval_variant(sub, feat_df, cfg):
    """feat_df를 X로, 랜덤/시간 R²."""
    y = sub[cfg['target']].values
    X = feat_df.values.astype(float)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=1-cfg['split_ratio'], random_state=42)
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=cfg['ridge_alpha']).fit(sc.transform(Xtr), ytr)
    r2_rand = r2_score(yte, m.predict(sc.transform(Xte)))

    si = int(len(sub) * cfg['split_ratio'])
    sc2 = StandardScaler().fit(X[:si])
    m2 = Ridge(alpha=cfg['ridge_alpha']).fit(sc2.transform(X[:si]), y[:si])
    pred = m2.predict(sc2.transform(X[si:]))
    r2_time = r2_score(y[si:], pred)
    mae = mean_absolute_error(y[si:], pred)
    return {'n_feat': X.shape[1], 'r2_random': round(r2_rand, 4),
            'r2_time': round(r2_time, 4), 'mae_time': round(mae, 4)}


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    sub, COND = prepare(cfg)
    EQP = cfg['eqp_col']
    print(f"[데이터] {len(sub)}행\n")

    temp_matrix = sub[cfg['temp_cols']].values
    other = sub[cfg['other_recipe'] + COND].reset_index(drop=True)
    dummies = (pd.get_dummies(sub[EQP], prefix='eqp').reset_index(drop=True)
               if cfg['use_eqp_dummy'] else pd.DataFrame(index=sub.index))

    # A: position 12개
    A = pd.concat([sub[cfg['temp_cols']].reset_index(drop=True), other, dummies], axis=1)
    # B: 요약
    summ = temp_summary(temp_matrix, cfg['temp_pos']).reset_index(drop=True)
    B = pd.concat([summ, other, dummies], axis=1)
    # C: 둘 다
    C = pd.concat([sub[cfg['temp_cols']].reset_index(drop=True), summ, other, dummies], axis=1)

    variants = {
        'A. position 12개 (inverse 가능)': A,
        'B. 요약 통계 (inverse 불가)': B,
        'C. position + 요약': C,
    }

    results = []
    for label, X_df in variants.items():
        r = eval_variant(sub, X_df, cfg)
        results.append({'variant': label, **r})
        print(f"  {label}: feature {r['n_feat']}개 | "
              f"랜덤={r['r2_random']:.3f} | 시간={r['r2_time']:.3f} | MAE={r['mae_time']:.3f}")

    res = pd.DataFrame(results)
    res.to_csv(pt.join(cfg['out_dir'], 'summary_feature_test.csv'),
               index=False, encoding='utf-8-sig')

    # 판정
    print(f"\n{'='*56}\n판정\n{'='*56}")
    r_a = res.iloc[0]['r2_time']
    r_b = res.iloc[1]['r2_time']
    r_c = res.iloc[2]['r2_time']
    print(f"  A(position) 시간 R²: {r_a:.3f}")
    print(f"  B(요약) 시간 R²:     {r_b:.3f}  (Δ vs A: {r_b-r_a:+.3f})")
    print(f"  C(둘 다) 시간 R²:    {r_c:.3f}  (Δ vs A: {r_c-r_a:+.3f})")
    print()
    if r_b > r_a + 0.02:
        print("  ℹ 요약이 position보다 나음 → position이 정보 손실 존재")
        print("    단, recipe 추천 위해 position 유지 불가피 (inverse 호환)")
        print("    → 예측 전용 보조 모델로 요약 고려 가능")
    elif abs(r_b - r_a) <= 0.02:
        print("  ✅ 요약 ≈ position → position 유지가 손해 아님 (정당성 확보)")
        print("    → recipe 추천은 position(A)로, 성능 손실 없음")
    else:
        print("  ✅ position이 요약보다 나음 → position 유지가 최선")

    _plot(res, cfg)
    print(f"\n💾 저장: {cfg['out_dir']}/")
    return res


def _plot(res, cfg):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(res))
    w = 0.35
    b1 = ax.bar(x - w/2, res['r2_random'], w, label='랜덤',
                color='#3498db', edgecolor='k', linewidth=0.5)
    b2 = ax.bar(x + w/2, res['r2_time'], w, label='시간 분할',
                color='#e74c3c', edgecolor='k', linewidth=0.5)
    for bars, vals in [(b1, res['r2_random']), (b2, res['r2_time'])]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(0.24, color='green', linestyle='--', alpha=0.6, label='상한 0.24')
    ax.set_xticks(x)
    ax.set_xticklabels(['A.position','B.요약','C.둘다'], fontsize=10)
    ax.set_ylabel('Test R²')
    ax.set_title('Feature 방식별 성능 (inverse는 A만 가능)', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'summary_feature_test.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
