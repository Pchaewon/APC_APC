# -*- coding: utf-8 -*-
"""
[1/3] BOW 타겟 비교: total vs tail vs seed vs mid
─────────────────────────────────────────
어떤 BOW 위치가 가장 예측 잘 되는지 확인.
같은 feature(temp 12 + tension + wait + ingot + 장비더미)로
타겟만 바꿔가며 랜덤/시간 분할 R² 비교.

inverse와 무관 (예측 정확도 탐색). 가장 예측 잘 되는 타겟을
recipe 추천의 주 타겟으로 쓸지 판단.
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
    'out_dir':    r'./target_comparison',
    'process_time': '13.3Hr',
    'targets': ['avg_bow_bf_total', 'avg_bow_bf_tail',
                'avg_bow_bf_seed', 'avg_bow_bf_mid'],
    'eqp_col':    'eqp_nm_3200',
    'date_col':   'date_3200',
    'leakage': {
        'test_eqps': ['BSWS38','BSWS42','BSWS44','BSWS52',
                      'BSWS54','BSWS55','BSWS56','BSWS61'],
        'test_start': '2026-03-01',
    },
    'recipe_cols': [
        'set_frame_temp_0pct','set_frame_temp_10pct','set_frame_temp_20pct',
        'set_frame_temp_30pct','set_frame_temp_40pct','set_frame_temp_50pct',
        'set_frame_temp_60pct','set_frame_temp_70pct','set_frame_temp_80pct',
        'set_frame_temp_90pct','set_frame_temp_99pct','set_frame_temp_100pct',
        'fdc_set_tension','fdc_wait_time','fdc_ingot_len',
    ],
    'condition_cols': ['range_slurry_temp_10_0'],
    'use_eqp_dummy': True,
    'ridge_alpha': 5.0,
    'split_ratio': 0.8,
    'encoding':   'utf-8',
}


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
    return df


def eval_target(df, target, cfg):
    DATE, EQP = cfg['date_col'], cfg['eqp_col']
    COND = [c for c in cfg['condition_cols'] if c in df.columns]
    base = cfg['recipe_cols'] + COND

    if target not in df.columns:
        return None

    sub = df[base + [target, DATE, EQP]].dropna().copy()
    sub = sub.sort_values(DATE).reset_index(drop=True)

    if cfg['use_eqp_dummy']:
        dummies = pd.get_dummies(sub[EQP], prefix='eqp')
        X_df = pd.concat([sub[base].reset_index(drop=True),
                          dummies.reset_index(drop=True)], axis=1)
    else:
        X_df = sub[base]
    X = X_df.values.astype(float)
    y = sub[target].values

    # 랜덤
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=1-cfg['split_ratio'], random_state=42)
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=cfg['ridge_alpha']).fit(sc.transform(Xtr), ytr)
    r2_rand = r2_score(yte, m.predict(sc.transform(Xte)))

    # 시간
    si = int(len(sub) * cfg['split_ratio'])
    sc2 = StandardScaler().fit(X[:si])
    m2 = Ridge(alpha=cfg['ridge_alpha']).fit(sc2.transform(X[:si]), y[:si])
    pred_t = m2.predict(sc2.transform(X[si:]))
    r2_time = r2_score(y[si:], pred_t)
    mae_time = mean_absolute_error(y[si:], pred_t)

    return {'target': target, 'n': len(sub),
            'y_std': round(float(np.std(y)), 3),
            'r2_random': round(r2_rand, 4),
            'r2_time': round(r2_time, 4),
            'mae_time': round(mae_time, 4)}


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    df = prepare(cfg)
    print(f"[데이터] {len(df)}행\n")

    results = []
    for tgt in cfg['targets']:
        r = eval_target(df, tgt, cfg)
        if r is None:
            print(f"  {tgt}: 컬럼 없음 — 스킵")
            continue
        results.append(r)
        print(f"  {tgt}: 랜덤={r['r2_random']:.3f} | 시간={r['r2_time']:.3f} | "
              f"MAE={r['mae_time']:.3f} | y_std={r['y_std']}")

    res = pd.DataFrame(results)
    res.to_csv(pt.join(cfg['out_dir'], 'target_comparison.csv'),
               index=False, encoding='utf-8-sig')

    # 최고 타겟
    if len(res) > 0:
        best = res.loc[res['r2_time'].idxmax()]
        print(f"\n{'='*56}")
        print(f"시간 분할 최고 타겟: {best['target']} (R²={best['r2_time']:.3f})")
        print(f"현재 사용 중(total)과 비교:")
        total_row = res[res['target'] == 'avg_bow_bf_total']
        if len(total_row) > 0:
            t = total_row.iloc[0]
            print(f"  total: {t['r2_time']:.3f} → {best['target']}: {best['r2_time']:.3f} "
                  f"(Δ {best['r2_time']-t['r2_time']:+.3f})")

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
    ax.set_xticklabels([t.replace('avg_bow_bf_', '') for t in res['target']],
                       fontsize=11)
    ax.set_ylabel('Test R²'); ax.set_xlabel('BOW 타겟 위치')
    ax.set_title('BOW 타겟별 예측 성능 비교', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'target_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
