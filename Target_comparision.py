# -*- coding: utf-8 -*-
"""
타겟 BOW 비교: total vs tail vs seed vs mid
─────────────────────────────────────────
어떤 BOW 타겟이 가장 예측 잘 되는지 (R² 상한이 더 높은지) 확인.
블록 위치(seed=시작, mid=중간, tail=끝)별로 온도-BOW 관계 강도가 다를 수 있음.

각 타겟에 대해:
  · Ridge 랜덤/시간 분할 Test R²
  · 온도 계수 부호 (공정지식 일치 여부)
  · 재현성 진단 (유사조건 Y차이 → 상한 추정)
"""
import os
import os.path as pt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
    'out_dir':    r'./target_comparison',
    'process_time': '13.3Hr',
    'targets':    ['avg_bow_bf_total', 'avg_bow_bf_tail',
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
    'ridge_alpha': 5.0,
    'split_ratio': 0.8,
    'ceiling_k': 5,        # 재현성 진단 이웃 수
    'encoding':   'utf-8',
}


def load(cfg):
    df = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    if cfg['process_time']:
        df = df[df['process_time'] == cfg['process_time']]
    DATE, EQP = cfg['date_col'], cfg['eqp_col']
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    cutoff = pd.to_datetime(cfg['leakage']['test_start'])
    mask = (df[EQP].isin(cfg['leakage']['test_eqps'])) & (df[DATE] >= cutoff)
    return df[~mask].copy()


def estimate_ceiling(X, y, k):
    """재현성 진단: 유사조건 이웃의 Y 차이로 R² 상한 추정."""
    Xs = StandardScaler().fit_transform(X)
    tree = cKDTree(Xs)
    _, idx = tree.query(Xs, k=k+1)
    neighbor_var = []
    for i in range(len(y)):
        neighbors = idx[i][1:]  # 자기 제외
        neighbor_var.append(np.var(y[neighbors]))
    noise_var = np.mean(neighbor_var)   # 설명 불가 분산
    total_var = np.var(y)
    ceiling = 1 - noise_var / total_var
    return max(ceiling, 0), noise_var, total_var


def eval_target(df, target, cfg):
    EQP, DATE = cfg['eqp_col'], cfg['date_col']
    COND = [c for c in cfg['condition_cols'] if c in df.columns]
    base = cfg['recipe_cols'] + COND

    sub = df[base + [target, DATE, EQP]].dropna().copy()
    if len(sub) < 100:
        return None

    # 장비 더미
    dummies = pd.get_dummies(sub[EQP], prefix='eqp')
    X_df = pd.concat([sub[base].reset_index(drop=True),
                      dummies.reset_index(drop=True)], axis=1)
    FEATURES = list(X_df.columns)
    X = X_df.values.astype(float)
    y = sub[target].values

    # 랜덤
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=1-cfg['split_ratio'], random_state=42)
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=cfg['ridge_alpha']).fit(sc.transform(Xtr), ytr)
    r2_rand = r2_score(yte, m.predict(sc.transform(Xte)))

    # 시간
    sub_t = sub.sort_values(DATE).reset_index(drop=True)
    Xt = X_df.iloc[sub_t.index].values.astype(float) if False else None
    # 재구성 (정렬 인덱스로)
    order = sub.sort_values(DATE).index
    pos = {idx: p for p, idx in enumerate(sub.index)}
    ord_pos = [pos[i] for i in order]
    Xo, yo = X[ord_pos], y[ord_pos]
    si = int(len(yo) * cfg['split_ratio'])
    sc2 = StandardScaler().fit(Xo[:si])
    m2 = Ridge(alpha=cfg['ridge_alpha']).fit(sc2.transform(Xo[:si]), yo[:si])
    r2_time = r2_score(yo[si:], m2.predict(sc2.transform(Xo[si:])))

    # 온도 계수 부호 (temp_60pct)
    coef60 = dict(zip(FEATURES, m.coef_)).get('set_frame_temp_60pct', np.nan)

    # 재현성 상한 (recipe만으로, 샘플 제한)
    samp = min(3000, len(sub))
    ridx = np.random.RandomState(0).choice(len(sub), samp, replace=False)
    ceiling, nv, tv = estimate_ceiling(sub[base].values[ridx],
                                        y[ridx], cfg['ceiling_k'])

    return {
        'target': target,
        'n': len(sub),
        'y_std': round(float(np.std(y)), 4),
        'r2_random': round(r2_rand, 4),
        'r2_time': round(r2_time, 4),
        'temp60_coef': round(float(coef60), 4),
        'temp60_sign': '↓BOW ✓' if coef60 < 0 else '↑BOW ✗',
        'ceiling_est': round(float(ceiling), 4),
    }


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    df = load(cfg)
    print(f"[데이터] {len(df)}행\n")

    results = []
    for target in cfg['targets']:
        if target not in df.columns:
            print(f"  ⚠ {target} 없음 — 스킵")
            continue
        r = eval_target(df, target, cfg)
        if r:
            results.append(r)
            print(f"  {target}: 랜덤={r['r2_random']:.3f} | 시간={r['r2_time']:.3f} "
                  f"| 상한추정={r['ceiling_est']:.3f} | temp60={r['temp60_sign']}")

    res = pd.DataFrame(results)
    res.to_csv(pt.join(cfg['out_dir'], 'target_comparison.csv'),
               index=False, encoding='utf-8-sig')

    # 판정
    print(f"\n{'='*60}\n판정\n{'='*60}")
    best = res.loc[res['r2_time'].idxmax()]
    print(f"  시간분할 최고 타겟: {best['target']} (R²={best['r2_time']:.3f})")
    cur = res[res['target'] == 'avg_bow_bf_total']
    if len(cur) > 0:
        cur_r2 = cur.iloc[0]['r2_time']
        if best['r2_time'] - cur_r2 > 0.03:
            print(f"  → {best['target']}이 현재 total({cur_r2:.3f})보다 "
                  f"+{best['r2_time']-cur_r2:.3f} 높음. 타겟 변경 검토")
        else:
            print(f"  → total과 큰 차이 없음. 현재 타겟 유지 무방")

    _plot(res, cfg)
    print(f"\n💾 저장: {cfg['out_dir']}/")
    return res


def _plot(res, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(res))
    w = 0.35
    ax1.bar(x - w/2, res['r2_random'], w, label='랜덤', color='#3498db',
            edgecolor='k', linewidth=0.5)
    ax1.bar(x + w/2, res['r2_time'], w, label='시간분할', color='#e74c3c',
            edgecolor='k', linewidth=0.5)
    ax1.plot(x, res['ceiling_est'], 'g^--', label='상한추정', markersize=10)
    ax1.axhline(0, color='k', linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace('avg_bow_bf_', '') for t in res['target']])
    ax1.set_ylabel('R²'); ax1.set_title('타겟별 성능 + 상한추정', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)

    # 상한 대비 달성률
    res['achieve'] = res['r2_time'] / res['ceiling_est'].replace(0, np.nan)
    ax2.bar(x, res['achieve']*100, color='#9b59b6', edgecolor='k', linewidth=0.5)
    for i, v in enumerate(res['achieve']*100):
        if pd.notna(v):
            ax2.text(i, v+1, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace('avg_bow_bf_', '') for t in res['target']])
    ax2.set_ylabel('상한 대비 달성률 (%)')
    ax2.set_title('시간분할 R² / 상한추정', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'target_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
