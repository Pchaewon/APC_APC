# -*- coding: utf-8 -*-
"""
직전 WG one-hot feature 효과 검증
─────────────────────────────────────────
질문: WG 구간 정보를 X에 넣으면 성능이 오르나?
      단, 당회 WG는 사후값(배포 불가) → 직전 run 기준으로 써야 배포 가능.

3가지 비교 (모두 시간 분할):
  [A] baseline: WG feature 없음 (현재)
  [B] 직전 WG one-hot: 장비별 직전 run WG를 구간화 (★ 배포 가능)
  [C] 당회 WG one-hot: 이번 lot WG 구간화 (배포 불가, 상한 참고용)

각각 랜덤/시간 분할 Test R² 비교.
B가 A보다 높으면 → 직전 WG가 유효, 배포 feature로 추가
C가 훨씬 높으면 → WG는 강력하나 사후값 → 동적 상태판정으로 우회
"""
import os
import os.path as pt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
    'out_dir':    r'./prev_wg_test',
    'process_time': '13.3Hr',
    'target':     'avg_bow_bf_total',
    'eqp_col':    'eqp_nm_3200',
    'date_col':   'date_3200',
    'wire_col':   'new_fdc_wire_id',
    'wg_col':     'range_wire_guide_10_99',   # 당회 WG (사후값)
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
    'n_wg_bins': 4,           # WG 구간 수
    'split_ratio': 0.8,
    'encoding':   'utf-8',
}


def add_prev_wg(sub, cfg):
    """장비별 직전 run WG 상태 추가 (shift로 사전값화)."""
    EQP = cfg['eqp_col']; DATE = cfg['date_col']; WG = cfg['wg_col']
    sub = sub.sort_values([EQP, DATE]).reset_index(drop=True)
    # 직전 run WG (같은 장비 내 shift(1)) — 이동중앙값(최근 3개)
    sub['prev_wg'] = (sub.groupby(EQP)[WG]
                      .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median()))
    return sub


def make_wg_onehot(values, bins_edges, prefix):
    """WG 값을 구간 one-hot으로."""
    binned = pd.cut(values, bins=bins_edges, labels=False, include_lowest=True)
    oh = pd.get_dummies(binned, prefix=prefix)
    return oh


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
    base = cfg['recipe_cols'] + COND
    WG = cfg['wg_col']
    keep = base + [cfg['target'], DATE, EQP, WG]
    sub = df[keep].dropna(subset=base + [cfg['target'], WG]).copy()

    # 직전 WG 추가
    sub = add_prev_wg(sub, cfg)
    sub = sub.dropna(subset=['prev_wg']).reset_index(drop=True)

    return sub, base


def build_X(sub, base, cfg, wg_mode):
    """wg_mode: 'none' | 'prev' | 'current'"""
    EQP = cfg['eqp_col']
    parts = [sub[base].reset_index(drop=True)]

    # 장비 더미
    if cfg['use_eqp_dummy']:
        parts.append(pd.get_dummies(sub[EQP], prefix='eqp').reset_index(drop=True))

    # WG one-hot
    if wg_mode == 'prev':
        # 직전 WG 구간 (train 기준 경계는 아래서 처리하나, 여기선 전체 분위)
        edges = np.quantile(sub['prev_wg'], np.linspace(0, 1, cfg['n_wg_bins']+1))
        edges[0] -= 1e-6; edges[-1] += 1e-6
        oh = make_wg_onehot(sub['prev_wg'].values, edges, 'prevwg')
        parts.append(oh.reset_index(drop=True))
    elif wg_mode == 'current':
        edges = np.quantile(sub[cfg['wg_col']], np.linspace(0, 1, cfg['n_wg_bins']+1))
        edges[0] -= 1e-6; edges[-1] += 1e-6
        oh = make_wg_onehot(sub[cfg['wg_col']].values, edges, 'curwg')
        parts.append(oh.reset_index(drop=True))

    X = pd.concat(parts, axis=1)
    return X.values.astype(float), list(X.columns)


def eval_split(X, y, sub, cfg):
    """랜덤 + 시간 분할 R²."""
    # 랜덤
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=1-cfg['split_ratio'], random_state=42)
    sc = StandardScaler().fit(Xtr)
    m = LinearRegression().fit(sc.transform(Xtr), ytr)
    r2_rand = r2_score(yte, m.predict(sc.transform(Xte)))

    # 시간 (sub는 이미 장비·날짜 정렬됨 → 날짜만 재정렬)
    order = sub.sort_values(cfg['date_col']).index
    Xo, yo = X[order], y[order]
    si = int(len(yo) * cfg['split_ratio'])
    sc2 = StandardScaler().fit(Xo[:si])
    m2 = LinearRegression().fit(sc2.transform(Xo[:si]), yo[:si])
    r2_time = r2_score(yo[si:], m2.predict(sc2.transform(Xo[si:])))

    return r2_rand, r2_time


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    sub, base = prepare(cfg)
    y = sub[cfg['target']].values
    print(f"[데이터] {len(sub)}행 (직전 WG 계산 후)")

    modes = {
        'A. baseline (WG 없음)':      'none',
        'B. 직전 WG one-hot (배포 가능)': 'prev',
        'C. 당회 WG one-hot (배포 불가)': 'current',
    }

    results = []
    for label, mode in modes.items():
        X, feats = build_X(sub, base, cfg, mode)
        r2_rand, r2_time = eval_split(X, y, sub, cfg)
        results.append({'mode': label, 'n_features': len(feats),
                        'r2_random': round(r2_rand, 4),
                        'r2_time': round(r2_time, 4)})
        print(f"  {label}: feature {len(feats)}개 | "
              f"랜덤={r2_rand:.3f} | 시간={r2_time:.3f}")

    res = pd.DataFrame(results)
    res.to_csv(pt.join(cfg['out_dir'], 'prev_wg_test.csv'),
               index=False, encoding='utf-8-sig')

    # 판정
    print(f"\n{'='*56}\n판정\n{'='*56}")
    r_a = res.iloc[0]['r2_time']
    r_b = res.iloc[1]['r2_time']
    r_c = res.iloc[2]['r2_time']
    print(f"  A(없음) 시간 R²:      {r_a:.3f}")
    print(f"  B(직전 WG) 시간 R²:   {r_b:.3f}  (Δ vs A: {r_b-r_a:+.3f})")
    print(f"  C(당회 WG) 시간 R²:   {r_c:.3f}  (Δ vs A: {r_c-r_a:+.3f})")
    print()
    if r_b - r_a > 0.02:
        print("  ✅ 직전 WG one-hot이 성능 개선 → 배포 feature로 추가 권장")
    else:
        print("  ⚠ 직전 WG one-hot 효과 미미 → 직전 WG는 예측력 약함")
    if r_c - r_b > 0.1:
        print("  ℹ 당회 WG는 강력하나 사후값 → 동적 상태판정으로 우회가 정답")

    _plot(res, cfg)
    print(f"\n💾 저장: {cfg['out_dir']}/")
    return res


def _plot(res, cfg):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(res))
    w = 0.35
    b1 = ax.bar(x - w/2, res['r2_random'], w, label='랜덤 분할',
                color='#3498db', edgecolor='k', linewidth=0.5)
    b2 = ax.bar(x + w/2, res['r2_time'], w, label='시간 분할 (배포)',
                color='#e74c3c', edgecolor='k', linewidth=0.5)
    for bars, vals in [(b1, res['r2_random']), (b2, res['r2_time'])]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    v+0.005 if v > 0 else v-0.02, f'{v:.3f}',
                    ha='center', va='bottom' if v > 0 else 'top',
                    fontsize=9, fontweight='bold')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axhline(0.24, color='green', linestyle='--', linewidth=1, alpha=0.6,
               label='상한 0.24')
    ax.set_xticks(x)
    ax.set_xticklabels(res['mode'], fontsize=9, rotation=10)
    ax.set_ylabel('Test R²')
    ax.set_title('직전 WG one-hot feature 효과 (시간 분할이 배포 성능)',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'prev_wg_test.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
