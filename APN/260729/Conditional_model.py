# -*- coding: utf-8 -*-
"""
[2/3] 조건부 모델: WG 상태별로 "될 때만 예측"
─────────────────────────────────────────
전체 R²는 상한(0.24)에 막히지만, WG 상태에 따라
온도-BOW 관계 강도가 다름 → 관계 뚜렷한 구간만 예측하면
그 부분집합 R²는 더 높음 (커버리지↓ 정확도↑).

방식:
  · 직전 WG 상태로 데이터를 구간 분할 (배포 가능)
  · 각 구간에서 별도 R² 측정 (position 유지 → inverse 호환)
  · "고신뢰 구간"의 R²와 커버리지 제시

배포 활용: 고신뢰 구간에서만 적극 추천, 나머지는 표준 유지.
"""
import os
import os.path as pt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
    'out_dir':    r'./conditional_model',
    'process_time': '13.3Hr',
    'target':     'avg_bow_bf_total',
    'eqp_col':    'eqp_nm_3200',
    'date_col':   'date_3200',
    'wg_col':     'range_wire_guide_10_99',
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
    # 직전 WG 상태 판정: 이동중앙값 임계 (HIGH_VAR/LOW_VAR)
    'wg_var_threshold': 11.6,   # 이전 분석의 HIGH/LOW 경계
    'wg_roll_k': 3,
    'encoding':   'utf-8',
}


def prepare(cfg):
    df = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    if cfg['process_time']:
        df = df[df['process_time'] == cfg['process_time']]
    DATE, EQP, WG = cfg['date_col'], cfg['eqp_col'], cfg['wg_col']
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    cutoff = pd.to_datetime(cfg['leakage']['test_start'])
    mask = (df[EQP].isin(cfg['leakage']['test_eqps'])) & (df[DATE] >= cutoff)
    df = df[~mask].copy()

    COND = [c for c in cfg['condition_cols'] if c in df.columns]
    base = cfg['recipe_cols'] + COND
    sub = df[base + [cfg['target'], DATE, EQP, WG]].dropna(
        subset=base + [cfg['target'], WG]).copy()
    sub = sub.sort_values([EQP, DATE]).reset_index(drop=True)

    # 직전 WG 상태 (이동중앙값, shift로 사전값화)
    sub['prev_wg'] = (sub.groupby(EQP)[WG]
                      .transform(lambda s: s.shift(1)
                                 .rolling(cfg['wg_roll_k'], min_periods=1).median()))
    sub = sub.dropna(subset=['prev_wg']).reset_index(drop=True)
    # 상태 라벨
    sub['wg_state'] = np.where(sub['prev_wg'] >= cfg['wg_var_threshold'],
                               'HIGH_VAR', 'LOW_VAR')
    return sub, base


def fit_predict(sub_tr, sub_te, base, cfg):
    """train으로 학습, test 예측 → R², MAE."""
    EQP = cfg['eqp_col']; TARGET = cfg['target']
    if cfg['use_eqp_dummy']:
        # 더미는 train 기준으로 맞춤
        dtr = pd.get_dummies(sub_tr[EQP], prefix='eqp')
        dte = pd.get_dummies(sub_te[EQP], prefix='eqp')
        dte = dte.reindex(columns=dtr.columns, fill_value=0)
        Xtr = pd.concat([sub_tr[base].reset_index(drop=True),
                         dtr.reset_index(drop=True)], axis=1).values.astype(float)
        Xte = pd.concat([sub_te[base].reset_index(drop=True),
                         dte.reset_index(drop=True)], axis=1).values.astype(float)
    else:
        Xtr = sub_tr[base].values.astype(float)
        Xte = sub_te[base].values.astype(float)
    ytr = sub_tr[TARGET].values
    yte = sub_te[TARGET].values

    if len(ytr) < 30 or len(yte) < 10:
        return None
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=cfg['ridge_alpha']).fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    return {'r2': r2_score(yte, pred),
            'mae': mean_absolute_error(yte, pred),
            'n_train': len(ytr), 'n_test': len(yte)}


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    sub, base = prepare(cfg)
    DATE = cfg['date_col']
    print(f"[데이터] {len(sub)}행")
    print(f"[WG 상태] HIGH_VAR={sum(sub['wg_state']=='HIGH_VAR')}, "
          f"LOW_VAR={sum(sub['wg_state']=='LOW_VAR')}\n")

    # 시간 분할
    sub_t = sub.sort_values(DATE).reset_index(drop=True)
    si = int(len(sub_t) * cfg['split_ratio'])
    tr, te = sub_t.iloc[:si], sub_t.iloc[si:]

    results = []

    # ① 전체 (baseline)
    r_all = fit_predict(tr, te, base, cfg)
    if r_all:
        results.append({'segment': '전체 (baseline)', 'coverage': 100.0, **r_all})
        print(f"  전체: R²={r_all['r2']:.3f}, MAE={r_all['mae']:.3f} "
              f"(n_test={r_all['n_test']})")

    # ② WG 상태별 (같은 모델, test만 상태별로 나눠 평가)
    #    → "고신뢰 구간에서 예측이 더 정확한가"
    for state in ['HIGH_VAR', 'LOW_VAR']:
        te_s = te[te['wg_state'] == state]
        if len(te_s) < 10:
            print(f"  {state}: test 샘플 부족")
            continue
        # 전체 train으로 학습, 해당 상태 test만 평가
        r = fit_predict(tr, te_s, base, cfg)
        if r:
            cov = len(te_s) / len(te) * 100
            results.append({'segment': f'{state} (test)', 'coverage': round(cov, 1),
                            **r})
            print(f"  {state}: R²={r['r2']:.3f}, MAE={r['mae']:.3f}, "
                  f"커버리지={cov:.0f}% (n_test={r['n_test']})")

    # ③ 상태별 전용 모델 (해당 상태 train으로만 학습)
    print(f"\n  [상태 전용 모델]")
    for state in ['HIGH_VAR', 'LOW_VAR']:
        tr_s = tr[tr['wg_state'] == state]
        te_s = te[te['wg_state'] == state]
        if len(tr_s) < 30 or len(te_s) < 10:
            continue
        r = fit_predict(tr_s, te_s, base, cfg)
        if r:
            cov = len(te_s) / len(te) * 100
            results.append({'segment': f'{state} 전용', 'coverage': round(cov, 1),
                            **r})
            print(f"  {state} 전용: R²={r['r2']:.3f}, MAE={r['mae']:.3f}")

    res = pd.DataFrame(results)
    res.to_csv(pt.join(cfg['out_dir'], 'conditional_model.csv'),
               index=False, encoding='utf-8-sig')

    # 판정
    print(f"\n{'='*56}\n판정\n{'='*56}")
    base_r2 = res[res['segment'] == '전체 (baseline)']['r2'].values
    high_r2 = res[res['segment'] == 'HIGH_VAR (test)']['r2'].values
    if len(base_r2) and len(high_r2):
        print(f"  전체 R²: {base_r2[0]:.3f}")
        print(f"  HIGH_VAR R²: {high_r2[0]:.3f} (Δ {high_r2[0]-base_r2[0]:+.3f})")
        if high_r2[0] - base_r2[0] > 0.05:
            hc = res[res['segment']=='HIGH_VAR (test)']['coverage'].values[0]
            print(f"  ✅ HIGH_VAR 구간에서 예측 더 정확 → 이 구간({hc:.0f}%) 집중 추천")
        else:
            print(f"  ⚠ 상태별 차이 미미 → 조건부 이득 작음")

    _plot(res, cfg)
    print(f"\n💾 저장: {cfg['out_dir']}/")
    return res


def _plot(res, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ['#95a5a6' if 'baseline' in s else
              '#2ecc71' if 'HIGH' in s else '#e67e22' for s in res['segment']]
    # R²
    ax1.barh(res['segment'], res['r2'], color=colors, edgecolor='k', linewidth=0.5)
    ax1.axvline(0.24, color='green', linestyle='--', alpha=0.6, label='상한 0.24')
    ax1.set_xlabel('Test R²'); ax1.set_title('구간별 예측 R²', fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(axis='x', alpha=0.3)
    for i, (r2, cov) in enumerate(zip(res['r2'], res['coverage'])):
        ax1.text(r2, i, f' {r2:.3f}', va='center', fontsize=8)
    # 커버리지 vs R²
    ax2.scatter(res['coverage'], res['r2'], s=100, c=colors, edgecolor='k')
    for _, row in res.iterrows():
        ax2.annotate(row['segment'].replace(' (test)','').replace(' (baseline)',''),
                     (row['coverage'], row['r2']), fontsize=7,
                     ha='center', va='bottom')
    ax2.set_xlabel('커버리지 (%)'); ax2.set_ylabel('Test R²')
    ax2.set_title('커버리지 vs 정확도 트레이드오프', fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'conditional_model.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
