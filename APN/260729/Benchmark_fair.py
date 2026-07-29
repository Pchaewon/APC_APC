# -*- coding: utf-8 -*-
"""
공정한 TabPFN 벤치마크
─────────────────────────────────────────
기존 benchmark의 두 문제 수정:
  1. 시간분할 서브샘플: 랜덤 → 최근 N개 (시간 순서 보존)
  2. 장비 더미 유무 비교: TabPFN은 희소 더미에 약할 수 있음

각 모델 × {장비더미 O/X} × {랜덤/시간분할} Test R² 비교.
"""
import os
import os.path as pt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
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
    'out_dir':    r'./benchmark_fair',
    'process_time': '13.3Hr',
    'target':     'avg_bow_bf_total',
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
    'split_ratio': 0.8,
    'tabpfn_max': 3000,
    'r2_ceiling': 0.24,
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

    COND = [c for c in cfg['condition_cols'] if c in df.columns]
    base = cfg['recipe_cols'] + COND
    sub = df[base + [cfg['target'], DATE, EQP]].dropna().copy()
    sub = sub.sort_values(DATE).reset_index(drop=True)
    print(f"[데이터] {len(sub)}행 (leakage 제거, 시간 정렬)")
    return sub, base


def get_splits(sub, base, cfg, use_eqp):
    """장비더미 유무에 따라 X 구성 + 랜덤/시간 분할 반환."""
    EQP = cfg['eqp_col']; TARGET = cfg['target']
    if use_eqp:
        dummies = pd.get_dummies(sub[EQP], prefix='eqp')
        X_df = pd.concat([sub[base], dummies], axis=1)
        FEATURES = list(X_df.columns)
    else:
        X_df = sub[base]
        FEATURES = base

    X = X_df.values.astype(float)
    y = sub[TARGET].values

    # 랜덤
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X, y, test_size=1-cfg['split_ratio'], random_state=42)
    # 시간 (이미 정렬됨)
    si = int(len(sub) * cfg['split_ratio'])
    Xtr_t, Xte_t = X[:si], X[si:]
    ytr_t, yte_t = y[:si], y[si:]

    return (Xtr_r, Xte_r, ytr_r, yte_r), (Xtr_t, Xte_t, ytr_t, yte_t), FEATURES


def fit_eval(model_fn, Xtr, Xte, ytr, yte, use_scaler, is_tabpfn, cfg,
             time_split=False):
    """단일 모델 학습·평가. TabPFN은 샘플 제한."""
    try:
        # TabPFN 샘플 제한
        if is_tabpfn and len(ytr) > cfg['tabpfn_max']:
            n = cfg['tabpfn_max']
            if time_split:
                # ★ 시간분할: 최근 N개 (랜덤 아님)
                Xtr, ytr = Xtr[-n:], ytr[-n:]
            else:
                idx = np.random.RandomState(42).choice(len(ytr), n, replace=False)
                Xtr, ytr = Xtr[idx], ytr[idx]

        if use_scaler:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = model_fn()
        m.fit(Xtr, ytr)
        return r2_score(yte, m.predict(Xte)), None
    except Exception as e:
        return None, str(e)


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    sub, base = prepare(cfg)

    # 모델 정의
    models = {
        'LinearRegression': (lambda: LinearRegression(), True, False),
        'Ridge(a=5)':       (lambda: Ridge(alpha=5.0), True, False),
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = (
            lambda: XGBRegressor(n_estimators=100, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 random_state=42), False, False)
    except ImportError:
        print("  ⚠ XGBoost 미설치")
    tabpfn_ok = False
    try:
        from tabpfn import TabPFNRegressor
        models['TabPFN'] = (lambda: TabPFNRegressor(), True, True)
        tabpfn_ok = True
    except ImportError:
        print("  ⚠ TabPFN 미설치")

    # 장비더미 O/X 두 경우
    results = []
    for use_eqp in [True, False]:
        tag = '장비더미O' if use_eqp else '장비더미X'
        (rand, time_, FEATURES) = get_splits(sub, base, cfg, use_eqp)
        print(f"\n[{tag}] feature {len(FEATURES)}개")

        for name, (fn, scaler, is_tab) in models.items():
            # 랜덤
            r2_rand, e1 = fit_eval(fn, rand[0], rand[1], rand[2], rand[3],
                                   scaler, is_tab, cfg, time_split=False)
            # 시간
            r2_time, e2 = fit_eval(fn, time_[0], time_[1], time_[2], time_[3],
                                   scaler, is_tab, cfg, time_split=True)
            if e1 or e2:
                print(f"  {name}: 오류 {e1 or e2}")
                continue
            results.append({
                'model': name, 'eqp_dummy': tag,
                'r2_random': round(r2_rand, 4),
                'r2_time': round(r2_time, 4),
            })
            print(f"  {name}: 랜덤={r2_rand:.3f} | 시간={r2_time:.3f}")

    res = pd.DataFrame(results)
    res.to_csv(pt.join(cfg['out_dir'], 'benchmark_fair.csv'),
               index=False, encoding='utf-8-sig')
    _plot(res, cfg)
    print(f"\n💾 저장: {cfg['out_dir']}/")

    # 핵심 관찰
    print(f"\n[핵심]")
    print(f"  · 시간분할 서브샘플을 '최근 N개'로 수정 → TabPFN 공정 평가")
    print(f"  · 장비더미 O/X 비교 → TabPFN이 더미 없이 나은지 확인")
    return res


def _plot(res, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, split in zip(axes, ['r2_random', 'r2_time']):
        pivot = res.pivot(index='model', columns='eqp_dummy', values=split)
        pivot.plot(kind='bar', ax=ax, edgecolor='k', linewidth=0.5)
        ax.axhline(cfg['r2_ceiling'], color='green', linestyle='--',
                   linewidth=1.5, label=f'상한 {cfg["r2_ceiling"]}')
        ax.axhline(0, color='k', linewidth=0.8)
        title = '랜덤 분할' if split == 'r2_random' else '시간 분할 (배포)'
        ax.set_title(f'{title} Test R²', fontweight='bold')
        ax.set_ylabel('Test R²'); ax.set_xlabel('')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle('공정한 벤치마크: 장비더미 유무 × 랜덤/시간분할',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'benchmark_fair.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
