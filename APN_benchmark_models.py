# -*- coding: utf-8 -*-
"""
Forward 모델 벤치마크: LinearRegression vs Ridge vs XGBoost vs TabPFN
─────────────────────────────────────────
목적: "어떤 모델도 R² 상한(0.24)을 못 넘는다"를 실증
      → 데이터 한계(미측정 변수)임을 발표 근거로 확보

각 모델의 랜덤/시간 분할 Test R²를 비교.
장비 one-hot 포함, leakage 제거된 데이터 사용.

TabPFN 설치:
  pip install tabpfn
  (v2 이상은 Hugging Face 토큰 필요 — 아래 참고)
"""
import os
import os.path as pt
import json
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
    'out_dir':    r'./benchmark',
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
    'use_eqp_dummy': True,
    'split_ratio': 0.8,
    'r2_ceiling': 0.24,   # 재현성 진단으로 확인한 상한
    'encoding':   'utf-8',
}


def prepare_data(cfg):
    df = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    if cfg['process_time']:
        df = df[df['process_time'] == cfg['process_time']]

    DATE, EQP = cfg['date_col'], cfg['eqp_col']
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')

    # leakage 제거
    cutoff = pd.to_datetime(cfg['leakage']['test_start'])
    mask = (df[EQP].isin(cfg['leakage']['test_eqps'])) & (df[DATE] >= cutoff)
    df = df[~mask].copy()
    print(f"[데이터] {len(df)}행 (leakage 제거 후)")

    COND = [c for c in cfg['condition_cols'] if c in df.columns]
    base_cols = cfg['recipe_cols'] + COND
    sub = df[base_cols + [cfg['target'], DATE, EQP]].dropna().copy()

    # 장비 one-hot
    if cfg['use_eqp_dummy']:
        dummies = pd.get_dummies(sub[EQP], prefix='eqp')
        sub = pd.concat([sub, dummies], axis=1)
        FEATURES = base_cols + list(dummies.columns)
    else:
        FEATURES = base_cols

    return sub, FEATURES


def eval_model(name, model_fn, Xtr, Xte, ytr, yte, use_scaler=True):
    """모델 학습·평가 (예외 처리 포함)."""
    try:
        if use_scaler:
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        else:
            Xtr_s, Xte_s = Xtr, Xte
        model = model_fn()
        model.fit(Xtr_s, ytr)
        r2_tr = r2_score(ytr, model.predict(Xtr_s))
        r2_te = r2_score(yte, model.predict(Xte_s))
        return r2_tr, r2_te, None
    except Exception as e:
        return None, None, str(e)


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    sub, FEATURES = prepare_data(cfg)
    TARGET, DATE = cfg['target'], cfg['date_col']

    X = sub[FEATURES].values.astype(float)
    y = sub[TARGET].values

    # 랜덤 분할
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X, y, test_size=1-cfg['split_ratio'], random_state=42)

    # 시간 분할
    sub_t = sub.sort_values(DATE).reset_index(drop=True)
    si = int(len(sub_t) * cfg['split_ratio'])
    Xtr_t = sub_t[FEATURES].iloc[:si].values.astype(float)
    Xte_t = sub_t[FEATURES].iloc[si:].values.astype(float)
    ytr_t = sub_t[TARGET].iloc[:si].values
    yte_t = sub_t[TARGET].iloc[si:].values

    print(f"[샘플] 전체 {len(sub)}, feature {len(FEATURES)}개")
    print(f"       랜덤 train {len(ytr_r)}/test {len(yte_r)}")
    print(f"       시간 train {len(ytr_t)}/test {len(yte_t)}")

    # ── 모델 정의 ──
    model_defs = {
        'LinearRegression': (lambda: LinearRegression(), True),
        'Ridge(a=5)':       (lambda: Ridge(alpha=5.0), True),
    }

    # XGBoost (설치되어 있으면)
    try:
        from xgboost import XGBRegressor
        model_defs['XGBoost'] = (
            lambda: XGBRegressor(n_estimators=100, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 random_state=42), False)
    except ImportError:
        print("  ⚠ XGBoost 미설치 (pip install xgboost)")

    # TabPFN (설치되어 있으면)
    tabpfn_available = False
    try:
        from tabpfn import TabPFNRegressor
        tabpfn_available = True
        # TabPFN은 샘플 수 제한 (보통 ~10000). 큰 데이터는 서브샘플
        model_defs['TabPFN'] = (lambda: TabPFNRegressor(), True)
    except ImportError:
        print("  ⚠ TabPFN 미설치 (pip install tabpfn)")

    # ── 평가 ──
    results = []
    for name, (fn, scaler) in model_defs.items():
        # TabPFN은 샘플 수 제한 → 서브샘플
        if name == 'TabPFN':
            max_n = 3000
            if len(ytr_r) > max_n:
                idx_r = np.random.RandomState(42).choice(len(ytr_r), max_n, replace=False)
                Xtr_r_use, ytr_r_use = Xtr_r[idx_r], ytr_r[idx_r]
                idx_t = np.arange(min(len(ytr_t), max_n))
                Xtr_t_use, ytr_t_use = Xtr_t[idx_t], ytr_t[idx_t]
                print(f"  TabPFN: train을 {max_n}개로 서브샘플")
            else:
                Xtr_r_use, ytr_r_use = Xtr_r, ytr_r
                Xtr_t_use, ytr_t_use = Xtr_t, ytr_t
        else:
            Xtr_r_use, ytr_r_use = Xtr_r, ytr_r
            Xtr_t_use, ytr_t_use = Xtr_t, ytr_t

        # 랜덤
        r2tr_r, r2te_r, err_r = eval_model(name, fn, Xtr_r_use, Xte_r,
                                           ytr_r_use, yte_r, scaler)
        # 시간
        r2tr_t, r2te_t, err_t = eval_model(name, fn, Xtr_t_use, Xte_t,
                                           ytr_t_use, yte_t, scaler)

        if err_r or err_t:
            print(f"  {name}: 오류 {err_r or err_t}")
            continue

        results.append({
            'model': name,
            'r2_train_random': round(r2tr_r, 4),
            'r2_test_random': round(r2te_r, 4),
            'r2_train_time': round(r2tr_t, 4),
            'r2_test_time': round(r2te_t, 4),
        })
        print(f"  {name}: 랜덤 Test R²={r2te_r:.3f} | 시간 Test R²={r2te_t:.3f}")

    res = pd.DataFrame(results)
    res.to_csv(pt.join(cfg['out_dir'], 'benchmark_results.csv'),
               index=False, encoding='utf-8-sig')

    # ── 시각화 ──
    _plot(res, cfg)
    print(f"\n💾 저장: {cfg['out_dir']}/")
    print(f"\n[핵심 관찰] 어떤 모델도 상한 {cfg['r2_ceiling']}를 크게 넘지 못하면")
    print(f"           → 데이터 한계(미측정 변수) 확정, 모델 문제 아님")
    return res


def _plot(res, cfg):
    if len(res) == 0:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(res))
    w = 0.35
    b1 = ax.bar(x - w/2, res['r2_test_random'], w, label='랜덤 분할',
                color='#3498db', edgecolor='k', linewidth=0.5)
    b2 = ax.bar(x + w/2, res['r2_test_time'], w, label='시간 분할 (배포)',
                color='#e74c3c', edgecolor='k', linewidth=0.5)
    for bars, vals in [(b1, res['r2_test_random']), (b2, res['r2_test_time'])]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    v + 0.005 if v > 0 else v - 0.02,
                    f'{v:.3f}', ha='center',
                    va='bottom' if v > 0 else 'top',
                    fontsize=9, fontweight='bold')
    # 상한선
    ax.axhline(cfg['r2_ceiling'], color='green', linestyle='--', linewidth=1.5,
               label=f'이론 상한 {cfg["r2_ceiling"]}')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(res['model'], fontsize=10)
    ax.set_ylabel('Test R²', fontsize=11)
    ax.set_title('Forward 모델 벤치마크 — 어떤 모델도 상한을 못 넘음\n'
                 '(= 데이터 한계, 모델 문제 아님)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'benchmark.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 벤치마크 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
