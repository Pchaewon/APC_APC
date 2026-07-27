# -*- coding: utf-8 -*-
"""
TabPFN 기반 Inverse (샘플링 방식)
─────────────────────────────────────────
TabPFN은 미분이 어렵고 느려서 SLSQP gradient 최적화가 부적합.
대신 "후보 recipe를 많이 생성 → BOW 예측 → 목표에 가장 가까운 것 선택"
하는 샘플링 기반 inverse를 사용.

주의:
  · TabPFN은 predict마다 train set 전체를 context로 넣어 느림
  · 후보 수(n_candidates)를 늘리면 정확도↑ 속도↓
  · Forward R²가 상한(0.24)에 막히면 inverse도 그 한계 안에서만 작동

설치: pip install tabpfn
"""
import os
import os.path as pt
import json
import numpy as np
import pandas as pd

CONFIG = {
    'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
    'test_csv':   r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',
    'out_dir':    r'./inverse_tabpfn',
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
    'other_recipe': ['fdc_set_tension','fdc_wait_time','fdc_ingot_len'],
    'condition_cols': ['range_slurry_temp_10_0'],
    'temp_eqps': ['BSWS38','BSWS42','BSWS44'],
    'max_train': 3000,        # TabPFN train 상한
    'n_candidates': 200,      # inverse 후보 recipe 수
    'candidate_std': 0.3,     # 후보 생성 시 실측 대비 표준편차
    'sample_rows': 30,        # 검증할 test 행 수 (느려서 제한)
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
    RECIPE = cfg['temp_cols'] + cfg['other_recipe']
    FEATURES = RECIPE + COND
    sub = df[FEATURES + [cfg['target'], EQP]].dropna().copy()

    # 장비 one-hot
    dummies = pd.get_dummies(sub[EQP], prefix='eqp')
    sub = pd.concat([sub, dummies], axis=1)
    FEATURES_ALL = FEATURES + list(dummies.columns)
    return sub, FEATURES, FEATURES_ALL, list(dummies.columns), RECIPE


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)

    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        print("❌ TabPFN 미설치. pip install tabpfn")
        return

    sub, FEATURES, FEATURES_ALL, eqp_cols, RECIPE = prepare(cfg)
    TARGET = cfg['target']; EQP = cfg['eqp_col']
    temp_cols = cfg['temp_cols']

    # 학습 (TabPFN은 fit이 사실상 context 저장)
    Xtr = sub[FEATURES_ALL].values.astype(float)
    ytr = sub[TARGET].values
    if len(ytr) > cfg['max_train']:
        idx = np.random.RandomState(42).choice(len(ytr), cfg['max_train'],
                                                replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"[TabPFN 학습] N={len(ytr)}, feature={len(FEATURES_ALL)}")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    model = TabPFNRegressor()
    model.fit(scaler.transform(Xtr), ytr)
    print("  학습(context 저장) 완료")

    # 통계 (후보 생성용)
    x_stats = {c: {'mean': sub[c].mean(), 'std': sub[c].std(),
                   'q01': sub[c].quantile(0.01), 'q99': sub[c].quantile(0.99)}
               for c in RECIPE}

    # test 로드
    test_df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                          encoding_errors='replace')

    def predict_batch(rows_df):
        """여러 후보를 한 번에 예측 (배치)."""
        X = rows_df[FEATURES_ALL].values.astype(float)
        return model.predict(scaler.transform(X))

    # ── 샘플링 기반 inverse ──
    rng = np.random.default_rng(42)
    results = []

    for eqp in cfg['temp_eqps']:
        esub = test_df[test_df[EQP] == eqp].dropna(subset=[TARGET] + temp_cols)
        if len(esub) == 0:
            continue
        if len(esub) > cfg['sample_rows']:
            esub = esub.sample(cfg['sample_rows'], random_state=42)
        print(f"\n[{eqp}] {len(esub)}행 inverse...")

        for _, row in esub.iterrows():
            target_y = float(row[TARGET])

            # 후보 recipe 생성 (실측 온도 주변 + 랜덤)
            n_cand = cfg['n_candidates']
            cand_rows = []
            for _ in range(n_cand):
                cand = row.to_dict()
                # 온도 12개를 실측 주변에서 흔들기
                for c in temp_cols:
                    base = float(row[c])
                    cand[c] = np.clip(base + rng.normal(0, cfg['candidate_std']),
                                      x_stats[c]['q01'], x_stats[c]['q99'])
                # 장비 더미
                for ec in eqp_cols:
                    cand[ec] = 1.0 if ec == f'eqp_{eqp}' else 0.0
                cand_rows.append(cand)

            cand_df = pd.DataFrame(cand_rows)
            # 결측 feature 채우기
            for c in FEATURES_ALL:
                if c not in cand_df.columns:
                    cand_df[c] = x_stats.get(c, {}).get('mean', 0.0)
                cand_df[c] = pd.to_numeric(cand_df[c], errors='coerce').fillna(
                    x_stats.get(c, {}).get('mean', 0.0))

            preds = predict_batch(cand_df)
            best_idx = np.argmin(np.abs(preds - target_y))
            best_cand = cand_df.iloc[best_idx]

            results.append({
                'eqp': eqp,
                'target_bow': round(target_y, 3),
                'pred_bow': round(float(preds[best_idx]), 3),
                'gap': round(abs(preds[best_idx] - target_y), 3),
                'actual_temp60': round(float(row['set_frame_temp_60pct']), 3),
                'rec_temp60': round(float(best_cand['set_frame_temp_60pct']), 3),
            })

    res = pd.DataFrame(results)
    res.to_csv(pt.join(cfg['out_dir'], 'tabpfn_inverse.csv'),
               index=False, encoding='utf-8-sig')

    print(f"\n{'='*56}\nTabPFN Inverse 결과\n{'='*56}")
    if len(res) > 0:
        print(f"평균 gap: {res['gap'].mean():.4f}")
        d = res['rec_temp60'] - res['actual_temp60']
        print(f"temp60 역산-실측: mean={d.mean():+.3f}, |mean|={d.abs().mean():.3f}")
        from scipy import stats
        if len(res) > 2:
            r = stats.pearsonr(res['actual_temp60'], res['rec_temp60'])[0]
            print(f"역산-실측 상관: r={r:.3f}")
    print(f"\n💾 저장: {cfg['out_dir']}/")
    return res


if __name__ == '__main__':
    main(CONFIG)
