# -*- coding: utf-8 -*-
"""
슬라이딩 윈도우 + TabPFN (및 Ridge 비교)
─────────────────────────────────────────
아이디어: TabPFN의 in-context learning은 "유사 이웃"에 강한데,
         먼 과거로 학습하면 그 이웃이 미래에 없어 무너짐.
         → 최근 W일로만 학습해서 다음 H일 예측하면
           유사 이웃이 가까운 과거에 있어 살아날 수 있음.

방식:
  · 시간순 정렬 후, [학습 W일] → [평가 H일] 윈도우를 S일씩 이동
  · 각 윈도우에서 TabPFN vs Ridge의 Test R² 측정
  · 윈도우별 성능을 평균±표준편차로 집계

핵심 비교:
  · 단일 시간분할(먼 과거) TabPFN −0.02  vs  슬라이딩(최근) TabPFN ?
  · TabPFN이 슬라이딩에서 살아나면 → 배포 시 재학습 방식으로 활용 가능
"""
import os
import os.path as pt
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
    'out_dir':    r'./sliding_tabpfn',
    'process_time': '13.3Hr',
    'target':     'avg_bow_bf_total',
    'eqp_col':    'eqp_nm_3200',
    'date_col':   'date_3200',
    'recipe_cols': [
        'set_frame_temp_0pct','set_frame_temp_10pct','set_frame_temp_20pct',
        'set_frame_temp_30pct','set_frame_temp_40pct','set_frame_temp_50pct',
        'set_frame_temp_60pct','set_frame_temp_70pct','set_frame_temp_80pct',
        'set_frame_temp_90pct','set_frame_temp_99pct','set_frame_temp_100pct',
        'fdc_set_tension','fdc_wait_time','fdc_ingot_len',
    ],
    'condition_cols': ['range_slurry_temp_10_0'],
    'use_eqp_dummy': True,        # TabPFN엔 False가 나을 수도 (실험)
    # 슬라이딩 윈도우 파라미터
    'train_days': 20,
    'eval_days':  10,
    'step_days':  10,
    'min_train':  100,            # 윈도우 최소 학습 샘플
    'min_eval':   20,             # 윈도우 최소 평가 샘플
    'tabpfn_max': 3000,
    'encoding':   'utf-8',
}


def prepare(cfg):
    df = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    if cfg['process_time']:
        df = df[df['process_time'] == cfg['process_time']]
    DATE = cfg['date_col']; EQP = cfg['eqp_col']
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')

    COND = [c for c in cfg['condition_cols'] if c in df.columns]
    base = cfg['recipe_cols'] + COND
    sub = df[base + [cfg['target'], DATE, EQP]].dropna().copy()
    sub = sub.sort_values(DATE).reset_index(drop=True)

    if cfg['use_eqp_dummy']:
        dummies = pd.get_dummies(sub[EQP], prefix='eqp')
        sub = pd.concat([sub, dummies], axis=1)
        FEATURES = base + list(dummies.columns)
    else:
        FEATURES = base
    return sub, FEATURES


def eval_window(model_fn, Xtr, Xte, ytr, yte, is_tabpfn, cfg):
    try:
        if is_tabpfn and len(ytr) > cfg['tabpfn_max']:
            Xtr, ytr = Xtr[-cfg['tabpfn_max']:], ytr[-cfg['tabpfn_max']:]
        sc = StandardScaler().fit(Xtr)
        m = model_fn()
        m.fit(sc.transform(Xtr), ytr)
        return r2_score(yte, m.predict(sc.transform(Xte))), None
    except Exception as e:
        return None, str(e)


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    sub, FEATURES = prepare(cfg)
    DATE = cfg['date_col']; TARGET = cfg['target']

    # 모델
    models = {'Ridge': (lambda: Ridge(alpha=5.0), False)}
    try:
        from tabpfn import TabPFNRegressor
        models['TabPFN'] = (lambda: TabPFNRegressor(), True)
    except ImportError:
        print("  ⚠ TabPFN 미설치 — Ridge만 실행")

    d_min, d_max = sub[DATE].min(), sub[DATE].max()
    print(f"[기간] {d_min.date()} ~ {d_max.date()}")
    print(f"[윈도우] 학습 {cfg['train_days']}일 / 평가 {cfg['eval_days']}일 / "
          f"이동 {cfg['step_days']}일")

    # 슬라이딩
    rows = []
    cur = d_min
    win_id = 0
    while True:
        tr_start = cur
        tr_end = tr_start + timedelta(days=cfg['train_days'])
        te_end = tr_end + timedelta(days=cfg['eval_days'])
        if te_end > d_max + timedelta(days=1):
            break

        tr = sub[(sub[DATE] >= tr_start) & (sub[DATE] < tr_end)]
        te = sub[(sub[DATE] >= tr_end) & (sub[DATE] < te_end)]

        if len(tr) >= cfg['min_train'] and len(te) >= cfg['min_eval']:
            Xtr = tr[FEATURES].values.astype(float)
            ytr = tr[TARGET].values
            Xte = te[FEATURES].values.astype(float)
            yte = te[TARGET].values

            row = {'window': win_id,
                   'train_start': tr_start.date(), 'train_end': tr_end.date(),
                   'n_train': len(tr), 'n_eval': len(te)}
            for name, (fn, is_tab) in models.items():
                r2, err = eval_window(fn, Xtr, Xte, ytr, yte, is_tab, cfg)
                row[f'{name}_r2'] = round(r2, 4) if r2 is not None else None
            rows.append(row)
            win_id += 1
            msg = ' | '.join(f"{n}={row.get(f'{n}_r2')}" for n in models)
            print(f"  W{win_id}: {tr_start.date()}~{te_end.date()} "
                  f"(tr{len(tr)}/te{len(te)}) {msg}")

        cur = cur + timedelta(days=cfg['step_days'])

    res = pd.DataFrame(rows)
    if len(res) == 0:
        print("⚠ 유효 윈도우 없음 — 파라미터(train_days 등) 조정 필요")
        return
    res.to_csv(pt.join(cfg['out_dir'], 'sliding_results.csv'),
               index=False, encoding='utf-8-sig')

    # 집계
    print(f"\n{'='*56}\n집계 (윈도우 {len(res)}개)\n{'='*56}")
    for name in models:
        col = f'{name}_r2'
        vals = res[col].dropna()
        if len(vals) > 0:
            print(f"  {name}: 평균 R²={vals.mean():+.3f} ± {vals.std():.3f} "
                  f"(범위 {vals.min():.3f}~{vals.max():.3f})")

    _plot(res, list(models.keys()), cfg)
    print(f"\n💾 저장: {cfg['out_dir']}/")
    print(f"\n[해석] 슬라이딩에서 TabPFN 평균 R²가 단일 시간분할(-0.02)보다")
    print(f"       크게 높으면 → 재학습 방식으로 TabPFN 활용 가능")
    return res


def _plot(res, model_names, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # 좌: 윈도우별 R² 추이
    for name in model_names:
        col = f'{name}_r2'
        if col in res.columns:
            ax1.plot(res['window'], res[col], 'o-', label=name, linewidth=1.5)
    ax1.axhline(0, color='k', linewidth=0.8)
    ax1.axhline(-0.023, color='red', linestyle=':', alpha=0.6,
                label='단일분할 TabPFN(-0.02)')
    ax1.set_xlabel('윈도우 번호 (시간 순)'); ax1.set_ylabel('Test R²')
    ax1.set_title('슬라이딩 윈도우별 성능 추이', fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 우: 모델별 분포 (박스)
    data, labels = [], []
    for name in model_names:
        col = f'{name}_r2'
        if col in res.columns:
            v = res[col].dropna()
            if len(v) > 0:
                data.append(v.values); labels.append(name)
    if data:
        bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db'); patch.set_alpha(0.5)
        for i, v in enumerate(data, 1):
            ax2.text(i, np.mean(v), f'μ={np.mean(v):.3f}', ha='center',
                     va='bottom', fontsize=9, fontweight='bold')
    ax2.axhline(0, color='k', linewidth=0.8)
    ax2.set_ylabel('Test R²')
    ax2.set_title('모델별 성능 분포 (전 윈도우)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('슬라이딩 윈도우 재학습: TabPFN vs Ridge',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'sliding_tabpfn.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
