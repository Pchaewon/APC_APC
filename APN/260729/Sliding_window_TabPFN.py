# -*- coding: utf-8 -*-
"""
장비별 슬라이딩 윈도우 + TabPFN (및 Ridge 비교)
─────────────────────────────────────────
★ 장비별로 슬라이딩 (전체 혼합 아님):
  각 장비 안에서 [최근 W일 학습] → [다음 H일 평가], S일씩 이동.
  이유:
    · Simpson's Paradox — 장비 섞으면 장비 간 차이가 관계 왜곡
    · 장비별 시간 이동이 서로 다름
    · 배포 현실 = "이 장비 최근 데이터 → 이 장비 다음 lot"

장비별로 나누면 윈도우당 샘플이 적어짐 → min_train 낮춤 / train_days 늘림으로 대응.

집계:
  · (장비, 윈도우)별 R² → 장비별 평균 + 전체 평균
  · TabPFN vs Ridge 비교
"""
import os
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
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
    # ★ 장비 더미 불필요 (장비별로 도니까 자동으로 장비 고정)
    # 슬라이딩 파라미터 (장비별이라 기간 넉넉히)
    'train_days': 60,     # 30→60 (평가 샘플·분산 확보)
    'eval_days':  20,     # 10→20 (평가 분산 확보, R² 폭발 방지)
    'step_days':  20,
    'min_train':  50,
    'min_eval':   20,     # 평가 최소 20개 (R² 안정)
    'min_eval_std': 0.1,  # ★ 평가 y 표준편차 하한 (미만이면 윈도우 제외)
    'min_windows_per_eqp': 2,
    'tabpfn_max': 2000,
    'encoding':   'utf-8',
}


def prepare(cfg):
    df = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    if cfg['process_time']:
        df = df[df['process_time'] == cfg['process_time']]
    DATE = cfg['date_col']
    df[DATE] = pd.to_datetime(df[DATE], errors='coerce')
    COND = [c for c in cfg['condition_cols'] if c in df.columns]
    FEATURES = cfg['recipe_cols'] + COND
    keep = FEATURES + [cfg['target'], DATE, cfg['eqp_col']]
    sub = df[keep].dropna().copy()
    # 장비별 시간 정렬
    sub = sub.sort_values([cfg['eqp_col'], DATE]).reset_index(drop=True)
    return sub, FEATURES


def eval_window(model_fn, Xtr, Xte, ytr, yte, is_tabpfn, cfg):
    try:
        if is_tabpfn and len(ytr) > cfg['tabpfn_max']:
            Xtr, ytr = Xtr[-cfg['tabpfn_max']:], ytr[-cfg['tabpfn_max']:]
        # 소샘플 방어
        if len(ytr) < cfg['min_train'] or len(yte) < cfg['min_eval']:
            return None, "샘플 부족", None
        # 평가 y 분산 체크 (분산 작으면 R² 폭발 → 제외)
        if np.std(yte) < cfg.get('min_eval_std', 0.1):
            return None, "평가 분산 too small", None

        # ★ 스케일링 방어 1: 분산 0인 feature 제거 (train 기준)
        train_std = Xtr.std(axis=0)
        valid_cols = train_std > 1e-8
        if valid_cols.sum() == 0:
            return None, "유효 feature 없음", None
        Xtr_v, Xte_v = Xtr[:, valid_cols], Xte[:, valid_cols]

        # 스케일링
        sc = StandardScaler().fit(Xtr_v)
        Xtr_s = sc.transform(Xtr_v)
        Xte_s = sc.transform(Xte_v)

        # ★ 스케일링 방어 2: test에서 극단 외삽 클리핑 (train 범위 ±5σ)
        Xte_s = np.clip(Xte_s, -5.0, 5.0)

        m = model_fn()
        m.fit(Xtr_s, ytr)
        pred = m.predict(Xte_s)

        # ★ 스케일링 방어 3: 예측값도 train y 범위로 클리핑
        y_lo, y_hi = ytr.min(), ytr.max()
        y_margin = (y_hi - y_lo) * 0.5 + 1e-6
        pred = np.clip(pred, y_lo - y_margin, y_hi + y_margin)

        r2 = r2_score(yte, pred)
        mae = float(np.mean(np.abs(pred - yte)))
        r2_clip = max(r2, -2.0)
        return r2_clip, None, mae
    except Exception as e:
        return None, str(e), None


def slide_one_eqp(esub, eqp, FEATURES, models, cfg):
    """단일 장비 내 슬라이딩."""
    DATE = cfg['date_col']; TARGET = cfg['target']
    d_min, d_max = esub[DATE].min(), esub[DATE].max()
    rows = []
    cur = d_min
    wid = 0
    while True:
        tr_end = cur + timedelta(days=cfg['train_days'])
        te_end = tr_end + timedelta(days=cfg['eval_days'])
        if te_end > d_max + timedelta(days=1):
            break
        tr = esub[(esub[DATE] >= cur) & (esub[DATE] < tr_end)]
        te = esub[(esub[DATE] >= tr_end) & (esub[DATE] < te_end)]
        if len(tr) >= cfg['min_train'] and len(te) >= cfg['min_eval']:
            Xtr = tr[FEATURES].values.astype(float); ytr = tr[TARGET].values
            Xte = te[FEATURES].values.astype(float); yte = te[TARGET].values
            row = {'eqp': eqp, 'window': wid,
                   'train_start': cur.date(), 'n_train': len(tr), 'n_eval': len(te)}
            for name, (fn, is_tab) in models.items():
                r2, err, mae = eval_window(fn, Xtr, Xte, ytr, yte, is_tab, cfg)
                row[f'{name}_r2'] = round(r2, 4) if r2 is not None else None
                row[f'{name}_mae'] = round(mae, 4) if mae is not None else None
            rows.append(row); wid += 1
        cur = cur + timedelta(days=cfg['step_days'])
    return rows


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    sub, FEATURES = prepare(cfg)
    EQP = cfg['eqp_col']
    print(f"[데이터] {len(sub)}행, {sub[EQP].nunique()}대 장비, feature {len(FEATURES)}개")

    models = {'Ridge': (lambda: Ridge(alpha=5.0), False)}
    try:
        from tabpfn import TabPFNRegressor
        models['TabPFN'] = (lambda: TabPFNRegressor(ignore_pretraining_limits=True), True)
    except ImportError:
        print("  ⚠ TabPFN 미설치 — Ridge만")

    # 장비별 슬라이딩
    all_rows = []
    eqp_list = sorted(sub[EQP].unique())
    print(f"[장비별 슬라이딩] 학습 {cfg['train_days']}일/평가 {cfg['eval_days']}일/"
          f"이동 {cfg['step_days']}일\n")

    for eqp in eqp_list:
        esub = sub[sub[EQP] == eqp].reset_index(drop=True)
        rows = slide_one_eqp(esub, eqp, FEATURES, models, cfg)
        if len(rows) >= cfg['min_windows_per_eqp']:
            all_rows.extend(rows)
            msg = []
            for name in models:
                vals = [r[f'{name}_r2'] for r in rows
                        if r.get(f'{name}_r2') is not None]
                if vals:
                    msg.append(f"{name} μ={np.mean(vals):+.3f}({len(vals)}win)")
            print(f"  {eqp}: {' | '.join(msg)}")

    if not all_rows:
        print("\n⚠ 유효 윈도우 없음 — train_days↑ 또는 min_train↓ 필요")
        return

    res = pd.DataFrame(all_rows)
    res.to_csv(pt.join(cfg['out_dir'], 'sliding_by_eqp.csv'),
               index=False, encoding='utf-8-sig')

    # ── 집계 ──
    print(f"\n{'='*60}\n집계\n{'='*60}")
    print(f"총 (장비,윈도우): {len(res)}개, 장비 {res['eqp'].nunique()}대")
    for name in models:
        col = f'{name}_r2'
        mcol = f'{name}_mae'
        if col not in res.columns: continue
        vals = res[col].dropna()
        if len(vals) == 0: continue
        print(f"\n  [{name}]")
        # R² (클리핑됨) — 중앙값도 함께 (평균은 이상치에 민감)
        print(f"    R² 평균: {vals.mean():+.3f} | 중앙값: {vals.median():+.3f}")
        eqp_means = res.groupby('eqp')[col].mean().dropna()
        print(f"    장비별 평균의 평균: {eqp_means.mean():+.3f} "
              f"(장비 {len(eqp_means)}대)")
        print(f"    장비별 중앙값의 중앙값: {res.groupby('eqp')[col].median().median():+.3f}")
        print(f"    양수 윈도우 비율: {(vals > 0).mean()*100:.0f}%")
        # MAE (R²보다 안정적)
        if mcol in res.columns:
            mvals = res[mcol].dropna()
            if len(mvals) > 0:
                print(f"    MAE 평균: {mvals.mean():.4f} (BOW 단위, 낮을수록 좋음)")

    _plot(res, list(models.keys()), cfg)
    print(f"\n💾 저장: {cfg['out_dir']}/")
    print(f"\n[해석]")
    print(f"  · 장비별 슬라이딩은 장비 효과 자동 제거 + 배포 현실 반영")
    print(f"  · TabPFN이 여기서 Ridge를 이기면 → 소샘플 재학습에 TabPFN 강점")
    return res


def _plot(res, model_names, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 좌: 장비별 평균 R² (막대)
    eqps = sorted(res['eqp'].unique())
    x = np.arange(len(eqps))
    w = 0.8 / max(len(model_names), 1)
    for i, name in enumerate(model_names):
        col = f'{name}_r2'
        if col not in res.columns: continue
        means = [res[res['eqp'] == e][col].mean() for e in eqps]
        ax1.bar(x + i*w, means, w, label=name, edgecolor='k', linewidth=0.4)
    ax1.axhline(0, color='k', linewidth=0.8)
    ax1.set_ylim(-2.1, 1.0)   # 클리핑된 R² 범위
    ax1.set_xticks(x + w*(len(model_names)-1)/2)
    ax1.set_xticklabels([e.replace('BSWS','') for e in eqps], fontsize=7,
                        rotation=45)
    ax1.set_xlabel('장비'); ax1.set_ylabel('장비별 평균 Test R²')
    ax1.set_title('장비별 슬라이딩 평균 성능', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)

    # 우: 모델별 전체 분포 (박스)
    data, labels = [], []
    for name in model_names:
        col = f'{name}_r2'
        if col in res.columns:
            v = res[col].dropna()
            if len(v) > 0:
                data.append(v.values); labels.append(name)
    if data:
        try:
            bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True)
        except TypeError:
            bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db'); patch.set_alpha(0.5)
        for i, v in enumerate(data, 1):
            ax2.text(i, np.mean(v), f'μ={np.mean(v):.3f}', ha='center',
                     va='bottom', fontsize=9, fontweight='bold')
    ax2.axhline(0, color='k', linewidth=0.8)
    ax2.set_ylabel('Test R² (전 윈도우)')
    ax2.set_title('모델별 성능 분포', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('장비별 슬라이딩 윈도우: TabPFN vs Ridge',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'sliding_by_eqp.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
