# Wire Saw APC TF : train
# -*- coding: utf-8 -*-
"""
Forward 모델 학습 + 저장 (train/test) — Inverse 재사용용
─────────────────────────────────────────
반영 사항:
  · 클러스터링: 타겟(BOW) 제거, WG 분위수 경계로 재현 가능하게 (경계 json 저장)
  · Feature: recipe(temp 12 + wait + ingot) + condition 분리
  · 모델: LinearRegression (현재 방식 유지) — 필요시 Ridge 전환 옵션
  · 평가: 랜덤 + 시간 분할 병기
  · 저장: model.pkl + scaler.pkl + meta.json (Inverse가 참조)

산출:
  ./apc_model/{cluster}/
    ├─ forward_model.pkl
    ├─ scaler.pkl
    ├─ feature_meta.json
    └─ train_log.json
  ./apc_model/cluster_bins.json   (WG 분위수 경계 — 배포 시 클러스터 판정용)
"""
import os
import os.path as pt
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'paths': {
        'input_csv':  r'./data/data.csv',
        'output_dir': r'./apc_model',
    },
    'filter': {
        'process_time': '13.3Hr',
        'meas_eqp':     'BSAFS08',      # None이면 전체
    },
    'cluster': {
        'wg_col':   'range_wire_guide_10_99',   # ⚠️ 배포 시 '직전 run' 값으로 대체 필요
        'n_bins':   4,
        'method':   'quantile',          # 'quantile'(권장) | 'kmeans'
    },
    'model': {
        'target':       'avg_bow_bf_tail',
        'model_type':   'linear',        # 'linear'(현재) | 'ridge'
        'ridge_alpha':  5.0,
        'split_ratio':  0.8,
        'random_state': 42,
        'use_scaler':   True,            # Inverse 호환 위해 True 권장
    },
    'features': {
        # ★ recipe (Inverse 최적화 대상)
        'recipe_cols': [
            'set_frame_temp_0pct',  'set_frame_temp_10pct', 'set_frame_temp_20pct',
            'set_frame_temp_30pct', 'set_frame_temp_40pct', 'set_frame_temp_50pct',
            'set_frame_temp_60pct', 'set_frame_temp_70pct', 'set_frame_temp_80pct',
            'set_frame_temp_90pct', 'set_frame_temp_99pct', 'set_frame_temp_100pct',
            'fdc_wait_time',
            'fdc_ingot_len',
        ],
        # condition (Inverse에서 고정)
        'condition_cols': [
            'range_slurry_temp_10_0',
            'fdc_set_tension',
            'VP_C',
        ],
    },
    'meta_cols': {
        'date': 'date_3200',
        'eqp':  'eqp_nm_3200',
    },
    'min_samples': 50,
    'encoding': 'utf-8',
}


# ============================================================
# 클러스터링 (타겟 제거, WG 분위수 경계)
# ============================================================
def make_clusters(df, cfg):
    """WG 값만으로 클러스터 부여. 분위수 경계는 배포 재현용으로 저장."""
    wg_col = cfg['cluster']['wg_col']
    n_bins = cfg['cluster']['n_bins']
    method = cfg['cluster']['method']

    d = df.dropna(subset=[wg_col]).copy()

    if method == 'quantile':
        # 분위수 경계 (배포 시 이 경계로 직전 run WG를 판정)
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = d[wg_col].quantile(quantiles).values
        bins[0]  -= 1e-6   # 경계 포함 보정
        bins[-1] += 1e-6
        d['clustering_group'] = pd.cut(d[wg_col], bins=bins,
                                        labels=range(n_bins)).astype('Int64')
        bin_info = {'method': 'quantile', 'wg_col': wg_col,
                    'bins': bins.tolist(),
                    'ranges': {int(i): [float(bins[i]), float(bins[i+1])]
                               for i in range(n_bins)}}
    else:  # kmeans (1D)
        from sklearn.cluster import KMeans
        X = d[[wg_col]].values
        km = KMeans(n_clusters=n_bins, random_state=0, n_init='auto').fit(X)
        d['clustering_group'] = km.labels_
        # 중심 순서로 정렬된 경계 정보
        centers = sorted(km.cluster_centers_.ravel().tolist())
        bin_info = {'method': 'kmeans', 'wg_col': wg_col,
                    'centers': centers}

    print(f"[클러스터링] method={method}, {wg_col} 기준")
    for g in sorted(d['clustering_group'].dropna().unique()):
        sub = d[d['clustering_group'] == g]
        print(f"  cluster {g}: N={len(sub)}, "
              f"WG {sub[wg_col].min():.2f}~{sub[wg_col].max():.2f}")
    return d, bin_info


# ============================================================
# 단일 클러스터 학습 + 저장
# ============================================================
def train_one_cluster(d, cluster_id, cfg, out_dir):
    RECIPE = cfg['features']['recipe_cols']
    COND = [c for c in cfg['features']['condition_cols'] if c in d.columns]
    FEATURES = RECIPE + COND
    TARGET = cfg['model']['target']
    DATE = cfg['meta_cols']['date']

    sub = d[FEATURES + [TARGET, DATE]].dropna().copy()
    if len(sub) < cfg['min_samples']:
        print(f"  [cluster {cluster_id}] N={len(sub)} < {cfg['min_samples']} → 스킵")
        return None

    # ── 랜덤 분할 + 시간 분할 병기 ──
    X_all = sub[FEATURES].values
    y_all = sub[TARGET].values

    # 랜덤
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X_all, y_all, test_size=1-cfg['model']['split_ratio'],
        random_state=cfg['model']['random_state'])

    # 시간
    sub[DATE] = pd.to_datetime(sub[DATE], errors='coerce')
    sub_t = sub.sort_values(DATE).reset_index(drop=True)
    split_idx = int(len(sub_t) * cfg['model']['split_ratio'])
    Xtr_t = sub_t[FEATURES].iloc[:split_idx].values
    Xte_t = sub_t[FEATURES].iloc[split_idx:].values
    ytr_t = sub_t[TARGET].iloc[:split_idx].values
    yte_t = sub_t[TARGET].iloc[split_idx:].values

    # ── 모델 팩토리 ──
    def make_model():
        if cfg['model']['model_type'] == 'ridge':
            return Ridge(alpha=cfg['model']['ridge_alpha'])
        return LinearRegression()

    def fit_eval(Xtr, Xte, ytr, yte):
        if cfg['model']['use_scaler']:
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        else:
            sc = None
            Xtr_s, Xte_s = Xtr, Xte
        m = make_model().fit(Xtr_s, ytr)
        r2_tr = r2_score(ytr, m.predict(Xtr_s))
        r2_te = r2_score(yte, m.predict(Xte_s))
        rmse = float(np.sqrt(mean_squared_error(yte, m.predict(Xte_s))))
        mae = float(mean_absolute_error(yte, m.predict(Xte_s)))
        return m, sc, r2_tr, r2_te, rmse, mae

    # 랜덤 분할 성능 (참고)
    _, _, r2tr_rand, r2te_rand, _, _ = fit_eval(Xtr_r, Xte_r, ytr_r, yte_r)
    # 시간 분할 성능 (배포 근사) + 최종 모델은 시간분할 train으로
    model, scaler, r2tr_time, r2te_time, rmse, mae = fit_eval(
        Xtr_t, Xte_t, ytr_t, yte_t)

    print(f"  [cluster {cluster_id}] N={len(sub)} | "
          f"랜덤 Test R²={r2te_rand:.3f} | 시간 Test R²={r2te_time:.3f}")

    # ── 저장 ──
    cdir = pt.join(out_dir, f'cluster{cluster_id}')
    os.makedirs(cdir, exist_ok=True)

    with open(pt.join(cdir, 'forward_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    if scaler is not None:
        with open(pt.join(cdir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    def stats_of(arr):
        return {'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
                'min': float(np.min(arr)), 'max': float(np.max(arr)),
                'q01': float(np.quantile(arr, 0.01)),
                'q99': float(np.quantile(arr, 0.99))}

    meta = {
        'cluster_id': int(cluster_id),
        'target': TARGET,
        'feature_cols': FEATURES,
        'recipe_cols': RECIPE,
        'condition_cols': COND,
        'use_scaler': cfg['model']['use_scaler'],
        'model_type': cfg['model']['model_type'],
        'x_stats': {c: stats_of(sub_t[c].values) for c in FEATURES},
        'y_stats': stats_of(y_all),
    }
    with open(pt.join(cdir, 'feature_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(sub),
        'metrics': {
            'random_split': {'r2_train': round(r2tr_rand, 4),
                             'r2_test': round(r2te_rand, 4)},
            'time_split':   {'r2_train': round(r2tr_time, 4),
                             'r2_test': round(r2te_time, 4),
                             'rmse': round(rmse, 4), 'mae': round(mae, 4)},
        },
        'coefficients': {c: float(v) for c, v in zip(FEATURES, model.coef_)},
        'intercept': float(model.intercept_),
    }
    with open(pt.join(cdir, 'train_log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return {'cluster': cluster_id, 'n': len(sub),
            'r2_random': round(r2te_rand, 4), 'r2_time': round(r2te_time, 4)}


# ============================================================
# 메인
# ============================================================
def main():
    cfg = CONFIG
    out_dir = cfg['paths']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    # 1. 로드 + 필터
    df = pd.read_csv(cfg['paths']['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    print(f"Loaded: {len(df)} rows")

    if cfg['filter']['process_time']:
        df = df[df['process_time'] == cfg['filter']['process_time']]
    if cfg['filter']['meas_eqp']:
        df = df[df['meas_eqp'] == cfg['filter']['meas_eqp']]
    print(f"필터 후: {len(df)} rows "
          f"(PT={cfg['filter']['process_time']}, meas={cfg['filter']['meas_eqp']})")

    # ⚠️ VP_C가 필요한 경우: 여기서 apply_vp_c로 df에 VP_C 컬럼 추가되어 있어야 함
    if 'VP_C' in cfg['features']['condition_cols'] and 'VP_C' not in df.columns:
        print("  ⚠ VP_C 컬럼이 없습니다. condition_cols에서 제거하거나 사전에 생성하세요.")
        cfg['features']['condition_cols'] = [
            c for c in cfg['features']['condition_cols'] if c != 'VP_C']

    # 2. 클러스터링 (타겟 제거, WG 분위수)
    d, bin_info = make_clusters(df, cfg)
    with open(pt.join(out_dir, 'cluster_bins.json'), 'w', encoding='utf-8') as f:
        json.dump(bin_info, f, indent=2, ensure_ascii=False)
    print(f"💾 클러스터 경계 저장: cluster_bins.json (배포 시 판정용)")

    # 3. 클러스터별 학습
    print(f"\n{'='*60}\n클러스터별 학습\n{'='*60}")
    summary = []
    for g in sorted(d['clustering_group'].dropna().unique()):
        res = train_one_cluster(d[d['clustering_group'] == g], int(g), cfg, out_dir)
        if res: summary.append(res)

    # 4. 요약
    sm = pd.DataFrame(summary)
    sm.to_csv(pt.join(out_dir, 'training_summary.csv'),
              index=False, encoding='utf-8-sig')
    print(f"\n{'='*60}\n요약\n{'='*60}")
    print(sm.to_string(index=False))
    print(f"\n💾 전체 저장: {out_dir}/")


if __name__ == '__main__':
    main()
