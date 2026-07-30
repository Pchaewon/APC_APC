# -*- coding: utf-8 -*-
"""
Rolling 모델의 0.829 진단 — 진짜 신호인가, 자기상관 leakage인가
─────────────────────────────────────────
의심: roll_avg_bow(직전 BOW 평균)가 현재 BOW와 자기상관이 강해서
      "직전 BOW를 복사"하는 것일 수 있음. 그러면 R²는 높아도
      온도로 BOW를 제어하는 능력(inverse)은 없음.

4가지 비교 (모두 시간 분할):
  [1] 온도만 (roll 없음)                    — 순수 온도 효과
  [2] 온도 + roll_조건 (roll_BOW 제외)      — 조건 예측력
  [3] 온도 + roll_BOW만                     — 자기상관만
  [4] 전체 (온도 + roll_BOW + roll_조건)    — 현재 0.829

+ 온도 계수 크기 확인 (inverse 가능성)
"""
import os
import os.path as pt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from rolling_run_features import add_rolling_run_features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    'input_csv':  r'D:\chaewon\APC\02.TF\260726\data\data.csv',
    'out_dir':    r'./rolling_diagnosis',
    'process_time': '13.3Hr',
    'target':     'avg_bow_bf_total',
    'wire_col':   'new_fdc_wire_id',
    'eqp_col':    'eqp_nm_3200',
    'date_col':   'date_3200',
    'temp_cols':  [f'set_frame_temp_{p}pct' for p in
                   [0,10,20,30,40,50,60,70,80,90,99,100]],
    'roll_source_cols': [
        'fdc_set_tension','fdc_wait_time','fdc_ingot_len','range_slurry_temp_10_0',
    ],
    'lag': 2, 'window': 10, 'min_runs': 3,
    'use_eqp_dummy': True,
    'ridge_alpha': 5.0,
    'split_ratio': 0.8,
    'encoding':   'utf-8',
}


def build(cfg):
    df = pd.read_csv(cfg['input_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    if cfg['process_time']:
        df = df[df['process_time'] == cfg['process_time']]
    roll_cfg = {
        'wire_col': cfg['wire_col'], 'date_col': cfg['date_col'],
        'eqp_col': cfg['eqp_col'], 'target_col': cfg['target'],
        'feature_cols': cfg['temp_cols'] + cfg['roll_source_cols'],
        'lag': cfg['lag'], 'window': cfg['window'], 'min_runs': cfg['min_runs'],
    }
    return add_rolling_run_features(df, roll_cfg)


def eval_feats(sub, feat_cols, cfg, return_coef=False):
    EQP = cfg['eqp_col']; TARGET = cfg['target']; DATE = cfg['date_col']
    use = sub[feat_cols + [TARGET, DATE, EQP]].dropna().copy()
    use = use.sort_values(DATE).reset_index(drop=True)
    if cfg['use_eqp_dummy']:
        dummies = pd.get_dummies(use[EQP], prefix='eqp')
        X_df = pd.concat([use[feat_cols].reset_index(drop=True),
                          dummies.reset_index(drop=True)], axis=1)
        all_feats = feat_cols + list(dummies.columns)
    else:
        X_df = use[feat_cols]; all_feats = feat_cols
    X = X_df.values.astype(float); y = use[TARGET].values
    si = int(len(use) * cfg['split_ratio'])
    sc = StandardScaler().fit(X[:si])
    m = Ridge(alpha=cfg['ridge_alpha']).fit(sc.transform(X[:si]), y[:si])
    pred = m.predict(sc.transform(X[si:]))
    r2 = r2_score(y[si:], pred); mae = mean_absolute_error(y[si:], pred)
    out = {'n': len(use), 'n_feat': len(all_feats),
           'r2_time': round(r2, 4), 'mae': round(mae, 4)}
    if return_coef:
        coef = dict(zip(all_feats, m.coef_))
        out['temp_coef_abs_mean'] = np.mean([abs(coef[c]) for c in cfg['temp_cols']
                                             if c in coef])
        rollbow = f'roll_{TARGET}'
        out['roll_bow_coef'] = coef.get(rollbow, None)
    return out


def main(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    df = build(cfg)
    TARGET = cfg['target']
    temp = cfg['temp_cols']
    roll_bow = [f'roll_{TARGET}']
    roll_cond = [f'roll_{c}' for c in cfg['roll_source_cols']]

    print(f"[데이터] {len(df)}행\n")

    configs = {
        '[1] 온도만':                    temp,
        '[2] 온도+roll조건(BOW제외)':    temp + roll_cond,
        '[3] 온도+roll_BOW만':           temp + roll_bow,
        '[4] 전체':                      temp + roll_bow + roll_cond,
    }
    results = []
    for label, feats in configs.items():
        r = eval_feats(df, feats, cfg, return_coef=True)
        results.append({'config': label, **r})
        extra = ""
        if r.get('roll_bow_coef') is not None:
            extra = f" | roll_BOW계수={r['roll_bow_coef']:+.3f}"
        print(f"  {label}: R²={r['r2_time']:.3f}, MAE={r['mae']:.3f}, "
              f"온도계수평균={r['temp_coef_abs_mean']:.4f}{extra}")

    res = pd.DataFrame(results)
    res.to_csv(pt.join(cfg['out_dir'], 'rolling_diagnosis.csv'),
               index=False, encoding='utf-8-sig')

    # 판정
    print(f"\n{'='*60}\n판정\n{'='*60}")
    r1 = res.iloc[0]['r2_time']  # 온도만
    r3 = res.iloc[2]['r2_time']  # 온도+roll_BOW
    r4 = res.iloc[3]['r2_time']  # 전체
    print(f"  온도만:          R²={r1:.3f}")
    print(f"  온도+roll_BOW:   R²={r3:.3f}  (roll_BOW 추가 효과: {r3-r1:+.3f})")
    print(f"  전체:            R²={r4:.3f}")
    print()
    if r3 - r1 > 0.3:
        print("  ⚠ roll_BOW가 성능의 대부분 → BOW 자기상관 지배 (leakage 성격)")
        print("    → 예측은 잘 되나 '직전 BOW 복사'에 가까움")
        temp_coef = res.iloc[3]['temp_coef_abs_mean']
        temp_only_coef = res.iloc[0]['temp_coef_abs_mean']
        print(f"    온도 계수: 온도만={temp_only_coef:.4f} → 전체={temp_coef:.4f}")
        if temp_coef < temp_only_coef * 0.3:
            print("    ❌ 전체 모델에서 온도 계수 위축 → inverse 온도 제어력 상실")
            print("       → recipe 추천엔 부적합. roll_BOW 빼고 [2] 사용 권장")
        else:
            print("    ⭕ 온도 계수 유지 → inverse 가능. 단 roll_BOW 효과 해석 주의")
    else:
        print("  ✅ roll_BOW 효과 제한적 → 자기상관 leakage 아님, 진짜 조건 예측력")

    _plot(res, cfg)
    print(f"\n💾 저장: {cfg['out_dir']}/")
    print(f"\n[권장]")
    print(f"  · recipe 추천용: [2] 온도+roll조건 (roll_BOW 제외) 사용")
    print(f"    → 온도 제어력 유지하면서 직전 상태 반영")
    print(f"  · 순수 BOW 예측(모니터링)용: [4] 전체 가능")
    return res


def _plot(res, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ['#3498db','#2ecc71','#e74c3c','#9b59b6']
    ax1.bar(range(len(res)), res['r2_time'], color=colors, edgecolor='k')
    ax1.set_xticks(range(len(res)))
    ax1.set_xticklabels(res['config'], fontsize=8, rotation=15)
    ax1.axhline(0.24, color='green', linestyle='--', alpha=0.6, label='기존상한 0.24')
    ax1.set_ylabel('Test R² (시간분할)')
    ax1.set_title('roll_BOW 유무별 성능\n(자기상관 leakage 확인)', fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(res['r2_time']):
        ax1.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    # 온도 계수
    ax2.bar(range(len(res)), res['temp_coef_abs_mean'], color=colors, edgecolor='k')
    ax2.set_xticks(range(len(res)))
    ax2.set_xticklabels(res['config'], fontsize=8, rotation=15)
    ax2.set_ylabel('온도 계수 절대값 평균')
    ax2.set_title('온도 계수 크기\n(작아지면 inverse 제어력 상실)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(pt.join(cfg['out_dir'], 'rolling_diagnosis.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 그림 저장")


if __name__ == '__main__':
    main(CONFIG)
