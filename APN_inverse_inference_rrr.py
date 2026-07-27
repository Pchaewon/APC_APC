# -*- coding: utf-8 -*-
"""
Inverse 검증/추론 (엔지니어 test 데이터)
─────────────────────────────────────────
train_test.py로 학습·저장한 모델(./apc_model/full/)을 로드해서:
  · test_df 8대를 장비군별로 검증
      15-19년식(38,42,44): recipe=temp 12개 역산 → 실제 temp 비교
      21년식(52,54,55,56,61): recipe=tension 역산 → 실제 tension 비교
  · 검증 2종: ① 역산값 vs 실제값(수치) ② 변경 방향 일치(Group 0→1)

※ Leakage 처리는 train_test.py에서 완료됨 (test 8대의 3월 이후 제외).
  여기서는 저장된 모델을 로드만 하므로 재학습 없음.
"""
import os
import os.path as pt
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'paths': {
        'model_dir': r'./apc_model/full',       # train_test.py 산출물
        'test_csv':  r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',
        'out_dir':   r'./inverse_inference',
    },
    'target': 'avg_bow_bf_total',   # train과 동일해야 함
    'eqp_col': 'eqp_nm_3200',
    'eqp_groups': {
        # 두 장비군 모두 temp + tension 함께 역산 ('both')
        '15-19': {'eqps': ['BSWS38','BSWS42','BSWS44'], 'optimize': 'both'},
        '21':    {'eqps': ['BSWS52','BSWS54','BSWS55','BSWS56','BSWS61'],
                  'optimize': 'both'},
    },
    'change_group_cols': {
        'temp':    'Frame_Group',      # 0=기존, 1=변경
        'tension': 'Tension_Group',    # 0=기존, 1/2=변경
    },
    'lambda_smooth': 0.01,
    'encoding': 'utf-8',
}


# ============================================================
# 로드
# ============================================================
def load_model(model_dir):
    with open(pt.join(model_dir, 'forward_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(pt.join(model_dir, 'feature_meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    scaler = None
    sc_path = pt.join(model_dir, 'scaler.pkl')
    if meta.get('use_scaler') and os.path.exists(sc_path):
        with open(sc_path, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler, meta


# ============================================================
# Inverse: 지정 변수만 최적화
# ============================================================
def inverse_for_target(model, scaler, meta, target_y, base_row,
                       optimize='temp', lambda_smooth=0.01, eqp_name=None):
    FEATURES = meta['feature_cols']
    X_STATS  = meta['x_stats']
    temp_cols = meta['temp_cols']
    tension_col = meta['tension_col']
    eqp_cols = meta.get('eqp_cols', [])
    eqp_prefix = meta.get('eqp_prefix', 'eqp_')

    if optimize == 'temp':
        opt_cols = temp_cols
    elif optimize == 'tension':
        opt_cols = [tension_col]
    elif optimize == 'both':
        opt_cols = temp_cols + [tension_col]   # temp 12개 + tension
    else:
        raise ValueError(optimize)
    opt_idx = [FEATURES.index(c) for c in opt_cols]

    # x_full 구성: recipe/condition은 base_row 값, 장비 더미는 해당 장비만 1
    def get_val(c):
        if c in eqp_cols:
            # 이 lot의 장비에 해당하는 더미만 1
            if eqp_name is not None:
                return 1.0 if c == f'{eqp_prefix}{eqp_name}' else 0.0
            return 0.0
        v = base_row.get(c, None)
        # NaN 또는 없으면 학습 평균으로 대체 (NaN 방어)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float(X_STATS.get(c, {}).get('mean', 0.0))
        return float(v)

    x_full = np.array([get_val(c) for c in FEATURES])

    def predict(x_vec):
        xx = x_vec.reshape(1, -1)
        if scaler is not None:
            xx = scaler.transform(xx)
        return float(model.predict(xx)[0])

    n_temp = len(temp_cols)
    def objective(opt_vec):
        x = x_full.copy()
        x[opt_idx] = opt_vec
        loss = (predict(x) - target_y) ** 2
        # smoothness는 온도 부분에만 (both일 때 앞 12개가 temp)
        if lambda_smooth > 0 and optimize in ('temp', 'both'):
            temp_part = opt_vec[:n_temp] if optimize == 'both' else opt_vec
            loss += lambda_smooth * np.sum(np.diff(temp_part) ** 2)
        return loss

    x0 = x_full[opt_idx].copy()
    bounds = [(X_STATS[c]['q01'], X_STATS[c]['q99']) for c in opt_cols]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})

    x_opt = x_full.copy()
    x_opt[opt_idx] = res.x
    y_pred = predict(x_opt)
    return {
        'optimized': {c: round(float(v), 4) for c, v in zip(opt_cols, res.x)},
        'predicted_bow': round(y_pred, 4),
        'gap': round(abs(y_pred - target_y), 4),
        'success': bool(res.success),
    }


# ============================================================
# 검증
# ============================================================
def validate(cfg):
    model, scaler, meta = load_model(cfg['paths']['model_dir'])
    print(f"[모델 로드] {cfg['paths']['model_dir']}")
    print(f"  feature {len(meta['feature_cols'])}개, target={meta['target']}")

    test_df = pd.read_csv(cfg['paths']['test_csv'],
                          encoding=cfg['encoding'], encoding_errors='replace')
    EQP = cfg['eqp_col']; TARGET = cfg['target']
    tension_col = meta['tension_col']

    out = cfg['paths']['out_dir']
    plot_dir = pt.join(out, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    numeric_rows, direction_rows = [], []

    for grp_name, grp_cfg in cfg['eqp_groups'].items():
        optimize = grp_cfg['optimize']
        # both면 장비군에 따라 주 변경컬럼 선택 (15-19=Frame, 21=Tension)
        if optimize == 'both':
            change_col = ('Frame_Group' if grp_name == '15-19'
                          else 'Tension_Group')
        else:
            change_col = cfg['change_group_cols'][optimize]
        print(f"\n{'='*56}\n[{grp_name}] 최적화={optimize} (변경컬럼={change_col})\n{'='*56}")

        for eqp in grp_cfg['eqps']:
            sub = test_df[test_df[EQP] == eqp].dropna(subset=[TARGET])
            if len(sub) == 0:
                print(f"  {eqp}: 데이터 없음")
                continue
            print(f"  {eqp}: N={len(sub)}")

            # 검증 ①: 각 행 실제 BOW → 역산 → 실제 recipe 비교
            for _, row in sub.iterrows():
                target_y = float(row[TARGET])
                inv = inverse_for_target(model, scaler, meta, target_y,
                                         base_row=row.to_dict(),
                                         optimize=optimize,
                                         lambda_smooth=cfg['lambda_smooth'],
                                         eqp_name=eqp)
                rec = inv['optimized']
                ro = {'eqp': eqp, 'group': grp_name, 'optimize': optimize,
                      'target_bow': round(target_y, 3),
                      'pred_bow': inv['predicted_bow'], 'gap': inv['gap'],
                      'wire_id': row.get('new_fdc_wire_id', np.nan),
                      change_col: row.get(change_col, np.nan)}
                # 온도 12개 (temp 또는 both일 때)
                if optimize in ('temp', 'both'):
                    for tc in meta['temp_cols']:
                        ro[f'rec_{tc}'] = rec.get(tc)
                        ro[f'actual_{tc}'] = row.get(tc, np.nan)
                # tension (tension 또는 both일 때)
                if optimize in ('tension', 'both'):
                    rec_t = rec.get(tension_col)
                    act_t = row.get(tension_col, np.nan)
                    ro['rec_tension'] = rec_t
                    ro['actual_tension'] = act_t
                    ro['diff_tension'] = (round(rec_t - act_t, 4)
                                          if pd.notna(act_t) and rec_t is not None
                                          else None)
                numeric_rows.append(ro)

            # 검증 ②: 변경 방향 (0 vs 1)
            if change_col in sub.columns:
                g0 = sub[sub[change_col] == 0]
                g1 = sub[sub[change_col] == 1]
                if len(g0) > 0 and len(g1) > 0:
                    bow_dir = np.sign(g1[TARGET].mean() - g0[TARGET].mean())
                    dr = {'eqp': eqp, 'group': grp_name, 'optimize': optimize,
                          'bow_0': round(g0[TARGET].mean(), 3),
                          'bow_1': round(g1[TARGET].mean(), 3),
                          'bow_change_dir': int(bow_dir),
                          'n_0': len(g0), 'n_1': len(g1)}
                    # temp 방향 (both 또는 temp)
                    if optimize in ('temp', 'both'):
                        tcol = 'set_frame_temp_60pct'
                        td = np.sign(g1[tcol].mean() - g0[tcol].mean())
                        dr['temp60_0'] = round(g0[tcol].mean(), 3)
                        dr['temp60_1'] = round(g1[tcol].mean(), 3)
                        dr['temp_change_dir'] = int(td)
                    # tension 방향 (both 또는 tension)
                    if optimize in ('tension', 'both'):
                        td = np.sign(g1[tension_col].mean() - g0[tension_col].mean())
                        dr['tension_0'] = round(g0[tension_col].mean(), 3)
                        dr['tension_1'] = round(g1[tension_col].mean(), 3)
                        dr['tension_change_dir'] = int(td)
                    direction_rows.append(dr)

    # 저장
    num_df = pd.DataFrame(numeric_rows)
    dir_df = pd.DataFrame(direction_rows)
    num_df.to_csv(pt.join(out, 'validation_numeric.csv'),
                  index=False, encoding='utf-8-sig')
    dir_df.to_csv(pt.join(out, 'validation_direction.csv'),
                  index=False, encoding='utf-8-sig')

    # 요약
    print(f"\n{'='*56}\n검증 요약\n{'='*56}")
    if len(num_df) > 0:
        print(f"수치 검증: {len(num_df)}행, 평균 gap={num_df['gap'].mean():.4f}")
        # tension 역산-실제 (diff_tension 있는 모든 행)
        if 'diff_tension' in num_df.columns:
            vd = num_df['diff_tension'].dropna()
            if len(vd) > 0:
                print(f"  [tension] 역산-실제 차이: "
                      f"mean={vd.mean():+.4f}, |mean|={vd.abs().mean():.4f}, "
                      f"std={vd.std():.4f}")
        # temp60 역산-실제 (rec_set_frame_temp_60pct 있는 행)
        if 'rec_set_frame_temp_60pct' in num_df.columns:
            d60 = (num_df['rec_set_frame_temp_60pct'] -
                   num_df['actual_set_frame_temp_60pct']).dropna()
            if len(d60) > 0:
                print(f"  [temp_60pct] 역산-실제 차이: "
                      f"mean={d60.mean():+.4f}, |mean|={d60.abs().mean():.4f}")
        # 장비군별로도 분리 출력
        for grp in num_df['group'].unique():
            gsub = num_df[num_df['group'] == grp]
            msg = f"  [{grp}] "
            if 'rec_set_frame_temp_60pct' in gsub.columns:
                dd = (gsub['rec_set_frame_temp_60pct'] -
                      gsub['actual_set_frame_temp_60pct']).dropna()
                if len(dd) > 0:
                    msg += f"temp60 |Δ|={dd.abs().mean():.3f} "
            if 'diff_tension' in gsub.columns:
                dt = gsub['diff_tension'].dropna()
                if len(dt) > 0:
                    msg += f"tension |Δ|={dt.abs().mean():.3f}"
            print(msg)
    if len(dir_df) > 0:
        print(f"\n방향 검증:")
        print(dir_df.to_string(index=False))

    # 시각화
    _plot_tension(num_df, plot_dir)
    _plot_temp(num_df, plot_dir)
    print(f"\n✅ 완료: {out}/")
    return num_df, dir_df


def _plot_tension(num_df, plot_dir):
    t = num_df[num_df['optimize'] == 'tension'].copy()
    if len(t) == 0: return
    t = t.dropna(subset=['rec_tension', 'actual_tension'])
    if len(t) == 0: return
    fig, ax = plt.subplots(figsize=(7, 7))
    for eqp in t['eqp'].unique():
        s = t[t['eqp'] == eqp]
        ax.scatter(s['actual_tension'], s['rec_tension'], s=40, alpha=0.6, label=eqp)
    lims = [min(t['actual_tension'].min(), t['rec_tension'].min()),
            max(t['actual_tension'].max(), t['rec_tension'].max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('실제 tension'); ax.set_ylabel('역산 tension')
    ax.set_title('21년식: tension 역산 vs 실제', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pt.join(plot_dir, 'tension_inverse_vs_actual.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 tension scatter 저장")


def _plot_temp(num_df, plot_dir):
    t = num_df[num_df['optimize'] == 'temp'].copy()
    if len(t) == 0 or 'rec_set_frame_temp_60pct' not in t.columns: return
    t = t.dropna(subset=['rec_set_frame_temp_60pct', 'actual_set_frame_temp_60pct'])
    if len(t) == 0: return
    fig, ax = plt.subplots(figsize=(7, 7))
    for eqp in t['eqp'].unique():
        s = t[t['eqp'] == eqp]
        ax.scatter(s['actual_set_frame_temp_60pct'],
                   s['rec_set_frame_temp_60pct'], s=40, alpha=0.6, label=eqp)
    lims = [min(t['actual_set_frame_temp_60pct'].min(),
                t['rec_set_frame_temp_60pct'].min()),
            max(t['actual_set_frame_temp_60pct'].max(),
                t['rec_set_frame_temp_60pct'].max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('실제 temp_60pct'); ax.set_ylabel('역산 temp_60pct')
    ax.set_title('15-19년식: frame_temp_60pct 역산 vs 실제', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pt.join(plot_dir, 'temp60_inverse_vs_actual.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 temp scatter 저장")


if __name__ == '__main__':
    validate(CONFIG)
