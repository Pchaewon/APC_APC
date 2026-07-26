# Wire Saw APC TF : Plot recipe comparision
# -*- coding: utf-8 -*-
"""
장비별 실측 vs 예측 recipe 라인 비교 plot
─────────────────────────────────────────
inverse_inference.py의 validation_numeric.csv를 입력으로 사용.

두 종류:
  ① temp: 각 wire_id별로 x축=position(0~100pct), y축=temp
          실측(actual) 실선 vs 역산(rec) 점선 비교, 장비별 서브플롯
  ② tension: x축=wire_id(또는 순번), y축=tension
             실측 vs 역산 라인 비교, 장비별

입력 CSV에 wire_id 컬럼이 있어야 라인 구분됨.
없으면 행 순번으로 대체.
"""
import os
import os.path as pt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'numeric_csv': r'./inverse_inference/validation_numeric.csv',
    'test_csv':    r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',  # wire_id·전체 temp 참조용
    'out_dir':     r'./inverse_inference/recipe_plots',
    'eqp_col':     'eqp_nm_3200',
    'wire_col':    'new_fdc_wire_id',    # 없으면 순번 사용
    'target':      'avg_bow_bf_total',
    'tension_col': 'fdc_set_tension',
    'temp_cols':   [f'set_frame_temp_{p}pct' for p in
                    [0,10,20,30,40,50,60,70,80,90,99,100]],
    'temp_positions': [0,10,20,30,40,50,60,70,80,90,99,100],
    'max_wires_per_eqp': 8,   # 너무 많으면 라인 겹침 → 대표 N개만
    'encoding': 'utf-8',
}


# ============================================================
# ① TEMP: position별 실측 vs 역산 (wire_id별 라인)
# ============================================================
def plot_temp_comparison(cfg):
    """
    각 장비 서브플롯: x=position, y=temp
    wire_id별로 실측 라인 + 역산 라인 비교.

    numeric_csv에는 rec_set_frame_temp_{0,60,100}pct만 있으므로,
    전체 12 position 역산값이 필요하면 test_csv에서 실측 temp를 가져오고
    역산은 대표 3점만 표시하거나, inverse를 재실행해야 함.
    → 여기서는 numeric_csv의 대표 3점(0/60/100) + 실측 전체를 비교.
    """
    num = pd.read_csv(cfg['numeric_csv'], encoding=cfg['encoding'])
    temp_part = num[num['optimize'] == 'temp'].copy()
    if len(temp_part) == 0:
        print("⚠ temp 데이터 없음")
        return

    os.makedirs(cfg['out_dir'], exist_ok=True)
    eqps = sorted(temp_part['eqp'].unique())

    # test_csv에서 전체 temp·wire_id 참조 (실측 전체 프로파일용)
    test_df = None
    if os.path.exists(cfg['test_csv']):
        test_df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                              encoding_errors='replace')

    rep_pos = [0, 60, 100]   # numeric_csv에 있는 대표 position
    rep_cols_rec = [f'rec_set_frame_temp_{p}pct' for p in rep_pos]
    rep_cols_act = [f'actual_set_frame_temp_{p}pct' for p in rep_pos]

    n = len(eqps)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), squeeze=False)
    fig.suptitle('15-19년식: Frame Temp 실측 vs 역산 (대표 position)',
                 fontsize=13, fontweight='bold')

    for ax, eqp in zip(axes[0], eqps):
        sub = temp_part[temp_part['eqp'] == eqp].reset_index(drop=True)
        # 대표 N개 행만 (라인 겹침 방지)
        if len(sub) > cfg['max_wires_per_eqp']:
            idx = np.linspace(0, len(sub)-1, cfg['max_wires_per_eqp']).astype(int)
            sub = sub.iloc[idx].reset_index(drop=True)

        cmap = plt.get_cmap('tab10')
        for i, (_, row) in enumerate(sub.iterrows()):
            c = cmap(i % 10)
            act = [row.get(col) for col in rep_cols_act]
            rec = [row.get(col) for col in rep_cols_rec]
            ax.plot(rep_pos, act, 'o-', color=c, alpha=0.7, linewidth=1.5,
                    label=f'#{i} 실측' if i < 3 else None)
            ax.plot(rep_pos, rec, 's--', color=c, alpha=0.7, linewidth=1.2,
                    label=f'#{i} 역산' if i < 3 else None)

        ax.set_xlabel('Position (%)'); ax.set_ylabel('set_frame_temp')
        ax.set_title(f'{eqp} (N={len(temp_part[temp_part["eqp"]==eqp])})',
                     fontweight='bold')
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout()
    fpath = pt.join(cfg['out_dir'], 'temp_actual_vs_rec_by_eqp.png')
    plt.savefig(fpath, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"📊 temp 비교 저장: {fpath}")


# ============================================================
# ①-b TEMP 전체 프로파일 (inverse 재실행으로 12점 전부)
# ============================================================
def plot_temp_full_profile(cfg, model_dir=r'./apc_model/full'):
    """
    12 position 전체 역산 프로파일이 필요할 때.
    inverse_inference의 함수를 재사용해 각 wire별 12점 역산.
    """
    import json, pickle
    from inverse_inference import load_model, inverse_for_target

    test_df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                          encoding_errors='replace')
    model, scaler, meta = load_model(model_dir)

    EQP = cfg['eqp_col']; WIRE = cfg['wire_col']; TARGET = cfg['target']
    TEMP = cfg['temp_cols']; POS = cfg['temp_positions']

    temp_eqps = ['BSWS38', 'BSWS42', 'BSWS44']   # 15-19년식
    os.makedirs(cfg['out_dir'], exist_ok=True)

    n = len(temp_eqps)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), squeeze=False)
    fig.suptitle('15-19년식: Frame Temp 전체 프로파일 (실측 vs 역산, wire별)',
                 fontsize=13, fontweight='bold')

    for ax, eqp in zip(axes[0], temp_eqps):
        sub = test_df[test_df[EQP] == eqp].dropna(subset=[TARGET])
        if len(sub) == 0:
            ax.set_title(f'{eqp} (없음)'); continue

        # wire별 그룹 (없으면 순번)
        if WIRE in sub.columns:
            groups = list(sub.groupby(WIRE))
        else:
            sub = sub.reset_index(drop=True)
            groups = [(i, sub.iloc[[i]]) for i in range(len(sub))]

        if len(groups) > cfg['max_wires_per_eqp']:
            idx = np.linspace(0, len(groups)-1,
                              cfg['max_wires_per_eqp']).astype(int)
            groups = [groups[i] for i in idx]

        cmap = plt.get_cmap('tab10')
        for i, (wid, g) in enumerate(groups):
            row = g.iloc[0]
            c = cmap(i % 10)
            # 실측 12점
            act = [row.get(col, np.nan) for col in TEMP]
            # 역산 12점
            target_y = float(row[TARGET])
            inv = inverse_for_target(model, scaler, meta, target_y,
                                     base_row=row.to_dict(),
                                     optimize='temp', eqp_name=eqp)
            rec = [inv['optimized'].get(col, np.nan) for col in TEMP]

            ax.plot(POS, act, 'o-', color=c, alpha=0.7, linewidth=1.4,
                    label=f'wire {wid} 실측')
            ax.plot(POS, rec, 's--', color=c, alpha=0.6, linewidth=1.1)

        ax.set_xlabel('Position (%)'); ax.set_ylabel('set_frame_temp')
        ax.set_title(f'{eqp}', fontweight='bold')
        ax.legend(fontsize=6); ax.grid(alpha=0.3)

    plt.tight_layout()
    fpath = pt.join(cfg['out_dir'], 'temp_full_profile_by_wire.png')
    plt.savefig(fpath, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"📊 temp 전체 프로파일 저장: {fpath}")


# ============================================================
# ② TENSION: wire_id별 실측 vs 역산 라인
# ============================================================
def plot_tension_comparison(cfg):
    """
    각 장비 서브플롯: x=wire_id(순번), y=tension
    실측 라인 vs 역산 라인.
    """
    num = pd.read_csv(cfg['numeric_csv'], encoding=cfg['encoding'])
    t_part = num[num['optimize'] == 'tension'].copy()
    if len(t_part) == 0:
        print("⚠ tension 데이터 없음")
        return

    os.makedirs(cfg['out_dir'], exist_ok=True)
    eqps = sorted(t_part['eqp'].unique())

    n = len(eqps)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow), squeeze=False)
    fig.suptitle('21년식: Tension 실측 vs 역산 (wire 순번별)',
                 fontsize=13, fontweight='bold')

    for idx, eqp in enumerate(eqps):
        ax = axes[idx // ncol][idx % ncol]
        sub = t_part[t_part['eqp'] == eqp].reset_index(drop=True)
        x = np.arange(len(sub))   # wire 순번

        ax.plot(x, sub['actual_tension'], 'o-', color='#3498db',
                linewidth=1.5, markersize=4, label='실측', alpha=0.8)
        ax.plot(x, sub['rec_tension'], 's--', color='#e74c3c',
                linewidth=1.3, markersize=4, label='역산', alpha=0.8)

        # 차이 음영
        ax.fill_between(x, sub['actual_tension'], sub['rec_tension'],
                        alpha=0.15, color='gray')

        mae = (sub['rec_tension'] - sub['actual_tension']).abs().mean()
        ax.set_xlabel('wire 순번'); ax.set_ylabel('tension')
        ax.set_title(f'{eqp} (N={len(sub)}, |Δ|평균={mae:.3f})',
                     fontweight='bold')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 빈 서브플롯 숨김
    for idx in range(n, nrow*ncol):
        axes[idx // ncol][idx % ncol].axis('off')

    plt.tight_layout()
    fpath = pt.join(cfg['out_dir'], 'tension_actual_vs_rec_by_eqp.png')
    plt.savefig(fpath, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"📊 tension 비교 저장: {fpath}")


# ============================================================
# 메인
# ============================================================
def main():
    cfg = CONFIG
    print("① temp 대표 position 비교...")
    plot_temp_comparison(cfg)

    print("\n② tension 비교...")
    plot_tension_comparison(cfg)

    # 전체 12점 프로파일 (inverse 재실행, 시간 걸림 — 선택)
    try:
        print("\n①-b temp 전체 프로파일 (재실행)...")
        plot_temp_full_profile(cfg)
    except Exception as e:
        print(f"  ⚠ 전체 프로파일 스킵: {e}")

    print(f"\n✅ 완료: {cfg['out_dir']}/")


if __name__ == '__main__':
    main()
