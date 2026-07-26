# -*- coding: utf-8 -*-
"""
장비별 temp 실측 vs 예측 라인 plot
─────────────────────────────────────────
구조:
  x축: wire_id 순서대로 나열, 각 wire_id 내부에 temp pct(0~100) 12점 펼침
       → wire1[0,10,...,100] | wire2[0,10,...,100] | ...
  y축: temp 값
  라인: 실측(actual) 실선 vs 예측(rec) 점선
  장비별로 서브플롯 분리

입력: inverse_inference.py의 validation_numeric.csv
      (temp 12개 rec_/actual_ 컬럼 + wire_id 필요)
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

CONFIG = {
    'numeric_csv': r'./inverse_inference/validation_numeric.csv',
    'out_dir':     r'./inverse_inference/temp_line_plots',
    'temp_pcts':   [0,10,20,30,40,50,60,70,80,90,99,100],
    'max_wires':   15,          # 장비당 최대 wire 수 (많으면 x축 과밀)
    'encoding':    'utf-8',
}


def plot_temp_by_wire(cfg):
    num = pd.read_csv(cfg['numeric_csv'], encoding=cfg['encoding'])
    temp_part = num[num['optimize'] == 'temp'].copy()
    if len(temp_part) == 0:
        print("⚠ temp 데이터 없음")
        return

    os.makedirs(cfg['out_dir'], exist_ok=True)
    PCTS = cfg['temp_pcts']
    rec_cols = [f'rec_set_frame_temp_{p}pct' for p in PCTS]
    act_cols = [f'actual_set_frame_temp_{p}pct' for p in PCTS]

    # 컬럼 존재 확인
    missing = [c for c in rec_cols + act_cols if c not in temp_part.columns]
    if missing:
        print(f"⚠ 누락 컬럼 {len(missing)}개 (예: {missing[:3]})")
        print("  → inverse_inference.py를 최신본으로 재실행해 12개 pct 저장 필요")
        return

    eqps = sorted(temp_part['eqp'].unique())

    for eqp in eqps:
        sub = temp_part[temp_part['eqp'] == eqp].reset_index(drop=True)

        # wire_id 있으면 그걸로, 없으면 행 순번
        if 'wire_id' in sub.columns and sub['wire_id'].notna().any():
            sub = sub.sort_values('wire_id').reset_index(drop=True)
            wire_ids = sub['wire_id'].tolist()
        else:
            wire_ids = list(range(len(sub)))

        # 너무 많으면 대표 N개 균등 샘플
        if len(sub) > cfg['max_wires']:
            idx = np.linspace(0, len(sub)-1, cfg['max_wires']).astype(int)
            sub = sub.iloc[idx].reset_index(drop=True)
            wire_ids = [wire_ids[i] for i in idx]

        n_wire = len(sub)
        n_pct = len(PCTS)

        fig, ax = plt.subplots(figsize=(max(12, n_wire * 1.2), 5))

        # 각 wire를 x축에서 연속 구간으로 배치
        # wire i의 pct들은 x = i*n_pct + [0..n_pct-1]
        x_all_act, y_all_act = [], []
        x_all_rec, y_all_rec = [], []

        for i, (_, row) in enumerate(sub.iterrows()):
            x_base = i * n_pct
            xs = [x_base + j for j in range(n_pct)]
            act = [row.get(c, np.nan) for c in act_cols]
            rec = [row.get(c, np.nan) for c in rec_cols]

            # wire 내부는 실선으로 연결, wire 간은 끊음 (NaN 삽입)
            x_all_act.extend(xs + [np.nan])
            y_all_act.extend(act + [np.nan])
            x_all_rec.extend(xs + [np.nan])
            y_all_rec.extend(rec + [np.nan])

            # wire 경계 세로선
            if i > 0:
                ax.axvline(x_base - 0.5, color='lightgray',
                           linewidth=0.8, linestyle=':')

        ax.plot(x_all_act, y_all_act, '-', color='#3498db', linewidth=1.6,
                marker='o', markersize=3, label='실측', alpha=0.85)
        ax.plot(x_all_rec, y_all_rec, '--', color='#e74c3c', linewidth=1.4,
                marker='s', markersize=3, label='예측(역산)', alpha=0.85)

        # x축 눈금: 각 wire 중앙에 wire_id 라벨
        tick_pos = [i * n_pct + (n_pct-1)/2 for i in range(n_wire)]
        tick_lab = [str(w)[:8] for w in wire_ids]   # 너무 길면 자름
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=45, ha='right', fontsize=7)

        ax.set_xlabel('wire_id (각 구간 내부: temp 0→100pct)', fontsize=10)
        ax.set_ylabel('set_frame_temp', fontsize=10)
        ax.set_title(f'{eqp} — Frame Temp 실측 vs 예측 (wire_id별)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        # gap 평균 텍스트
        if 'gap' in sub.columns:
            ax.text(0.01, 0.98, f"평균 gap(예측BOW-목표): {sub['gap'].mean():.3f}",
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        plt.tight_layout()
        fpath = pt.join(cfg['out_dir'], f'temp_line_{eqp}.png')
        plt.savefig(fpath, dpi=140, bbox_inches='tight')
        plt.close()
        print(f"📊 {eqp}: {fpath}")

    print(f"\n✅ 완료: {cfg['out_dir']}/")


if __name__ == '__main__':
    plot_temp_by_wire(CONFIG)
