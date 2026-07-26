# Wire Saw APC TF : 뭐가 문제 일까? 알아보자,
# -*- coding: utf-8 -*-
"""
방향 검증 문제 진단
─────────────────────────────────────────
질문: 변경 그룹(Frame_Group/Tension_Group 0→1)에서 BOW가 변한 것이
      정말 recipe 변경 때문인가? 아니면 다른 요인(교란변수)인가?

진단 5종:
  ① 변경 그룹 간 recipe 실제 변화량 확인 (진짜 바뀌었나?)
  ② 변경 그룹 간 다른 조건들의 변화 (wait/ingot/slurry/WG가 같이 움직였나?)
  ③ 변경 그룹 간 기간 분포 (시점이 다른가 → 시간 교란)
  ④ 장비 내 recipe-BOW 상관 (그 장비에서 recipe가 BOW를 실제 설명하나?)
  ⑤ 다변량: 변경 외 요인 통제 시 recipe 효과가 남는가
"""
import os
import os.path as pt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8

CONFIG = {
    'test_csv': r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',
    'out_dir':  r'./direction_diagnosis',
    'eqp_col':  'eqp_nm_3200',
    'date_col': 'date_3200',
    'target':   'avg_bow_bf_total',
    'tension_col': 'fdc_set_tension',
    'temp_rep': 'set_frame_temp_60pct',
    'eqp_groups': {
        '15-19': {'eqps': ['BSWS38','BSWS42','BSWS44'],
                  'optimize': 'temp', 'change_col': 'Frame_Group',
                  'recipe_col': 'set_frame_temp_60pct'},
        '21':    {'eqps': ['BSWS52','BSWS54','BSWS55','BSWS56','BSWS61'],
                  'optimize': 'tension', 'change_col': 'Tension_Group',
                  'recipe_col': 'fdc_set_tension'},
    },
    # 교란 후보 (변경 그룹 간 이것들도 바뀌었는지 확인)
    'confound_cols': ['fdc_wait_time', 'fdc_ingot_len',
                      'range_slurry_temp_10_0', 'range_wire_guide_10_99'],
    'encoding': 'utf-8',
}


def diagnose(cfg):
    df = pd.read_csv(cfg['test_csv'], encoding=cfg['encoding'],
                     encoding_errors='replace')
    df[cfg['date_col']] = pd.to_datetime(df[cfg['date_col']], errors='coerce')
    os.makedirs(cfg['out_dir'], exist_ok=True)
    EQP, TARGET = cfg['eqp_col'], cfg['target']

    all_rows = []

    for grp_name, gc in cfg['eqp_groups'].items():
        change_col = gc['change_col']
        recipe_col = gc['recipe_col']
        print(f"\n{'='*64}\n[{grp_name}] 변경={change_col}, recipe={recipe_col}\n{'='*64}")

        for eqp in gc['eqps']:
            sub = df[df[EQP] == eqp].dropna(subset=[TARGET, change_col])
            if len(sub) == 0:
                continue
            g0 = sub[sub[change_col] == 0]
            g1 = sub[sub[change_col] == 1]
            if len(g0) < 5 or len(g1) < 5:
                print(f"  {eqp}: 그룹 샘플 부족 (0:{len(g0)}, 1:{len(g1)})")
                continue

            row = {'eqp': eqp, 'group': grp_name, 'n_0': len(g0), 'n_1': len(g1)}

            # ── ① recipe 실제 변화 (진짜 바뀌었나?) ──
            r0, r1 = g0[recipe_col].mean(), g1[recipe_col].mean()
            r_change = r1 - r0
            r_change_pct = r_change / (abs(r0) + 1e-9) * 100
            row['recipe_0'] = round(r0, 3)
            row['recipe_1'] = round(r1, 3)
            row['recipe_Δ'] = round(r_change, 3)
            # t-test: recipe 변화가 유의한가
            _, p_recipe = stats.ttest_ind(g0[recipe_col].dropna(),
                                          g1[recipe_col].dropna(), equal_var=False)
            row['recipe_Δ_p'] = round(p_recipe, 4)

            # ── BOW 변화 ──
            b0, b1 = g0[TARGET].mean(), g1[TARGET].mean()
            row['bow_0'] = round(b0, 3)
            row['bow_1'] = round(b1, 3)
            row['bow_Δ'] = round(b1 - b0, 3)
            _, p_bow = stats.ttest_ind(g0[TARGET].dropna(),
                                       g1[TARGET].dropna(), equal_var=False)
            row['bow_Δ_p'] = round(p_bow, 4)

            # 방향 일치 여부
            row['dir_match'] = int(np.sign(r_change) == np.sign(b1 - b0))

            # ── ② 교란 변수도 같이 바뀌었나? ──
            confound_changes = []
            for cc in cfg['confound_cols']:
                if cc in sub.columns:
                    c0, c1 = g0[cc].mean(), g1[cc].mean()
                    if pd.notna(c0) and pd.notna(c1) and abs(c0) > 1e-9:
                        pct = (c1 - c0) / abs(c0) * 100
                        row[f'{cc}_Δ%'] = round(pct, 1)
                        # 10% 이상 변한 교란 변수 카운트
                        if abs(pct) > 10:
                            confound_changes.append(f"{cc}({pct:+.0f}%)")
            row['confounds_changed'] = '; '.join(confound_changes) if confound_changes else '없음'

            # ── ③ 기간 분포 (시간 교란) ──
            d0_mid = g0[cfg['date_col']].mean()
            d1_mid = g1[cfg['date_col']].mean()
            if pd.notna(d0_mid) and pd.notna(d1_mid):
                gap_days = abs((d1_mid - d0_mid).days)
                row['기간차_일'] = gap_days
                # 시간이 완전히 분리되어 있으면 = recipe 아닌 시간 효과 가능성
                overlap = (g0[cfg['date_col']].max() >= g1[cfg['date_col']].min()) and \
                          (g1[cfg['date_col']].max() >= g0[cfg['date_col']].min())
                row['기간_겹침'] = '겹침' if overlap else '분리'

            # ── ④ 장비 내 recipe-BOW 상관 (recipe가 실제 BOW 설명?) ──
            s2 = sub[[recipe_col, TARGET]].dropna()
            if len(s2) > 10 and s2[recipe_col].std() > 1e-9:
                r_corr, p_corr = stats.pearsonr(s2[recipe_col], s2[TARGET])
                row['recipe_BOW_r'] = round(r_corr, 3)
                row['recipe_BOW_p'] = round(p_corr, 4)
            else:
                row['recipe_BOW_r'] = None

            all_rows.append(row)

            # 콘솔 요약
            print(f"  {eqp}: recipe {r0:.2f}→{r1:.2f} (Δ{r_change:+.2f}, p={p_recipe:.3f}) | "
                  f"BOW {b0:.2f}→{b1:.2f} (Δ{b1-b0:+.2f}) | "
                  f"{'방향일치' if row['dir_match'] else '방향반대'}")
            if confound_changes:
                print(f"      ⚠ 교란 변수 변화: {row['confounds_changed']}")
            if row.get('기간_겹침') == '분리':
                print(f"      ⚠ 변경 전/후 기간 분리 ({row.get('기간차_일')}일차) → 시간 교란 가능")

    # 저장
    res = pd.DataFrame(all_rows)
    res.to_csv(pt.join(cfg['out_dir'], 'direction_diagnosis.csv'),
               index=False, encoding='utf-8-sig')

    # ── 종합 판정 ──
    print(f"\n{'='*64}\n종합 판정\n{'='*64}")
    _summarize(res)

    # ── 시각화 ──
    _plot(res, cfg['out_dir'])

    print(f"\n💾 저장: {cfg['out_dir']}/direction_diagnosis.csv")
    return res


def _summarize(res):
    if len(res) == 0:
        print("데이터 없음")
        return
    n = len(res)
    n_match = int(res['dir_match'].sum())
    print(f"방향 일치: {n_match}/{n} ({n_match/n*100:.0f}%)")

    # recipe가 유의하게 변했는지
    if 'recipe_Δ_p' in res.columns:
        n_sig_recipe = int((res['recipe_Δ_p'] < 0.05).sum())
        print(f"recipe가 유의하게 변한 장비: {n_sig_recipe}/{n}")

    # 교란 변수가 같이 변한 장비
    n_confound = int((res['confounds_changed'] != '없음').sum())
    print(f"교란 변수도 크게 변한 장비: {n_confound}/{n}")

    # 기간 분리된 장비 (시간 교란)
    if '기간_겹침' in res.columns:
        n_sep = int((res['기간_겹침'] == '분리').sum())
        print(f"변경 전/후 기간이 분리된 장비: {n_sep}/{n} (시간 교란 위험)")

    # recipe-BOW 상관이 유의한 장비
    if 'recipe_BOW_p' in res.columns:
        valid = res['recipe_BOW_p'].dropna()
        n_corr_sig = int((valid < 0.05).sum())
        print(f"recipe-BOW 상관이 유의한 장비: {n_corr_sig}/{len(valid)}")

    # 결론 가이드
    print(f"\n[해석 가이드]")
    if n_confound > n / 2:
        print("  ⚠ 교란 변수가 함께 변한 장비가 많음 → BOW 변화가 recipe 단독 효과가 아닐 수 있음")
    if '기간_겹침' in res.columns and int((res['기간_겹침']=='분리').sum()) > n/2:
        print("  ⚠ 변경 전/후 기간이 분리됨 → 시간에 따른 다른 요인(정비/계절)의 효과일 가능성")
    if n_match < n / 2:
        print("  ⚠ 방향 불일치가 많음 → recipe→BOW 관계가 이 test 세트에서 약하거나 교란됨")
        print("     → inverse 방향성 주장은 신중히. 개별 장비별로 조건 통제 후 재확인 필요")


def _plot(res, out_dir):
    if len(res) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ① recipe Δ vs BOW Δ (방향 일치 확인)
    ax = axes[0]
    for grp in res['group'].unique():
        s = res[res['group'] == grp]
        ax.scatter(s['recipe_Δ'], s['bow_Δ'], s=100, alpha=0.7, label=grp)
        for _, r in s.iterrows():
            ax.annotate(r['eqp'].replace('BSWS',''), (r['recipe_Δ'], r['bow_Δ']),
                        fontsize=7, ha='center', va='bottom')
    ax.axhline(0, color='k', linewidth=0.6); ax.axvline(0, color='k', linewidth=0.6)
    ax.set_xlabel('recipe 변화 (Δ)'); ax.set_ylabel('BOW 변화 (Δ)')
    ax.set_title('recipe 변화 vs BOW 변화\n(1,3사분면=방향일치)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ② 장비별 recipe-BOW 상관
    ax = axes[1]
    if 'recipe_BOW_r' in res.columns:
        s = res.dropna(subset=['recipe_BOW_r'])
        colors = ['#2ecc71' if p < 0.05 else '#95a5a6'
                  for p in s['recipe_BOW_p']]
        ax.barh(s['eqp'], s['recipe_BOW_r'], color=colors, edgecolor='k',
                linewidth=0.4)
        ax.axvline(0, color='k', linewidth=0.8)
        ax.set_xlabel('장비 내 recipe-BOW 상관 (r)')
        ax.set_title('장비별 recipe-BOW 상관\n(초록=p<0.05 유의)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(pt.join(out_dir, 'direction_diagnosis.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("📊 진단 그림 저장")


if __name__ == '__main__':
    diagnose(CONFIG)
