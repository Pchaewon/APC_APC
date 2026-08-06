# -*- coding: utf-8 -*-
"""
Wire Saw APC Report 생성 (HTML, 엔지니어 공유용)
─────────────────────────────────────────
구성:
  · Title / Date / 장비명
  · 추천 Recipe:
      ① Frame Temp 프로파일 (0~100%) + 테이블
      ② Slurry Temp 프로파일 (0~100%) + 테이블
      ③ 적용 시 예상 BOW 범위
  · Warp Trend: Total/Seed/Mid/Tail (최근 10 lot)
  · Bow Trend:  Total/Seed/Mid/Tail (최근 10 lot)

입력:
  · 학습 모델 (train_inverse_rolling.py 산출)
  · test/최근 데이터 (해당 장비의 Trend용)
  · 역산 결과 (Frame/Slurry 추천)

출력: report_{장비명}_{날짜}.html
"""
import os
import os.path as pt
import json
import pickle
import base64
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CONFIG = {
    # dual 모델 디렉토리 (frame/, slurry/ 하위 폴더)
    'model_dir':  r'./apc_model_dual',
    'recent_csv': r'D:\chaewon\APC\02.TF\260726\data\test_df.csv',
    'out_dir':    r'./reports',
    'eqp_name':   'BSWS30',           # 리포트 대상 장비
    'target_bow': 1.75,
    'eqp_col':    'eqp_nm_3200',
    'date_col':   'date_3200',
    'wire_col':   'new_fdc_wire_id',
    # 프로파일 컬럼
    'frame_cols':  [f'set_frame_temp_{p}pct' for p in
                    [0,10,20,30,40,50,60,70,80,90,99,100]],
    'slurry_cols': [f'set_slurry_temp_{p}pct' for p in
                    [0,10,20,30,40,50,60,70,80,90,99,100]],
    'temp_pcts':   [0,10,20,30,40,50,60,70,80,90,99,100],
    # Trend 대상 (BOW/Warp × Total/Seed/Mid/Tail)
    'bow_cols':  {'Total':'avg_bow_bf_total', 'Seed':'avg_bow_bf_seed',
                  'Mid':'avg_bow_bf_mid', 'Tail':'avg_bow_bf_tail'},
    'warp_cols': {'Total':'avg_warp_bf_total', 'Seed':'avg_warp_bf_seed',
                  'Mid':'avg_warp_bf_mid', 'Tail':'avg_warp_bf_tail'},
    'trend_n': 10,                    # 최근 10 lot
    'roll_source_cols': ['fdc_set_tension','fdc_wait_time','fdc_ingot_len',
                         'range_slurry_temp_10_0'],
    'lag': 2, 'window': 10, 'min_runs': 3,
    # BOW 스펙 (양품 범위) + 목표선
    'bow_spec': (1.5, 2.0),           # 스펙 하한/상한
    'bow_target': 1.75,               # 목표선
    # 신뢰도 판정 (직전 WG 상태)
    'wg_col': 'range_wire_guide_10_99',
    'wg_var_threshold': 11.6,
    'report_title': 'Wire Saw APC Report',
    # Frame inverse 시작점 default (0pct)
    'frame_start_default': 28.0,
    # ── 다중 장비 (일괄 생성) ──
    'batch_eqps': {
        '15-19': ['BSWS30','BSWS31','BSWS35','BSWS42','BSWS48'],
        '21':    ['BSWS51','BSWS53','BSWS55','BSWS57','BSWS58'],
    },
    # ── X인자 프로파일 섹션 ──
    # pct 프로파일 있는 인자 (선 10개 = 10 lot, x축 pct)
    'xprofile_pct': {
        'Frame Temp':  [f'set_frame_temp_{p}pct' for p in
                        [0,10,20,30,40,50,60,70,80,90,99,100]],
        'Slurry Temp': [f'set_slurry_temp_{p}pct' for p in
                        [0,10,20,30,40,50,60,70,80,90,99,100]],
        'WireGuide L': [f'shift_of_wireguide_l_{p}pct' for p in
                        [0,10,20,30,40,50,60,70,80,90,99,100]],
        'WireGuide R': [f'shift_of_wireguide_r_{p}pct' for p in
                        [0,10,20,30,40,50,60,70,80,90,99,100]],
    },
    # lot당 값 1개 인자 (선 1개, x축 최근 10 lot)
    'xprofile_scalar': {
        'Ingot Length': 'fdc_ingot_len',
        'Wait Time':    'fdc_wait_time',
        'Warm-up Time': 'warm_up_time',
    },
    'encoding': 'utf-8',
}


# ═══════════════════════════════════════
# Dual 모델 로드 + 역산 (Frame/Slurry 각 별도 모델)
# ═══════════════════════════════════════
def load_profile_model(model_dir, name):
    """frame 또는 slurry 하위 폴더의 모델 로드."""
    mdir = pt.join(model_dir, name)
    if not os.path.exists(pt.join(mdir, 'model.pkl')):
        return None
    with open(pt.join(mdir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(pt.join(mdir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(pt.join(mdir, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    return model, scaler, meta


def inverse_profile(model_dir, name, target_bow, roll_values, eqp_name,
                    frame_start=None):
    """해당 프로파일(frame/slurry) 전용 모델로 역산."""
    loaded = load_profile_model(model_dir, name)
    if loaded is None:
        return None
    model, scaler, meta = loaded
    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    profile_cols = meta['profile_cols']; roll_cols = meta.get('roll_cols', [])
    eqp_cols = meta.get('eqp_cols', []); pfx = meta.get('eqp_prefix', 'eqp_')
    opt_cols = [c for c in profile_cols if c in FEATURES]
    if len(opt_cols) == 0:
        return None

    def gv(c, override):
        if c in eqp_cols:
            return 1.0 if c == f'{pfx}{eqp_name}' else 0.0
        if override is not None and c in opt_cols:
            return float(override[opt_cols.index(c)])
        if c in roll_cols:
            return float(roll_values.get(c, X_STATS.get(c, {}).get('mean', 0.0)))
        return float(X_STATS.get(c, {}).get('mean', 0.0))

    def predict(vec):
        x = np.array([gv(c, vec) for c in FEATURES]).reshape(1, -1)
        return float(model.predict(scaler.transform(x))[0])

    def objective(vec):
        return (predict(vec) - target_bow) ** 2

    x0 = np.array([X_STATS.get(c, {}).get('mean', 29.0) for c in opt_cols])
    # Frame 시작점(0pct) default 반영
    if name == 'frame' and frame_start is not None:
        for i, c in enumerate(opt_cols):
            if c.endswith('_0pct'):
                x0[i] = frame_start
    bounds = [(X_STATS.get(c, {}).get('q01', x0[i]-1),
               X_STATS.get(c, {}).get('q99', x0[i]+1))
              for i, c in enumerate(opt_cols)]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})
    rec = {c: round(float(v), 2) for c, v in zip(opt_cols, res.x)}
    return {'recipe': rec, 'predicted_bow': round(predict(res.x), 3),
            'cols': opt_cols, 'mae': meta.get('metrics', {}).get('mae', 0.1)}


# ═══════════════════════════════════════
# 그림 → base64 (HTML 임베드)
# ═══════════════════════════════════════
def fig_to_base64(fig):
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_profile(pcts, values, title, ylabel='Temp (°C)', accent='#1a1a1a'):
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(pcts, values, 'o-', color=accent, linewidth=2, markersize=5,
            markerfacecolor='white', markeredgewidth=1.5, zorder=3)
    ax.fill_between(pcts, values, min(v for v in values if v is not None),
                    alpha=0.05, color='#000000', zorder=1)
    ax.set_xlabel('Position (%)', fontsize=9, color='#333333')
    ax.set_ylabel(ylabel, fontsize=9, color='#333333')
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
    ax.tick_params(labelsize=8, colors='#4a5568')
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left','bottom']:
        ax.spines[spine].set_color('#cbd5e0')
    fig.patch.set_facecolor('white')
    return fig_to_base64(fig)


def plot_trend_row(recent, cols_map, trend_n, title_prefix, lot_labels=None,
                   spec=None, target=None, accent='#1a1a1a'):
    """Total/Seed/Mid/Tail 4개 그래프. 스펙선·목표선 옵션, x축 겹침 방지."""
    imgs = {}
    for pos, col in cols_map.items():
        fig, ax = plt.subplots(figsize=(3.1, 2.6))
        if col in recent.columns:
            s = recent[col].dropna().tail(trend_n)
            y = s.values
            if len(y) == 0:
                ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='#a0aec0')
            else:
                x = np.arange(len(y))
                # 스펙/목표 (BOW Trend에만)
                if spec is not None:
                    ax.axhspan(spec[0], spec[1], color='#000000', alpha=0.05, zorder=0)
                    ax.axhline(spec[0], color='#666666', lw=0.9, ls='--', alpha=0.8, zorder=1)
                    ax.axhline(spec[1], color='#666666', lw=0.9, ls='--', alpha=0.8, zorder=1)
                if target is not None:
                    ax.axhline(target, color='#1a1a1a', lw=1.0, ls=':', alpha=0.9, zorder=1)
                ax.plot(x, y, 'o-', color=accent, linewidth=1.8, markersize=4,
                        markerfacecolor='white', markeredgewidth=1.2, zorder=3)
                # x축 겹침 방지: lot 라벨을 최대 5개만 (처음/끝 포함 균등)
                if lot_labels is not None:
                    labels = lot_labels[-len(y):]
                    n = len(labels)
                    step = max(1, n // 5)
                    show_idx = list(range(0, n, step))
                    if n-1 not in show_idx:
                        show_idx.append(n-1)
                    ax.set_xticks([x[i] for i in show_idx])
                    ax.set_xticklabels([labels[i] for i in show_idx],
                                       fontsize=6.5, rotation=45, ha='right',
                                       color='#4a5568')
            ax.set_title(pos, fontsize=10, fontweight='bold', color='#1a1a1a')
        else:
            ax.text(0.5, 0.5, '컬럼 없음', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='#a0aec0')
            ax.set_title(pos, fontsize=10, fontweight='bold', color='#1a1a1a')
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=7, colors='#4a5568')
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left','bottom']:
            ax.spines[spine].set_color('#cbd5e0')
        imgs[pos] = fig_to_base64(fig)
    return imgs


def plot_xfactor_pct(esub, wire_col, date_col, cols, title, trend_n=10,
                     lot_col=None):
    """pct 프로파일 인자를 wire id별로 구획해 가로 연결.
    구조: wire id > lot id > pct
      · x축을 wire id 구간으로 나눔 (시간순)
      · 각 wire 구간 = 그 wire의 lot들 pct 평균 (선 1개)
      · 구간마다 pct(0~100%)로 프로파일을 가로로 펼침
    """
    avail = [c for c in cols if c in esub.columns]
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    if len(avail) == 0:
        ax.text(0.5, 0.5, f'{title}\n컬럼 없음', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='#a0aec0')
        ax.set_title(title, fontsize=10, fontweight='bold', color='#1a1a1a')
        return fig_to_base64(fig), 0

    n_pct = len(avail)
    # 최근 trend_n개 wire (시간순)
    wire_order = (esub.sort_values(date_col)[wire_col].drop_duplicates()
                  .tail(trend_n).tolist())
    if len(wire_order) == 0:
        ax.text(0.5, 0.5, 'wire 데이터 없음', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='#a0aec0')
        ax.set_title(title, fontsize=10, fontweight='bold', color='#1a1a1a')
        return fig_to_base64(fig), 0

    x_cursor = 0
    tick_positions = []   # wire 구간 중앙 (라벨용)
    boundary_positions = []  # wire 구간 경계 (구분선)
    n_drawn = 0
    gap = 1  # wire 구간 사이 간격

    for wi, w in enumerate(wire_order):
        wsub = esub[esub[wire_col] == w]
        if len(wsub) == 0:
            continue
        # 이 wire의 lot들 pct 평균 (여러 lot → 평균 프로파일)
        prof = [wsub[c].mean() for c in avail]
        # x 좌표: 이 구간의 pct 위치
        xs = list(range(x_cursor, x_cursor + n_pct))
        ax.plot(xs, prof, '-', color='#1a1a1a', linewidth=1.3, zorder=3)
        ax.plot(xs, prof, '.', color='#1a1a1a', markersize=2, zorder=3)
        # 구간 중앙 (라벨)
        tick_positions.append((w, x_cursor + n_pct/2 - 0.5))
        n_drawn += 1
        x_cursor += n_pct
        # 구간 경계선
        if wi < len(wire_order) - 1:
            boundary_positions.append(x_cursor + gap/2 - 0.5)
        x_cursor += gap

    # wire 구간 경계 (연한 세로선)
    for bx in boundary_positions:
        ax.axvline(bx, color='#d0d0d0', linewidth=0.8, zorder=1)

    # x축: wire id 라벨 (구간 중앙)
    ax.set_xticks([pos for _, pos in tick_positions])
    ax.set_xticklabels([str(w) for w, _ in tick_positions],
                       fontsize=6.5, rotation=45, ha='right', color='#333333')
    ax.set_xlabel('Wire ID (각 구간: pct 0→100%, 시간순)', fontsize=8.5,
                  color='#333333')
    ax.set_title(f'{title}  (wire {n_drawn}개 · lot평균 프로파일)', fontsize=10,
                 fontweight='bold', color='#1a1a1a')
    ax.grid(axis='y', alpha=0.22, linewidth=0.6)
    ax.tick_params(labelsize=7, colors='#333333')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#cbd5e0')
    return fig_to_base64(fig), n_drawn


def plot_xfactor_scalar(esub, date_col, col, title, lot_labels=None, trend_n=10):
    """lot당 값 1개 인자: 최근 10 lot 추세 (선 1개)."""
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    if col not in esub.columns:
        ax.text(0.5, 0.5, f'{title}\n컬럼 없음', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='#a0aec0')
        ax.set_title(title, fontsize=10, fontweight='bold', color='#1a1a1a')
        return fig_to_base64(fig)
    s = esub.sort_values(date_col)[col].dropna().tail(trend_n)
    y = s.values
    if len(y) == 0:
        ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='#a0aec0')
    else:
        x = np.arange(len(y))
        ax.plot(x, y, 'o-', color='#1a1a1a', linewidth=1.8, markersize=4,
                markerfacecolor='white', markeredgewidth=1.2)
        if lot_labels is not None:
            labels = lot_labels[-len(y):]
            n = len(labels); step = max(1, n // 5)
            idx = list(range(0, n, step))
            if n-1 not in idx: idx.append(n-1)
            ax.set_xticks([x[i] for i in idx])
            ax.set_xticklabels([labels[i] for i in idx], fontsize=6.5,
                               rotation=45, ha='right', color='#333333')
    ax.set_title(title, fontsize=10, fontweight='bold', color='#1a1a1a')
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.tick_params(labelsize=7, colors='#333333')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#cbd5e0')
    return fig_to_base64(fig)


# ═══════════════════════════════════════
# 리포트 생성
# ═══════════════════════════════════════
def build_report(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    eqp = cfg['eqp_name']
    today = datetime.now().strftime('%Y.%m.%d')

    # roll_cols는 frame 모델 meta에서 가져옴 (frame/slurry 공통)
    frame_loaded = load_profile_model(cfg['model_dir'], 'frame')
    roll_cols = (frame_loaded[2].get('roll_cols', []) if frame_loaded else [])

    # 최근 데이터 (Trend + 직전 평균)
    recent = pd.read_csv(cfg['recent_csv'], encoding=cfg['encoding'],
                         encoding_errors='replace')
    recent[cfg['date_col']] = pd.to_datetime(recent[cfg['date_col']], errors='coerce')
    esub = recent[recent[cfg['eqp_col']] == eqp].sort_values(cfg['date_col'])

    # 직전 평균 조건 (최근 window run 평균)
    roll_values = {}
    for rc in roll_cols:
        src = rc.replace('roll_', '')
        if src in esub.columns and len(esub) > 0:
            roll_values[rc] = float(esub[src].tail(cfg['window']).mean())

    # ── 신뢰도 판정 (직전 WG 상태) ──
    confidence, conf_note = 'unknown', '직전 WG 정보 부족'
    wg_col = cfg.get('wg_col')
    if wg_col and wg_col in esub.columns and len(esub) >= 2:
        # 최근(현재 직전) WG 이동중앙값
        prev_wg = esub[wg_col].iloc[:-1].tail(cfg['window']).median()
        if pd.notna(prev_wg):
            if prev_wg >= cfg['wg_var_threshold']:
                confidence = 'high'
                conf_note = '직전 Wire Guide 고변동 · 온도–BOW 관계 뚜렷 (예측 R²≈0.44) · 적극 반영 권장'
            else:
                confidence = 'low'
                conf_note = '직전 Wire Guide 저변동 · 관계 약함 (예측 R²≈0.09) · 참고용, 현재 recipe 유지 고려'

    # ── 역산: Frame / Slurry (각 전용 모델) ──
    frame_inv = inverse_profile(cfg['model_dir'], 'frame', cfg['target_bow'],
                                roll_values, eqp,
                                frame_start=cfg.get('frame_start_default'))
    slurry_inv = inverse_profile(cfg['model_dir'], 'slurry', cfg['target_bow'],
                                 roll_values, eqp)

    # 예상 BOW 범위 (frame 모델 기준, 예측 ± MAE)
    #   앙상블은 이득 없음 확인됨 → frame 모델로 예측
    mae = frame_inv['mae'] if frame_inv else 0.1
    pred_bow = frame_inv['predicted_bow'] if frame_inv else cfg['target_bow']
    bow_lo, bow_hi = round(pred_bow - mae, 1), round(pred_bow + mae, 1)

    # ── 프로파일 그림 ──
    pcts = cfg['temp_pcts']
    frame_vals = ([frame_inv['recipe'].get(c) for c in cfg['frame_cols']]
                  if frame_inv else [None]*12)
    slurry_vals = ([slurry_inv['recipe'].get(c) for c in cfg['slurry_cols']]
                   if slurry_inv else [None]*12)
    frame_img = (plot_profile(pcts, frame_vals, 'Frame in Temp', accent='#1a1a1a')
                 if frame_inv else None)
    slurry_img = (plot_profile(pcts, slurry_vals, 'Slurry in Temp', accent='#555555')
                  if slurry_inv else None)

    # ── Trend 그림 ──
    print(f"\n[Trend 진단] 장비 {eqp} 데이터 {len(esub)}행")
    if len(esub) == 0:
        print(f"  ⚠ '{eqp}' 데이터가 recent_csv에 없음 → Trend 전부 빈칸")
        print(f"     recent_csv의 장비 목록 확인 필요")
    # x축 lot 이름 (최근 trend_n개)
    lot_col = cfg.get('lot_col', cfg['wire_col'])
    # 지정 컬럼 없으면 wire 관련 컬럼 자동 탐색 (이름 순서 헷갈림 방어)
    if lot_col not in esub.columns:
        cands = [c for c in esub.columns if 'wire' in c.lower() and 'id' in c.lower()]
        if cands:
            print(f"  ℹ lot_col '{lot_col}' 없음 → '{cands[0]}' 사용")
            lot_col = cands[0]
    lot_labels = None
    if lot_col in esub.columns and len(esub) > 0:
        lot_labels = esub[lot_col].tail(cfg['trend_n']).astype(str).tolist()
        print(f"  lot 이름 컬럼: '{lot_col}', 예시: {lot_labels[:2]}")
    warp_imgs = plot_trend_row(esub, cfg['warp_cols'], cfg['trend_n'], 'Warp',
                               lot_labels=lot_labels, accent='#555555')
    bow_imgs = plot_trend_row(esub, cfg['bow_cols'], cfg['trend_n'], 'Bow',
                              lot_labels=lot_labels, spec=cfg.get('bow_spec'),
                              target=cfg.get('bow_target'), accent='#1a1a1a')

    # ── 테이블 HTML (세로형: %와 추천값 2열) ──
    def profile_table(vals, cols):
        if vals[0] is None:
            return '<p class="muted">해당 프로파일 데이터 없음</p>'
        rows = ''.join(
            f'<tr><td class="pct">{p}%</td>'
            f'<td class="val">{v:.1f}</td></tr>' if v is not None
            else f'<tr><td class="pct">{p}%</td><td class="val">–</td></tr>'
            for p, v in zip(pcts, vals))
        return (f'<table class="recipe-tbl"><thead>'
                f'<tr><th>Position</th><th>Temp (°C)</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>')

    frame_tbl = profile_table(frame_vals, cfg['frame_cols'])
    slurry_tbl = profile_table(slurry_vals, cfg['slurry_cols'])

    # ── X인자 프로파일 (새 섹션) ──
    wire_col = lot_col  # 위에서 확정된 lot/wire 컬럼
    xpct_imgs = {}   # {라벨: (img, n)}
    for label, cols in cfg.get('xprofile_pct', {}).items():
        img, n = plot_xfactor_pct(esub, wire_col, cfg['date_col'], cols,
                                  label, cfg['trend_n'], lot_col=wire_col)
        xpct_imgs[label] = img
    xscalar_imgs = {}
    for label, col in cfg.get('xprofile_scalar', {}).items():
        img = plot_xfactor_scalar(esub, cfg['date_col'], col, label,
                                  lot_labels=lot_labels, trend_n=cfg['trend_n'])
        xscalar_imgs[label] = img

    # ── HTML 조립 ──
    html = _render_html(cfg, eqp, today, frame_img, frame_tbl,
                        slurry_img, slurry_tbl, bow_lo, bow_hi,
                        warp_imgs, bow_imgs, confidence, conf_note,
                        pred_bow, mae, xpct_imgs, xscalar_imgs)

    out_path = pt.join(cfg['out_dir'],
                       f'report_{eqp}_{today.replace(".", "")}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ 리포트 저장: {out_path}")
    return out_path


def _img_tag(b64, alt=''):
    if b64 is None:
        return '<p class="muted">데이터 없음</p>'
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}"/>'


def _render_html(cfg, eqp, today, frame_img, frame_tbl, slurry_img, slurry_tbl,
                 bow_lo, bow_hi, warp_imgs, bow_imgs, confidence, conf_note,
                 pred_bow, mae, xpct_imgs, xscalar_imgs):
    def trend_block(imgs):
        cells = ''.join(
            f'<div class="trend-cell">{_img_tag(imgs.get(p))}</div>'
            for p in ['Total','Seed','Mid','Tail'])
        return f'<div class="trend-grid">{cells}</div>'

    # 신뢰도 배지
    conf_map = {
        'high': ('신뢰도 높음', '#1a1a1a', '#f2f2f2', '#1a1a1a'),
        'low':  ('신뢰도 낮음', '#666666', '#f7f7f7', '#999999'),
        'unknown': ('신뢰도 판정 불가', '#999999', '#fafafa', '#cccccc'),
    }
    clabel, ccolor, cbg, cborder = conf_map.get(confidence, conf_map['unknown'])

    spec_lo, spec_hi = cfg.get('bow_spec', (1.5, 2.0))
    tgt = cfg.get('bow_target', 1.75)

    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cfg["report_title"]} — {eqp}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --navy: #111111; --steel: #1a1a1a; --slate: #555555;
    --line: #e2e2e2; --line-strong: #b0b0b0;
    --bg: #ededed; --paper: #ffffff;
    --accent: #1a1a1a; --warn: #666666;
  }}
  body {{
    font-family: "Segoe UI", -apple-system, "Malgun Gothic", sans-serif;
    background: var(--bg); color: #1a1a1a; line-height: 1.55;
    padding: 28px 14px; -webkit-font-smoothing: antialiased;
  }}
  .doc {{
    max-width: 1000px; margin: 0 auto; background: var(--paper);
    border: 1px solid var(--line-strong);
  }}
  /* ── 문서 헤더 (레터헤드) ── */
  .letterhead {{
    border-top: 5px solid var(--navy);
    padding: 26px 40px 20px; border-bottom: 1px solid var(--line);
    display: flex; justify-content: space-between; align-items: flex-start;
  }}
  .lh-left .eyebrow {{
    font-size: 11px; letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--slate); font-weight: 600; margin-bottom: 6px;
  }}
  .lh-left h1 {{
    font-size: 25px; color: var(--navy); font-weight: 700;
    letter-spacing: -0.3px;
  }}
  .lh-meta {{ text-align: right; font-size: 13px; color: var(--slate); }}
  .lh-meta table {{ border-collapse: collapse; }}
  .lh-meta td {{ padding: 2px 0 2px 16px; }}
  .lh-meta .k {{ color: #888888; font-weight: 600; text-align: right;
    font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; }}
  .lh-meta .v {{ color: var(--navy); font-weight: 700;
    font-variant-numeric: tabular-nums; }}
  /* ── 신뢰도 배너 ── */
  .conf-banner {{
    margin: 0 40px; margin-top: 20px; padding: 12px 18px;
    background: {cbg}; border: 1px solid {cborder}33;
    border-left: 4px solid {cborder}; border-radius: 3px;
    display: flex; align-items: center; gap: 14px;
  }}
  .conf-badge {{
    font-size: 12px; font-weight: 700; color: #fff; background: {ccolor};
    padding: 4px 12px; border-radius: 3px; white-space: nowrap;
    letter-spacing: 0.3px;
  }}
  .conf-note {{ font-size: 13px; color: #3a4453; }}
  /* ── 섹션 ── */
  .section {{ padding: 26px 40px; border-top: 1px solid var(--line); }}
  .section:first-of-type {{ border-top: none; }}
  .sec-head {{
    display: flex; align-items: baseline; gap: 12px; margin-bottom: 18px;
  }}
  .sec-num {{
    font-size: 12px; font-weight: 700; color: #fff; background: var(--navy);
    width: 22px; height: 22px; border-radius: 3px; display: inline-flex;
    align-items: center; justify-content: center; flex-shrink: 0;
  }}
  .sec-title {{ font-size: 17px; font-weight: 700; color: var(--navy);
    letter-spacing: -0.2px; }}
  .sec-sub {{ font-size: 12px; color: #888888; margin-left: auto;
    font-weight: 500; }}
  /* ── Recipe (그래프+테이블 가로 배치, 여백 활용) ── */
  .recipe-item {{ margin-bottom: 22px; }}
  .recipe-item:last-child {{ margin-bottom: 0; }}
  .recipe-label {{
    font-size: 14px; font-weight: 700; color: var(--steel);
    margin-bottom: 12px; padding-left: 10px;
    border-left: 3px solid var(--steel);
  }}
  .recipe-body {{ display: grid; grid-template-columns: 1.55fr 1fr;
    gap: 24px; align-items: start; }}
  .recipe-chart img {{ width: 100%; border: 1px solid var(--line);
    border-radius: 3px; }}
  .recipe-tbl {{ width: 100%; border-collapse: collapse; font-size: 12px;
    font-variant-numeric: tabular-nums; }}
  .recipe-tbl thead th {{
    background: var(--navy); color: #fff; padding: 7px 10px;
    font-weight: 600; font-size: 11px; letter-spacing: 0.5px;
    text-transform: uppercase; text-align: center;
  }}
  .recipe-tbl td {{ border-bottom: 1px solid #eef1f4; padding: 5px 10px;
    text-align: center; }}
  .recipe-tbl .pct {{ color: var(--slate); font-weight: 600;
    background: #f7f7f7; width: 45%; }}
  .recipe-tbl .val {{ color: var(--navy); font-weight: 700; }}
  .recipe-tbl tbody tr:hover td {{ background: #f4f4f4; }}
  /* ── 예상 BOW ── */
  .bow-forecast {{
    background: #f7f7f7; border: 1px solid var(--line);
    border-radius: 4px; padding: 18px 22px; display: flex;
    align-items: center; gap: 28px; flex-wrap: wrap;
  }}
  .bow-forecast .big {{ font-size: 30px; font-weight: 800; color: var(--steel);
    font-variant-numeric: tabular-nums; letter-spacing: -0.5px; }}
  .bow-forecast .desc {{ font-size: 12.5px; color: var(--slate); }}
  .bow-forecast .desc b {{ color: var(--navy); }}
  .spec-chip {{ display: inline-block; font-size: 11px; padding: 2px 9px;
    border-radius: 3px; background: #f0f0f0; color: #1a1a1a;
    font-weight: 600; margin-left: 4px; border: 1px solid #d5d5d5; }}
  /* ── Trend ── */
  .trend-note {{ font-size: 12px; color: #888888; margin-bottom: 14px;
    display: flex; gap: 16px; align-items: center; }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 5px; }}
  .legend-line {{ width: 16px; height: 0; border-top: 2px solid; }}
  .trend-grid {{ display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 14px; }}
  .trend-cell img {{ width: 100%; border: 1px solid var(--line);
    border-radius: 3px; }}
  /* ── X-Factor ── */
  .xfactor-grid {{ display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 14px; }}
  .xfactor-wide {{ display: flex; flex-direction: column; gap: 18px; }}
  .xf-wide-cell {{ width: 100%; }}
  .xf-wide-cell img {{ width: 100%; border: 1px solid var(--line);
    border-radius: 3px; }}
  .xf-cell {{ display: flex; flex-direction: column; }}
  .xf-label {{ font-size: 12px; font-weight: 700; color: var(--steel);
    margin-bottom: 6px; }}
  .xf-cell img {{ width: 100%; border: 1px solid var(--line);
    border-radius: 3px; }}
  /* ── 푸터 ── */
  .footer {{ padding: 18px 40px; border-top: 2px solid var(--navy);
    font-size: 11.5px; color: #888888; display: flex;
    justify-content: space-between; }}
  @media (max-width: 720px) {{
    .recipe-body {{ grid-template-columns: 1fr; }}
    .trend-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .letterhead {{ flex-direction: column; gap: 14px; }}
    .lh-meta {{ text-align: left; }}
  }}
</style>
</head>
<body>
<div class="doc">
  <div class="letterhead">
    <div class="lh-left">
      <div class="eyebrow">Advanced Process Control · Wire Saw</div>
      <h1>{cfg["report_title"]}</h1>
    </div>
    <div class="lh-meta">
      <table>
        <tr><td class="k">Date</td><td class="v">{today}</td></tr>
        <tr><td class="k">Equipment</td><td class="v">{eqp}</td></tr>
        <tr><td class="k">Target BOW</td><td class="v">{tgt}</td></tr>
      </table>
    </div>
  </div>

  <div class="conf-banner">
    <span class="conf-badge">{clabel}</span>
    <span class="conf-note">{conf_note}</span>
  </div>

  <div class="section">
    <div class="sec-head">
      <span class="sec-num">1</span>
      <span class="sec-title">추천 Recipe</span>
      <span class="sec-sub">직전 {cfg["window"]} lot 평균 조건 기반 역산</span>
    </div>

    <div class="recipe-item">
      <div class="recipe-label">① Frame in Temp</div>
      <div class="recipe-body">
        <div class="recipe-chart">{_img_tag(frame_img)}</div>
        <div>{frame_tbl}</div>
      </div>
    </div>

    <div class="recipe-item">
      <div class="recipe-label">② Slurry in Temp</div>
      <div class="recipe-body">
        <div class="recipe-chart">{_img_tag(slurry_img)}</div>
        <div>{slurry_tbl}</div>
      </div>
    </div>

    <div class="recipe-item">
      <div class="recipe-label">③ Recipe 적용 시 예상 BOW</div>
      <div class="bow-forecast">
        <span class="big">{bow_lo} ~ {bow_hi}</span>
        <span class="desc">
          예측 <b>{pred_bow}</b> ± MAE <b>{mae:.2f}</b><br>
          양품 스펙 <span class="spec-chip">{spec_lo} ~ {spec_hi}</span>
          · 목표 <b>{tgt}</b>
        </span>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="sec-head">
      <span class="sec-num">2</span>
      <span class="sec-title">Warp Trend</span>
      <span class="sec-sub">최근 {cfg["trend_n"]} lot</span>
    </div>
    <div class="trend-note">X축: 최근 lot ID (시간 순)</div>
    {trend_block(warp_imgs)}
  </div>

  <div class="section">
    <div class="sec-head">
      <span class="sec-num">3</span>
      <span class="sec-title">Bow Trend</span>
      <span class="sec-sub">최근 {cfg["trend_n"]} lot</span>
    </div>
    <div class="trend-note">
      <span>X축: 최근 lot ID (시간 순)</span>
      <span class="legend-item"><span class="legend-line" style="border-color:#666666;border-top-style:dashed;"></span>스펙</span>
      <span class="legend-item"><span class="legend-line" style="border-color:#1a1a1a;border-top-style:dotted;"></span>목표</span>
    </div>
    {trend_block(bow_imgs)}
  </div>

  <div class="section">
    <div class="sec-head">
      <span class="sec-num">4</span>
      <span class="sec-title">X-Factor Profile</span>
      <span class="sec-sub">최근 {cfg["trend_n"]} lot</span>
    </div>
    <div class="trend-note">
      <span>Wire ID별 구획 · 각 구간은 그 wire의 lot 평균 프로파일 (pct 0→100%)</span>
    </div>
    <div class="xfactor-wide">
      {''.join(f'<div class="xf-wide-cell"><div class="xf-label">{lbl}</div>{_img_tag(img)}</div>' for lbl, img in xpct_imgs.items())}
    </div>
    <div class="trend-note" style="margin-top:16px;">
      <span>단일 값 인자: 최근 {cfg["trend_n"]} lot 추세</span>
    </div>
    <div class="xfactor-grid">
      {''.join(f'<div class="xf-cell"><div class="xf-label">{lbl}</div>{_img_tag(img)}</div>' for lbl, img in xscalar_imgs.items())}
    </div>
  </div>

  <div class="footer">
    <span>Wire Saw APC 자동 추천 시스템</span>
    <span>본 추천은 엔지니어 검토 후 반영 여부를 결정합니다.</span>
  </div>
</div>
</body>
</html>'''


def build_batch(cfg):
    """다중 장비 일괄 리포트 생성."""
    import copy
    all_eqps = []
    for group, eqps in cfg.get('batch_eqps', {}).items():
        all_eqps.extend(eqps)
    print(f"[일괄 생성] {len(all_eqps)}개 장비: {all_eqps}\n")
    paths = []
    for eqp in all_eqps:
        c = copy.deepcopy(cfg)
        c['eqp_name'] = eqp
        try:
            p = build_report(c)
            paths.append(p)
        except Exception as e:
            print(f"  ⚠ {eqp} 실패: {e}")
    print(f"\n✅ 총 {len(paths)}개 리포트 생성 완료")
    return paths


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'single'
    if mode == 'batch':
        build_batch(CONFIG)
    else:
        build_report(CONFIG)
