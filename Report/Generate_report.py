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
    'model_dir':  r'./apc_model_rolling',
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
    'encoding': 'utf-8',
}


# ═══════════════════════════════════════
# 모델 로드 + 역산 (Frame/Slurry)
# ═══════════════════════════════════════
def load_model(model_dir):
    with open(pt.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(pt.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(pt.join(model_dir, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    return model, scaler, meta


def inverse_profile(model, scaler, meta, target_bow, roll_values, eqp_name,
                    profile_cols):
    """지정 프로파일 컬럼(frame 또는 slurry)을 역산."""
    FEATURES = meta['feature_cols']; X_STATS = meta['x_stats']
    eqp_cols = meta.get('eqp_cols', []); pfx = meta.get('eqp_prefix', 'eqp_')
    roll_cols = meta.get('roll_cols', [])
    # 최적화 대상: profile_cols 중 모델 feature에 있는 것만
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
        v = X_STATS.get(c, {}).get('mean', 0.0)
        return float(v)

    def predict(vec):
        x = np.array([gv(c, vec) for c in FEATURES]).reshape(1, -1)
        return float(model.predict(scaler.transform(x))[0])

    def objective(vec):
        return (predict(vec) - target_bow) ** 2

    x0 = np.array([X_STATS.get(c, {}).get('mean', 29.0) for c in opt_cols])
    bounds = [(X_STATS.get(c, {}).get('q01', x0[i]-1),
               X_STATS.get(c, {}).get('q99', x0[i]+1))
              for i, c in enumerate(opt_cols)]
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 300, 'ftol': 1e-9})
    rec = {c: round(float(v), 2) for c, v in zip(opt_cols, res.x)}
    return {'recipe': rec, 'predicted_bow': round(predict(res.x), 3),
            'cols': opt_cols}


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


def plot_profile(pcts, values, title, ylabel='Temp'):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(pcts, values, 'o-', color='#2c6e9c', linewidth=2, markersize=5)
    ax.set_xlabel('Position (%)'); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
    return fig_to_base64(fig)


def plot_trend_row(recent, cols_map, trend_n, title_prefix):
    """Total/Seed/Mid/Tail 4개 그래프를 한 줄로."""
    imgs = {}
    for pos, col in cols_map.items():
        fig, ax = plt.subplots(figsize=(3, 2.3))
        if col in recent.columns:
            y = recent[col].dropna().tail(trend_n).values
            x = np.arange(1, len(y)+1)
            ax.plot(x, y, 'o-', color='#c0563b', linewidth=1.8, markersize=4)
            ax.set_title(pos, fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'{col}\n없음', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(pos, fontsize=10, fontweight='bold')
        ax.set_xlabel('최근 lot', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        imgs[pos] = fig_to_base64(fig)
    return imgs


# ═══════════════════════════════════════
# 리포트 생성
# ═══════════════════════════════════════
def build_report(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    model, scaler, meta = load_model(cfg['model_dir'])
    eqp = cfg['eqp_name']
    today = datetime.now().strftime('%Y.%m.%d')

    # 최근 데이터 (Trend + 직전 평균)
    recent = pd.read_csv(cfg['recent_csv'], encoding=cfg['encoding'],
                         encoding_errors='replace')
    recent[cfg['date_col']] = pd.to_datetime(recent[cfg['date_col']], errors='coerce')
    esub = recent[recent[cfg['eqp_col']] == eqp].sort_values(cfg['date_col'])

    # 직전 평균 조건 (최근 값 기준)
    roll_cols = meta.get('roll_cols', [])
    roll_values = {}
    for rc in roll_cols:
        src = rc.replace('roll_', '')
        if src in esub.columns and len(esub) > 0:
            roll_values[rc] = float(esub[src].tail(cfg['window']).mean())

    # ── 역산: Frame / Slurry ──
    frame_inv = inverse_profile(model, scaler, meta, cfg['target_bow'],
                                roll_values, eqp, cfg['frame_cols'])
    slurry_inv = inverse_profile(model, scaler, meta, cfg['target_bow'],
                                 roll_values, eqp, cfg['slurry_cols'])

    # 예상 BOW 범위 (예측 ± MAE)
    mae = meta.get('metrics', {}).get('mae', 0.1)
    pred_bow = frame_inv['predicted_bow'] if frame_inv else cfg['target_bow']
    bow_lo, bow_hi = round(pred_bow - mae, 1), round(pred_bow + mae, 1)

    # ── 프로파일 그림 ──
    pcts = cfg['temp_pcts']
    frame_vals = ([frame_inv['recipe'].get(c) for c in cfg['frame_cols']]
                  if frame_inv else [None]*12)
    slurry_vals = ([slurry_inv['recipe'].get(c) for c in cfg['slurry_cols']]
                   if slurry_inv else [None]*12)
    frame_img = (plot_profile(pcts, frame_vals, 'Frame in Temp')
                 if frame_inv else None)
    slurry_img = (plot_profile(pcts, slurry_vals, 'Slurry in Temp')
                  if slurry_inv else None)

    # ── Trend 그림 ──
    warp_imgs = plot_trend_row(esub, cfg['warp_cols'], cfg['trend_n'], 'Warp')
    bow_imgs = plot_trend_row(esub, cfg['bow_cols'], cfg['trend_n'], 'Bow')

    # ── 테이블 HTML ──
    def profile_table(vals, cols):
        if vals[0] is None:
            return '<p class="muted">해당 프로파일 컬럼 없음</p>'
        head = ''.join(f'<th>{p}</th>' for p in pcts)
        body = ''.join(f'<td>{v:.1f}</td>' if v is not None else '<td>-</td>'
                       for v in vals)
        return (f'<table class="recipe-tbl"><thead><tr><th>%</th>{head}</tr>'
                f'</thead><tbody><tr><th>추천 Temp</th>{body}</tr></tbody></table>')

    frame_tbl = profile_table(frame_vals, cfg['frame_cols'])
    slurry_tbl = profile_table(slurry_vals, cfg['slurry_cols'])

    # ── HTML 조립 ──
    html = _render_html(cfg, eqp, today, frame_img, frame_tbl,
                        slurry_img, slurry_tbl, bow_lo, bow_hi,
                        warp_imgs, bow_imgs)

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
                 bow_lo, bow_hi, warp_imgs, bow_imgs):
    def trend_block(imgs):
        cells = ''.join(
            f'<div class="trend-cell">{_img_tag(imgs.get(p))}</div>'
            for p in ['Total','Seed','Mid','Tail'])
        return f'<div class="trend-row">{cells}</div>'

    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wire Saw APC Report - {eqp}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, 'Malgun Gothic', 'Segoe UI', sans-serif;
    background: #f5f6f8; color: #1f2933; line-height: 1.5;
    padding: 32px 16px;
  }}
  .report {{
    max-width: 960px; margin: 0 auto; background: #fff;
    border: 1px solid #dfe3e8; border-radius: 8px;
    padding: 40px 44px; box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  }}
  .header {{
    border-bottom: 3px solid #2c6e9c; padding-bottom: 20px; margin-bottom: 28px;
  }}
  .header h1 {{
    font-size: 26px; color: #1a4971; letter-spacing: -0.5px; margin-bottom: 12px;
  }}
  .meta-row {{ display: flex; gap: 32px; font-size: 14px; color: #52606d; }}
  .meta-row .label {{ font-weight: 700; color: #2c6e9c; margin-right: 6px; }}
  .section {{ margin-bottom: 36px; }}
  .section-title {{
    font-size: 18px; font-weight: 700; color: #1a4971;
    background: #eef4f9; padding: 8px 14px; border-radius: 5px;
    margin-bottom: 18px; border-left: 4px solid #2c6e9c;
  }}
  .subsection-title {{
    font-size: 15px; font-weight: 700; color: #2c6e9c;
    margin: 18px 0 10px;
  }}
  .recipe-block {{
    display: flex; gap: 20px; align-items: flex-start;
    margin-bottom: 24px; flex-wrap: wrap;
  }}
  .recipe-block img {{ max-width: 420px; border: 1px solid #e5e9ee; border-radius: 5px; }}
  .recipe-tbl {{
    border-collapse: collapse; font-size: 12px; margin-top: 6px;
  }}
  .recipe-tbl th, .recipe-tbl td {{
    border: 1px solid #cfd6de; padding: 5px 8px; text-align: center;
  }}
  .recipe-tbl thead th {{ background: #2c6e9c; color: #fff; font-weight: 600; }}
  .recipe-tbl tbody th {{ background: #eef4f9; color: #1a4971; }}
  .bow-range {{
    background: #fff8e6; border: 1px solid #f0d98a; border-radius: 6px;
    padding: 16px 20px; font-size: 16px; margin-top: 8px;
  }}
  .bow-range .val {{ font-size: 22px; font-weight: 700; color: #c0563b; }}
  .trend-row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .trend-cell {{ flex: 1; min-width: 180px; text-align: center; }}
  .trend-cell img {{ width: 100%; border: 1px solid #e5e9ee; border-radius: 4px; }}
  .trend-note {{ font-size: 12px; color: #7b8794; margin-bottom: 10px; }}
  .muted {{ color: #9aa5b1; font-size: 13px; font-style: italic; }}
  .footer {{
    margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e9ee;
    font-size: 12px; color: #9aa5b1; text-align: center;
  }}
</style>
</head>
<body>
<div class="report">
  <div class="header">
    <h1>Wire Saw APC Report</h1>
    <div class="meta-row">
      <div><span class="label">Date</span>{today}</div>
      <div><span class="label">장비명</span>{eqp}</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">추천 Recipe</div>

    <div class="subsection-title">① Frame in Temp</div>
    <div class="recipe-block">
      {_img_tag(frame_img, 'Frame Temp')}
      <div>{frame_tbl}</div>
    </div>

    <div class="subsection-title">② Slurry in Temp</div>
    <div class="recipe-block">
      {_img_tag(slurry_img, 'Slurry Temp')}
      <div>{slurry_tbl}</div>
    </div>

    <div class="subsection-title">③ Recipe 적용 시 예상 BOW 범위</div>
    <div class="bow-range">
      <span class="val">{bow_lo} ~ {bow_hi}</span>
      <span style="color:#7b8794; font-size:13px; margin-left:12px;">
        (목표 {cfg['target_bow']}, 예측 ± MAE 기준)</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Warp Trend</div>
    <div class="trend-note">X축: 시간 순서 기준 최근 {cfg['trend_n']} lot</div>
    {trend_block(warp_imgs)}
  </div>

  <div class="section">
    <div class="section-title">Bow Trend</div>
    <div class="trend-note">X축: 시간 순서 기준 최근 {cfg['trend_n']} lot</div>
    {trend_block(bow_imgs)}
  </div>

  <div class="footer">
    Wire Saw APC 자동 추천 시스템 · 본 추천은 엔지니어 검토 후 반영 여부를 결정하세요.
  </div>
</div>
</body>
</html>'''


if __name__ == '__main__':
    build_report(CONFIG)
