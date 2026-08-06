# -*- coding: utf-8 -*-
"""
build_recommend_report.py  (추천 + 실제 통합판)
─────────────────────────────────────────
두 소스를 읽어 장비별 리포트 HTML 생성.

  recommend_future.csv → ①②③ 추천 (rec_ 온도, pred_bow)   [미래·역산]
  field_store.csv      → 실제 영역                          [과거·실측]
      · Bow/Warp Trend (최근 N wire, Total)
      · X-Factor: 실제 frame/slurry 온도 프로파일 (최근 N wire 겹침)
      · X-Factor: 단일값 조건 (ingot/wait/warmup) 최근 N wire 추세

  두 소스는 eqp(장비)로 매칭. field_store 없으면 추천만 표시.

사용:
  python build_recommend_report.py
  python build_recommend_report.py ./recommend_future.csv ./data/field_store.csv ./reports/recipe.html
"""
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ── 설정 ──
PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RANGE = 0.15
TARGET_BOW = 1.75
SPEC_LO, SPEC_HI = 1.5, 2.0      # 양품 스펙 (엔지니어 확인 예정)
PROCESS_TIME = '13.3Hr'
RECENT_N = 10                    # 실제 trend/프로파일에 쓸 최근 wire 수

REC_FRAME = 'rec_set_frame_temp_{p}pct'
REC_SLURRY = 'rec_set_slurry_temp_{p}pct'
ACT_FRAME = 'set_frame_temp_{p}pct'       # 실제 (rec_ 없음)
ACT_SLURRY = 'set_slurry_temp_{p}pct'
ACT_WG_L = 'shift_amount_wireguide_l_{p}pct'   # wire guide L 프로파일
ACT_WG_R = 'shift_amount_wireguide_r_{p}pct'   # wire guide R 프로파일
ACT_WG_L = 'shift_amount_wireguide_l_{p}pct'
ACT_WG_R = 'shift_amount_wireguide_r_{p}pct'

STORE_CFG = {
    'eqp':  'eqp_nm_3200',
    'wire': 'fdc_new_wire_id',
    'date': 'date_3200',
    'bow':  'avg_bow_bf_total',
    'warp': 'avg_warp_bf_total',
    'ingot':'fdc_ingot_len',
    'wait': 'fdc_wait_time',
    'warm': 'fdc_warm_up_time',
}


def _profile(row, tmpl):
    out = []
    for p in PCTS:
        v = row.get(tmpl.format(p=p))
        out.append(None if pd.isna(v) else round(float(v), 2))
    return out


def _pred_bow(row):
    for c in ['frame_pred_bow', 'slurry_pred_bow']:
        if c in row.index and pd.notna(row[c]):
            return round(float(row[c]), 3)
    return TARGET_BOW


def load_recommend(csv_path):
    df = pd.read_csv(csv_path)
    eqp_col = 'eqp' if 'eqp' in df.columns else df.columns[0]
    recs = {}
    for _, r in df.iterrows():
        frame = _profile(r, REC_FRAME)
        slurry = _profile(r, REC_SLURRY)
        if all(v is None for v in frame) and all(v is None for v in slurry):
            print(f"  ⚠ {r.get(eqp_col)}: 추천 온도 없음 — 스킵")
            continue
        eqp = str(r.get(eqp_col, '?'))
        recs[eqp] = {
            'eqp': eqp,
            'waf': int(r['n_waf_used']) if 'n_waf_used' in r.index
                   and pd.notna(r['n_waf_used']) else 0,
            'wire': str(r.get('latest_wire', '')),
            'bow': _pred_bow(r),
            'rec_frame': [v if v is not None else 0 for v in frame],
            'rec_slurry': [v if v is not None else 0 for v in slurry],
        }
    return recs


def load_actuals(store_path):
    """field_store에서 장비별 최근 N wire의 실제값 추출."""
    if not os.path.exists(store_path):
        print(f"  ⚠ field_store 없음: {store_path} — 실제 영역 생략")
        return {}
    df = pd.read_csv(store_path)
    C = STORE_CFG
    if C['eqp'] not in df.columns:
        print(f"  ⚠ {C['eqp']} 컴럼 없음 — 실제 영역 생략")
        return {}

    acts = {}
    LOT = 'lot_id'
    for eqp, g in df.groupby(C['eqp']):
        if C['date'] in g.columns:
            g = g.sort_values(C['date'])

        # 최근 N wire 선택 (wire 등장 순서 기준, 최근 것)
        wire_col = C['wire']
        wire_order = list(dict.fromkeys(g[wire_col].astype(str).tolist()))  # 등장순 유지
        recent_wires = wire_order[-RECENT_N:]

        has_lot = LOT in g.columns

        def lot_profiles(sub, tmpl):
            """sub(한 wire의 행들)에서 lot별 프로파일 리스트 반환."""
            out = []
            if has_lot:
                for lot, lg in sub.groupby(LOT, sort=False):
                    # lot 안 여러 행이면 평균 (보통 1행)
                    prof = []
                    for p in PCTS:
                        c = tmpl.format(p=p)
                        v = lg[c].mean() if c in lg.columns else None
                        prof.append(None if pd.isna(v) else round(float(v), 2))
                    if not all(v is None for v in prof):
                        out.append({'lot': str(lot),
                                    'prof': [v if v is not None else 0 for v in prof]})
            else:
                for _, r in sub.iterrows():
                    prof = _profile(r, tmpl)
                    if not all(v is None for v in prof):
                        out.append({'lot': '',
                                    'prof': [v if v is not None else 0 for v in prof]})
            return out

        def lot_scalars(sub, name):
            """lot별 단일값 리스트."""
            key = C[name]
            out = []
            if has_lot:
                for lot, lg in sub.groupby(LOT, sort=False):
                    v = lg[key].mean() if key in lg.columns else None
                    out.append({'lot': str(lot),
                                'val': round(float(v), 1) if pd.notna(v) else None})
            else:
                for _, r in sub.iterrows():
                    v = r.get(key)
                    out.append({'lot': '',
                                'val': round(float(v), 1) if pd.notna(v) else None})
            return out

        # wire별로 lot 계층 구성
        wire_blocks = []      # [{wire, frame:[{lot,prof}], slurry:[...], ...}]
        bow_pts, warp_pts, bow_wires = [], [], []
        for w in recent_wires:
            sub = g[g[wire_col].astype(str) == w]
            wire_blocks.append({
                'wire': w,
                'frame':  lot_profiles(sub, ACT_FRAME),
                'slurry': lot_profiles(sub, ACT_SLURRY),
                'wg_l':   lot_profiles(sub, ACT_WG_L),
                'wg_r':   lot_profiles(sub, ACT_WG_R),
                'ingot':  lot_scalars(sub, 'ingot'),
                'wait':   lot_scalars(sub, 'wait'),
                'warm':   lot_scalars(sub, 'warm'),
            })
            # bow/warp trend는 wire 단위 (wire 내 평균)
            for metric, arr in [('bow', bow_pts), ('warp', warp_pts)]:
                col_nm = C[metric]
                v = sub[col_nm].mean() if col_nm in sub.columns else None
                arr.append(round(float(v), 3) if pd.notna(v) else None)
            bow_wires.append(w)

        acts[str(eqp)] = {
            'wires': bow_wires,
            'bow':  bow_pts,
            'warp': warp_pts,
            'blocks': wire_blocks,   # wire>lot 계층
            'has_lot': has_lot,
        }
    return acts


def merge(recs, acts):
    out = []
    for eqp, r in recs.items():
        r['actual'] = acts.get(eqp)
        out.append(r)
    return out


def render_html(records):
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    frame_start = records[0]['rec_frame'][0] if records else 28.0
    return (TEMPLATE
            .replace('__DATA__', json.dumps(records, ensure_ascii=False))
            .replace('__PCTS__', json.dumps(PCTS))
            .replace('__RANGE__', str(RANGE))
            .replace('__NEQP__', str(len(records)))
            .replace('__TARGET__', str(TARGET_BOW))
            .replace('__SPECLO__', str(SPEC_LO))
            .replace('__SPECHI__', str(SPEC_HI))
            .replace('__PTIME__', PROCESS_TIME)
            .replace('__FSTART__', str(frame_start))
            .replace('__RECENTN__', str(RECENT_N))
            .replace('__STAMP__', stamp))


TEMPLATE = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wire Saw APC — 온도 Recipe 추천 리포트</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#1a1a1a; --ink-soft:#404040; --ink-faint:#808080;
    --paper:#ffffff; --panel:#f2f2f2; --panel-line:#dcdcdc; --line:#cccccc;
    --frame:#1a1a1a; --frame-soft:#eeeeee;
    --slurry:#1a1a1a; --slurry-soft:#eeeeee;
    --target:#1a1a1a; --spec:#666666; --actual:#4d4d4d;
    --low:#4d4d4d; --low-bg:#f0f0f0;
    --rec-bg:#ececec; --act-bg:#f5f5f5;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Noto Sans KR',system-ui,sans-serif;color:var(--ink);background:var(--panel);line-height:1.6;padding:0 0 80px;}
  code,.mono{font-family:'JetBrains Mono',monospace;}
  .wrap{max-width:1280px;margin:0 auto;padding:0 16px;}
  .toolbar{position:sticky;top:0;z-index:50;background:rgba(255,255,255,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--panel-line);padding:11px 0;}
  .toolbar .wrap{display:flex;align-items:center;justify-content:space-between;gap:16px;}
  .toolbar .t-title{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--ink-faint);letter-spacing:.06em;}
  .btn{font-family:'Noto Sans KR',sans-serif;font-size:13px;font-weight:700;background:var(--frame);color:#fff;border:none;padding:9px 18px;border-radius:7px;cursor:pointer;}
  .btn:hover{background:#0c4a70;}
  .masthead{background:var(--ink);color:#fff;padding:38px 0 30px;border-bottom:4px solid var(--frame);}
  .eyebrow{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.28em;text-transform:uppercase;color:#7fb8dc;font-weight:500;margin-bottom:13px;}
  .masthead h1{font-size:30px;font-weight:900;letter-spacing:-.01em;line-height:1.2;}
  .masthead .sub{margin-top:9px;color:#a9bccb;font-size:14.5px;font-weight:300;max-width:660px;}
  .meta-row{display:flex;flex-wrap:wrap;gap:26px;margin-top:24px;padding-top:20px;border-top:1px solid rgba(255,255,255,.14);}
  .meta-item .k{font-family:'JetBrains Mono',monospace;color:#6f8598;font-size:11px;letter-spacing:.12em;text-transform:uppercase;display:block;margin-bottom:4px;}
  .meta-item .v{color:#dbe6ee;font-weight:500;font-size:13.5px;}
  .summary{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0 4px;}
  .stat{background:var(--paper);border:1px solid var(--panel-line);border-radius:9px;padding:16px 18px;}
  .stat .sk{font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-faint);margin-bottom:7px;}
  .stat .sv{font-size:24px;font-weight:900;letter-spacing:-.01em;}
  .stat .su{font-size:12px;color:var(--ink-faint);font-weight:400;margin-left:3px;}
  .note{background:var(--low-bg);border-left:3px solid var(--low);padding:13px 18px;margin:22px 0 6px;border-radius:0 6px 6px 0;font-size:13.5px;color:#6b5d1f;}
  .note strong{font-weight:700;}
  .eqp{background:var(--paper);border:1px solid var(--panel-line);border-radius:11px;margin-top:22px;overflow:hidden;box-shadow:0 1px 2px rgba(18,32,46,.03);}
  .eqp-head{display:flex;align-items:center;gap:16px;padding:20px 26px;border-bottom:1px solid var(--panel-line);background:linear-gradient(180deg,#fbfcfd,#fff);}
  .eqp-name{font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;}
  .eqp-head .grow{flex:1;}
  .badge{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.05em;padding:5px 11px;border-radius:6px;}
  .badge-waf{background:var(--low-bg);color:var(--low);border:1px solid #e6dcae;}
  .badge-time{background:var(--frame-soft);color:var(--frame);margin-left:8px;}
  .sec-label{display:flex;align-items:center;gap:10px;padding:12px 26px;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:700;}
  .sec-rec{background:var(--rec-bg);color:var(--frame);border-top:1px solid var(--panel-line);border-bottom:1px solid var(--panel-line);}
  .sec-act{background:var(--act-bg);color:#8a6d4a;border-top:1px solid var(--panel-line);border-bottom:1px solid var(--panel-line);}
  .sec-label .tag{font-size:9px;padding:2px 7px;border-radius:4px;color:#fff;letter-spacing:.06em;}
  .sec-rec .tag{background:var(--frame);}
  .sec-act .tag{background:#8a6d4a;}
  .bow-band{display:flex;align-items:center;gap:20px;padding:16px 26px;background:var(--frame-soft);flex-wrap:wrap;}
  .bow-band .lbl{font-size:12.5px;color:var(--ink-soft);font-weight:500;}
  .bow-val{font-size:22px;font-weight:900;}
  .bow-range{font-family:'JetBrains Mono',monospace;font-size:15px;color:var(--frame);font-weight:700;}
  .bow-target{margin-left:auto;font-size:12px;color:var(--ink-faint);}
  .bow-target b{color:var(--target);font-weight:700;}
  .profiles{display:grid;grid-template-columns:1fr 1fr;gap:0;}
  @media(max-width:760px){.profiles{grid-template-columns:1fr;}}
  .profile{padding:20px 24px;}
  .profile:first-child{border-right:1px solid var(--panel-line);}
  @media(max-width:760px){.profile:first-child{border-right:none;border-bottom:1px solid var(--panel-line);}}
  .profile h3, .xf h3{font-size:14px;font-weight:700;display:flex;align-items:center;gap:9px;margin-bottom:4px;font-family:'Noto Sans KR',system-ui,sans-serif;}
  .profile h3 .sw, .xf h3 .sw{width:11px;height:11px;border-radius:3px;}
  .profile .unit, .xf .unit{font-size:11.5px;color:var(--ink-faint);font-family:'JetBrains Mono',monospace;margin-bottom:12px;}
  .xfactor{padding:4px 0;}
  .xf{padding:16px 24px;border-bottom:1px solid var(--panel-line);}
  .xf:last-child{border-bottom:none;}
  .chart{width:100%;height:150px;margin-bottom:6px;}
  .chart-tall{height:170px;}
  .tbl-wrap{width:100%;overflow-x:auto;margin-top:8px;}
  .tbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:9.5px;table-layout:fixed;}
  .tbl th{background:var(--panel);color:var(--ink-faint);font-weight:500;padding:4px 1px;text-align:center;border-bottom:1px solid var(--panel-line);}
  .tbl td{padding:4px 1px;text-align:center;border-bottom:1px solid #eef1f4;color:var(--ink-soft);}
  .tbl td.v{font-weight:700;color:var(--ink);}
  .tbl th:first-child,.tbl td:first-child{width:24px;color:var(--ink-faint);}
  .trend-row{display:grid;grid-template-columns:1fr 1fr;gap:0;}
  @media(max-width:760px){.trend-row{grid-template-columns:1fr;}}
  .cap{font-size:11px;color:var(--ink-faint);margin-top:2px;}
  .cond-row{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border-top:1px solid var(--panel-line);}
  @media(max-width:620px){.cond-row{grid-template-columns:1fr;}}
  .cond{padding:16px 20px;border-right:1px solid var(--panel-line);}
  .cond:last-child{border-right:none;}
  .cond h4{font-size:12px;font-weight:700;color:var(--ink-soft);margin-bottom:8px;font-family:'JetBrains Mono',monospace;}
  .no-actual{padding:18px 26px;font-size:13px;color:var(--ink-faint);background:var(--act-bg);}
  .foot{max-width:1280px;margin:34px auto 0;padding:22px 16px 0;border-top:1px solid var(--line);font-size:11.5px;color:var(--ink-faint);display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;font-family:'JetBrains Mono',monospace;}
  @media(max-width:620px){.summary{grid-template-columns:repeat(2,1fr);}.masthead h1{font-size:24px;}.wrap{padding:0 18px;}}
  @media print{body{background:#fff;}.toolbar{display:none;}.eqp{box-shadow:none;break-inside:avoid;}.masthead,.bow-band,.badge,.tbl th,.stat,.sec-rec,.sec-act{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
</style>
</head>
<body>
<header class="masthead"><div class="wrap">
  <div class="eyebrow">Wire Saw APC · Temperature Recipe Recommendation</div>
  <h1>온도 Recipe 추천 리포트</h1>
  <p class="sub">위쪽 <b>추천</b>(미래·역산), 아래쪽 <b>실제 추이</b>(최근 __RECENTN__ wire·실측). 최근 상태를 보고 추천 수용을 판단합니다.</p>
  <div class="meta-row">
    <div class="meta-item"><span class="k">Process Time</span><span class="v">__PTIME__</span></div>
    <div class="meta-item"><span class="k">Target BOW</span><span class="v">__TARGET__ µm</span></div>
    <div class="meta-item"><span class="k">양품 스펙</span><span class="v">__SPECLO__ ~ __SPECHI__ µm</span></div>
    <div class="meta-item"><span class="k">생성 시각</span><span class="v">__STAMP__</span></div>
  </div>
</div></header>
<div class="wrap">
  <div class="summary">
    <div class="stat"><div class="sk">추천 장비</div><div class="sv">__NEQP__<span class="su">/ 10대</span></div></div>
    <div class="stat"><div class="sk">Target BOW</div><div class="sv">__TARGET__<span class="su">µm</span></div></div>
    <div class="stat"><div class="sk">예상 범위</div><div class="sv">±__RANGE__<span class="su">µm</span></div></div>
    <div class="stat"><div class="sk">실제 추이</div><div class="sv">__RECENTN__<span class="su">wire</span></div></div>
  </div>
  <div id="cards"></div>
</div>
<div class="foot"><span>WIRESAW_APC · DUAL_MODEL · SLSQP_INVERSE + FIELD_ACTUALS</span><span>GENERATED __STAMP__</span></div>
<script>
const PCTS=__PCTS__, DATA=__DATA__, RANGE=__RANGE__, TARGET=__TARGET__;
const SPEC_LO=__SPECLO__, SPEC_HI=__SPECHI__;

function lineChart(values,color,soft){
  const W=300,H=150,padL=34,padR=10,padT=14,padB=24;
  const mn=Math.min(...values),mx=Math.max(...values),sp=(mx-mn)||1;
  const lo=mn-sp*0.15,hi=mx+sp*0.15,rng=hi-lo;
  const x=i=>padL+(W-padL-padR)*(i/(values.length-1)),y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  let grid='';for(let g=0;g<=2;g++){const val=lo+rng*g/2,yy=y(val);
    grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
    grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(1)}</text>`;}
  let xl='';[0,5,10].forEach(i=>{if(i<values.length)xl+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${PCTS[i]}</text>`;});
  const pts=values.map((v,i)=>`${x(i)},${y(v)}`).join(' ');
  const area=`M${padL},${H-padB} L`+values.map((v,i)=>`${x(i)},${y(v)}`).join(' L')+` L${W-padR},${H-padB} Z`;
  const dots=values.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="2.5" fill="${color}"/>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}">${grid}<path d="${area}" fill="${soft}" opacity="0.6"/><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>${dots}${xl}</svg>`;
}

// 사진 방식: wire별 구획을 가로로 이어 각 구획 안에서 0->100pct 진행
// wire > lot > pct 3계층 프로파일
//  blocks: [{wire:'W1', lots:[[...11pct], [...]]}, ...]  (각 wire의 lot 프로파일들)
function horizonChart(blocks,color){
  const W=760,H=220,padL=42,padR=12,padT=16,padB=58,wireGap=0.12,lotGap=0.12;
  const all=blocks.flatMap(b=>b.lots.flat());
  if(!all.length)return '<div class="cap">데이터 없음</div>';
  const mn=Math.min(...all),mx=Math.max(...all),sp=(mx-mn)||1;
  const lo=mn-sp*0.12,hi=mx+sp*0.12,rng=hi-lo;
  const nW=blocks.length, plotW=W-padL-padR, wireSlot=plotW/nW;
  const y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  // y grid
  let grid='';for(let g=0;g<=3;g++){const val=lo+rng*g/3,yy=y(val);
    grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
    grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(2)}</text>`;}
  let body='';
  blocks.forEach((blk,wi)=>{
    const wireX=padL+wireSlot*wi;
    // wire 구획 배경 (짝수 옅게)
    if(wi%2===1) body+=`<rect x="${wireX}" y="${padT}" width="${wireSlot}" height="${H-padT-padB}" fill="#f6f6f6"/>`;
    // wire 경계 (굵은 선) — 첫 wire 제외 왼쪽에
    if(wi>0) body+=`<line x1="${wireX}" y1="${padT-4}" x2="${wireX}" y2="${H-padB}" stroke="#b8bfc6" stroke-width="1.6"/>`;
    const inner=wireSlot*(1-wireGap), wStart=wireX+wireSlot*wireGap/2;
    const nL=blk.lots.length, lotSlot=inner/nL;
    blk.lots.forEach((prof,li)=>{
      const lotX=wStart+lotSlot*li, lotInner=lotSlot*(1-lotGap), lStart=lotX+lotSlot*lotGap/2;
      // lot 경계 (얕은 선) — 첫 lot 제외
      if(li>0) body+=`<line x1="${lotX}" y1="${padT}" x2="${lotX}" y2="${H-padB}" stroke="#e5e8eb" stroke-width="0.8" stroke-dasharray="2 2"/>`;
      const xx=i=>lStart+lotInner*(i/(prof.length-1));
      const pts=prof.map((v,i)=>`${xx(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
      body+=`<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/>`;
      body+=prof.map((v,i)=>`<circle cx="${xx(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="1.3" fill="${color}"/>`).join('');
    });
    // wire id 라벨 (구획 중앙, 하단 세로)
    const wid=(blk.wire||'').toString(), cx=wireX+wireSlot/2;
    body+=`<text x="${cx}" y="${H-padB+16}" text-anchor="end" font-family="JetBrains Mono" font-size="8" fill="#7a8896" transform="rotate(-40 ${cx} ${H-padB+16})">${wid}</text>`;
  });
  const xaxis=`<text x="${padL+plotW/2}" y="${H-4}" text-anchor="middle" font-family="JetBrains Mono" font-size="8.5" fill="#7a8896">Wire ID &gt; lot &gt; pct 0→100% (굵은선=wire, 얕은선=lot, 시간순)</text>`;
  return `<svg class="chart chart-horizon" viewBox="0 0 ${W} ${H}">${grid}${body}${xaxis}</svg>`;
}

// 단일값 인자: wire > lot 계층, lot마다 막대 하나
//  blocks: [{wire:'W1', vals:[v1,v2,...]}, ...]
function barSlotChart(blocks,color,unit){
  const W=760,H=200,padL=42,padR=12,padT=16,padB=58,wireGap=0.12,lotGap=0.25;
  const all=blocks.flatMap(b=>b.vals).filter(v=>v!=null);
  if(!all.length)return '<div class="cap">데이터 없음</div>';
  const mn=Math.min(...all),mx=Math.max(...all),sp=(mx-mn)||1;
  const lo=mn-sp*0.15,hi=mx+sp*0.15,rng=hi-lo;
  const nW=blocks.length, plotW=W-padL-padR, wireSlot=plotW/nW;
  const y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  let grid='';for(let g=0;g<=3;g++){const val=lo+rng*g/3,yy=y(val);
    grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
    grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(1)}</text>`;}
  let body='';
  blocks.forEach((blk,wi)=>{
    const wireX=padL+wireSlot*wi;
    if(wi%2===1) body+=`<rect x="${wireX}" y="${padT}" width="${wireSlot}" height="${H-padT-padB}" fill="#f6f6f6"/>`;
    if(wi>0) body+=`<line x1="${wireX}" y1="${padT-4}" x2="${wireX}" y2="${H-padB}" stroke="#b8bfc6" stroke-width="1.6"/>`;
    const inner=wireSlot*(1-wireGap), wStart=wireX+wireSlot*wireGap/2;
    const nL=blk.vals.length, lotSlot=inner/nL, barW=lotSlot*(1-lotGap);
    blk.vals.forEach((v,li)=>{
      const lotX=wStart+lotSlot*li+lotSlot*lotGap/2;
      if(v!=null){
        const yy=y(v), h=(H-padB)-yy;
        body+=`<rect x="${lotX.toFixed(1)}" y="${yy.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" fill="${color}" opacity="0.75" rx="1"/>`;
      }
    });
    const wid=(blk.wire||'').toString(), cx=wireX+wireSlot/2;
    body+=`<text x="${cx}" y="${H-padB+16}" text-anchor="end" font-family="JetBrains Mono" font-size="8" fill="#7a8896" transform="rotate(-40 ${cx} ${H-padB+16})">${wid}</text>`;
  });
  const xaxis=`<text x="${padL+plotW/2}" y="${H-4}" text-anchor="middle" font-family="JetBrains Mono" font-size="8.5" fill="#7a8896">Wire ID &gt; lot (굵은선=wire, 시간순) · ${unit}</text>`;
  return `<svg class="chart chart-horizon" viewBox="0 0 ${W} ${H}">${grid}${body}${xaxis}</svg>`;
}

function trendChart(values,wires,opts){
  const W=460,H=180,padL=38,padR=12,padT=16,padB=40;
  const vals=values.filter(v=>v!=null);if(!vals.length)return '<div class="cap">데이터 없음</div>';
  let mn=Math.min(...vals),mx=Math.max(...vals);
  if(opts.target!=null){mn=Math.min(mn,opts.target);mx=Math.max(mx,opts.target);}
  if(opts.specLo!=null){mn=Math.min(mn,opts.specLo);mx=Math.max(mx,opts.specHi);}
  const sp=(mx-mn)||1,lo=mn-sp*0.12,hi=mx+sp*0.12,rng=hi-lo;
  const n=values.length;
  const x=i=>padL+(W-padL-padR)*(n>1?i/(n-1):0.5),y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  let grid='';for(let g=0;g<=3;g++){const val=lo+rng*g/3,yy=y(val);
    grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
    grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(2)}</text>`;}
  let band='';
  if(opts.specLo!=null){band=`<rect x="${padL}" y="${y(opts.specHi)}" width="${W-padL-padR}" height="${y(opts.specLo)-y(opts.specHi)}" fill="#0f766e" opacity="0.05"/>
    <line x1="${padL}" y1="${y(opts.specLo)}" x2="${W-padR}" y2="${y(opts.specLo)}" stroke="#c0392b" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>
    <line x1="${padL}" y1="${y(opts.specHi)}" x2="${W-padR}" y2="${y(opts.specHi)}" stroke="#c0392b" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>`;}
  let tline='';
  if(opts.target!=null){tline=`<line x1="${padL}" y1="${y(opts.target)}" x2="${W-padR}" y2="${y(opts.target)}" stroke="#0f766e" stroke-width="1.5" stroke-dasharray="6 3"/>
    <text x="${W-padR}" y="${y(opts.target)-4}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#0f766e">target ${opts.target}</text>`;}
  let xl='';const step=Math.ceil(n/5);
  values.forEach((v,i)=>{if(i%step===0||i===n-1){const w=(wires[i]||'').slice(-4);
    xl+=`<text x="${x(i)}" y="${H-20}" text-anchor="middle" font-family="JetBrains Mono" font-size="8" fill="#9aa6b2">${w}</text>`;}});
  let seg=[],segs=[];
  values.forEach((v,i)=>{if(v==null){if(seg.length){segs.push(seg);seg=[];}}else seg.push(`${x(i)},${y(v)}`);});
  if(seg.length)segs.push(seg);
  const line=segs.map(s=>`<polyline points="${s.join(' ')}" fill="none" stroke="${opts.color}" stroke-width="2" stroke-linejoin="round"/>`).join('');
  const dots=values.map((v,i)=>v==null?'':`<circle cx="${x(i)}" cy="${y(v)}" r="3" fill="${opts.color}"/>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" style="height:186px">${grid}${band}${tline}${line}${dots}${xl}<text x="${padL}" y="${H-6}" font-family="JetBrains Mono" font-size="8" fill="#c3ccd4">← wire (과거→최근) →</text></svg>`;
}

function tbl(values){
  const head='<tr><th>pct</th>'+PCTS.map(p=>`<th>${p}</th>`).join('')+'</tr>';
  const row='<tr><td>°C</td>'+values.map(v=>`<td class="v">${v.toFixed(2)}</td>`).join('')+'</tr>';
  return `<div class="tbl-wrap"><table class="tbl">${head}${row}</table></div>`;
}
function condTrend(label,arr,unit){
  const vals=arr.filter(v=>v!=null);
  const last=vals.length?vals[vals.length-1]:'—';
  const spark=(function(){
    if(!vals.length)return '';const W=120,H=34,mn=Math.min(...vals),mx=Math.max(...vals),sp=(mx-mn)||1;
    const x=i=>2+(W-4)*(i/Math.max(1,arr.length-1)),y=v=>2+(H-4)*(1-(v-mn)/sp);
    const pts=arr.map((v,i)=>v==null?null:`${x(i)},${y(v)}`).filter(Boolean).join(' ');
    return `<svg width="${W}" height="${H}"><polyline points="${pts}" fill="none" stroke="#5a6b7a" stroke-width="1.5"/></svg>`;
  })();
  return `<div class="cond"><h4>${label}</h4>${spark}<div class="cap">최근값 <b>${last}</b> ${unit}</div></div>`;
}

function card(d){
  const lo=(d.bow-RANGE).toFixed(2),hi=(d.bow+RANGE).toFixed(2);
  const a=d.actual;
  let actualHTML='';
  if(a){
    // blocks(wire>lot)에서 인자별 구조 추출
    const B=a.blocks||[];
    const prof=(key)=>B.map(b=>({wire:b.wire, lots:(b[key]||[]).map(x=>x.prof)}))
                        .filter(b=>b.lots.length);
    const scal=(key)=>B.map(b=>({wire:b.wire, vals:(b[key]||[]).map(x=>x.val)}))
                        .filter(b=>b.vals.length);
    const frB=prof('frame'), slB=prof('slurry'), wlB=prof('wg_l'), wrB=prof('wg_r');
    const inB=scal('ingot'), waB=scal('wait'), wmB=scal('warm');
    const nW=a.wires.length;

    actualHTML=`
    <div class="sec-label sec-act"><span class="tag">ACTUAL</span> 실제 추이 · 최근 ${nW} wire</div>
    <div class="trend-row">
      <div class="profile" style="border-right:1px solid var(--panel-line)">
        <h3><span class="sw" style="background:var(--actual)"></span>Bow Trend (Total)</h3>
        <div class="unit">avg_bow_bf_total · 실측 · wire순</div>
        ${trendChart(a.bow,a.wires,{color:'#0f5c8c',target:TARGET,specLo:SPEC_LO,specHi:SPEC_HI})}
      </div>
      <div class="profile">
        <h3><span class="sw" style="background:var(--actual)"></span>Warp Trend (Total)</h3>
        <div class="unit">avg_warp_bf_total · 실측 · wire순</div>
        ${trendChart(a.warp,a.wires,{color:'#b8531f'})}
      </div>
    </div>
    <div class="sec-label sec-act" style="border-top:1px solid var(--panel-line);background:#ececec;color:#4d4d4d"><span class="tag" style="background:#4d4d4d">X-FACTOR</span> 실제 인자 · wire &gt; lot 계층 (pct 0→100%, 시간순)</div>
    <div class="xfactor">
      <div class="xf">
        <h3><span class="sw" style="background:var(--frame)"></span>Frame Temp</h3>
        <div class="unit">set_frame_temp · 최근 ${nW} wire · wire&gt;lot&gt;pct</div>
        ${horizonChart(frB,'#0f5c8c')}
      </div>
      <div class="xf">
        <h3><span class="sw" style="background:var(--slurry)"></span>Slurry Temp</h3>
        <div class="unit">set_slurry_temp · 최근 ${nW} wire · wire&gt;lot&gt;pct</div>
        ${horizonChart(slB,'#b8531f')}
      </div>
      ${wlB.length?`<div class="xf">
        <h3><span class="sw" style="background:#5a6b7a"></span>Wire Guide L</h3>
        <div class="unit">shift_amount_wireguide_l · wire&gt;lot&gt;pct</div>
        ${horizonChart(wlB,'#5a6b7a')}
      </div>`:''}
      ${wrB.length?`<div class="xf">
        <h3><span class="sw" style="background:#8a6d4a"></span>Wire Guide R</h3>
        <div class="unit">shift_amount_wireguide_r · wire&gt;lot&gt;pct</div>
        ${horizonChart(wrB,'#8a6d4a')}
      </div>`:''}
      <div class="xf">
        <h3><span class="sw" style="background:#0f766e"></span>ingot_len</h3>
        <div class="unit">fdc_ingot_len · wire&gt;lot 단일값</div>
        ${barSlotChart(inB,'#0f766e','mm')}
      </div>
      <div class="xf">
        <h3><span class="sw" style="background:#0f766e"></span>wait_time</h3>
        <div class="unit">fdc_wait_time · wire&gt;lot 단일값</div>
        ${barSlotChart(waB,'#7a8896','')}
      </div>
      <div class="xf">
        <h3><span class="sw" style="background:#0f766e"></span>warm_up_time</h3>
        <div class="unit">fdc_warm_up_time · wire&gt;lot 단일값</div>
        ${barSlotChart(wmB,'#b8531f','')}
      </div>
    </div>`;
  }else{
    actualHTML=`<div class="no-actual">⚠ 이 장비의 field_store 실제 데이터가 없어 실제 영역을 생략합니다.</div>`;
  }
  return `<section class="eqp">
    <div class="eqp-head"><span class="eqp-name">${d.eqp}</span><div class="grow"></div>
      <span class="badge badge-waf">WAF ${d.waf}개</span><span class="badge badge-time">__PTIME__</span></div>

    <div class="sec-label sec-rec"><span class="tag">RECOMMEND</span> 추천 · 미래 lot (역산)</div>
    <div class="bow-band"><span class="lbl">예상 BOW</span><span class="bow-val">${d.bow.toFixed(2)}</span>
      <span class="bow-range">${lo} ~ ${hi} µm</span>
      <span class="bow-target">Target <b>${TARGET}</b> · 최근 wire <code>${d.wire}</code></span></div>
    <div class="profiles">
      <div class="profile"><h3><span class="sw" style="background:var(--frame)"></span>① Frame Temp 추천</h3>
        <div class="unit">rec_set_frame_temp · °C · 0→100pct</div>${lineChart(d.rec_frame,'#0f5c8c','#cfe2ef')}${tbl(d.rec_frame)}</div>
      <div class="profile"><h3><span class="sw" style="background:var(--slurry)"></span>② Slurry Temp 추천</h3>
        <div class="unit">rec_set_slurry_temp · °C · 0→100pct</div>${lineChart(d.rec_slurry,'#b8531f','#f0d9c9')}${tbl(d.rec_slurry)}</div>
    </div>
    ${actualHTML}
  </section>`;
}
document.getElementById('cards').innerHTML = DATA.map(card).join('');
</script>
</body>
</html>'''


def main():
    args = sys.argv[1:]
    rec_csv   = args[0] if len(args) > 0 else './recommend_future.csv'
    store_csv = args[1] if len(args) > 1 else './data/field_store.csv'
    out_path  = args[2] if len(args) > 2 else './reports/recipe_report.html'

    if not os.path.exists(rec_csv):
        print(f"❌ 추천 CSV 없음: {rec_csv}"); sys.exit(1)

    print(f"[리포트] 추천: {rec_csv}")
    recs = load_recommend(rec_csv)
    if not recs:
        print("❌ 추천 0건 — 리포트 생성 안 함"); sys.exit(1)

    print(f"[리포트] 실제: {store_csv}")
    acts = load_actuals(store_csv)
    records = merge(recs, acts)

    html = render_html(records)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ 리포트 생성: {out_path} ({len(records)}개 장비)")
    for r in records:
        has_act = '실제O' if r.get('actual') else '실제X'
        print(f"   · {r['eqp']}: WAF {r['waf']}개, 예측 BOW {r['bow']}, {has_act}")


if __name__ == '__main__':
    main()
