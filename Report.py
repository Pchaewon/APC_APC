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
            print(f"  \u26a0 {r.get(eqp_col)}: \ucd94\ucc9c \uc628\ub3c4 \uc5c6\uc74c \u2014 \uc2a4\ud0b5")
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
    """field_store\uc5d0\uc11c \uc7a5\ube44\ubcc4 \ucd5c\uadfc N wire\uc758 \uc2e4\uc81c\uac12 \ucd94\ucd9c."""
    if not os.path.exists(store_path):
        print(f"  \u26a0 field_store \uc5c6\uc74c: {store_path} \u2014 \uc2e4\uc81c \uc601\uc5ed \uc0dd\ub7b5")
        return {}
    df = pd.read_csv(store_path)
    C = STORE_CFG
    if C['eqp'] not in df.columns:
        print(f"  \u26a0 {C['eqp']} \ucef4\ub7fc \uc5c6\uc74c \u2014 \uc2e4\uc81c \uc601\uc5ed \uc0dd\ub7b5")
        return {}

    acts = {}
    for eqp, g in df.groupby(C['eqp']):
        if C['date'] in g.columns:
            g = g.sort_values(C['date'])
        g = g.tail(RECENT_N)
        wires = g[C['wire']].astype(str).tolist() if C['wire'] in g else \
                [str(i) for i in range(len(g))]

        def col(name):
            return g[C[name]].tolist() if C[name] in g.columns else []

        def profiles(tmpl):
            rows = []
            for _, r in g.iterrows():
                p = _profile(r, tmpl)
                if not all(v is None for v in p):
                    rows.append([v if v is not None else 0 for v in p])
            return rows

        acts[str(eqp)] = {
            'wires': wires,
            'bow':  [round(float(v), 3) if pd.notna(v) else None for v in col('bow')],
            'warp': [round(float(v), 3) if pd.notna(v) else None for v in col('warp')],
            # 프로파일 인자 (wire별 11pct) — 사진처럼 가로 연결로 표시
            'act_frame':  profiles(ACT_FRAME),
            'act_slurry': profiles(ACT_SLURRY),
            'act_wg_l':   profiles(ACT_WG_L),
            'act_wg_r':   profiles(ACT_WG_R),
            # 단일값 인자 (wire별 값 1개) — wire 구획에 점/막대로
            'ingot': [round(float(v), 1) if pd.notna(v) else None for v in col('ingot')],
            'wait':  [round(float(v), 1) if pd.notna(v) else None for v in col('wait')],
            'warm':  [round(float(v), 1) if pd.notna(v) else None for v in col('warm')],
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
<title>Wire Saw APC \u2014 \uc628\ub3c4 Recipe \ucd94\ucc9c \ub9ac\ud3ec\ud2b8</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#12202e; --ink-soft:#3d4d5c; --ink-faint:#7a8896;
    --paper:#ffffff; --panel:#f4f6f8; --panel-line:#e2e7ec; --line:#d4dae0;
    --frame:#0f5c8c; --frame-soft:#e8f1f7;
    --slurry:#b8531f; --slurry-soft:#fbeee6;
    --target:#0f766e; --spec:#c0392b; --actual:#5a6b7a;
    --low:#8a7a2e; --low-bg:#fbf6e3;
    --rec-bg:#f0f6fa; --act-bg:#f6f4f1;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Noto Sans KR',system-ui,sans-serif;color:var(--ink);background:var(--panel);line-height:1.6;padding:0 0 80px;}
  code,.mono{font-family:'JetBrains Mono',monospace;}
  .wrap{max-width:980px;margin:0 auto;padding:0 32px;}
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
  .profile h3{font-size:14px;font-weight:700;display:flex;align-items:center;gap:9px;margin-bottom:4px;}
  .profile h3 .sw{width:11px;height:11px;border-radius:3px;}
  .profile .unit{font-size:11.5px;color:var(--ink-faint);font-family:'JetBrains Mono',monospace;margin-bottom:12px;}
  .chart{width:100%;height:150px;margin-bottom:6px;}
  .chart-tall{height:170px;}
  .tbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:11px;margin-top:8px;}
  .tbl th{background:var(--panel);color:var(--ink-faint);font-weight:500;padding:5px 4px;text-align:center;border-bottom:1px solid var(--panel-line);}
  .tbl td{padding:5px 4px;text-align:center;border-bottom:1px solid #eef1f4;color:var(--ink-soft);}
  .tbl td.v{font-weight:700;color:var(--ink);}
  .trend-row{display:grid;grid-template-columns:1fr 1fr;gap:0;}
  @media(max-width:760px){.trend-row{grid-template-columns:1fr;}}
  .cap{font-size:11px;color:var(--ink-faint);margin-top:2px;}
  .cond-row{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border-top:1px solid var(--panel-line);}
  @media(max-width:620px){.cond-row{grid-template-columns:1fr;}}
  .cond{padding:16px 20px;border-right:1px solid var(--panel-line);}
  .cond:last-child{border-right:none;}
  .cond h4{font-size:12px;font-weight:700;color:var(--ink-soft);margin-bottom:8px;font-family:'JetBrains Mono',monospace;}
  .no-actual{padding:18px 26px;font-size:13px;color:var(--ink-faint);background:var(--act-bg);}
  .foot{max-width:980px;margin:34px auto 0;padding:22px 32px 0;border-top:1px solid var(--line);font-size:11.5px;color:var(--ink-faint);display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;font-family:'JetBrains Mono',monospace;}
  @media(max-width:620px){.summary{grid-template-columns:repeat(2,1fr);}.masthead h1{font-size:24px;}.wrap{padding:0 18px;}}
  @media print{body{background:#fff;}.toolbar{display:none;}.eqp{box-shadow:none;break-inside:avoid;}.masthead,.bow-band,.badge,.tbl th,.stat,.sec-rec,.sec-act{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
</style>
</head>
<body>
<div class="toolbar"><div class="wrap">
  <span class="t-title">WIRESAW_APC \u00b7 RECIPE_RECOMMEND \u00b7 __PTIME__</span>
  <button class="btn" onclick="window.print()">\u2399 \uc778\uc1c4 / PDF \uc800\uc7a5</button>
</div></div>
<header class="masthead"><div class="wrap">
  <div class="eyebrow">Wire Saw APC \u00b7 Temperature Recipe Recommendation</div>
  <h1>\uc628\ub3c4 Recipe \ucd94\ucc9c \ub9ac\ud3ec\ud2b8</h1>
  <p class="sub">\uc704\ucabd <b>\ucd94\ucc9c</b>(\ubbf8\ub798\u00b7\uc5ed\uc0b0), \uc544\ub798\ucabd <b>\uc2e4\uc81c \ucd94\uc774</b>(\ucd5c\uadfc __RECENTN__ wire\u00b7\uc2e4\uce21). \ucd5c\uadfc \uc0c1\ud0dc\ub97c \ubcf4\uace0 \ucd94\ucc9c \uc218\uc6a9\uc744 \ud310\ub2e8\ud569\ub2c8\ub2e4.</p>
  <div class="meta-row">
    <div class="meta-item"><span class="k">Process Time</span><span class="v">__PTIME__</span></div>
    <div class="meta-item"><span class="k">Target BOW</span><span class="v">__TARGET__ \u00b5m</span></div>
    <div class="meta-item"><span class="k">\uc591\ud488 \uc2a4\ud399</span><span class="v">__SPECLO__ ~ __SPECHI__ \u00b5m</span></div>
    <div class="meta-item"><span class="k">\uc0dd\uc131 \uc2dc\uac01</span><span class="v">__STAMP__</span></div>
  </div>
</div></header>
<div class="wrap">
  <div class="summary">
    <div class="stat"><div class="sk">\ucd94\ucc9c \uc7a5\ube44</div><div class="sv">__NEQP__<span class="su">/ 10\ub300</span></div></div>
    <div class="stat"><div class="sk">Target BOW</div><div class="sv">__TARGET__<span class="su">\u00b5m</span></div></div>
    <div class="stat"><div class="sk">\uc608\uc0c1 \ubc94\uc704</div><div class="sv">\u00b1__RANGE__<span class="su">\u00b5m</span></div></div>
    <div class="stat"><div class="sk">\uc2e4\uc81c \ucd94\uc774</div><div class="sv">__RECENTN__<span class="su">wire</span></div></div>
  </div>
  <div class="note">
    <strong>\uad6c\uc870:</strong> \uac01 \uc7a5\ube44\ub294 <b>\ucd94\ucc9c \uc601\uc5ed</b>(\uc628\ub3c4 recipe\u00b7\uc608\uc0c1 BOW)\uacfc <b>\uc2e4\uc81c \uc601\uc5ed</b>(Bow/Warp \ucd94\uc774\u00b7\uc2e4\uc81c \uc628\ub3c4 \ud504\ub85c\ud30c\uc77c\u00b7\uc870\uac74)\uc73c\ub85c \ub098\ub25c\ub2c8\ub2e4.
    \ucd94\ucc9c BOW\ub294 \ubaa9\ud45c \ub2ec\uc131\uc744 \uc704\ud574 \uc5ed\uc0b0\ub41c \uac12\uc774\uba70, WAF \uac1c\uc218\uac00 \uc2e0\ub8b0\ub3c4 \uc9c0\ud45c\uc785\ub2c8\ub2e4. seed/mid/tail\uc740 \uc5d4\uc9c0\ub2c8\uc5b4 \uae30\uc900 \ud655\uc778 \ud6c4 \ucd94\uac00\ub429\ub2c8\ub2e4.
  </div>
  <div id="cards"></div>
</div>
<div class="foot"><span>WIRESAW_APC \u00b7 DUAL_MODEL \u00b7 SLSQP_INVERSE + FIELD_ACTUALS</span><span>GENERATED __STAMP__</span></div>
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
function horizonChart(series,wires,color){
  const W=760,H=210,padL=42,padR=12,padT=16,padB=52,gap=0.18;
  const all=series.flat();if(!all.length)return '<div class="cap">\ub370\uc774\ud130 \uc5c6\uc74c</div>';
  const mn=Math.min(...all),mx=Math.max(...all),sp=(mx-mn)||1;
  const lo=mn-sp*0.12,hi=mx+sp*0.12,rng=hi-lo;
  const n=series.length, plotW=W-padL-padR, slotW=plotW/n, innerW=slotW*(1-gap);
  const y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  // y grid
  let grid='';for(let g=0;g<=3;g++){const val=lo+rng*g/3,yy=y(val);
    grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
    grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(2)}</text>`;}
  // 각 wire 구획: 프로파일 라인 + 점 + 구획 경계(옅게) + wire id 라벨
  let body='';
  series.forEach((s,wi)=>{
    const x0=padL+slotW*wi+slotW*gap/2;
    const xx=i=>x0+innerW*(i/(s.length-1));
    // 구획 배경 (짝수만 옅게)
    if(wi%2===1) body+=`<rect x="${padL+slotW*wi}" y="${padT}" width="${slotW}" height="${H-padT-padB}" fill="#f7f9fb"/>`;
    const pts=s.map((v,i)=>`${xx(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    body+=`<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.8" stroke-linejoin="round"/>`;
    body+=s.map((v,i)=>`<circle cx="${xx(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="1.6" fill="${color}"/>`).join('');
    // wire id (하단 세로 라벨)
    const wid=(wires[wi]||'').toString();
    const cx=x0+innerW/2;
    body+=`<text x="${cx}" y="${H-padB+14}" text-anchor="end" font-family="JetBrains Mono" font-size="8" fill="#8a97a3" transform="rotate(-40 ${cx} ${H-padB+14})">${wid}</text>`;
  });
  const xaxis=`<text x="${padL+plotW/2}" y="${H-4}" text-anchor="middle" font-family="JetBrains Mono" font-size="8.5" fill="#7a8896">Wire ID (\uac01 \uad6c\uac04: pct 0\u2192100%, \uc2dc\uac04\uc21c)</text>`;
  return `<svg class="chart chart-horizon" viewBox="0 0 ${W} ${H}">${grid}${body}${xaxis}</svg>`;
}

// 단일값 인자: wire별 값 1개를 구획에 막대로 (시간순)
function barSlotChart(arr,wires,color,unit){
  const W=760,H=190,padL=42,padR=12,padT=16,padB=52,gap=0.3;
  const vals=arr.filter(v=>v!=null);if(!vals.length)return '<div class="cap">\ub370\uc774\ud130 \uc5c6\uc74c</div>';
  const mn=Math.min(...vals),mx=Math.max(...vals),sp=(mx-mn)||1;
  const lo=mn-sp*0.15,hi=mx+sp*0.15,rng=hi-lo;
  const n=arr.length, plotW=W-padL-padR, slotW=plotW/n, barW=slotW*(1-gap);
  const y=v=>padT+(H-padT-padB)*(1-(v-lo)/rng);
  let grid='';for(let g=0;g<=3;g++){const val=lo+rng*g/3,yy=y(val);
    grid+=`<line x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}" stroke="#eef1f4"/>`;
    grid+=`<text x="${padL-6}" y="${yy+3}" text-anchor="end" font-family="JetBrains Mono" font-size="9" fill="#9aa6b2">${val.toFixed(1)}</text>`;}
  let body='';
  arr.forEach((v,i)=>{
    const x0=padL+slotW*i+slotW*gap/2;
    if(v!=null){
      const yy=y(v), h=(H-padB)-yy;
      body+=`<rect x="${x0.toFixed(1)}" y="${yy.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" fill="${color}" opacity="0.75" rx="1.5"/>`;
      body+=`<text x="${(x0+barW/2).toFixed(1)}" y="${(yy-3).toFixed(1)}" text-anchor="middle" font-family="JetBrains Mono" font-size="7.5" fill="#5a6b7a">${v}</text>`;
    }
    const wid=(wires[i]||'').toString();const cx=x0+barW/2;
    body+=`<text x="${cx}" y="${H-padB+14}" text-anchor="end" font-family="JetBrains Mono" font-size="8" fill="#8a97a3" transform="rotate(-40 ${cx} ${H-padB+14})">${wid}</text>`;
  });
  const xaxis=`<text x="${padL+plotW/2}" y="${H-4}" text-anchor="middle" font-family="JetBrains Mono" font-size="8.5" fill="#7a8896">Wire ID (\uc2dc\uac04\uc21c) \u00b7 ${unit}</text>`;
  return `<svg class="chart chart-horizon" viewBox="0 0 ${W} ${H}">${grid}${body}${xaxis}</svg>`;
}

function trendChart(values,wires,opts){
  const W=460,H=180,padL=38,padR=12,padT=16,padB=40;
  const vals=values.filter(v=>v!=null);if(!vals.length)return '<div class="cap">\ub370\uc774\ud130 \uc5c6\uc74c</div>';
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
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" style="height:186px">${grid}${band}${tline}${line}${dots}${xl}<text x="${padL}" y="${H-6}" font-family="JetBrains Mono" font-size="8" fill="#c3ccd4">\u2190 wire (\uacfc\uac70\u2192\ucd5c\uadfc) \u2192</text></svg>`;
}

function tbl(values){
  const head='<tr><th>pct</th>'+PCTS.map(p=>`<th>${p}</th>`).join('')+'</tr>';
  const row='<tr><td>\u00b0C</td>'+values.map(v=>`<td class="v">${v.toFixed(2)}</td>`).join('')+'</tr>';
  return `<table class="tbl">${head}${row}</table>`;
}
function condTrend(label,arr,unit){
  const vals=arr.filter(v=>v!=null);
  const last=vals.length?vals[vals.length-1]:'\u2014';
  const spark=(function(){
    if(!vals.length)return '';const W=120,H=34,mn=Math.min(...vals),mx=Math.max(...vals),sp=(mx-mn)||1;
    const x=i=>2+(W-4)*(i/Math.max(1,arr.length-1)),y=v=>2+(H-4)*(1-(v-mn)/sp);
    const pts=arr.map((v,i)=>v==null?null:`${x(i)},${y(v)}`).filter(Boolean).join(' ');
    return `<svg width="${W}" height="${H}"><polyline points="${pts}" fill="none" stroke="#5a6b7a" stroke-width="1.5"/></svg>`;
  })();
  return `<div class="cond"><h4>${label}</h4>${spark}<div class="cap">\ucd5c\uadfc\uac12 <b>${last}</b> ${unit}</div></div>`;
}

function card(d){
  const lo=(d.bow-RANGE).toFixed(2),hi=(d.bow+RANGE).toFixed(2);
  const a=d.actual;
  let actualHTML='';
  if(a){
    actualHTML=`
    <div class="sec-label sec-act"><span class="tag">ACTUAL</span> \uc2e4\uc81c \ucd94\uc774 \u00b7 \ucd5c\uadfc ${a.wires.length} wire</div>
    <div class="trend-row">
      <div class="profile" style="border-right:1px solid var(--panel-line)">
        <h3><span class="sw" style="background:var(--actual)"></span>Bow Trend (Total)</h3>
        <div class="unit">avg_bow_bf_total \u00b7 \uc2e4\uce21 \u00b7 wire\uc21c</div>
        ${trendChart(a.bow,a.wires,{color:'#0f5c8c',target:TARGET,specLo:SPEC_LO,specHi:SPEC_HI})}
      </div>
      <div class="profile">
        <h3><span class="sw" style="background:var(--actual)"></span>Warp Trend (Total)</h3>
        <div class="unit">avg_warp_bf_total \u00b7 \uc2e4\uce21 \u00b7 wire\uc21c</div>
        ${trendChart(a.warp,a.wires,{color:'#b8531f'})}
      </div>
    </div>
    <div class="sec-label sec-act" style="border-top:1px solid var(--panel-line);background:#f0f4f7;color:#4a5b6a"><span class="tag" style="background:#4a5b6a">X-FACTOR</span> \uc2e4\uc81c \uc778\uc790 \ud504\ub85c\ud30c\uc77c \u00b7 wire\ubcc4 \uad6c\ud68d (pct 0\u2192100%, \uc2dc\uac04\uc21c)</div>
    <div class="xfactor">
      <div class="xf">
        <h3><span class="sw" style="background:var(--frame)"></span>Frame Temp</h3>
        <div class="unit">set_frame_temp \u00b7 wire ${a.act_frame.length}\uac1c \u00b7 lot\ud3c9\uade0 \ud504\ub85c\ud30c\uc77c</div>
        ${horizonChart(a.act_frame,a.wires,'#0f5c8c')}
      </div>
      <div class="xf">
        <h3><span class="sw" style="background:var(--slurry)"></span>Slurry Temp</h3>
        <div class="unit">set_slurry_temp \u00b7 wire ${a.act_slurry.length}\uac1c \u00b7 lot\ud3c9\uade0 \ud504\ub85c\ud30c\uc77c</div>
        ${horizonChart(a.act_slurry,a.wires,'#b8531f')}
      </div>
      ${a.act_wg_l.length?`<div class="xf">
        <h3><span class="sw" style="background:#5a6b7a"></span>Wire Guide L</h3>
        <div class="unit">shift_amount_wireguide_l \u00b7 wire ${a.act_wg_l.length}\uac1c</div>
        ${horizonChart(a.act_wg_l,a.wires,'#5a6b7a')}
      </div>`:''}
      ${a.act_wg_r.length?`<div class="xf">
        <h3><span class="sw" style="background:#8a6d4a"></span>Wire Guide R</h3>
        <div class="unit">shift_amount_wireguide_r \u00b7 wire ${a.act_wg_r.length}\uac1c</div>
        ${horizonChart(a.act_wg_r,a.wires,'#8a6d4a')}
      </div>`:''}
      <div class="xf">
        <h3><span class="sw" style="background:#0f766e"></span>ingot_len</h3>
        <div class="unit">fdc_ingot_len \u00b7 wire\ubcc4 \ub2e8\uc77c\uac12</div>
        ${barSlotChart(a.ingot,a.wires,'#0f766e','mm')}
      </div>
      <div class="xf">
        <h3><span class="sw" style="background:#0f766e"></span>wait_time</h3>
        <div class="unit">fdc_wait_time \u00b7 wire\ubcc4 \ub2e8\uc77c\uac12</div>
        ${barSlotChart(a.wait,a.wires,'#7a8896','')}
      </div>
      <div class="xf">
        <h3><span class="sw" style="background:#0f766e"></span>warm_up_time</h3>
        <div class="unit">fdc_warm_up_time \u00b7 wire\ubcc4 \ub2e8\uc77c\uac12</div>
        ${barSlotChart(a.warm,a.wires,'#b8531f','')}
      </div>
    </div>`;
  }else{
    actualHTML=`<div class="no-actual">\u26a0 \uc774 \uc7a5\ube44\uc758 field_store \uc2e4\uc81c \ub370\uc774\ud130\uac00 \uc5c6\uc5b4 \uc2e4\uc81c \uc601\uc5ed\uc744 \uc0dd\ub7b5\ud569\ub2c8\ub2e4.</div>`;
  }
  return `<section class="eqp">
    <div class="eqp-head"><span class="eqp-name">${d.eqp}</span><div class="grow"></div>
      <span class="badge badge-waf">WAF ${d.waf}\uac1c</span><span class="badge badge-time">__PTIME__</span></div>

    <div class="sec-label sec-rec"><span class="tag">RECOMMEND</span> \ucd94\ucc9c \u00b7 \ubbf8\ub798 lot (\uc5ed\uc0b0)</div>
    <div class="bow-band"><span class="lbl">\uc608\uc0c1 BOW</span><span class="bow-val">${d.bow.toFixed(2)}</span>
      <span class="bow-range">${lo} ~ ${hi} \u00b5m</span>
      <span class="bow-target">Target <b>${TARGET}</b> \u00b7 \ucd5c\uadfc wire <code>${d.wire}</code></span></div>
    <div class="profiles">
      <div class="profile"><h3><span class="sw" style="background:var(--frame)"></span>\u2460 Frame Temp \ucd94\ucc9c</h3>
        <div class="unit">rec_set_frame_temp \u00b7 \u00b0C \u00b7 0\u2192100pct</div>${lineChart(d.rec_frame,'#0f5c8c','#cfe2ef')}${tbl(d.rec_frame)}</div>
      <div class="profile"><h3><span class="sw" style="background:var(--slurry)"></span>\u2461 Slurry Temp \ucd94\ucc9c</h3>
        <div class="unit">rec_set_slurry_temp \u00b7 \u00b0C \u00b7 0\u2192100pct</div>${lineChart(d.rec_slurry,'#b8531f','#f0d9c9')}${tbl(d.rec_slurry)}</div>
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
        print(f"\u274c \ucd94\ucc9c CSV \uc5c6\uc74c: {rec_csv}"); sys.exit(1)

    print(f"[\ub9ac\ud3ec\ud2b8] \ucd94\ucc9c: {rec_csv}")
    recs = load_recommend(rec_csv)
    if not recs:
        print("\u274c \ucd94\ucc9c 0\uac74 \u2014 \ub9ac\ud3ec\ud2b8 \uc0dd\uc131 \uc548 \ud568"); sys.exit(1)

    print(f"[\ub9ac\ud3ec\ud2b8] \uc2e4\uc81c: {store_csv}")
    acts = load_actuals(store_csv)
    records = merge(recs, acts)

    html = render_html(records)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\u2705 \ub9ac\ud3ec\ud2b8 \uc0dd\uc131: {out_path} ({len(records)}\uac1c \uc7a5\ube44)")
    for r in records:
        has_act = '\uc2e4\uc81cO' if r.get('actual') else '\uc2e4\uc81cX'
        print(f"   \u00b7 {r['eqp']}: WAF {r['waf']}\uac1c, \uc608\uce21 BOW {r['bow']}, {has_act}")


if __name__ == '__main__':
    main()
