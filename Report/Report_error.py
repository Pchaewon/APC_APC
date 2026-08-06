<!DOCTYPE html>
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
  <p class="sub">위쪽 <b>추천</b>(미래·역산), 아래쪽 <b>실제 추이</b>(최근 wire·실측). 최근 상태를 보고 추천 수용을 판단합니다.</p>
  <div class="meta-row">
    <div class="meta-item"><span class="k">Process Time</span><span class="v">13.3Hr</span></div>
    <div class="meta-item"><span class="k">Target BOW</span><span class="v">1.75 µm</span></div>
    <div class="meta-item"><span class="k">양품 스펙</span><span class="v">1.5 ~ 2.0 µm</span></div>
    <div class="meta-item"><span class="k">생성 시각</span><span class="v">2026-08-06 23:33</span></div>
  </div>
</div></header>
<div class="wrap">
  <div class="summary">
    <div class="stat"><div class="sk">추천 장비</div><div class="sv">2<span class="su">/ 10대</span></div></div>
    <div class="stat"><div class="sk">Target BOW</div><div class="sv">1.75<span class="su">µm</span></div></div>
    <div class="stat"><div class="sk">예상 범위</div><div class="sv">±0.15<span class="su">µm</span></div></div>
    <div class="stat"><div class="sk">실제 추이(최대)</div><div class="sv">1<span class="su">wire</span></div></div>
  </div>
  <div id="cards"></div>
</div>
<div class="foot"><span>WIRESAW_APC · DUAL_MODEL · SLSQP_INVERSE + FIELD_ACTUALS</span><span>GENERATED 2026-08-06 23:33</span></div>
<script>
const PCTS=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], DATA=[{"eqp": "BSWS35", "waf": 1, "wire": "LGA202607590", "bow": 1.75, "rec_frame": [28.0, 28.08, 28.16, 28.24, 28.32, 28.4, 28.48, 28.56, 28.64, 28.72, 28.8], "rec_slurry": [21.15, 22.56, 21.88, 22.45, 22.96, 22.08, 22.0, 21.14, 21.54, 22.0, 22.36], "actual": {"wires": ["LGA202607590"], "bow": [1.801], "warp": [3.178], "blocks": [{"wire": "LGA202607590", "frame": [{"lot": "LGA202607590-L0", "prof": [28.0, 28.08, 28.16, 28.24, 28.32, 28.4, 28.48, 28.56, 28.64, 28.72, 28.8]}, {"lot": "LGA202607590-L1", "prof": [28.0, 28.08, 28.16, 28.24, 28.32, 28.4, 28.48, 28.56, 28.64, 28.72, 28.8]}], "slurry": [{"lot": "LGA202607590-L0", "prof": [21.59, 21.94, 21.99, 21.24, 21.01, 21.83, 21.93, 21.46, 21.77, 21.87, 21.61]}, {"lot": "LGA202607590-L1", "prof": [21.27, 21.28, 21.12, 21.91, 21.03, 21.67, 21.07, 21.36, 21.42, 21.18, 21.52]}], "wg_l": [{"lot": "LGA202607590-L0", "prof": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}, {"lot": "LGA202607590-L1", "prof": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}], "wg_r": [{"lot": "LGA202607590-L0", "prof": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}, {"lot": "LGA202607590-L1", "prof": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}], "ingot": [{"lot": "LGA202607590-L0", "val": 200.0}, {"lot": "LGA202607590-L1", "val": 200.0}], "wait": [{"lot": "LGA202607590-L0", "val": 30.0}, {"lot": "LGA202607590-L1", "val": 30.0}], "warm": [{"lot": "LGA202607590-L0", "val": 10.0}, {"lot": "LGA202607590-L1", "val": 10.0}]}], "has_lot": true}}, {"eqp": "BSWS48", "waf": 1, "wire": "LGA202607594", "bow": 1.75, "rec_frame": [28.0, 28.08, 28.16, 28.24, 28.32, 28.4, 28.48, 28.56, 28.64, 28.72, 28.8], "rec_slurry": [22.61, 21.76, 21.13, 21.58, 22.82, 21.43, 21.9, 22.86, 21.05, 22.2, 22.9], "actual": {"wires": ["LGA202607590"], "bow": [1.798], "warp": [3.296], "blocks": [{"wire": "LGA202607590", "frame": [{"lot": "LGA202607590-L0", "prof": [28.0, 28.08, 28.16, 28.24, 28.32, 28.4, 28.48, 28.56, 28.64, 28.72, 28.8]}, {"lot": "LGA202607590-L1", "prof": [28.0, 28.08, 28.16, 28.24, 28.32, 28.4, 28.48, 28.56, 28.64, 28.72, 28.8]}], "slurry": [{"lot": "LGA202607590-L0", "prof": [21.74, 21.16, 21.19, 21.35, 21.38, 21.21, 21.92, 21.83, 21.11, 21.37, 21.23]}, {"lot": "LGA202607590-L1", "prof": [21.5, 21.92, 21.38, 21.65, 21.6, 21.75, 21.06, 21.74, 21.95, 21.6, 21.29]}], "wg_l": [{"lot": "LGA202607590-L0", "prof": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}, {"lot": "LGA202607590-L1", "prof": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}], "wg_r": [{"lot": "LGA202607590-L0", "prof": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}, {"lot": "LGA202607590-L1", "prof": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}], "ingot": [{"lot": "LGA202607590-L0", "val": 200.0}, {"lot": "LGA202607590-L1", "val": 200.0}], "wait": [{"lot": "LGA202607590-L0", "val": 30.0}, {"lot": "LGA202607590-L1", "val": 30.0}], "warm": [{"lot": "LGA202607590-L0", "val": 10.0}, {"lot": "LGA202607590-L1", "val": 10.0}]}], "has_lot": true}}], RANGE=0.15, TARGET=1.75;
const SPEC_LO=1.5, SPEC_HI=2.0;

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
      <span class="badge badge-waf">WAF ${d.waf}개</span><span class="badge badge-time">13.3Hr</span></div>

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
</html>
