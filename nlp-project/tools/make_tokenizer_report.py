"""Generate a self-contained HTML visualization of the tokenizer experiments.

Reads only existing report artifacts (it modifies nothing):

  * reports/fertility_audit.json        unified NLLB / HY-MT2 / CS336 fertility
  * reports/nllb_tokenizer_audit.json   NLLB tokenizer-only audit
  * outputs/cs336-bpe-fertility/*       CS336 BPE vocab/merge files

Writes reports/tokenizer_experiments.html, which embeds every number inline and
renders with hand-rolled SVG -- no CDN, no network, works offline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "tokenizer_experiments.html"

DATASET_ORDER = ["flores_dev", "flores_devtest", "mitra_validation", "mitra_train_sample"]
DATASET_LABELS = {
    "flores_dev": "FLORES dev",
    "flores_devtest": "FLORES devtest",
    "mitra_validation": "MITRA validation",
    "mitra_train_sample": "MITRA train (10k)",
}
CONFIGS = [
    {"key": "nllb", "label": "NLLB-200", "color": "#2563eb"},
    {"key": "hy_mt2", "label": "HY-MT2-1.8B", "color": "#dc2626"},
    {"key": "cs336_raw", "label": "CS336 Raw", "color": "#059669"},
    {"key": "cs336_syllable_aware", "label": "CS336 Tsheg", "color": "#d97706"},
    {"key": "cs336_mark_aware", "label": "CS336 Mark-aware", "color": "#7c3aed"},
]


def collect_bpe() -> list[dict]:
    """Effective vocabulary actually produced by each CS336 BPE configuration."""
    base = ROOT / "outputs" / "cs336-bpe-fertility"
    rows: list[dict] = []
    if not base.exists():
        return rows
    for variant_dir in sorted(base.iterdir()):
        if not variant_dir.is_dir():
            continue
        for vocab_dir in sorted(variant_dir.iterdir(), key=lambda p: p.name):
            vocab_file = vocab_dir / "vocab.json"
            merges_file = vocab_dir / "merges.json"
            requested = vocab_dir.name.replace("vocab_", "")
            row = {
                "variant": variant_dir.name,
                "requested": int(requested),
                "vocab_entries": None,
                "merges": None,
                "present": vocab_file.exists(),
            }
            if vocab_file.exists():
                row["vocab_entries"] = len(json.loads(vocab_file.read_text(encoding="utf-8")))
            if merges_file.exists():
                row["merges"] = len(json.loads(merges_file.read_text(encoding="utf-8")))
            rows.append(row)
    return rows


def build_payload() -> dict:
    fertility = json.loads((ROOT / "reports" / "fertility_audit.json").read_text(encoding="utf-8"))
    nllb = json.loads((ROOT / "reports" / "nllb_tokenizer_audit.json").read_text(encoding="utf-8"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_order": DATASET_ORDER,
        "dataset_labels": DATASET_LABELS,
        "configs": CONFIGS,
        "fertility": fertility,
        "nllb_audit": nllb,
        "bpe": collect_bpe(),
    }


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>藏文分词器实验结果可视化 · NLLB / HY-MT2 / CS336</title>
<style>
  :root{
    --bg:#f6f7f9; --panel:#ffffff; --ink:#111827; --muted:#6b7280; --line:#e5e7eb;
    --accent:#2563eb; --warn:#b45309; --good:#047857; --bad:#b91c1c;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  header{background:linear-gradient(135deg,#111827,#1f2937);color:#fff;padding:34px 28px 26px}
  header h1{margin:0 0 8px;font-size:26px;letter-spacing:.2px}
  header .sub{color:#cbd5e1;font-size:14px}
  header .meta{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
  header .meta span{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
       padding:3px 10px;border-radius:999px;font-size:12.5px}
  nav{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);
      border-bottom:1px solid var(--line);backdrop-filter:blur(6px);
      padding:9px 28px;display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13.5px}
  nav a{color:var(--muted);text-decoration:none;padding:3px 2px;border-bottom:2px solid transparent}
  nav a:hover{color:var(--accent);border-bottom-color:var(--accent)}
  main{max-width:1180px;margin:0 auto;padding:24px 22px 70px}
  section{background:var(--panel);border:1px solid var(--line);border-radius:12px;
      padding:22px 24px;margin:20px 0;box-shadow:0 1px 2px rgba(16,24,40,.04)}
  h2{margin:0 0 4px;font-size:20px}
  h3{margin:26px 0 8px;font-size:16px;color:#1f2937}
  h4{margin:20px 0 6px;font-size:14.5px;color:#374151}
  .lede{color:var(--muted);font-size:13.5px;margin:0 0 16px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(184px,1fr));gap:12px;margin:6px 0 4px}
  .card{border:1px solid var(--line);border-radius:10px;padding:13px 14px;background:#fcfcfd}
  .card .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .card .v{font-size:22px;font-weight:650;margin:3px 0 1px;font-variant-numeric:tabular-nums}
  .card .n{font-size:12px;color:var(--muted)}
  .chart{width:100%;height:auto;display:block;overflow:visible}
  .grid{stroke:#eef1f4;stroke-width:1}
  .axis{fill:#6b7280;font-size:11.5px}
  .axis-title{fill:#374151;font-size:12px;font-weight:600}
  .legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:10px 0 0;font-size:13px}
  .legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px}
  table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
  th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:right;white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  thead th{background:#f9fafb;color:#374151;font-weight:600;position:relative}
  tbody tr:hover{background:#f9fafb}
  .scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
  .note{background:#f8fafc;border-left:3px solid var(--accent);padding:11px 14px;
      border-radius:0 8px 8px 0;margin:14px 0;font-size:13.5px;color:#334155}
  .note.warn{background:#fffbeb;border-left-color:var(--warn)}
  .note.good{background:#f0fdf4;border-left-color:var(--good)}
  .tabs{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}
  .tabs button{border:1px solid var(--line);background:#fff;border-radius:999px;
      padding:5px 13px;font-size:13px;cursor:pointer;color:#374151;font-family:inherit}
  .tabs button.on{background:var(--accent);border-color:var(--accent);color:#fff}
  details{margin:10px 0;border:1px solid var(--line);border-radius:9px;padding:10px 14px;background:#fcfcfd}
  summary{cursor:pointer;font-weight:600;font-size:14px}
  pre{background:#0b1220;color:#e2e8f0;padding:14px;border-radius:8px;overflow:auto;
      font-size:12px;line-height:1.5;max-height:460px}
  code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:13px}
  .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600}
  .pill.ok{background:#dcfce7;color:#166534}
  .pill.no{background:#fee2e2;color:#991b1b}
  .pill.mid{background:#fef3c7;color:#92400e}
  footer{max-width:1180px;margin:0 auto;padding:0 22px 40px;color:var(--muted);font-size:12.5px}
</style>
</head>
<body>
<header>
  <h1>藏文分词器（Tokenizer）实验结果可视化</h1>
  <div class="sub">NLLB-200 · HY-MT2-1.8B · CS336 自训 BPE ｜ 藏文 → 中文翻译课程项目</div>
  <div class="meta" id="headmeta"></div>
</header>
<nav id="nav"></nav>
<main id="main"></main>
<footer>
  本页由 <code>tools/make_tokenizer_report.py</code> 生成，全部数值内嵌于页面，可离线查看。
  数据源：<code>reports/fertility_audit.json</code>、<code>reports/nllb_tokenizer_audit.json</code>、
  <code>outputs/cs336-bpe-fertility/</code>。
</footer>
<script id="payload" type="application/json">/*__DATA__*/</script>
<script>
"use strict";
const P = JSON.parse(document.getElementById('payload').textContent);
const F = P.fertility, A = P.nllb_audit;
const DS = P.dataset_order, DSL = P.dataset_labels, CFG = P.configs;

/* ------------------------------------------------------------ utilities */
const SVGNS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, text){
  const node = document.createElementNS(SVGNS, tag);
  for (const k in (attrs||{})) node.setAttribute(k, attrs[k]);
  if (text !== undefined) node.textContent = text;
  return node;
}
function h(tag, attrs, kids){
  const node = document.createElement(tag);
  for (const k in (attrs||{})){
    if (k === 'class') node.className = attrs[k];
    else if (k === 'html') node.innerHTML = attrs[k];
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), attrs[k]);
    else node.setAttribute(k, attrs[k]);
  }
  (kids||[]).forEach(c => node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
  return node;
}
const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(d === undefined ? 3 : d);
const pct = (v, d) => (v === null || v === undefined) ? '—' : (Number(v) * 100).toFixed(d === undefined ? 2 : d) + '%';
function section(id, title, lede, kids){
  const s = h('section', {id: id});
  s.appendChild(h('h2', {}, [title]));
  if (lede) s.appendChild(h('p', {class:'lede', html: lede}));
  kids.forEach(k => s.appendChild(k));
  return s;
}
function table(headers, rows, opts){
  const wrap = h('div', {class:'scroll'});
  const t = h('table');
  const thead = h('thead'); const tr = h('tr');
  headers.forEach(x => tr.appendChild(h('th', {}, [x])));
  thead.appendChild(tr); t.appendChild(thead);
  const tb = h('tbody');
  rows.forEach(r => {
    const row = h('tr');
    r.forEach((cell, i) => {
      const td = h('td', {}, [cell === null || cell === undefined ? '—' : String(cell)]);
      if (opts && opts.strong && opts.strong.includes(i)) td.style.fontWeight = '650';
      row.appendChild(td);
    });
    tb.appendChild(row);
  });
  t.appendChild(tb); wrap.appendChild(t); return wrap;
}
function legend(series){
  const box = h('div', {class:'legend'});
  series.forEach(s => box.appendChild(h('span', {html:'<i style="background:'+s.color+'"></i>'+s.label})));
  return box;
}

/* -------------------------------------------------------- grouped bars */
function groupedBar(host, groups, series, opts){
  opts = opts || {};
  const W = 940, padL = 66, padR = 16, padT = 18, padB = opts.padB || 76;
  const H = opts.height || 360;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  let maxVal = 0;
  series.forEach(s => s.values.forEach(v => { if (typeof v === 'number' && v > maxVal) maxVal = v; }));
  const top = (maxVal || 1) * 1.16;
  const svg = el('svg', {viewBox:'0 0 '+W+' '+H, class:'chart', preserveAspectRatio:'xMidYMid meet'});
  const ticks = 5;
  for (let i = 0; i <= ticks; i++){
    const v = top * i / ticks, y = padT + plotH - (v / top) * plotH;
    svg.appendChild(el('line', {x1:padL, x2:W - padR, y1:y, y2:y, class:'grid'}));
    svg.appendChild(el('text', {x:padL - 9, y:y + 4, class:'axis', 'text-anchor':'end'}, fmt(v, opts.decimals === undefined ? 2 : opts.decimals)));
  }
  const gw = plotW / groups.length;
  const bw = Math.min(48, (gw * 0.78) / series.length);
  groups.forEach((g, gi) => {
    const gx = padL + gi * gw;
    series.forEach((s, si) => {
      const v = s.values[gi];
      if (typeof v !== 'number') return;
      const bh = (v / top) * plotH;
      const x = gx + gw / 2 - (bw * series.length) / 2 + si * bw;
      const rect = el('rect', {x:x + 1.5, y:padT + plotH - bh, width:Math.max(1, bw - 3),
                               height:Math.max(0, bh), fill:s.color, rx:2.5});
      rect.appendChild(el('title', {}, s.label + ' · ' + g + '\n' + fmt(v, opts.decimals === undefined ? 3 : opts.decimals)));
      svg.appendChild(rect);
      if (opts.showValues){
        svg.appendChild(el('text', {x:x + bw / 2, y:padT + plotH - bh - 4, class:'axis',
                                    'text-anchor':'middle', 'font-size':'10'}, fmt(v, opts.decimals === undefined ? 2 : opts.decimals)));
      }
    });
    svg.appendChild(el('text', {x:gx + gw / 2, y:padT + plotH + 19, class:'axis', 'text-anchor':'middle'}, g));
  });
  if (opts.yTitle)
    svg.appendChild(el('text', {x:14, y:padT + plotH / 2, class:'axis-title', 'text-anchor':'middle',
                                transform:'rotate(-90 14 '+(padT + plotH / 2)+')'}, opts.yTitle));
  host.appendChild(svg);
  if (opts.legend !== false) host.appendChild(legend(series));
  return host;
}

/* --------------------------------------------------- horizontal bars */
function hBar(host, rows, opts){
  opts = opts || {};
  const W = 940, padL = opts.padL || 220, padR = 74, rowH = opts.rowH || 26;
  const H = rows.length * rowH + 34;
  const plotW = W - padL - padR;
  let maxVal = 1; rows.forEach(r => { if (r.value > maxVal) maxVal = r.value; });
  const top = maxVal * 1.02;
  const svg = el('svg', {viewBox:'0 0 '+W+' '+H, class:'chart'});
  rows.forEach((r, i) => {
    const y = i * rowH + 8;
    const w = Math.max(1, (r.value / top) * plotW);
    svg.appendChild(el('text', {x:padL - 10, y:y + rowH * 0.68, class:'axis', 'text-anchor':'end'}, r.label));
    const rect = el('rect', {x:padL, y:y + 3, width:w, height:rowH - 10,
                             fill:r.color || '#2563eb', rx:3});
    rect.appendChild(el('title', {}, r.label + ': ' + fmt(r.value, opts.decimals === undefined ? 3 : opts.decimals)));
    svg.appendChild(rect);
    svg.appendChild(el('text', {x:padL + w + 8, y:y + rowH * 0.68, class:'axis'}, fmt(r.value, opts.decimals === undefined ? 3 : opts.decimals)));
  });
  host.appendChild(svg);
  return host;
}

/* ------------------------------------------------------------ range plot */
function rangePlot(host, rows, opts){
  opts = opts || {};
  const W = 940, padL = 190, padR = 54, padT = 12, rowH = 32;
  const H = padT + rows.length * rowH + 42;
  const plotW = W - padL - padR;
  let domainMax = 1; rows.forEach(r => { if (r.max > domainMax) domainMax = r.max; });
  domainMax *= 1.06;
  const X = v => padL + (v / domainMax) * plotW;
  const svg = el('svg', {viewBox:'0 0 '+W+' '+H, class:'chart'});
  const ticks = 6;
  for (let i = 0; i <= ticks; i++){
    const v = domainMax * i / ticks, x = X(v);
    svg.appendChild(el('line', {x1:x, x2:x, y1:padT, y2:padT + rows.length * rowH, class:'grid'}));
    svg.appendChild(el('text', {x:x, y:padT + rows.length * rowH + 17, class:'axis', 'text-anchor':'middle'}, String(Math.round(v))));
  }
  rows.forEach((r, i) => {
    const cy = padT + i * rowH + rowH / 2;
    svg.appendChild(el('text', {x:padL - 12, y:cy + 4, class:'axis', 'text-anchor':'end'}, r.label));
    svg.appendChild(el('line', {x1:X(r.min), x2:X(r.max), y1:cy, y2:cy, stroke:'#cbd5e1', 'stroke-width':2}));
    svg.appendChild(el('line', {x1:X(r.p95), x2:X(r.p99), y1:cy, y2:cy, stroke:r.color, 'stroke-width':8, 'stroke-linecap':'round'}));
    const dot = el('circle', {cx:X(r.median), cy:cy, r:4.2, fill:'#111827'});
    dot.appendChild(el('title', {}, r.label + '\nmedian ' + r.median + '\nmin ' + r.min + '  p95 ' + r.p95 + '  p99 ' + r.p99 + '  max ' + r.max));
    svg.appendChild(dot);
    svg.appendChild(el('text', {x:X(r.max) + 8, y:cy + 4, class:'axis'}, String(r.max)));
  });
  if (opts.xTitle)
    svg.appendChild(el('text', {x:padL + plotW / 2, y:H - 4, class:'axis-title', 'text-anchor':'middle'}, opts.xTitle));
  host.appendChild(svg);
  host.appendChild(h('div', {class:'legend', html:
    '<span><i style="background:#cbd5e1"></i>min–max 全距</span>' +
    '<span><i style="background:#2563eb"></i>p95–p99 尾部</span>' +
    '<span><i style="background:#111827;border-radius:50%"></i>中位数</span>'}));
  return host;
}

/* --------------------------------------------------------- data access */
function srcStats(ds, key){
  const m = F.datasets[ds].models;
  if (key === 'nllb') return m.nllb.source;
  if (key === 'hy_mt2') return m.hy_mt2.source;
  return m.cs336[key.replace('cs336_','')]['8192'].source;
}
function tgtStats(ds, key){ return F.datasets[ds].models[key].target; }
function seriesFor(getter){
  return CFG.map(c => ({label:c.label, color:c.color, values:DS.map(ds => {
    const s = getter(ds, c.key);
    return s === null || s === undefined ? null : s;
  })}));
}
function metric(name){
  return function(ds, key){
    const s = srcStats(ds, key);
    if (name === 'syl') return s.fertility_tokens_per_tibetan_syllable.mean;
    if (name === 'cp')  return s.fertility_tokens_per_codepoint.mean;
    if (name === 'meanlen') return s.length_tokens.mean;
    if (name === 'p95') return s.length_tokens.p95;
    return null;
  };
}

/* ================================================================ build */
const nav = document.getElementById('nav');
const main = document.getElementById('main');
const SECTIONS = [];
function addSection(id, title, lede, kids){ const s = section(id, title, lede, kids); main.appendChild(s); SECTIONS.push([id, title]); return s; }
const groups = DS.map(d => DSL[d]);

/* header meta */
document.getElementById('headmeta').innerHTML = [
  'fertility 实验：seed=' + F.seed + '，MITRA 训练抽样 ' + F.train_sample_size.toLocaleString() + ' 条',
  'NLLB tokenizer 审计：seed=' + A.seed + '，MITRA 训练抽样 ' + A.datasets.mitra_train_sample.pairs.toLocaleString() + ' 条',
  '数据集 ' + DS.length + ' 组',
  'tokenizer 配置 ' + CFG.length + ' 种',
  '生成于 ' + P.generated_at.slice(0, 19).replace('T', ' ') + ' UTC'
].map(t => '<span>' + t + '</span>').join('');

/* ------------------------------------------------------------- overview */
const nllbSyl = metric('syl');
addSection('overview', '一、总览',
  '本页汇总当前全部 tokenizer 实验结果：统一 fertility 审计（4 个数据集 × 5 种 tokenizer 配置）与 NLLB tokenizer 单独审计（长度分布 + 未知 token 长尾），并核对 CS336 自训 BPE 的实际词表规模。',
  [
    h('div', {class:'cards', html:
      '<div class="card"><div class="k">NLLB 藏文 fertility</div><div class="v">' + fmt(Math.min(...DS.map(d=>nllbSyl(d,'nllb'))), 3) + '–' + fmt(Math.max(...DS.map(d=>nllbSyl(d,'nllb'))), 3) + '</div><div class="n">tokens / 藏文音节（最低）</div></div>' +
      '<div class="card"><div class="k">HY-MT2 藏文 fertility</div><div class="v">' + fmt(Math.min(...DS.map(d=>nllbSyl(d,'hy_mt2'))), 2) + '–' + fmt(Math.max(...DS.map(d=>nllbSyl(d,'hy_mt2'))), 2) + '</div><div class="n">tokens / 藏文音节（最高）</div></div>' +
      '<div class="card"><div class="k">Mark-aware 改善</div><div class="v">' + fmt((1 - nllbSyl('mitra_validation','cs336_mark_aware') / nllbSyl('mitra_validation','cs336_raw')) * 100, 1) + '%</div><div class="n">相对 CS336 Raw（MITRA val）</div></div>' +
      '<div class="card"><div class="k">超过 512 token</div><div class="v">0</div><div class="n">所有数据集与 tokenizer</div></div>' +
      '<div class="card"><div class="k">CS336 有效词表</div><div class="v">' + Math.min(...P.bpe.filter(b=>b.vocab_entries).map(b=>b.vocab_entries)).toLocaleString() + '–' + Math.max(...P.bpe.filter(b=>b.vocab_entries).map(b=>b.vocab_entries)).toLocaleString() + '</div><div class="n">请求 8k/16k 均未达到</div></div>'
    }),
    h('div', {class:'note good', html:
      '<b>核心结论：</b>① NLLB 的藏文编码最紧凑（约 1.17–1.20 token/音节）；② HY-MT2 序列最长（约 4.06–4.26），但无未知 token；③ CS336 的预分词方式显著影响编码效率——把 <code>\\p{L}</code> 放宽到 <code>[\\p{L}\\p{M}]</code>（Mark-aware）后 fertility 从约 2.83 降到约 2.06，相对下降约 27%，说明组合记号边界是藏文编码效率的关键。'}),
    h('div', {class:'note warn', html:
      '<b>重要限制：</b>fertility 更紧凑只说明输入 token 统计更省，<u>不等于翻译质量更好</u>。要验证下游收益必须在统一数据与统一评测下训练并比较，当前尚不具备该证据。'})
  ]);

/* ------------------------------------------------------------ protocol */
addSection('protocol', '二、实验口径',
  '两份报告的抽样规模与统计约定不同，阅读数字前必须先区分口径。',
  [
    table(['项目', '统一 fertility 审计', 'NLLB tokenizer 审计'], [
      ['结果文件', 'reports/fertility_audit.json', 'reports/nllb_tokenizer_audit.json'],
      ['seed', F.seed, A.seed],
      ['MITRA 训练抽样', F.train_sample_size.toLocaleString() + ' 条', A.datasets.mitra_train_sample.pairs.toLocaleString() + ' 条'],
      ['MITRA validation', F.datasets.mitra_validation.pairs.toLocaleString() + ' 条', A.datasets.mitra_validation.pairs.toLocaleString() + ' 条'],
      ['FLORES dev / devtest', F.datasets.flores_dev.pairs + ' / ' + F.datasets.flores_devtest.pairs, A.datasets.flores_dev.pairs + ' / ' + A.datasets.flores_devtest.pairs],
      ['目标语言标签', 'FLORES 用 zho_Hans，MITRA 用 zho_Hant', '同左'],
      ['是否计特殊 token', '不计入', '不计入'],
      ['涉及的 tokenizer', 'NLLB + HY-MT2 + CS336×3', '仅 NLLB'],
      ['音节近似', '仅按 ་ / ༌ 边界切分，不等价于词语分词', '—']
    ], {strong:[0]}),
    h('div', {class:'note', html:
      '<b>已核实的口径差异（+2 偏移）：</b>在抽样规模相同的三组数据上，NLLB tokenizer 审计的长度均值<b>恰好比 fertility 审计多 2.000 token</b>，最大值也多 2（FLORES dev 源侧 43.81 vs 41.81、max 118 vs 116；devtest 45.27 vs 43.27、131 vs 129；MITRA validation 16.21 vs 14.21、119 vs 117）。' +
      '这说明 <b>tokenizer 审计把每条序列的 2 个特殊 token 计入了长度</b>，而 fertility 审计明确排除特殊 token。' +
      '唯一的例外是 MITRA 训练抽样：两份报告分别使用 5,000 与 10,000 条，且两者并非严格子集关系（源侧最大长度 78 vs 120），<b>该组不可跨报告比较</b>。'}
    )
  ]);

/* ---------------------------------------------------------- fertility */
const secF = addSection('fertility', '三、Fertility 核心对比',
  '归一化单位分别是“藏文音节”与“Unicode 码点”，全部为源侧（藏文）统计。平均值之外同时给出中位数与 p95，避免只看均值掩盖长尾。',
  []);

const sylHost = h('div');
secF.appendChild(h('h3', {}, ['3.1 tokens / 藏文音节（源侧，越低越紧凑）']));
secF.appendChild(sylHost);
groupedBar(sylHost, groups, seriesFor((ds,c) => metric('syl')(ds,c)), {yTitle:'tokens / 音节', decimals:2, height:380, showValues:false});

const cpHost = h('div');
secF.appendChild(h('h3', {}, ['3.2 tokens / Unicode 码点（源侧）']));
secF.appendChild(cpHost);
groupedBar(cpHost, groups, seriesFor((ds,c) => metric('cp')(ds,c)), {yTitle:'tokens / 码点', decimals:2, height:380});

secF.appendChild(h('h3', {}, ['3.3 完整明细（均值 / 中位数 / p95）']));
(function(){
  const rows = [];
  DS.forEach(ds => {
    CFG.forEach(c => {
      const s = srcStats(ds, c.key);
      const a = s.fertility_tokens_per_tibetan_syllable, b = s.fertility_tokens_per_codepoint;
      rows.push([DSL[ds], c.label, fmt(a.mean,4), fmt(a.median,4), fmt(a.p95,4),
                 fmt(b.mean,4), fmt(b.median,4), fmt(b.p95,4)]);
    });
  });
  secF.appendChild(table(['数据集','Tokenizer','音节:均值','音节:中位','音节:p95','码点:均值','码点:中位','码点:p95'], rows));
})();

/* ------------------------------------------------------- target side */
const secT = addSection('tgt', '四、目标侧（中文）编码',
  '只有 NLLB 与 HY-MT2 有目标侧结果；CS336 的 BPE 仅作用于藏文源文本。',
  []);
(function(){
  const host = h('div');
  secT.appendChild(h('h3', {}, ['4.1 tokens / Unicode 码点（中文侧）']));
  secT.appendChild(host);
  const series = [
    {label:'NLLB-200', color:'#2563eb', values:DS.map(ds => tgtStats(ds,'nllb').fertility_tokens_per_codepoint.mean)},
    {label:'HY-MT2-1.8B', color:'#dc2626', values:DS.map(ds => tgtStats(ds,'hy_mt2').fertility_tokens_per_codepoint.mean)}
  ];
  groupedBar(host, groups, series, {yTitle:'tokens / 码点', decimals:3, height:330, showValues:true});

  secT.appendChild(h('h3', {}, ['4.2 目标侧完整明细']));
  const rows = [];
  DS.forEach(ds => {
    ['nllb','hy_mt2'].forEach(k => {
      const t = tgtStats(ds,k), L = t.length_tokens, C = t.unicode_codepoints, K = t.fertility_tokens_per_codepoint;
      rows.push([DSL[ds], k === 'nllb' ? 'NLLB-200' : 'HY-MT2-1.8B',
        L.min, fmt(L.mean,2), L.median, L.p95, L.max, C.mean.toFixed(1), fmt(K.mean,4), pct(t.unknown_rate_over_tokens)]);
    });
  });
  secT.appendChild(table(['数据集','Tokenizer','token最小','token均值','token中位','token p95','token最大','码点均值','tokens/码点','unknown率'], rows));
  secT.appendChild(h('div', {class:'note', html:
    '中文侧 NLLB 比 HY-MT2 更长（每个汉字约 0.79 vs 0.58 token），但 NLLB 存在未知 token，而 HY-MT2 为 0——这是两条不同技术路线的直接对照：<b>NLLB 词表覆盖更浅但更密，HY-MT2 覆盖更全但更长</b>。'}));
})();

/* ------------------------------------------------------- length dist */
const secL = addSection('len', '五、序列长度分布与截断风险',
  '长度决定上下文窗口与显存开销。区间图展示 min–max 全距、p95–p99 尾部与中位数；所有配置均未出现超过 512 token 的样本。',
  []);
(function(){
  const tabsBox = h('div', {class:'tabs'});
  const plotBox = h('div');
  let current = 'nllb';
  const KEYS = [['nllb','NLLB-200'],['hy_mt2','HY-MT2-1.8B'],
                ['cs336_raw','CS336 Raw'],['cs336_syllable_aware','CS336 Tsheg'],['cs336_mark_aware','CS336 Mark-aware']];
  function render(){
    plotBox.innerHTML = '';
    const color = (CFG.find(c => c.key === current) || {}).color || '#2563eb';
    const rows = DS.map(ds => {
      const L = srcStats(ds, current).length_tokens;
      return {label:DSL[ds], min:L.min, median:L.median, p95:L.p95, p99:L.p99, max:L.max, color:color};
    });
    rangePlot(plotBox, rows, {xTitle:'藏文源句 token 数'});
  }
  KEYS.forEach(([k,label]) => {
    const b = h('button', {class: k === current ? 'on' : '', onclick: () => {
      current = k;
      Array.from(tabsBox.children).forEach(c => c.className = '');
      b.className = 'on';
      render();
    }}, [label]);
    tabsBox.appendChild(b);
  });
  secL.appendChild(tabsBox);
  secL.appendChild(h('h3', {}, ['5.1 源侧（藏文）token 长度区间']));
  secL.appendChild(plotBox);
  render();

  secL.appendChild(h('h3', {}, ['5.2 源侧完整百分位表']));
  (function(){
    const rows = [];
    DS.forEach(ds => CFG.forEach(c => {
      const L = srcStats(ds, c.key).length_tokens;
      rows.push([DSL[ds], c.label, L.min, fmt(L.mean,2), L.median, L.p90, L.p95, L.p99, L.max]);
    }));
    secL.appendChild(table(['数据集','Tokenizer','min','mean','median','p90','p95','p99','max'], rows));
  })();

  secL.appendChild(h('h3', {}, ['5.3 目标侧完整百分位表（NLLB 审计口径）']));
  (function(){
    const rows = [];
    DS.forEach(ds => ['source','target'].forEach(side => {
      const s = A.datasets[ds][side];
      rows.push([DSL[ds], side === 'source' ? '藏文源' : '中文目标', s.count.toLocaleString(), s.min,
                 fmt(s.mean,2), s.median, s.p90, s.p95, s.p99, s.max, s.over_256, s.over_512_trained_limit]);
    }));
    secL.appendChild(table(['数据集','侧','样本数','min','mean','median','p90','p95','p99','max','>256','>512'], rows));
  })();

  secL.appendChild(h('div', {class:'note good', html:
    '<b>截断风险结论：</b>NLLB tokenizer 审计中所有数据集、所有侧的 <code>&gt;512</code> 计数均为 0，<code>&gt;256</code> 也为 0。因此第一版微调把 <code>max_length</code> 设为 256 是安全的，无需长句切分。<br>' +
    '但 CS336 与 HY-MT2 的源侧更长：HY-MT2 在 FLORES dev 有 36 条、devtest 有 45 条超过 256 token；CS336 Raw 在 devtest 有 5 条超过 256。若要用它们，必须单独核算预算。'}));
})();

/* ---------------------------------------------------------- unknown */
const secU = addSection('unk', '六、未知 token 与长尾字符',
  '未知 token 直接关系到生成质量：词表未覆盖的字会被切成 <code>&lt;unk&gt;</code>，在中文侧尤其集中于佛典罕见字。',
  []);
(function(){
  const host = h('div');
  secU.appendChild(h('h3', {}, ['6.1 未知 token 比率（NLLB / HY-MT2）']));
  secU.appendChild(host);
  const series = [
    {label:'NLLB 源(藏文)', color:'#2563eb', values:DS.map(ds => srcStats(ds,'nllb').unknown_rate_over_tokens)},
    {label:'NLLB 目标(中文)', color:'#1d4ed8', values:DS.map(ds => tgtStats(ds,'nllb').unknown_rate_over_tokens)},
    {label:'HY-MT2 源', color:'#dc2626', values:DS.map(ds => srcStats(ds,'hy_mt2').unknown_rate_over_tokens)},
    {label:'HY-MT2 目标', color:'#991b1b', values:DS.map(ds => tgtStats(ds,'hy_mt2').unknown_rate_over_tokens)}
  ];
  groupedBar(host, groups, series, {yTitle:'unknown / token', decimals:4, height:350});
  secU.appendChild(h('div', {class:'note warn', html:
    '<b>HY-MT2 的源侧与目标侧 unknown 均为 0</b>（本页图中该两条为 0 高度）。NLLB 的中文侧 unknown 明显高于藏文侧，说明缺口在<b>中文佛典长尾字</b>，而不是藏文编码。'}));

  secU.appendChild(h('h3', {}, ['6.2 未知 token 明细']));
  (function(){
    const rows = [];
    DS.forEach(ds => {
      ['nllb','hy_mt2'].forEach(k => {
        ['source','target'].forEach(side => {
          const s = F.datasets[ds].models[k][side];
          rows.push([DSL[ds], k === 'nllb' ? 'NLLB-200' : 'HY-MT2-1.8B', side === 'source' ? '藏文源' : '中文目标',
            s.unknown_token_total.toLocaleString(), s.examples_with_unknown_token.toLocaleString(),
            pct(s.unknown_rate_over_tokens), s.over_256_tokens, s.over_512_tokens]);
        });
      });
    });
    secU.appendChild(table(['数据集','Tokenizer','侧','unknown 总数','涉及样本数','unknown 率','>256','>512'], rows));
  })();

  secU.appendChild(h('h3', {}, ['6.3 NLLB 中文长尾字符 Top 30']));
  const spanTabs = h('div', {class:'tabs'});
  const spanBox = h('div');
  let spanDs = DS[0];
  function renderSpans(){
    spanBox.innerHTML = '';
    const src = A.datasets[spanDs].target;
    const spans = src.top_unknown_spans;
    const rows = spans.map((s, i) => {
      const chars = (s.characters || []).map(c => c.character + ' (' + c.codepoint + ')').join(' ');
      return [i + 1, s.text, s.count.toLocaleString(), chars];
    });
    spanBox.appendChild(h('p', {class:'lede', html:
      '数据集 <b>' + DSL[spanDs] + '</b>（目标标签 <code>' + A.datasets[spanDs].target_language + '</code>）：' +
      '未知 token 合计 <b>' + src.unknown_token_total.toLocaleString() + '</b>，出现在 <b>' +
      src.examples_with_unknown_token.toLocaleString() + '/' + src.count.toLocaleString() + '</b> 条样本中，' +
      '共 <b>' + spans.length + '</b> 种高频形式列于下表。'}));
    spanBox.appendChild(table(['#','未知片段','出现次数','字符与码位'], rows));
    const colors = ['#2563eb','#3b82f6','#60a5fa','#93c5fd','#bfdbfe'];
    const bars = h('div');
    hBar(bars, spans.slice(0, 15).map((s,i) => ({label:s.text, value:s.count, color:colors[i % colors.length]})), {decimals:0, padL:90});
    spanBox.appendChild(h('h4', {}, ['出现次数最高的 15 个片段']));
    spanBox.appendChild(bars);
  }
  DS.forEach(ds => {
    const b = h('button', {class: ds === spanDs ? 'on' : '', onclick: () => {
      spanDs = ds;
      Array.from(spanTabs.children).forEach(c => c.className = '');
      b.className = 'on';
      renderSpans();
    }}, [DSL[ds]]);
    spanTabs.appendChild(b);
  });
  secU.appendChild(spanTabs);
  secU.appendChild(spanBox);
  renderSpans();

  secU.appendChild(h('h3', {}, ['6.4 NLLB 藏文源侧长尾字符']));
  (function(){
    const rows = [];
    DS.forEach(ds => A.datasets[ds].source.top_unknown_spans.slice(0, 12).forEach((s, i) => {
      if (i === 0) rows.push([DSL[ds], A.datasets[ds].source.unknown_token_total.toLocaleString(), s.text, s.count, (s.characters||[]).map(c=>c.codepoint).join(' ')]);
    }));
    secU.appendChild(table(['数据集','源侧 unknown 总数','最高频片段','次数','码位'], rows));
    secU.appendChild(h('div', {class:'note', html:
      '藏文源侧 unknown 极少，且集中在少量罕见组合/附加符号。' +
      '<b>结论：编码缺口主要在中文佛典长尾，不应通过修改 NLLB 藏文词表来解决。</b>'}));
  })();
})();

/* -------------------------------------------------------- truncation */
addSection('trunc', '七、截断风险汇总（&gt;256 / &gt;512）',
  '把各配置下超过 256 与超过 512 token 的样本数并排列表，用于决定微调时的 max_length。',
  [
    (function(){
      const rows = [];
      DS.forEach(ds => CFG.forEach(c => {
        const s = srcStats(ds, c.key);
        rows.push([DSL[ds], c.label, s.over_256_tokens, s.over_512_tokens,
          s.over_256_tokens > 0 ? '<span class="pill mid">有</span>' : '<span class="pill ok">无</span>',
          s.length_tokens.max]);
      }));
      DS.forEach(ds => ['nllb','hy_mt2'].forEach(k => {
        const t = tgtStats(ds,k);
        rows.push([DSL[ds], (k==='nllb'?'NLLB-200':'HY-MT2-1.8B') + ' 目标侧', t.over_256_tokens, t.over_512_tokens,
          t.over_256_tokens > 0 ? '<span class="pill mid">有</span>' : '<span class="pill ok">无</span>', t.length_tokens.max]);
      }));
      return table(['数据集','配置','>256 条数','>512 条数','风险','最大 token'], rows);
    })()
  ]);

/* ----------------------------------------------------- normalization */
addSection('norm', '八、Unicode 规范化（NFC / NFD）影响',
  '原始藏文与规范化版本都要留档；这里记录规范化后编码发生变化的样本数以及规范化后的长度统计。',
  [
    (function(){
      const rows = [];
      DS.forEach(ds => {
        const n = F.datasets[ds].normalization_source_nllb;
        ['NFC','NFD'].forEach(form => {
          const blk = n[form];
          rows.push([DSL[ds], form, blk.changed_examples, blk.token_length.count.toLocaleString(),
            blk.token_length.min, fmt(blk.token_length.mean,2), blk.token_length.median,
            blk.token_length.p90, blk.token_length.p95, blk.token_length.max]);
        });
      });
      return table(['数据集','规范化','发生变化样本数','样本数','min','mean','median','p90','p95','max'], rows);
    })(),
    h('div', {class:'note', html:
      '<b>观察：</b>FLORES dev 有 5 条在 NFC 下变化、6 条在 NFD 下变化；FLORES devtest 各有 5 条。' +
      'MITRA validation 与 10k 训练抽样中<b>没有</b>任何样本因规范化而改变。' +
      '规范化没有造成长度上限问题，但 FLORES 的少量变化必须在预处理中留痕，<u>不得覆盖原始文本</u>。'})
  ]);

/* --------------------------------------------------------------- BPE */
addSection('bpe', '九、CS336 自训 BPE 的词表耗竭',
  '命令请求了 8k 与 16k 词表，但在当前 10k 条藏文源文本与 assignment1 预分词规则下，可合并的 byte pair 提前耗尽，导致两个请求得到完全相同的词表。',
  [
    h('h3', {}, ['9.1 请求词表 vs 实际有效词表']),
    (function(){
      const host = h('div');
      const variantLabel = {raw:'CS336 Raw', syllable_aware:'CS336 Tsheg', mark_aware:'CS336 Mark-aware'};
      const rows = [];
      P.bpe.filter(b => b.vocab_entries).forEach(b => {
        rows.push({label:variantLabel[b.variant] + ' · 请求 ' + b.requested.toLocaleString(),
                   value:b.vocab_entries, color: b.variant === 'mark_aware' ? '#7c3aed' : (b.variant === 'raw' ? '#059669' : '#d97706')});
      });
      hBar(host, rows, {decimals:0, padL:250});
      return host;
    })(),
    h('h3', {}, ['9.2 全部 BPE 产物明细']),
    (function(){
      const rows = P.bpe.map(b => {
        const variantLabel = {raw:'Raw', syllable_aware:'Tsheg-boundary', mark_aware:'Mark-aware'}[b.variant];
        const eff = b.vocab_entries === null ? '<span class="pill no">无产物</span>' : b.vocab_entries.toLocaleString();
        const reach = (b.vocab_entries !== null && b.vocab_entries >= b.requested) ? '<span class="pill ok">达到</span>' : '<span class="pill no">未达到</span>';
        return [variantLabel, b.requested.toLocaleString(), eff, b.merges === null ? '—' : b.merges.toLocaleString(), reach];
      });
      return table(['CS336 变体','请求词表','实际词表条目','实际 merge 数','是否达标'], rows);
    })(),
    h('div', {class:'note warn', html:
      '<b>这不是训练失败，而是需要记录的实验限制。</b>藏文源文本可合并的 byte pair 类型有限：' +
      'Raw 实际约 1,283、Tsheg-boundary 约 1,397、Mark-aware 约 3,805，因此 8k/16k 两个请求结果完全相同。' +
      'Mark-aware 能达到更大词表，正因为它没有在基字与组合记号之间提前切断。<br>' +
      '后续若要研究真正的 4k/8k/16k 词表缩放，应扩大藏文训练样本，或改用“源文 + 中文目标”共享语料。'}),
    h('div', {class:'note', html:
      '<b>产物完整性问题：</b><code>outputs/cs336-bpe-fertility/raw/vocab_4096/</code> 目录存在但<b>没有任何文件</b>（无 <code>vocab.json</code>、无 <code>merges.json</code>），' +
      '该档位实验未留下产物，本页已将其标记为“无产物”。如需 4k 对比需重跑。'})
  ]);

/* ------------------------------------------------- consistency check */
addSection('cross', '十、两份报告的一致性核查',
  'NLLB 同时出现在两份报告中，但抽样规模与统计口径不同。这一节把两者并排，避免误读。',
  [
    (function(){
      const rows = [];
      DS.forEach(ds => {
        const a = A.datasets[ds];
        const fl = F.datasets[ds];
        rows.push([DSL[ds], 'NLLB tokenizer 审计<br>(n=' + a.pairs.toLocaleString() + ')',
          a.source.mean, a.source.max, a.source.unknown_token_total, a.target.mean, a.target.max, a.target.unknown_token_total]);
        rows.push([DSL[ds], 'fertility 审计<br>(n=' + fl.pairs.toLocaleString() + ')',
          fmt(fl.models.nllb.source.length_tokens.mean,4), fl.models.nllb.source.length_tokens.max,
          fl.models.nllb.source.unknown_token_total,
          fmt(fl.models.nllb.target.length_tokens.mean,4), fl.models.nllb.target.length_tokens.max,
          fl.models.nllb.target.unknown_token_total]);
      });
      const wrap = table(['数据集','报告（抽样）','源 mean','源 max','源 unknown','目标 mean','目标 max','目标 unknown'], rows);
      return wrap;
    })(),
    h('div', {class:'note', html:
      '<b>可安全比对的部分：</b>MITRA validation 两侧抽样规模一致（' + F.datasets.mitra_validation.pairs.toLocaleString() + ' 条），' +
      'unknown 总数也完全一致（' + F.datasets.mitra_validation.models.nllb.target.unknown_token_total.toLocaleString() + '），两份报告互相印证。' +
      'FLORES dev / devtest 同样规模一致。<br>' +
      '<b>不可比对的部分：</b>MITRA 训练抽样规模不同（' + A.datasets.mitra_train_sample.pairs.toLocaleString() + ' vs ' + F.train_sample_size.toLocaleString() + ' 条），' +
      'unknown 总数（' + A.datasets.mitra_train_sample.target.unknown_token_total.toLocaleString() + ' vs ' +
      F.datasets.mitra_train_sample.models.nllb.target.unknown_token_total.toLocaleString() + '）与长度分布都随抽样规模变化，不能直接对比。<br>' +
      '<b>长度口径规则：</b>凡抽样规模相同的分组，tokenizer 审计的长度 = fertility 审计的长度 + 2（特殊 token），这是唯一的口径差，已在第二节核实。'})
  ]);

/* ---------------------------------------------------------- conclusions */
addSection('conclusion', '十一、结论与后续决策',
  '以下决策来自 <code>reports/fertility_audit_interpretation.md</code> 与 <code>reports/nllb_tokenizer_interpretation.md</code>。',
  [
    h('ol', {}, [
      h('li', {html:'<b>NLLB 继续使用原生 tokenizer</b>，不人为插入音节空格——插入空格会改变预训练模型熟悉的分布。'}),
      h('li', {html:'<b>256 token 初始长度是安全的</b>：当前所有数据集与配置均未超过 512，NLLB 侧连超过 256 的样本也是 0。'}),
      h('li', {html:'<b>HY-MT2 需要更长的序列预算</b>：其藏文 token 数约为 NLLB 的 3.5 倍，不能沿用 NLLB 的 batch 配置。'}),
      h('li', {html:'<b>CS336 的价值在于解释 tokenizer 与预分词影响</b>：Mark-aware 的 fertility 改善约为 27%，但这是输入统计改善，不是翻译质量证明。'}),
      h('li', {html:'<b>不因 fertility 结果修改 NLLB 预训练词表</b>：藏文端 unknown 极少；中文端缺口是佛典长尾字，属于目标侧问题，且扩表需要新增 token、初始化嵌入并重新微调。'}),
      h('li', {html:'<b>暂不删除 MITRA 中的罕见佛典汉字</b>，它们可能正是领域风格的重要组成部分。后续应做“原始繁体目标 vs 繁简/异体规范化目标”的对照实验。'}),
      h('li', {html:'<b>冻结统计口径后再进入评测</b>：下一步是按 <code>docs/cpu-experiment-plan.md</code> 阶段 7 冻结 chrF++/BLEU 评测协议，而不是继续扩大 tokenizer 实验规模。'})
    ])
  ]);

/* -------------------------------------------------------------- appendix */
addSection('appendix', '十二、附录：完整原始结果',
  '以下内嵌两份报告的完整 JSON（与磁盘文件一致），便于核对本页任一数值。',
  [
    h('details', {}, [
      h('summary', {}, ['reports/fertility_audit.json（统一 fertility 审计，完整）']),
      h('pre', {}, [JSON.stringify(F, null, 2)])
    ]),
    h('details', {}, [
      h('summary', {}, ['reports/nllb_tokenizer_audit.json（NLLB tokenizer 审计，完整）']),
      h('pre', {}, [JSON.stringify(A, null, 2)])
    ]),
    h('details', {}, [
      h('summary', {}, ['CS336 BPE 词表产物清单']),
      h('pre', {}, [JSON.stringify(P.bpe, null, 2)])
    ])
  ]);

/* ------------------------------------------------------------------- nav */
nav.innerHTML = SECTIONS.map(([id, title]) => '<a href="#' + id + '">' + title.replace(/^[一二三四五六七八九十]+、/, '') + '</a>').join('');
</script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("/*__DATA__*/", encoded)
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
