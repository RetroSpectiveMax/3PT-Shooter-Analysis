// Buckets-style shot explorer. Algorithms mirror src/shot_chart.py: keep constants in sync.
"use strict";

const NX = 50, NY = 52, X0 = 25, Y0 = 5;
const SIGMA = 2.0, SHRINK_K = 3.0, DIST_MAX = 30, THREE = 23.75;
const BREAKS = [-0.09, -0.03, 0.03, 0.09];
const COLORS = ["#4659a7", "#7fa0d0", "#f1e3a8", "#f4a45b", "#d1344a"];
const LEGEND_LABELS = ["< -9%", "-9 to -3%", "-3 to +3%", "+3 to +9%", "> +9%"];
const RED = "#c8102e", TEAL = "#17a2b8", GREY = "#777", BLUE = "#1f4e9c", LEFT_C = "#2ca02c", RIGHT_C = "#9467bd";
const POS_CODE = { League: -1, Guards: 0, Forwards: 1, Bigs: 2 };

const CELL_X = new Float64Array(NX * NY), CELL_Y = new Float64Array(NX * NY), CELL_D = new Float64Array(NX * NY);
for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
  const k = j * NX + i;
  CELL_X[k] = i - X0 + 0.5; CELL_Y[k] = j - Y0 + 0.5; CELL_D[k] = Math.hypot(CELL_X[k], CELL_Y[k]);
}

const $ = (id) => document.getElementById(id);
const state = { meta: null, data: {}, season: null };

// ------------------------------------------------------------------ data
async function loadSeason(season) {
  if (state.data[season]) return state.data[season];
  const d = await (await fetch(`data/${season}.json`, { cache: "no-cache" })).json();
  // index cells by player: cells are sorted by player id
  const idx = new Map();
  const pid = d.cells.pid;
  for (let i = 0; i < pid.length; i++) {
    const e = idx.get(pid[i]);
    if (e) e[1] = i + 1; else idx.set(pid[i], [i, i + 1]);
  }
  d.idx = idx;
  d.byId = new Map(d.players.id.map((id, i) => [id, i]));
  state.data[season] = d;
  return d;
}

function gridFromCells(c, lo, hi, filt) {
  const att = new Float64Array(NX * NY), made = new Float64Array(NX * NY);
  let a3 = 0, m3 = 0;
  for (let i = lo; i < hi; i++) {
    const hw = c.hw[i];
    if (filt.venue === "Home" && hw < 2) continue;
    if (filt.venue === "Away" && hw >= 2) continue;
    if (filt.result === "Wins" && hw % 2 === 0) continue;
    if (filt.result === "Losses" && hw % 2 === 1) continue;
    if (filt.opp !== null && c.opp[i] !== filt.opp) continue;
    const k = c.cy[i] * NX + c.cx[i];
    att[k] += c.a[i]; made[k] += c.m[i];
    if (c.t[i]) { a3 += c.a[i]; m3 += c.m[i]; }
  }
  return [att, made, a3, m3];
}

function baselineGrid(b, pos) {
  const att = new Float64Array(NX * NY), made = new Float64Array(NX * NY);
  for (let i = 0; i < b.pos.length; i++) {
    if (b.pos[i] !== pos) continue;
    const k = b.cy[i] * NX + b.cx[i];
    att[k] += b.a[i]; made[k] += b.m[i];
  }
  return [att, made];
}

// ------------------------------------------------------------------ math
function kernel(sigma) {
  const r = Math.ceil(3 * sigma), k = [];
  let s = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * (i / sigma) ** 2); k.push(v); s += v; }
  return k.map((v) => v / s);
}
function conv1(v, k) {  // zero padded, same length
  const n = v.length, r = (k.length - 1) / 2, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let t = -r; t <= r; t++) { const j = i + t; if (j >= 0 && j < n) s += v[j] * k[t + r]; }
    out[i] = s;
  }
  return out;
}
function smooth2d(g, sigma = SIGMA) {
  const k = kernel(sigma), tmp = new Float64Array(g.length), out = new Float64Array(g.length);
  const row = new Float64Array(NX), col = new Float64Array(NY);
  for (let j = 0; j < NY; j++) {                      // along x
    for (let i = 0; i < NX; i++) row[i] = g[j * NX + i];
    const r = conv1(row, k); for (let i = 0; i < NX; i++) tmp[j * NX + i] = r[i];
  }
  for (let i = 0; i < NX; i++) {                      // along y
    for (let j = 0; j < NY; j++) col[j] = tmp[j * NX + i];
    const r = conv1(col, k); for (let j = 0; j < NY; j++) out[j * NX + i] = r[j];
  }
  return out;
}
const sum = (a) => a.reduce((s, v) => s + v, 0);

function leagueFg(la, lm) {
  const overall = sum(lm) / Math.max(sum(la), 1);
  const sa = smooth2d(la), sm = smooth2d(lm), out = new Float64Array(la.length);
  for (let k = 0; k < out.length; k++) out[k] = sa[k] > 1e-6 ? (sm[k] + overall) / (sa[k] + 1) : overall;
  return out;
}
const colorFor = (d) => COLORS[BREAKS.filter((b) => d >= b).length];

function courtMarks(att, made, la, lm, mode, o) {
  const lg = leagueFg(la, lm), n = att.length;
  let a, m, fg = new Float64Array(n), keep = new Array(n).fill(false);
  if (mode === "Raw") {
    a = att; m = made;
    for (let k = 0; k < n; k++) { fg[k] = a[k] > 0 ? m[k] / a[k] : 0; keep[k] = a[k] >= 1; }
  } else {
    a = smooth2d(att).map((v) => v * 4); m = smooth2d(made).map((v) => v * 4);
    for (let k = 0; k < n; k++) {
      fg[k] = (m[k] + SHRINK_K * 4 * lg[k]) / (a[k] + SHRINK_K * 4);
      const j = Math.floor(k / NX), i = k % NX;
      keep[k] = j % 2 === 0 && i % 2 === 0 && a[k] >= 0.25;
    }
  }
  let peak = 0;
  for (let k = 0; k < n; k++) if (CELL_Y[k] <= 42 && a[k] > peak) peak = a[k];
  const out = [];
  for (let k = 0; k < n; k++) {
    if (!keep[k] || CELL_Y[k] > 42) continue;
    if (a[k] < o.minShots || a[k] > o.maxShots) continue;
    if (fg[k] < o.fgLo || fg[k] > o.fgHi) continue;
    const diff = fg[k] - lg[k];
    out.push({ x: CELL_X[k], y: CELL_Y[k], att: a[k], fg: fg[k], diff, size: Math.sqrt(a[k] / peak), color: colorFor(diff) });
  }
  return out;
}

const ZONE_DEFS = [
  ["Left corner 3", (x, y, d) => x <= -22 && y <= 8.94],
  ["Right corner 3", (x, y, d) => x >= 22 && y <= 8.94],
  ["Restricted area", (x, y, d) => d < 4],
  ["Paint (non-RA)", (x, y, d) => d >= 4 && Math.abs(x) <= 8 && y <= 13.75 && d < THREE],
  ["Mid-range left", (x, y, d) => d < THREE && x < -8 && y < 22],
  ["Mid-range right", (x, y, d) => d < THREE && x > 8 && y < 22],
  ["Mid-range top", (x, y, d) => d < THREE],
  ["Above break 3 left", (x, y, d) => d >= THREE && x < -8],
  ["Above break 3 center", (x, y, d) => d >= THREE && Math.abs(x) <= 8],
  ["Above break 3 right", (x, y, d) => d >= THREE && x > 8],
];
const ZONE_MASKS = (() => {
  const taken = new Array(NX * NY).fill(false);
  return ZONE_DEFS.map(([name, fn]) => {
    const idx = [];
    for (let k = 0; k < taken.length; k++) {
      if (!taken[k] && CELL_Y[k] <= 42 && fn(CELL_X[k], CELL_Y[k], CELL_D[k])) { taken[k] = true; idx.push(k); }
    }
    return { name, idx };
  });
})();

function zoneTable(att, made, la, lm) {
  let total = 0;
  for (let k = 0; k < att.length; k++) if (CELL_Y[k] <= 42) total += att[k];
  return ZONE_MASKS.map(({ name, idx }) => {
    let a = 0, m = 0, l = 0, lk = 0, wx = 0, wy = 0, w = 0;
    for (const k of idx) { a += att[k]; m += made[k]; l += la[k]; lk += lm[k]; wx += CELL_X[k] * (la[k] + 1e-9); wy += CELL_Y[k] * (la[k] + 1e-9); w += la[k] + 1e-9; }
    const fg = a ? m / a : NaN, lfg = l ? lk / l : NaN;
    return { name, att: a, fgm: m, fg, lfg, diff: a ? fg - lfg : NaN, freq: total ? a / total : 0, cx: wx / w, cy: wy / w };
  });
}

function distProfile(att, made) {
  const z = () => new Float64Array(DIST_MAX + 1);
  const p = { aL: z(), aR: z(), mL: z(), mR: z() };
  for (let k = 0; k < att.length; k++) {
    if (CELL_Y[k] > 42) continue;
    const b = Math.floor(CELL_D[k]);
    if (b > DIST_MAX) continue;
    if (CELL_X[k] < 0) { p.aL[b] += att[k]; p.mL[b] += made[k]; } else { p.aR[b] += att[k]; p.mR[b] += made[k]; }
  }
  p.a = p.aL.map((v, i) => v + p.aR[i]);
  p.m = p.mL.map((v, i) => v + p.mR[i]);
  return p;
}
const smooth1d = (v, s) => conv1(v, kernel(s));
function freqCurve(a) { const t = sum(a); return Array.from(smooth1d(a, 0.8), (v) => (t ? (v / t) * 100 : 0)); }
function fgCurve(a, m, prior) {
  const sa = smooth1d(a, 1.0), sm = smooth1d(m, 1.0);
  return Array.from(sa, (v, i) => (prior ? ((sm[i] + (3 * prior[i]) / 100) / (v + 3)) * 100 : v > 0 ? (sm[i] / v) * 100 : NaN));
}

// ------------------------------------------------------------------ drawing helpers
const tip = d3.select("body").append("div").attr("class", "tip");
const showTip = (ev, html) => tip.style("display", "block").html(html).style("left", ev.clientX + 12 + "px").style("top", ev.clientY + 12 + "px");
const hideTip = () => tip.style("display", "none");

function addExport(container, svg, name) {
  d3.select(container).insert("a", ":first-child").attr("class", "export").text("Export as PNG").on("click", () => exportPNG(svg.node(), name));
}
function exportPNG(svgNode, name) {
  const clone = svgNode.cloneNode(true);
  const w = +svgNode.getAttribute("width"), h = +svgNode.getAttribute("height");
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml" }));
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas"); c.width = w * 2; c.height = h * 2;
    const g = c.getContext("2d"); g.fillStyle = "#fff"; g.fillRect(0, 0, c.width, c.height); g.scale(2, 2); g.drawImage(img, 0, 0, w, h);
    const a = document.createElement("a"); a.download = name + ".png"; a.href = c.toDataURL("image/png"); a.click();
    URL.revokeObjectURL(url);
  };
  img.src = url;
}
const txt = (g, x, y, s, o = {}) => g.append("text").attr("x", x).attr("y", y).attr("font-family", "Helvetica, Arial, sans-serif")
  .attr("font-size", o.size || 11).attr("fill", o.fill || "#444").attr("text-anchor", o.anchor || "start")
  .attr("font-weight", o.bold ? "bold" : null).attr("transform", o.rotate ? `rotate(${o.rotate} ${x} ${y})` : null).text(s);

// ------------------------------------------------------------------ court
const CW = 520, CH = 500, M = 10, PX = (CW - 2 * M) / 50;
const CX = (x) => M + (x + 25) * PX, CY = (y) => M + (42 - y) * PX;

function drawCourtLines(g) {
  const ln = (x1, y1, x2, y2) => g.append("line").attr("x1", CX(x1)).attr("y1", CY(y1)).attr("x2", CX(x2)).attr("y2", CY(y2));
  const arc = (cx, cy, r, a0, a1) => {
    const pts = d3.range(61).map((i) => { const t = ((a0 + ((a1 - a0) * i) / 60) * Math.PI) / 180; return [CX(cx + r * Math.cos(t)), CY(cy + r * Math.sin(t))]; });
    g.append("path").attr("d", d3.line()(pts));
  };
  const rect = (x0, y0, x1, y1) => g.append("rect").attr("x", CX(x0)).attr("y", CY(y1)).attr("width", (x1 - x0) * PX).attr("height", (y1 - y0) * PX);
  rect(-25, -5.25, 25, 42); rect(-8, -5.25, 8, 13.75); rect(-6, -5.25, 6, 13.75);
  arc(0, 13.75, 6, 0, 180); arc(0, 0, 4, 0, 180);
  const a = (Math.asin(8.94 / THREE) * 180) / Math.PI;
  arc(0, 0, THREE, a, 180 - a);
  ln(-22, -5.25, -22, 8.94); ln(22, -5.25, 22, 8.94); ln(-3, -1.25, 3, -1.25);
  g.append("circle").attr("cx", CX(0)).attr("cy", CY(0)).attr("r", 0.75 * PX);
  arc(0, 41.75, 6, 180, 360);
  g.selectAll("line,path,rect,circle").attr("fill", "none").attr("stroke", "#666").attr("stroke-width", 1.3);
}

function drawCourt(container, title, marks, zones, mode, opt) {
  const svg = d3.select(container).append("svg").attr("width", CW).attr("height", CH + (opt.legend ? 28 : 0));
  svg.append("rect").attr("width", CW).attr("height", CH + 28).attr("fill", "#fff");
  drawCourtLines(svg.append("g"));
  txt(svg, 12, 18, title, { size: 13, bold: true, fill: "#222" });
  const g = svg.append("g");
  if (mode === "Zones") {
    const maxF = Math.max(...zones.map((z) => z.freq), 1e-9);
    for (const z of zones.filter((z) => z.att > 0)) {
      const s = (opt.encode ? 18 + 46 * Math.sqrt(z.freq / maxF) : 38) * (PX / 10.5);
      g.append("rect").attr("x", CX(z.cx) - s / 2).attr("y", CY(z.cy) - s / 2).attr("width", s).attr("height", s)
        .attr("fill", colorFor(z.diff)).attr("stroke", "#fff")
        .on("mousemove", (ev) => showTip(ev, `<b>${z.name}</b><br>${z.fgm}/${z.att} FG (${(z.fg * 100).toFixed(1)}%)<br>League ${(z.lfg * 100).toFixed(1)}%<br>${(z.freq * 100).toFixed(1)}% of shots`)).on("mouseleave", hideTip);
      txt(g, CX(z.cx), CY(z.cy) + 4, (z.fg * 100).toFixed(0) + "%", { anchor: "middle", fill: "#000" });
    }
  } else {
    const smoothed = mode === "Smoothed";
    g.selectAll("rect").data(marks).join("rect")
      .attr("width", (d) => (smoothed ? (opt.encode ? 6 + 16 * d.size : 15) : opt.encode ? 3 + 9 * d.size : 8) * (PX / 10.5))
      .attr("height", function () { return d3.select(this).attr("width"); })
      .attr("x", function (d) { return CX(d.x) - d3.select(this).attr("width") / 2; })
      .attr("y", function (d) { return CY(d.y) - d3.select(this).attr("height") / 2; })
      .attr("fill", (d) => d.color)
      .on("mousemove", (ev, d) => showTip(ev, `${d.att.toFixed(1)} shots<br>FG ${(d.fg * 100).toFixed(1)}%<br>vs league ${d.diff >= 0 ? "+" : ""}${(d.diff * 100).toFixed(1)}%`))
      .on("mouseleave", hideTip);
  }
  if (opt.legend) {
    txt(svg, 12, CH + 18, "FG% vs league:", { size: 11 });
    COLORS.forEach((c, i) => {
      const x = 100 + i * 82;
      svg.append("rect").attr("x", x).attr("y", CH + 8).attr("width", 11).attr("height", 11).attr("fill", c);
      txt(svg, x + 15, CH + 18, LEGEND_LABELS[i]);
    });
  }
  addExport(container, svg, title.replace(/\s+/g, "_"));
}

// ------------------------------------------------------------------ line charts
function lineChart(container, title, series, yLabel, opt) {
  const W = 500, H = 320, m = { l: 48, r: 14, t: 38, b: 40 };
  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  svg.append("rect").attr("width", W).attr("height", H).attr("fill", "#fff");
  const x = d3.scaleLinear([0, DIST_MAX], [m.l, W - m.r]);
  const ymax = opt.ymax || Math.ceil(d3.max(series, (s) => d3.max(s.v)) / 2) * 2 + 1;
  const y = d3.scaleLinear([0, ymax], [H - m.b, m.t]);
  const titleW = txt(svg, m.l, 16, title, { size: 13, bold: true, fill: "#222" }).node().getComputedTextLength();
  const axis = svg.append("g");
  y.ticks(5).forEach((t) => {
    axis.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", "#eee");
    txt(axis, m.l - 6, y(t) + 4, t + (opt.pct ? "%" : ""), { anchor: "end" });
  });
  x.ticks(6).forEach((t) => { txt(axis, x(t), H - m.b + 15, t, { anchor: "middle" }); });
  txt(svg, (m.l + W - m.r) / 2, H - 6, "Distance (ft)", { anchor: "middle" });
  txt(svg, 12, (m.t + H - m.b) / 2, yLabel, { anchor: "middle", rotate: -90 });
  svg.append("line").attr("x1", x(THREE)).attr("x2", x(THREE)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", "#999").attr("stroke-dasharray", "3,3");
  txt(svg, x(THREE) + 3, m.t + 10, "3pt", { fill: "#888" });
  series.forEach((s) => {
    const line = d3.line().defined((d) => !isNaN(d)).x((d, i) => x(i)).y((d) => y(Math.min(d, ymax)));
    svg.append("path").attr("d", line(s.v)).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2);
  });
  let lx = m.l + titleW + 18;
  series.forEach((s) => {
    svg.append("line").attr("x1", lx).attr("x2", lx + 14).attr("y1", 12).attr("y2", 12).attr("stroke", s.color).attr("stroke-width", 2);
    const t = txt(svg, lx + 18, 16, s.name, { size: 11 }); lx += 24 + t.node().getComputedTextLength() + 6;
  });
  // hover
  const hov = svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b).attr("fill", "transparent");
  hov.on("mousemove", (ev) => {
    const i = Math.max(0, Math.min(DIST_MAX, Math.round(x.invert(d3.pointer(ev)[0]))));
    showTip(ev, `<b>${i} ft</b><br>` + series.map((s) => `${s.name}: ${isNaN(s.v[i]) ? "-" : s.v[i].toFixed(1)}${opt.pct ? "%" : ""}`).join("<br>"));
  }).on("mouseleave", hideTip);
  addExport(container, svg, title.replace(/\W+/g, "_"));
}

function lrChart(container, title, p, kind) {
  const W = 500, H = 400, m = { l: 48, r: 14, t: 56, b: 42 };
  const svg = d3.select(container).append("svg").attr("width", W).attr("height", H);
  svg.append("rect").attr("width", W).attr("height", H).attr("fill", "#fff");
  const d = d3.range(DIST_MAX + 1);
  let lv, rv, diff, xdom, ticks, fmt, lo, hi;
  if (kind === "freq") {
    lv = Array.from(p.aL); rv = Array.from(p.aR);
    const ls = smooth1d(p.aL, 0.8), rs = smooth1d(p.aR, 0.8), t = sum(ls) + sum(rs) || 1;
    diff = Array.from(rs, (v, i) => v - ls[i]); lo = sum(ls) / t; hi = sum(rs) / t;
    const mx = Math.max(10, Math.ceil(Math.max(...lv, ...rv) / 20) * 20);
    xdom = [-mx, mx]; ticks = [-mx, -mx / 2, 0, mx / 2, mx]; fmt = (v) => Math.abs(v);
  } else {
    const f = (a, mm) => Array.from(smooth1d(a, 1.0), (v, i) => { const s = smooth1d(mm, 1.0)[i]; return v > 0 ? (s / v) * 100 : 0; });
    lv = f(p.aL, p.mL); rv = f(p.aR, p.mR); diff = rv.map((v, i) => v - lv[i]);
    const la = sum(p.aL), ra = sum(p.aR); lo = la ? sum(p.mL) / la : 0; hi = ra ? sum(p.mR) / ra : 0;
    xdom = [-100, 100]; ticks = [-100, -50, 0, 50, 100]; fmt = (v) => Math.abs(v) + "%";
  }
  const x = d3.scaleLinear(xdom, [m.l, W - m.r]), y = d3.scaleLinear([0, DIST_MAX], [H - m.b, m.t]);
  txt(svg, m.l, 16, title, { size: 13, bold: true, fill: "#222" });
  txt(svg, m.l, 32, `Overall: left ${(lo * 100).toFixed(0)}% | right ${(hi * 100).toFixed(0)}%`, { size: 11, fill: "#666" });
  const items = [["Left", LEFT_C], ["Right", RIGHT_C], ["Difference (R - L)", "#333"]];
  let lx = m.l;
  items.forEach(([n, c]) => {
    svg.append("rect").attr("x", lx).attr("y", 38).attr("width", 10).attr("height", 10).attr("fill", c).attr("opacity", n.startsWith("Diff") ? 1 : 0.5);
    const t = txt(svg, lx + 14, 47, n, { size: 11 }); lx += 28 + t.node().getComputedTextLength() + 4;
  });
  ticks.forEach((t) => {
    svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.t + 8).attr("y2", H - m.b).attr("stroke", t === 0 ? "#bbb" : "#eee");
    txt(svg, x(t), H - m.b + 15, fmt(t), { anchor: "middle" });
  });
  y.ticks(6).forEach((t) => txt(svg, m.l - 6, y(t) + 4, t, { anchor: "end" }));
  txt(svg, (m.l + W - m.r) / 2, H - 6, kind === "freq" ? "# of shots" : "Field goal %", { anchor: "middle" });
  txt(svg, 12, (m.t + H - m.b) / 2, "Distance (ft)", { anchor: "middle", rotate: -90 });
  const bh = Math.abs(y(0) - y(1));
  d.forEach((i) => {
    const v = [lv[i], rv[i]];
    svg.append("rect").attr("x", x(-v[0])).attr("y", y(i + 1)).attr("width", x(0) - x(-v[0])).attr("height", bh).attr("fill", LEFT_C).attr("opacity", 0.35);
    svg.append("rect").attr("x", x(0)).attr("y", y(i + 1)).attr("width", x(v[1]) - x(0)).attr("height", bh).attr("fill", RIGHT_C).attr("opacity", 0.35);
  });
  const line = d3.line().x((v) => x(Math.max(xdom[0], Math.min(xdom[1], v)))).y((v, i) => y(i + 0.5));
  svg.append("path").attr("d", line(diff)).attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 2);
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(THREE)).attr("y2", y(THREE)).attr("stroke", "#999").attr("stroke-dasharray", "3,3");
  txt(svg, m.l + 3, y(THREE) - 3, "3pt", { fill: "#888" });
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b).attr("fill", "transparent")
    .on("mousemove", (ev) => {
      const i = Math.max(0, Math.min(DIST_MAX, Math.floor(y.invert(d3.pointer(ev)[1]))));
      const u = kind === "freq" ? "" : "%";
      showTip(ev, `<b>${i} ft</b><br>Left: ${lv[i].toFixed(kind === "freq" ? 0 : 1)}${u}<br>Right: ${rv[i].toFixed(kind === "freq" ? 0 : 1)}${u}`);
    }).on("mouseleave", hideTip);
  addExport(container, svg, title.replace(/\W+/g, "_"));
}

// ------------------------------------------------------------------ app
function analysis(d, playerId, filt) {
  const [lo, hi] = d.idx.get(playerId) || [0, 0];
  const [att, made, a3, m3] = gridFromCells(d.cells, lo, hi, filt);
  const row = d.byId.get(playerId), P = d.players;
  const pos = P.pos[row];
  return {
    id: playerId, name: P.name[row], team: P.team[row], pos, height: P.height[row], pts: P.pts[row], gp: P.gp[row],
    att, made, a3, m3, lg: baselineGrid(d.base, -1), grp: baselineGrid(d.base, POS_CODE[pos]),
  };
}

function playerCard(container, a) {
  const fga = sum(a.att), fgm = sum(a.made);
  const t3a = a.a3, t3m = a.m3;
  const el = d3.select(container).append("div").attr("class", "card");
  el.append("img").attr("src", `https://cdn.nba.com/headshots/nba/latest/260x190/${a.id}.png`).attr("alt", a.name).on("error", function () { this.style.visibility = "hidden"; });
  el.append("h2").text(a.name);
  el.append("div").attr("class", "meta").text(`${a.team} | ${a.pos} | ${a.height}`);
  const rows = [["PTS/G", a.pts >= 0 ? a.pts.toFixed(1) : "-"], ["Games", a.gp], ["FGA", fga.toLocaleString()], ["FGM", fgm.toLocaleString()],
    ["FG%", fga ? ((fgm / fga) * 100).toFixed(1) + "%" : "-"], ["3PA", t3a], ["3PM", t3m], ["3P%", t3a ? ((t3m / t3a) * 100).toFixed(1) + "%" : "-"]];
  const tb = el.append("table");
  rows.forEach(([k, v]) => { const tr = tb.append("tr"); tr.append("td").text(k); tr.append("td").text(v); });
}

async function render() {
  const season = $("season").value;
  const d = await loadSeason(season);
  const compare = $("compare").checked;
  $("playerBWrap").hidden = !compare;
  const filt = { venue: $("venue").value, result: $("result").value, opp: $("opp").value === "All" ? null : state.meta.teams.indexOf($("opp").value) };
  const ids = [+$("playerA").value].concat(compare ? [+$("playerB").value] : []);
  const mode = document.querySelector("input[name=mode]:checked").value;
  const opt = { encode: $("encode").checked, legend: $("legend").checked };
  const nMin = +$("nMin").value, nMax = +$("nMax").value;
  const fgMin = +$("fgMin").value, fgMax = +$("fgMax").value;
  $("nLabel").textContent = `${nMin} - ${nMax >= 15 ? "15+" : nMax}`;
  $("fgLabel").textContent = `${fgMin}% - ${fgMax}%`;
  const o = { minShots: nMin, maxShots: nMax >= 15 ? Infinity : nMax, fgLo: fgMin / 100, fgHi: fgMax / 100 };

  ["courts", "freqChart", "fgChart", "lrCharts"].forEach((id) => { $(id).innerHTML = ""; });
  $("status").textContent = "";
  const A = ids.map((id) => analysis(d, id, filt));
  if (A.some((a) => sum(a.att) === 0)) { $("status").textContent = "No shots match the current filters for at least one player."; return; }

  A.forEach((a) => {
    const panel = d3.select("#courts").append("div").attr("class", "player-panel");
    playerCard(panel.node(), a);
    const wrap = panel.append("div").attr("class", "courtwrap").node();
    const marks = mode === "Zones" ? [] : courtMarks(a.att, a.made, ...a.lg, mode, o);
    const zones = mode === "Zones" ? zoneTable(a.att, a.made, ...a.lg) : null;
    drawCourt(wrap, `${a.name} ${season}`, marks, zones, mode, opt);
  });

  const profs = A.map((a) => distProfile(a.att, a.made));
  const lgp = distProfile(...A[0].lg);
  const lgFg = fgCurve(lgp.a, lgp.m);
  const series = (fn) => {
    const s = A.map((a, i) => ({ name: a.name, color: i === 0 ? RED : BLUE, v: fn(profs[i], false) }));
    s.push({ name: "League", color: TEAL, v: fn(lgp, true) });
    if (!compare) { const g = distProfile(...A[0].grp); s.push({ name: A[0].pos, color: GREY, v: fn(g, false) }); }
    return s;
  };
  lineChart($("freqChart"), "Shot frequency % by distance", series((p) => freqCurve(p.a)), "Frequency %", { pct: true });
  lineChart($("fgChart"), "Field goal % by distance", series((p, isLg) => fgCurve(p.a, p.m, isLg ? null : lgFg)), "Field goal %", { ymax: 100, pct: true });
  A.forEach((a, i) => {
    const row = d3.select("#lrCharts").append("div").style("display", "flex").style("gap", "20px").style("flex-wrap", "wrap");
    const nm = compare ? a.name + ": " : "";
    lrChart(row.append("figure").node(), nm + "Shot frequency: left vs. right", profs[i], "freq");
    lrChart(row.append("figure").node(), nm + "Field goal %: left vs. right", profs[i], "fg");
  });
}

function fillPlayers(d, selA, selB) {
  const P = d.players;
  const order = P.id.map((id, i) => i).sort((i, j) => P.fga[j] - P.fga[i]);
  for (const sel of [$("playerA"), $("playerB")]) {
    const keep = sel.value;
    sel.innerHTML = "";
    order.forEach((i) => sel.add(new Option(`${P.name[i]} (${P.team[i]})`, P.id[i])));
    sel.value = keep;
  }
  const exists = (v) => P.id.includes(+v);
  $("playerA").value = exists(selA) ? selA : P.id[order[0]];
  $("playerB").value = exists(selB) ? selB : P.id[order[1]];
}

async function init() {
  state.meta = await (await fetch("data/meta.json", { cache: "no-cache" })).json();
  const seasons = [...state.meta.seasons].reverse();
  seasons.forEach((s) => $("season").add(new Option(s, s)));
  ["All", ...[...new Set(state.meta.teams)].sort()].forEach((t) => $("opp").add(new Option(t, t)));
  const qs = new URLSearchParams(location.search);
  if (qs.get("season")) $("season").value = qs.get("season");
  const d = await loadSeason($("season").value);
  fillPlayers(d, qs.get("player") || 201935, qs.get("vs") || 201939);
  if (qs.get("compare")) $("compare").checked = true;
  if (qs.get("mode")) document.querySelector(`input[name=mode][value=${qs.get("mode")}]`).checked = true;
  const rerender = () => render().catch((e) => { $("status").textContent = "Error: " + e.message; console.error(e); });
  $("season").addEventListener("change", async () => {
    const dd = await loadSeason($("season").value);
    fillPlayers(dd, $("playerA").value, $("playerB").value);
    rerender();
  });
  document.querySelectorAll("#controls select:not(#season), #controls input").forEach((el) => el.addEventListener("input", rerender));
  rerender();
}
init();
