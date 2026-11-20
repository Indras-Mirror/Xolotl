#!/usr/bin/env python3
"""Loss graphs for Xolotl training runs.

usage: lossplot.py [RUNS_DIR=runs] [--png FILE] [--watch]

Every trainer writes RUNS_DIR/<run>/metrics.jsonl, one JSON object per line:
    {"step": 12, "loss": 1.84, "eval_loss": 1.91 (optional), "lr": 2e-4, "gnorm": 0.8, "tokens": 51200, "t": 812.4, ...}
and RUNS_DIR/<run>/config.json (whatever the trainer was started with).

This writes RUNS_DIR/index.html: one self-contained page (no network, no libraries) with every run overlaid -
train loss (raw + EMA-smoothed), eval loss, learning rate, gradient norm - against step or tokens, a run picker, and
a 15 s auto-refresh while a run is going. Open it in a browser: file://<RUNS_DIR>/index.html
--png FILE also renders a static PNG (matplotlib) for a quick look in a terminal session.
--watch regenerates the page every 15 s until interrupted.
"""
import json
import sys
import time
from pathlib import Path

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Xolotl training runs</title>
<meta http-equiv="refresh" content="15">
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--grid:#e6e3dc;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#9a988f;--grid:#2c2b27;--card:#1e1d1a}}
body{margin:0;padding:16px 20px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px} .sub{color:var(--mut);margin-bottom:12px}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:12px}
.runs label{margin-right:10px;white-space:nowrap} .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:10px}
.card h2{font-size:13px;margin:0 0 6px;color:var(--mut);font-weight:600} canvas{width:100%;height:260px}
table{border-collapse:collapse;margin-top:14px;font-size:13px} td,th{padding:3px 10px;border-bottom:1px solid var(--grid);text-align:left}
</style></head><body>
<h1>Xolotl training runs</h1><div class="sub" id="sub"></div>
<div class="bar"><span class="runs" id="runs"></span>
<label>x: <select id="xa"><option value="step">step</option><option value="tokens">tokens</option><option value="t">time (s)</option></select></label>
<label>smoothing <input id="sm" type="range" min="0" max="0.99" step="0.01" value="0.9"></label>
<label><input id="logy" type="checkbox"> log y</label></div>
<div class="grid">
<div class="card"><h2>train loss (faint: raw, solid: EMA) &middot; eval loss (dots)</h2><canvas id="c_loss"></canvas></div>
<div class="card"><h2>learning rate</h2><canvas id="c_lr"></canvas></div>
<div class="card"><h2>gradient norm</h2><canvas id="c_gn"></canvas></div>
<div class="card"><h2>tokens / s</h2><canvas id="c_tps"></canvas></div>
</div>
<table id="tab"></table>
<script>
const RUNS = __DATA__;
const COLORS=["#2a6fdb","#e4572e","#17a398","#a259c4","#f3a712","#4d4d4d","#d62a8b","#6a994e"];
const sel = JSON.parse(localStorage.getItem("xol_runs_sel")||"null") || Object.fromEntries(RUNS.map(r=>[r.name,true]));
let st={}; try{st=JSON.parse(localStorage.getItem("xol_runs_ui")||"{}")}catch(e){}
const $=id=>document.getElementById(id);
if(st.xa) $("xa").value=st.xa; if(st.sm!=null) $("sm").value=st.sm; if(st.logy) $("logy").checked=true;
$("sub").textContent = RUNS.length+" run(s) - page written "+new Date(__NOW__*1000).toLocaleString()+" (refreshes every 15 s)";
$("runs").innerHTML = RUNS.map((r,i)=>`<label><input type="checkbox" data-r="${r.name}" ${sel[r.name]!==false?"checked":""}><span class="sw" style="background:${COLORS[i%COLORS.length]}"></span>${r.name}</label>`).join("");
document.querySelectorAll("#runs input").forEach(b=>b.onchange=()=>{sel[b.dataset.r]=b.checked;localStorage.setItem("xol_runs_sel",JSON.stringify(sel));draw()});
["xa","sm","logy"].forEach(id=>$(id).oninput=()=>{localStorage.setItem("xol_runs_ui",JSON.stringify({xa:$("xa").value,sm:$("sm").value,logy:$("logy").checked}));draw()});
function ema(ys,a){let o=[],m=null;for(const y of ys){if(y==null||!isFinite(y)){o.push(null);continue}m=m==null?y:a*m+(1-a)*y;o.push(m)}return o}
function plot(cv,series,logy){
  const dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;
  const g=cv.getContext("2d");g.scale(dpr,dpr);g.clearRect(0,0,W,H);
  const css=getComputedStyle(document.documentElement),fg=css.getPropertyValue("--mut"),grid=css.getPropertyValue("--grid");
  let xs=[],ys=[];series.forEach(s=>s.pts.forEach(p=>{if(p[1]!=null&&isFinite(p[1])&&(!logy||p[1]>0)){xs.push(p[0]);ys.push(logy?Math.log10(p[1]):p[1])}}));
  if(!xs.length){g.fillStyle=fg;g.fillText("no data",10,20);return}
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);if(x1==x0)x1=x0+1;if(y1==y0){y1+=1e-3;y0-=1e-3}
  const pad=(y1-y0)*0.06;y0-=pad;y1+=pad;const L=52,R=10,T=8,B=24;
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);
  g.strokeStyle=grid;g.fillStyle=fg;g.font="11px system-ui";g.lineWidth=1;
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4,yy=Y(v);g.beginPath();g.moveTo(L,yy);g.lineTo(W-R,yy);g.stroke();
    const lab=logy?Math.pow(10,v):v;g.fillText(Math.abs(lab)<1e-2||Math.abs(lab)>=1e4?lab.toExponential(1):lab.toPrecision(3),4,yy+4)}
  for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;g.fillText(v>=1e4?v.toExponential(1):Math.round(v).toString(),X(v)-10,H-6)}
  series.forEach(s=>{g.strokeStyle=s.color;g.fillStyle=s.color;g.globalAlpha=s.alpha||1;g.lineWidth=s.w||1.5;
    if(s.dots){s.pts.forEach(p=>{if(p[1]==null)return;const y=logy?Math.log10(p[1]):p[1];g.beginPath();g.arc(X(p[0]),Y(y),3,0,7);g.fill()})}
    else{g.beginPath();let on=false;s.pts.forEach(p=>{if(p[1]==null||!isFinite(p[1])||(logy&&p[1]<=0)){on=false;return}const y=logy?Math.log10(p[1]):p[1];on?g.lineTo(X(p[0]),Y(y)):g.moveTo(X(p[0]),Y(y));on=true});g.stroke()}
    g.globalAlpha=1});
}
function draw(){
  const xa=$("xa").value,a=+$("sm").value,logy=$("logy").checked;const L=[],LR=[],GN=[],TP=[];
  RUNS.forEach((r,i)=>{if(sel[r.name]===false)return;const c=COLORS[i%COLORS.length],m=r.m,x=m.map(e=>e[xa]??e.step);
    const lo=m.map(e=>e.loss??null);L.push({color:c,alpha:.22,w:1,pts:x.map((v,j)=>[v,lo[j]])});
    const sm=ema(lo,a);L.push({color:c,w:2,pts:x.map((v,j)=>[v,sm[j]])});
    const ev=m.map((e,j)=>e.eval_loss!=null?[x[j],e.eval_loss]:null).filter(Boolean);if(ev.length)L.push({color:c,dots:true,pts:ev});
    LR.push({color:c,pts:x.map((v,j)=>[v,m[j].lr??null])});GN.push({color:c,alpha:.5,pts:x.map((v,j)=>[v,m[j].gnorm??null])});
    TP.push({color:c,pts:x.map((v,j)=>[v,j>0&&m[j].t!=null&&m[j-1].t!=null&&m[j].tokens!=null?(m[j].tokens-m[j-1].tokens)/Math.max(1e-6,m[j].t-m[j-1].t):null])});});
  plot($("c_loss"),L,logy);plot($("c_lr"),LR,false);plot($("c_gn"),GN,false);plot($("c_tps"),TP,false);
  $("tab").innerHTML="<tr><th>run</th><th>steps</th><th>last loss</th><th>best eval</th><th>tokens</th><th>config</th></tr>"+RUNS.map(r=>{const l=r.m[r.m.length-1]||{},ev=r.m.filter(e=>e.eval_loss!=null).map(e=>e.eval_loss);
    return `<tr><td>${r.name}</td><td>${l.step??"-"}</td><td>${l.loss!=null?l.loss.toFixed(4):"-"}</td><td>${ev.length?Math.min(...ev).toFixed(4):"-"}</td><td>${l.tokens??"-"}</td><td><code>${JSON.stringify(r.cfg).slice(0,160)}</code></td></tr>`}).join("");
}
draw();window.onresize=draw;
</script></body></html>"""


def load_runs(root: Path):
    runs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mf = d / "metrics.jsonl"
        if not mf.exists():
            continue
        m = []
        for line in mf.read_text().splitlines():
            try:
                m.append(json.loads(line))
            except ValueError:
                pass                     # a half-written last line while the run is going
        cfg = {}
        if (d / "config.json").exists():
            try:
                cfg = json.loads((d / "config.json").read_text())
            except ValueError:
                pass
        runs.append({"name": d.name, "m": m, "cfg": cfg})
    return runs


def write(root: Path, png=None):
    runs = load_runs(root)
    page = HTML.replace("__DATA__", json.dumps(runs)).replace("__NOW__", str(int(time.time())))
    (root / "index.html").write_text(page)
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for r in runs:
            s = [e["step"] for e in r["m"] if "loss" in e]
            lo = [e["loss"] for e in r["m"] if "loss" in e]
            if not lo:
                continue
            sm, m = [], None
            for v in lo:
                m = v if m is None else 0.9 * m + 0.1 * v
                sm.append(m)
            line, = ax[0].plot(s, lo, alpha=0.25)
            ax[0].plot(s, sm, color=line.get_color(), label=r["name"])
            ev = [(e["step"], e["eval_loss"]) for e in r["m"] if "eval_loss" in e]
            if ev:
                ax[0].scatter(*zip(*ev), color=line.get_color(), s=12)
            g = [(e["step"], e["gnorm"]) for e in r["m"] if "gnorm" in e]
            if g:
                ax[1].plot(*zip(*g), color=line.get_color(), alpha=0.6, label=r["name"])
        ax[0].set_title("train loss (EMA 0.9) / eval (dots)")
        ax[1].set_title("gradient norm")
        for x in ax:
            x.set_xlabel("step")
            x.grid(alpha=0.3)
            x.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(png, dpi=110)
    return len(runs)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0] if args else "runs")
    root.mkdir(parents=True, exist_ok=True)
    png = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None
    if "--watch" in sys.argv:
        while True:
            write(root, png)
            time.sleep(15)
    n = write(root, png)
    print(f"{n} run(s) -> {root / 'index.html'}" + (f", {png}" if png else ""))


if __name__ == "__main__":
    main()
