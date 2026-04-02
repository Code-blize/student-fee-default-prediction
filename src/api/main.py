from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.api.schemas import PredictionRequest, PredictionResponse
from src.features.build_features import build_feature_table
from src.models.predict import load_model, predict_single

ml_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path("models/best_model.pkl")
    ml_state["model"] = load_model(model_path)
    yield
    ml_state.clear()


app = FastAPI(
    title="Fee Defaulter Prediction API",
    version="1.0.0",
    description="API for predicting student fee-default risk.",
    lifespan=lifespan,
)

@app.get("/", response_class=HTMLResponse)
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Fee Defaulter Prediction API</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#09090b;--surface:#111113;--surface2:#18181b;
  --border:rgba(255,255,255,0.07);--border-med:rgba(255,255,255,0.13);
  --text:#f4f4f5;--muted:#71717a;--muted2:#52525b;
  --accent:#4ade80;--accent2:#22d3ee;--warn:#fb923c;--red:#f87171;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif}
body{min-height:100vh;display:flex;flex-direction:column;position:relative;overflow-x:hidden}
body::before{
  content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(255,255,255,0.022) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,255,255,0.022) 1px,transparent 1px);
  background-size:56px 56px;pointer-events:none;z-index:0
}
.blob{position:fixed;border-radius:50%;filter:blur(90px);pointer-events:none;z-index:0}
.b1{width:480px;height:480px;background:rgba(74,222,128,0.065);top:-100px;left:-60px}
.b2{width:380px;height:380px;background:rgba(34,211,238,0.055);bottom:60px;right:-80px}
nav{
  position:relative;z-index:2;
  display:flex;align-items:center;justify-content:space-between;
  padding:18px 40px;border-bottom:1px solid var(--border)
}
.nav-logo{display:flex;align-items:center;gap:10px;font-family:'DM Mono',monospace;font-size:12px;color:var(--muted);letter-spacing:.04em}
.dot{width:7px;height:7px;background:var(--accent);border-radius:50%;box-shadow:0 0 8px var(--accent);animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.nav-links{display:flex;gap:6px}
.nl{font-size:12px;font-family:'DM Mono',monospace;color:var(--muted);text-decoration:none;padding:5px 12px;border-radius:6px;border:1px solid transparent;transition:all .15s}
.nl:hover{color:var(--text);border-color:var(--border-med);background:var(--surface2)}
main{position:relative;z-index:1;max-width:880px;margin:0 auto;padding:72px 36px 56px;flex:1}
.tag{
  display:inline-flex;align-items:center;gap:7px;
  font-family:'DM Mono',monospace;font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--accent);border:1px solid rgba(74,222,128,0.22);
  padding:5px 13px;border-radius:99px;margin-bottom:30px
}
h1{
  font-family:'Syne',sans-serif;
  font-size:clamp(2.1rem,5.5vw,3.6rem);
  font-weight:700;line-height:1.08;letter-spacing:-.025em;margin-bottom:22px
}
h1 em{font-style:normal;color:var(--accent)}
.sub{
  font-size:1.05rem;color:var(--muted);line-height:1.75;
  max-width:540px;margin-bottom:14px;font-weight:300
}
.sub-note{
  font-size:.9rem;font-style:italic;color:var(--muted2);
  margin-bottom:42px;line-height:1.6
}
.cta-row{display:flex;flex-wrap:wrap;gap:11px;margin-bottom:60px}
.btn{
  text-decoration:none;font-size:13.5px;font-weight:500;
  padding:10px 22px;border-radius:8px;transition:all .15s;
  display:inline-flex;align-items:center;gap:8px
}
.btn-p{background:var(--accent);color:#09090b}
.btn-p:hover{background:#86efac;transform:translateY(-1px)}
.btn-o{background:transparent;color:var(--text);border:1px solid var(--border-med)}
.btn-o:hover{background:var(--surface2);border-color:rgba(255,255,255,.2)}
.sl{font-family:'DM Mono',monospace;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted2);margin-bottom:18px}
/* Endpoints */
.ep-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:10px;margin-bottom:48px}
.ep{
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:16px 18px;transition:border-color .15s;text-decoration:none
}
.ep:hover{border-color:var(--border-med)}
.method{
  font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.08em;font-weight:500;
  padding:3px 8px;border-radius:4px;margin-bottom:11px;display:inline-block
}
.get{background:rgba(34,211,238,.12);color:var(--accent2)}
.post{background:rgba(74,222,128,.12);color:var(--accent)}
.ep-path{font-family:'DM Mono',monospace;font-size:14px;color:var(--text);margin-bottom:5px}
.ep-desc{font-size:12px;color:var(--muted);line-height:1.5}
/* Risk tiers */
.tiers{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:48px}
.tier{
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:18px;position:relative;overflow:hidden
}
.tier::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.tl::before{background:var(--accent)}
.tm::before{background:var(--warn)}
.th::before{background:var(--red)}
.tier-tag{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.07em;text-transform:uppercase;margin-bottom:10px}
.tl .tier-tag{color:var(--accent)}.tm .tier-tag{color:var(--warn)}.th .tier-tag{color:var(--red)}
.tier-val{font-family:'Syne',sans-serif;font-size:1.45rem;font-weight:600;color:var(--text);margin-bottom:5px}
.tier-note{font-size:11.5px;color:var(--muted);line-height:1.5}
/* Model stats */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:48px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px}
.stat-val{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:600;color:var(--text);margin-bottom:4px}
.stat-label{font-size:11.5px;color:var(--muted);line-height:1.4}
/* Stack */
.stack{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:48px}
.badge{
  font-family:'DM Mono',monospace;font-size:11px;
  padding:5px 12px;border-radius:6px;
  background:var(--surface2);border:1px solid var(--border);color:var(--muted);letter-spacing:.04em
}
/* Divider */
.divider{height:1px;background:var(--border);margin:48px 0}
footer{
  position:relative;z-index:1;border-top:1px solid var(--border);
  padding:18px 40px;display:flex;justify-content:space-between;align-items:center;
  font-family:'DM Mono',monospace;font-size:11.5px;color:var(--muted)
}
footer a{color:var(--muted);text-decoration:none}
footer a:hover{color:var(--text)}
@media(max-width:640px){
  nav{padding:14px 18px}.nav-links{display:none}
  main{padding:48px 20px 40px}
  .tiers{grid-template-columns:1fr}
  footer{flex-direction:column;gap:8px;text-align:center;padding:16px 20px}
}
</style>
</head>
<body>
<div class="blob b1"></div>
<div class="blob b2"></div>
<nav>
  <div class="nav-logo">
    <span class="dot"></span>
    fee-defaulter-api &middot; v1.0.0
  </div>
  <div class="nav-links">
    <a href="/health" class="nl">health</a>
    <a href="/docs" class="nl">docs</a>
    <a href="/predict" class="nl">predict</a>
  </div>
</nav>
<main>
  <div class="tag">
    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
      <circle cx="5" cy="5" r="4" stroke="currentColor" stroke-width="1.4"/>
      <path d="M5 3v2.2L6.4 6.6" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
    </svg>
    FastAPI &middot; scikit-learn &middot; Render
  </div>
  <h1>Student Fee Defaulter<br><em>Prediction</em> API</h1>
  <p class="sub">
    End-to-end ML inference for assessing student fee-default risk &mdash;
    returning a calibrated probability score and tiered risk classification
    from academic, family, and financial support indicators.
  </p>
  <p class="sub-note">
    Target engineered via academic distress signals &middot; SHAP explainability &middot; Cost-sensitive threshold analysis
  </p>
  <div class="cta-row">
    <a href="/docs" class="btn btn-p">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M2 4h12M2 8h8M2 12h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      Open API Docs
    </a>
    <a href="/health" class="btn btn-o">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M5 8h1.5l1-2 1.5 4 1-2H11" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Health Check
    </a>
  </div>

  <p class="sl">Model Performance</p>
  <div class="stats">
    <div class="stat">
      <div class="stat-val">99.6%</div>
      <div class="stat-label">ROC-AUC<br>Random Forest</div>
    </div>
    <div class="stat">
      <div class="stat-val">94.9%</div>
      <div class="stat-label">Accuracy<br>Full feature set</div>
    </div>
    <div class="stat">
      <div class="stat-val">93.3%</div>
      <div class="stat-label">F1 Score<br>Champion model</div>
    </div>
    <div class="stat">
      <div class="stat-val">395</div>
      <div class="stat-label">Students<br>in training data</div>
    </div>
  </div>

  <p class="sl">Risk Classification</p>
  <div class="tiers">
    <div class="tier tl">
      <p class="tier-tag">Low Risk</p>
      <p class="tier-val">p &lt; 0.20</p>
      <p class="tier-note">Unlikely to default. No immediate action required.</p>
    </div>
    <div class="tier tm">
      <p class="tier-tag">Moderate Risk</p>
      <p class="tier-val">0.20 &ndash; 0.30</p>
      <p class="tier-note">Warrants monitoring and early outreach.</p>
    </div>
    <div class="tier th">
      <p class="tier-tag">High Risk</p>
      <p class="tier-val">p &ge; 0.30</p>
      <p class="tier-note">Intervention recommended. Payment plan or bursary review.</p>
    </div>
  </div>

  <p class="sl">Endpoints</p>
  <div class="ep-grid">
    <div class="ep">
      <span class="method get">GET</span>
      <p class="ep-path">/</p>
      <p class="ep-desc">This landing page</p>
    </div>
    <a href="/health" class="ep">
      <span class="method get">GET</span>
      <p class="ep-path">/health</p>
      <p class="ep-desc">Service liveness check</p>
    </a>
    <a href="/docs" class="ep">
      <span class="method get">GET</span>
      <p class="ep-path">/docs</p>
      <p class="ep-desc">Interactive Swagger UI</p>
    </a>
    <div class="ep">
      <span class="method post">POST</span>
      <p class="ep-path">/predict</p>
      <p class="ep-desc">Returns prediction, probability score, and risk label</p>
    </div>
  </div>

  <p class="sl">Built with</p>
  <div class="stack">
    <span class="badge">FastAPI</span>
    <span class="badge">scikit-learn</span>
    <span class="badge">pandas</span>
    <span class="badge">pydantic v2</span>
    <span class="badge">SHAP</span>
    <span class="badge">joblib</span>
    <span class="badge">Python 3.11</span>
    <span class="badge">Render</span>
  </div>
</main>
<footer>
  <span>Built by <a href="https://github.com/Code-blize">Blessing Obasi-Uzoma</a> &middot; <a href="https://linkedin.com/in/blessing-obasi-uzoma">LinkedIn</a></span>
  <span>Student Fee Defaulter Prediction API &middot; 2025</span>
</footer>
</body>
</html>"""
