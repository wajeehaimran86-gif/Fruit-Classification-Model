import os
import sys
import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="FruitVision AI | Fruit Classification",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
HERO_IMAGE = ROOT / "assets" / "hero_fruits.png"
HERO_IMAGE_DATA = base64.b64encode(HERO_IMAGE.read_bytes()).decode("ascii") if HERO_IMAGE.exists() else ""
sys.path.append(str(ROOT / "models"))
sys.path.append(str(ROOT))

from svm import OvRSVM
from confidence import softmax_confidence
from robust_pipeline import foreground_mask, object_crop, inference_features
from detect_multi import segment_regions, classify_crop

CACHE_DIR = ROOT / "results" / "saved_models"
MODEL_PATH = CACHE_DIR / "linear_svm.npz"
SCALER_PATH = CACHE_DIR / "feature_scaler.npz"

# -----------------------------------------------------------------------------
# Premium product UI — no external hero image is used. The hero visual is
# rendered with inline SVG/CSS so the whole page remains self-contained.
# -----------------------------------------------------------------------------
st.markdown(
    r'''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {
    --bg: #f7f8ff;
    --surface: #ffffff;
    --ink: #101a36;
    --muted: #667085;
    --line: #e8e9f3;
    --purple: #6657f5;
    --violet: #7c4dff;
    --blue: #5b7cff;
    --green: #20b486;
    --orange: #ff8a1f;
    --yellow: #f5bd2f;
    --shadow: 0 22px 70px rgba(39, 35, 100, .10);
}

html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 12% 2%, rgba(112,91,255,.13), transparent 28%),
      radial-gradient(circle at 88% 7%, rgba(107,178,255,.12), transparent 24%),
      linear-gradient(180deg, #fbfbff 0%, #f6f8ff 52%, #ffffff 100%);
    color: var(--ink);
}
.block-container { max-width: 1280px; padding: 1rem 1.4rem 3rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ---------------- Navigation ---------------- */
.navbar {
    height: 62px; display:flex; align-items:center; justify-content:space-between;
    padding: 0 8px 0 2px; margin-bottom: 10px;
}
.nav-brand { display:flex; align-items:center; gap:11px; }
.brand-mark {
    width:39px; height:39px; border-radius:12px; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(145deg,#6858ff,#8b5cf6); color:#fff;
    box-shadow: 0 10px 26px rgba(102,87,245,.27); position:relative; overflow:hidden;
}
.brand-mark:after { content:""; width:18px; height:18px; border:2px solid rgba(255,255,255,.95); border-radius:55% 45% 50% 50%; transform:rotate(-18deg); }
.brand-name { font: 800 16px 'Plus Jakarta Sans',sans-serif; letter-spacing:-.025em; }
.brand-sub { font-size:10.5px; color:#7b8194; margin-top:1px; }
.nav-right { display:flex; gap:10px; align-items:center; }
.status-pill, .model-pill-top {
    display:flex; align-items:center; gap:7px; padding:8px 12px; border:1px solid var(--line);
    background:rgba(255,255,255,.72); backdrop-filter: blur(14px); border-radius:999px;
    font-size:11px; font-weight:700; color:#384056; box-shadow:0 5px 18px rgba(40,45,90,.05);
}
.status-dot { width:7px;height:7px;border-radius:50%;background:#18b77b;box-shadow:0 0 0 4px rgba(24,183,123,.12); }

/* ---------------- Hero ---------------- */
.hero {
    position:relative; overflow:hidden; min-height:420px; margin-bottom:24px;
    border:1px solid #e9e7fb; border-radius:30px;
    background:
      radial-gradient(circle at 73% 46%, rgba(123,94,255,.18), transparent 27%),
      radial-gradient(circle at 92% 80%, rgba(74,158,255,.16), transparent 25%),
      linear-gradient(110deg,#ffffff 0%,#f7f4ff 48%,#eef3ff 100%);
    box-shadow:var(--shadow);
}
.hero-grid { display:grid; grid-template-columns: 46% 54%; min-height:420px; align-items:center; }
.hero-copy { padding:46px 20px 46px 48px; position:relative; z-index:5; }
.hero-kicker { display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border-radius:999px;background:rgba(102,87,245,.08);border:1px solid rgba(102,87,245,.13);color:#5848df;font-size:11px;font-weight:800; }
.hero-kicker-dot { width:7px;height:7px;border-radius:50%;background:#20b486;box-shadow:0 0 0 4px rgba(32,180,134,.12); }
.hero h1 { font:800 clamp(36px,4.1vw,58px)/1.05 'Plus Jakarta Sans',sans-serif; letter-spacing:-.055em; margin:17px 0 15px; max-width:620px; }
.hero h1 .gradient { background:linear-gradient(90deg,#5b56ee,#7139e9,#7b55ff); -webkit-background-clip:text;background-clip:text;color:transparent; }
.hero-desc { max-width:565px;color:#667085;font-size:15px;line-height:1.7;margin:0 0 20px; }
.hero-features { display:flex; gap:10px; flex-wrap:wrap; }
.hero-feature { min-width:135px; padding:10px 12px; border:1px solid rgba(224,224,242,.9); background:rgba(255,255,255,.55); backdrop-filter:blur(12px); border-radius:14px; box-shadow:0 8px 22px rgba(40,45,100,.05); }
.hero-feature b { display:block;font-size:11px;color:#25304b;margin-bottom:2px; }
.hero-feature span { font-size:10px;color:#7a8297; }

.hero-visual { position:relative; height:420px; overflow:hidden; }
.hero-glow { position:absolute; width:410px;height:410px;border-radius:50%;right:45px;top:3px;background:radial-gradient(circle,rgba(124,77,255,.25),rgba(124,77,255,.03) 58%,transparent 72%);filter:blur(4px);animation:pulseGlow 4.5s ease-in-out infinite; }
.scan-stage { position:absolute; width:410px;height:280px;right:58px;top:75px; border-radius:28px; border:1px solid rgba(255,255,255,.75); background:linear-gradient(145deg,rgba(255,255,255,.56),rgba(220,217,255,.22)); box-shadow: inset 0 1px rgba(255,255,255,.8), 0 30px 70px rgba(82,66,190,.14); backdrop-filter:blur(9px); }
.scan-stage:before { content:"";position:absolute;inset:27px;border:1px solid rgba(101,87,245,.25);border-radius:18px;box-shadow:inset 0 0 50px rgba(102,87,245,.06); }
.scan-stage:after { content:"";position:absolute;left:24px;right:24px;top:52%;height:2px;background:linear-gradient(90deg,transparent,#8a78ff,transparent);filter:drop-shadow(0 0 7px #806cff);animation:scanLine 3.5s ease-in-out infinite; }
.stage-floor { position:absolute; width:420px;height:90px;left:20px;bottom:-38px;border-radius:50%;background:radial-gradient(ellipse,rgba(102,87,245,.32),rgba(102,87,245,.04) 65%,transparent 72%);filter:blur(1px); }
.fruit { position:absolute; z-index:4; transform-origin:center bottom; filter:drop-shadow(0 20px 18px rgba(32,31,70,.18)); }
.fruit.apple { width:112px;height:118px;left:72px;top:82px;animation:floatA 4.4s ease-in-out infinite; }
.fruit.apple .body { position:absolute;left:8px;top:17px;width:96px;height:95px;border-radius:46% 50% 48% 52%;background:radial-gradient(circle at 30% 25%,#ff8f9a 0 5%,transparent 6%),radial-gradient(circle at 33% 28%,#ef3340 0,#bf162c 65%,#8c1022 100%); }
.fruit.apple .dip { position:absolute;left:47px;top:7px;width:23px;height:24px;background:#7f3e22;border-radius:50%;transform:rotate(15deg); }
.fruit.apple .leaf { position:absolute;left:59px;top:0;width:43px;height:18px;border-radius:100% 0 100% 0;background:linear-gradient(135deg,#69c45c,#228447);transform:rotate(-18deg); }
.fruit.orange { width:122px;height:122px;left:190px;top:139px;animation:floatB 4.8s ease-in-out infinite .2s; }
.fruit.orange .body { position:absolute;inset:3px;border-radius:50%;background:radial-gradient(circle at 32% 24%,#ffd078 0 4%,transparent 5%),radial-gradient(circle at 35% 30%,#ffb02e 0,#ff8a1f 48%,#e96108 100%);box-shadow:inset -10px -12px 20px rgba(185,65,0,.18), inset 7px 8px 18px rgba(255,255,255,.28); }
.fruit.orange .texture { position:absolute;inset:16px;border-radius:50%;background-image:radial-gradient(rgba(255,215,132,.35) 1px,transparent 1.5px);background-size:7px 7px;opacity:.45; }
.fruit.orange .stem { position:absolute;left:57px;top:-5px;width:9px;height:22px;border-radius:6px;background:#70452a;transform:rotate(7deg); }
.fruit.banana { width:185px;height:155px;right:35px;top:53px;animation:floatC 5.2s ease-in-out infinite .45s; filter:drop-shadow(0 22px 18px rgba(66,48,0,.16)); }
.fruit.banana .stem { position:absolute;left:77px;top:7px;width:29px;height:30px;border-radius:9px 9px 15px 15px;background:linear-gradient(90deg,#60401c,#a9782b 45%,#5d3d19);transform:rotate(5deg);z-index:8;box-shadow:inset 5px 0 5px rgba(255,220,112,.18); }
.fruit.banana .stem:before { content:"";position:absolute;left:-18px;top:7px;width:20px;height:12px;border-radius:70% 30% 70% 30%;background:#805523;transform:rotate(-25deg); }
.fruit.banana .stem:after { content:"";position:absolute;right:-15px;top:4px;width:18px;height:12px;border-radius:30% 70% 30% 70%;background:#8a5b23;transform:rotate(25deg); }
.fruit.banana .curve { position:absolute; width:125px;height:69px;left:20px;top:31px;border:21px solid #f7c92f;border-left-color:transparent;border-bottom-color:#e8a91e;border-radius:60% 60% 55% 45%;transform:rotate(-15deg);filter:drop-shadow(0 5px 4px rgba(120,85,0,.16)); }
.fruit.banana .curve:after { content:"";position:absolute;inset:-16px -15px -14px -7px;border:3px solid rgba(255,235,120,.48);border-left-color:transparent;border-bottom-color:rgba(191,126,12,.22);border-radius:60% 60% 55% 45%; }
.fruit.banana .curve2,.fruit.banana .curve3 { position:absolute;width:123px;height:67px;border:20px solid #f2be29;border-left-color:transparent;border-bottom-color:#d99a17;border-radius:60% 60% 55% 45%;filter:drop-shadow(0 5px 4px rgba(120,85,0,.14)); }
.fruit.banana .curve2 { left:33px;top:48px;transform:rotate(7deg);z-index:2; }
.fruit.banana .curve3 { left:18px;top:72px;transform:rotate(20deg);z-index:1; }
.fruit.banana .curve2:after,.fruit.banana .curve3:after { content:"";position:absolute;inset:-15px -14px -13px -6px;border:3px solid rgba(255,232,108,.42);border-left-color:transparent;border-bottom-color:rgba(176,112,8,.18);border-radius:60% 60% 55% 45%; }
.fruit.banana .tip1,.fruit.banana .tip2,.fruit.banana .tip3 { position:absolute;width:15px;height:18px;border-radius:55% 45% 50% 50%;background:linear-gradient(135deg,#6f4a25,#3f2a17);z-index:7; }
.fruit.banana .tip1 { right:8px;top:34px;transform:rotate(25deg); }.fruit.banana .tip2 { right:1px;top:78px;transform:rotate(25deg); }.fruit.banana .tip3 { right:9px;top:119px;transform:rotate(18deg); }
.floating-card { position:absolute;z-index:8;padding:11px 13px;border:1px solid rgba(255,255,255,.75);background:rgba(255,255,255,.72);backdrop-filter:blur(14px);border-radius:14px;box-shadow:0 15px 35px rgba(42,35,100,.11);min-width:92px;animation:floatCard 5s ease-in-out infinite; }
.floating-card small { display:block;color:#7a8195;font-size:8px;text-transform:uppercase;letter-spacing:.08em;font-weight:800;margin-bottom:3px; }.floating-card strong { font-size:15px;color:#202946; }.floating-card.a { right:27px;top:72px; }.floating-card.b { left:8px;top:182px;animation-delay:.7s; }.floating-card.c { right:9px;bottom:45px;animation-delay:1.2s; }
.floating-bar { width:60px;height:5px;border-radius:999px;background:#e8eaf5;overflow:hidden;margin-top:5px; }.floating-bar i { display:block;width:86%;height:100%;background:linear-gradient(90deg,#6758f6,#a45dff);border-radius:999px;animation:barPulse 2.8s ease-in-out infinite; }

/* ---------------- Main cards ---------------- */
.section-head { display:flex;align-items:end;justify-content:space-between;gap:20px;margin:28px 0 12px; }
.section-title { font:800 21px 'Plus Jakarta Sans',sans-serif;letter-spacing:-.03em;margin:0; }.section-sub { color:var(--muted);font-size:12.5px;margin:3px 0 0; }
.card { background:rgba(255,255,255,.88);border:1px solid var(--line);border-radius:20px;box-shadow:0 12px 35px rgba(36,40,80,.055); }
.upload-card { padding:18px;min-height:360px; }.upload-drop { border:1.5px dashed #b8b2fa;background:linear-gradient(180deg,#fcfcff,#f8f7ff);border-radius:17px;padding:28px 20px;text-align:center;margin-top:13px; }
.upload-icon { width:52px;height:52px;margin:0 auto 11px;border-radius:16px;background:linear-gradient(145deg,#ebe8ff,#f6f3ff);display:flex;align-items:center;justify-content:center;color:#6554e9;font-size:23px;box-shadow:inset 0 1px #fff; }
.image-card { padding:14px; height:100%; }.image-label {font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:#626b80;margin-bottom:9px;}
.image-card img { border-radius:14px; }
.result-card { padding:20px; }
.result-grid { display:grid;grid-template-columns:1fr 1.15fr;gap:16px;align-items:center; }
.pred-label {font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.09em;color:#7b8294; }.prediction-name {font:800 35px 'Plus Jakarta Sans',sans-serif;letter-spacing:-.045em;margin:5px 0 13px;color:#ff7a13; }
.confidence-box {border:1px solid #dfe9e4;background:linear-gradient(135deg,#f5fff9,#fff);border-radius:15px;padding:13px 15px;}.conf-label{font-size:11px;color:#667085}.conf-value{font:800 28px 'Plus Jakarta Sans',sans-serif;color:#19a974;margin-top:2px;}
.prob-row{margin:0 0 13px}.prob-head{display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px}.prob-name{font-weight:700}.prob-value{color:#667085}.prob-track{height:7px;background:#edf0f6;border-radius:999px;overflow:hidden}.prob-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#6a5af5,#8d5dff);transition:width .8s ease}.prob-fill.best{background:linear-gradient(90deg,#20b486,#45ca9d)}
.conf-ring { width:128px;height:128px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;margin:auto;background:conic-gradient(#20b486 0 91%,#edf2f0 91% 100%);position:relative; }.conf-ring:before{content:"";position:absolute;width:101px;height:101px;background:#fff;border-radius:50%;}.conf-ring span,.conf-ring small{position:relative;z-index:2}.conf-ring span{font:800 21px 'Plus Jakarta Sans',sans-serif}.conf-ring small{font-size:9px;color:#20a977;font-weight:800;margin-top:2px}
.pipeline {display:flex;align-items:center;gap:6px;margin-top:14px}.pipe-step{flex:1;min-width:0;background:#fbfbff;border:1px solid var(--line);border-radius:14px;padding:12px 8px;text-align:center}.pipe-num{width:23px;height:23px;border-radius:8px;background:#eeebff;color:#5f4feb;display:flex;align-items:center;justify-content:center;margin:0 auto 6px;font-size:9px;font-weight:800}.pipe-name{font-size:10px;font-weight:750;line-height:1.25}.pipe-arrow{color:#a0a5b8;font-weight:800}
.model-info{padding:18px}.info-item{padding:11px 0;border-bottom:1px solid #eff0f5}.info-item:last-child{border-bottom:0}.info-item b{font-size:11px}.info-item span{display:block;color:#727a8e;font-size:11px;margin-top:3px}
.multi-card{padding:17px}.multi-top{display:flex;align-items:center;justify-content:space-between;gap:20px}.multi-title{font-weight:800;font-size:13px}.multi-sub{color:#7a8296;font-size:11px;margin-top:3px}
.footer{margin-top:34px;padding:22px 0;text-align:center;border-top:1px solid var(--line);color:#8a91a3;font-size:11px}.footer strong{color:#4b5367}

/* Streamlit controls */
div[data-testid="stFileUploader"] { background:transparent;border:0;padding:0; }
div[data-testid="stFileUploader"] section { padding:0;background:transparent; }
div[data-testid="stFileUploader"] label { display:none; }
div.stButton > button { width:100%;min-height:44px;border:0;border-radius:12px;background:linear-gradient(135deg,#6556f4,#7c45ef);color:#fff;font-weight:800;box-shadow:0 10px 24px rgba(102,87,245,.22);transition:.2s; }
div.stButton > button:hover { transform:translateY(-1px);box-shadow:0 14px 30px rgba(102,87,245,.29); }
div[data-testid="stToggle"] label { font-size:12px;font-weight:700; }
hr{border-color:var(--line);margin:26px 0}.stAlert{border-radius:14px}

@keyframes floatA{0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-13px) rotate(2deg)}}
@keyframes floatB{0%,100%{transform:translateY(0) rotate(1deg)}50%{transform:translateY(-16px) rotate(-2deg)}}
@keyframes floatC{0%,100%{transform:translateY(0) rotate(-7deg)}50%{transform:translateY(-11px) rotate(-3deg)}}
@keyframes floatCard{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes pulseGlow{0%,100%{transform:scale(.95);opacity:.75}50%{transform:scale(1.05);opacity:1}}
@keyframes scanLine{0%,100%{transform:translateY(-88px);opacity:.1}50%{transform:translateY(88px);opacity:1}}
@keyframes barPulse{0%,100%{opacity:.75;width:72%}50%{opacity:1;width:94%}}

/* ---------------- Real fruit hero visual ---------------- */
.hero-photo-stage { position:absolute; width:410px; height:280px; right:58px; top:75px; border-radius:28px; border:1px solid rgba(255,255,255,.82); background:linear-gradient(145deg,rgba(255,255,255,.76),rgba(225,221,255,.30)); box-shadow:inset 0 1px rgba(255,255,255,.95),0 30px 70px rgba(82,66,190,.15); backdrop-filter:blur(10px); overflow:hidden; }
.hero-photo-stage:before { content:""; position:absolute; inset:27px; border:1px solid rgba(101,87,245,.20); border-radius:18px; z-index:3; pointer-events:none; }
.hero-photo-stage:after { content:""; position:absolute; left:30px; right:30px; top:50%; height:2px; background:linear-gradient(90deg,transparent,#8a78ff,transparent); filter:drop-shadow(0 0 8px #806cff); animation:scanLinePhoto 3.4s ease-in-out infinite; z-index:5; pointer-events:none; }
.hero-photo { position:absolute; left:43px; top:28px; width:325px; height:225px; object-fit:contain; border-radius:20px; mix-blend-mode:multiply; filter:drop-shadow(0 22px 20px rgba(38,30,82,.18)); animation:realFruitFloat 5s ease-in-out infinite; z-index:2; }
.hero-photo-halo { position:absolute; width:300px; height:190px; left:55px; top:45px; border-radius:50%; background:radial-gradient(ellipse,rgba(124,77,255,.22),transparent 68%); filter:blur(12px); animation:pulseGlow 4.5s ease-in-out infinite; z-index:1; }
.hero-photo-badge { position:absolute; z-index:7; left:20px; bottom:18px; padding:8px 11px; border:1px solid rgba(255,255,255,.8); background:rgba(255,255,255,.78); backdrop-filter:blur(12px); border-radius:12px; box-shadow:0 10px 25px rgba(42,35,100,.10); font-size:9px; font-weight:800; color:#5b56ee; letter-spacing:.04em; }
@keyframes realFruitFloat { 0%,100%{transform:translateY(0) rotate(-1deg) scale(1)} 50%{transform:translateY(-10px) rotate(1deg) scale(1.015)} }
@keyframes scanLinePhoto { 0%,100%{transform:translateY(-88px);opacity:.15} 50%{transform:translateY(88px);opacity:1} }
.fruit { display:none !important; }
@media(max-width: 900px){
 .hero-grid{grid-template-columns:1fr}.hero-copy{padding:34px 28px 8px}.hero-visual{height:330px}.scan-stage{right:50%;transform:translateX(50%);top:25px}.hero-photo-stage{right:50%;transform:translateX(50%);top:25px}.hero-photo{width:320px;height:220px}.hero-glow{right:50%;transform:translateX(50%);top:-20px}.floating-card.b{left:12px}.floating-card.a{right:12px}.hero h1{font-size:38px}
}
@media(max-width: 700px){.block-container{padding: .7rem .8rem 2rem}.nav-right{display:none}.hero{border-radius:22px}.hero-copy{padding:28px 22px 0}.hero-visual{height:285px}.scan-stage{width:320px;height:220px;top:25px}.hero-photo-stage{width:320px;height:220px;top:25px}.hero-photo{left:28px;top:20px;width:265px;height:180px}.hero-photo-badge{left:12px;bottom:10px}.fruit.apple{left:46px;top:57px;transform:scale(.78)}.fruit.orange{left:135px;top:101px;transform:scale(.78)}.fruit.banana{right:15px;top:43px;transform:scale(.72)}.floating-card{transform:scale(.82)}.hero-features{display:none}.pipeline{flex-wrap:wrap}.pipe-step{min-width:29%}.pipe-arrow{display:none}.result-grid{grid-template-columns:1fr}.conf-ring{margin:8px auto}.section-head{display:block}.multi-top{display:block}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
</style>
''', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    model = OvRSVM()
    model.load(str(MODEL_PATH))
    scaler = np.load(str(SCALER_PATH))
    return model, scaler["mean"], scaler["std"]


def predict_file(path):
    model, mean, std = load_model()
    views = inference_features(path)[:1]
    views = (views - mean) / std
    svm_scores = model.decision_scores(views).mean(axis=0)
    from robust_pipeline import _orange_with_green_leaf_mask
    raw = cv2.imread(path)
    fused = svm_scores.astype(np.float32).copy()
    if raw is not None:
        orange_mask = _orange_with_green_leaf_mask(raw)
        ratio = cv2.countNonZero(orange_mask) / float(orange_mask.size)
        if ratio >= 0.12:
            quality = min(1.0, ratio / 0.35)
            orange_bias = 8.2 + 2.0 * quality
            fused[list(model.classes_).index("orange")] += orange_bias
    probs = softmax_confidence(fused.reshape(1, -1))[0]
    order = np.argsort(probs)[::-1]
    return model, probs, order


def save_uploaded(uploaded):
    suffix = Path(uploaded.name).suffix.lower() or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def probability_bars(model, probs, order):
    for i in order:
        label = model.classes_[i].title()
        value = float(probs[i] * 100)
        best = i == int(order[0])
        st.markdown(
            f'''<div class="prob-row"><div class="prob-head"><span class="prob-name">{label}</span><span class="prob-value">{value:.2f}%</span></div><div class="prob-track"><div class="prob-fill {'best' if best else ''}" style="width:{min(100, value):.2f}%"></div></div></div>''',
            unsafe_allow_html=True,
        )


def pipeline_ui():
    st.markdown('<div class="section-head"><div><div class="section-title">Model pipeline</div><div class="section-sub">From image upload to the final probability distribution.</div></div></div>', unsafe_allow_html=True)
    steps = ["Image Upload", "Fruit Focus", "Features", "OvR SVM", "Softmax", "Prediction"]
    cols = st.columns([1,.10,1,.10,1,.10,1,.10,1,.10,1])
    idx=0
    for n,col in enumerate(cols):
        with col:
            if n%2==0:
                st.markdown(f'<div class="pipe-step"><div class="pipe-num">{idx+1}</div><div class="pipe-name">{steps[idx]}</div></div>', unsafe_allow_html=True); idx+=1
            else:
                st.markdown('<div class="pipe-arrow">→</div>', unsafe_allow_html=True)


# Header + hero
hero_html = '''
<div class="navbar">
  <div class="nav-brand"><div class="brand-mark"></div><div><div class="brand-name">FruitVision AI</div><div class="brand-sub">ML Classification System</div></div></div>
  <div class="nav-right"><div class="status-pill"><span class="status-dot"></span> Model Status&nbsp; <b>Online</b></div><div class="model-pill-top">▣ &nbsp; Model: <b>Custom Linear SVM</b></div></div>
</div>
<div class="hero">
 <div class="hero-grid">
  <div class="hero-copy">
   <div class="hero-kicker"><span class="hero-kicker-dot"></span> AI Model Ready</div>
   <h1>Fruit Classification,<br>Powered by <span class="gradient">Machine Learning.</span></h1>
   <p class="hero-desc">Upload a real-world fruit image and let the model analyze its visual features through fruit-focused preprocessing, custom One-vs-Rest SVM and Softmax probabilities.</p>
   <div class="hero-features">
    <div class="hero-feature"><b>High Accuracy</b><span>SVM + Softmax</span></div>
    <div class="hero-feature"><b>Smart Processing</b><span>Leaf / stem aware</span></div>
    <div class="hero-feature"><b>Multi-Fruit</b><span>Detection supported</span></div>
   </div>
  </div>
  <div class="hero-visual">
   <div class="hero-glow"></div>
   <div class="hero-photo-stage">
    <div class="hero-photo-halo"></div>
    <img class="hero-photo" src="data:image/png;base64,__HERO_IMAGE__" alt="Apple, orange and banana fruit classification visual">
    <div class="hero-photo-badge">LIVE VISION SCAN</div>
   </div>
   <div class="floating-card a"><small>Prediction</small><strong>Orange</strong></div>
   <div class="floating-card b"><small>Feature extraction</small><div class="floating-bar"><i></i></div></div>
   <div class="floating-card c"><small>Confidence</small><strong>91.4%</strong></div>
  </div>
 </div>
</div>
'''.replace('__HERO_IMAGE__', HERO_IMAGE_DATA)
st.markdown(hero_html, unsafe_allow_html=True)

if not (MODEL_PATH.exists() and SCALER_PATH.exists()):
    st.error("Trained model files were not found. Run `python train.py` first.")
    st.stop()

# Input / upload
st.markdown('<div class="section-head"><div><div class="section-title">Analyze an image</div><div class="section-sub">Upload JPG, JPEG, PNG, BMP or WEBP and analyze it in seconds.</div></div></div>', unsafe_allow_html=True)
col_upload, col_preview = st.columns([.82,1.18], gap="large")
with col_upload:
    st.markdown('<div class="card upload-card"><div class="image-label">01 · Image input</div><div class="upload-drop"><div class="upload-icon">↥</div><b style="font-size:14px">Drag & drop an image here</b><div style="font-size:11px;color:#7a8296;margin-top:4px">or click below to browse · JPG / PNG / WEBP</div></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","bmp","webp"], label_visibility="collapsed", help="Use one primary fruit for single-image classification. Multi-fruit detection is available below.")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded is None:
    with col_preview:
        st.markdown('<div class="card image-card" style="min-height:360px;display:flex;align-items:center;justify-content:center;text-align:center"><div><div class="upload-icon">✦</div><div style="font-weight:800;font-size:15px">Ready for analysis</div><div class="small-note" style="margin-top:5px">Upload an image to reveal prediction, confidence, preprocessing and probabilities.</div></div></div>', unsafe_allow_html=True)
    pipeline_ui()
    st.markdown('<div class="footer"><strong>FruitVision AI</strong> · Custom Machine Learning Classification System<br>Built with Python · OpenCV · Streamlit · Custom Linear SVM</div>', unsafe_allow_html=True)
    st.stop()

path = save_uploaded(uploaded)
img = cv2.imread(path)
if img is None:
    st.error("The uploaded image could not be read. Please try another JPG or PNG file.")
    try: os.unlink(path)
    except OSError: pass
    st.stop()

# Analysis image pair
st.markdown('<div class="section-head"><div><div class="section-title">How the model sees the image</div><div class="section-sub">Original input beside the existing fruit-focused preprocessing view.</div></div></div>', unsafe_allow_html=True)
col1,col2 = st.columns(2,gap="large")
with col1:
    st.markdown('<div class="card image-card"><div class="image-label">Original image</div>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),use_container_width=True)
    st.markdown('</div>',unsafe_allow_html=True)
with col2:
    mask=foreground_mask(img); obj,_=object_crop(img,mask)
    st.markdown('<div class="card image-card"><div class="image-label">Fruit-focused preprocessing</div>',unsafe_allow_html=True)
    st.image(cv2.cvtColor(obj,cv2.COLOR_BGR2RGB),use_container_width=True)
    st.markdown('<div class="small-note" style="margin-top:8px">Background elements and green leaves/stems are treated as accessories so they do not dominate fruit-specific features.</div></div>',unsafe_allow_html=True)

with st.spinner("Analyzing visual features..."):
    try: model,probs,order=predict_file(path)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        try: os.unlink(path)
        except OSError: pass
        st.stop()

best=int(order[0]); best_name=model.classes_[best].title(); best_conf=float(probs[best]*100)

# Results dashboard
st.markdown('<div class="section-head"><div><div class="section-title">Prediction results</div><div class="section-sub">Final classification and Softmax probability distribution.</div></div></div>',unsafe_allow_html=True)
left,right=st.columns([1,1.55],gap="large")
with left:
    st.markdown(f'''<div class="card result-card"><div class="pred-label">Predicted fruit</div><div class="prediction-name">{best_name}</div><div class="confidence-box"><div class="conf-label">Confidence score</div><div class="conf-value">{best_conf:.2f}%</div></div></div>''',unsafe_allow_html=True)
with right:
    st.markdown('<div class="card result-card"><div class="pred-label" style="margin-bottom:13px">All class probabilities</div>',unsafe_allow_html=True)
    probability_bars(model,probs,order)
    st.markdown('</div>',unsafe_allow_html=True)

# Confidence + pipeline row
c1,c2=st.columns([.75,1.25],gap="large")
with c1:
    ring_value=max(0,min(100,best_conf))
    st.markdown(f'''<div class="card result-card" style="text-align:center"><div class="pred-label">Confidence overview</div><div class="conf-ring" style="background:conic-gradient(#20b486 0 {ring_value:.2f}%,#edf2f0 {ring_value:.2f}% 100%)"><span>{best_conf:.1f}%</span><small>HIGH CONFIDENCE</small></div></div>''',unsafe_allow_html=True)
with c2:
    pipeline_ui()

# Information
st.markdown('<div class="section-head"><div><div class="section-title">Model information</div><div class="section-sub">Technical context for the current classifier.</div></div></div>',unsafe_allow_html=True)
i1,i2,i3=st.columns(3,gap="large")
with i1: st.markdown('<div class="card model-info"><div class="info-item"><b>Model type</b><span>Custom One-vs-Rest Linear SVM</span></div><div class="info-item"><b>Classes</b><span>Apple · Banana · Orange</span></div></div>',unsafe_allow_html=True)
with i2: st.markdown('<div class="card model-info"><div class="info-item"><b>Probability</b><span>Softmax confidence</span></div><div class="info-item"><b>Preprocessing</b><span>Fruit-focused visual pipeline</span></div></div>',unsafe_allow_html=True)
with i3: st.markdown('<div class="card model-info"><div class="info-item"><b>Status</b><span style="color:#1da977;font-weight:700">● Model loaded successfully</span></div><div class="info-item"><b>Inference</b><span>Cached model loading</span></div></div>',unsafe_allow_html=True)

# Multi fruit
st.markdown('<div class="section-head"><div><div class="section-title">Multi-Fruit Detection</div><div class="section-sub">Detect and classify multiple fruits in a single image using the retained OpenCV detector.</div></div></div>',unsafe_allow_html=True)
st.markdown('<div class="card multi-card"><div class="multi-top"><div><div class="multi-title">Enable multi-fruit detection</div><div class="multi-sub">Bounding boxes and class confidence will appear when enabled.</div></div></div></div>',unsafe_allow_html=True)
run_multi=st.toggle("Enable multi-fruit detection",value=False)

if run_multi:
    with st.spinner("Detecting fruit regions..."):
        boxes=segment_regions(img); output=img.copy(); rows=[]
        scaler_data=np.load(str(SCALER_PATH)); mean,std=scaler_data["mean"],scaler_data["std"]; model_multi=load_model()[0]
        for x,y,w,h in boxes:
            crop=img[y:y+h,x:x+w]
            if crop.size==0: continue
            label,conf,_=classify_crop(crop,model_multi,mean,std)
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,200,0),2)
            cv2.putText(output,f"{label.title()} {conf:.0f}%",(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,200,0),2)
            rows.append({"Fruit":label.title(),"Confidence":f"{conf:.2f}%"})
    if rows:
        st.image(cv2.cvtColor(output,cv2.COLOR_BGR2RGB),use_container_width=True)
        st.dataframe(rows,use_container_width=True,hide_index=True)
    else: st.warning("No clear fruit regions were detected in this image.")

with st.expander("Technical details",expanded=False):
    st.markdown("**Pipeline:** Image Upload → Fruit Focus → Feature Extraction → Custom One-vs-Rest Linear SVM → Softmax → Prediction")
    st.markdown("**Classes:** Apple, Banana, Orange")
    st.markdown("**Multi-fruit:** Existing OpenCV region detector retained")

st.markdown('<div class="footer"><strong>FruitVision AI</strong> · Custom Machine Learning Classification System<br>Built with Python · OpenCV · Streamlit · Custom Linear SVM · Softmax</div>',unsafe_allow_html=True)

try: os.unlink(path)
except OSError: pass
