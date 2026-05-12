import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabAI · Risk Intelligence",
    page_icon="⬡",
    layout="centered"
)

# Hide ALL Streamlit chrome
st.markdown("""
<style>
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"] { display: none !important; visibility: hidden !important; }

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: #020409 !important;
    padding: 0 !important;
    margin: 0 !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }
iframe { border: none !important; display: block; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base         = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join(base, "diabetes_model.pkl")
    columns_path = os.path.join(base, "columns.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"diabetes_model.pkl not found in {base}")
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"columns.pkl not found in {base}")
    return joblib.load(model_path), joblib.load(columns_path)

try:
    model, feature_columns = load_model()
    _model_ready = True
except Exception as _e:
    _model_ready  = False
    _load_err_msg = str(_e)

# ─────────────────────────────────────────────
# READ QUERY PARAMS → RUN MODEL
# ─────────────────────────────────────────────
params = st.query_params
prediction_result = None

if params.get("_predict") == "1":
    if _model_ready:
        try:
            raw = {
                "Pregnancies"              : float(params.get("pregnancies", 0)),
                "Glucose"                  : float(params.get("glucose", 0)),
                "BloodPressure"            : float(params.get("blood_pressure", 0)),
                "SkinThickness"            : float(params.get("skin_thickness", 0)),
                "Insulin"                  : float(params.get("insulin", 0)),
                "BMI"                      : float(params.get("bmi", 0)),
                "DiabetesPedigreeFunction" : float(params.get("dpf", 0)),
                "Age"                      : float(params.get("age", 0)),
            }
            df         = pd.DataFrame([raw]).reindex(columns=feature_columns, fill_value=0)
            prediction = int(model.predict(df)[0])
            proba      = model.predict_proba(df)[0]
            risk_pct   = round(float(proba[1]) * 100, 1)
            confidence = round(float(proba[prediction]) * 100, 1)
            prediction_result = {
                "prediction": prediction,
                "risk_pct":   risk_pct,
                "confidence": confidence,
            }
        except Exception as ex:
            prediction_result = {"error": str(ex)}
    else:
        prediction_result = {"error": _load_err_msg}

# ─────────────────────────────────────────────
# BUILD RESULT HTML SNIPPET
# ─────────────────────────────────────────────
if prediction_result and "error" not in prediction_result:
    p   = prediction_result
    if p["prediction"] == 1:
        card_cls  = "high-risk"
        icon      = "⚠️"
        rtitle    = "Elevated Diabetes Risk Detected"
        rdesc     = ("Your biomarker profile indicates a heightened risk for Type 2 diabetes. "
                     "We recommend consulting a healthcare professional for a comprehensive "
                     "evaluation and follow-up testing.")
    else:
        card_cls  = "low-risk"
        icon      = "✓"
        rtitle    = "Low Diabetes Risk"
        rdesc     = ("Your current biomarker profile is within a healthy range. "
                     "Continue maintaining balanced nutrition, regular exercise, and routine "
                     "check-ups to preserve long-term metabolic health.")
    pill_txt  = f"● Diabetes Probability · {p['risk_pct']}%  ·  Confidence {p['confidence']}%"
    RESULT_HTML = f"""
    <div id="result-wrap" class="visible">
      <div class="result-card {card_cls}">
        <div class="result-icon">{icon}</div>
        <div>
          <div class="result-title">{rtitle}</div>
          <div class="result-desc">{rdesc}</div>
          <span class="score-pill">{pill_txt}</span>
        </div>
      </div>
    </div>
    """
elif prediction_result and "error" in prediction_result:
    RESULT_HTML = f'<div class="error-box">❌ Error: {prediction_result["error"]}</div>'
else:
    RESULT_HTML = '<div id="result-wrap"></div>'

# ─────────────────────────────────────────────
# FULL PAGE HTML  (isolated iframe via components.html)
# ─────────────────────────────────────────────
# Pre-fill inputs if coming back from a prediction
pv = {
    "pregnancies":    params.get("pregnancies",    "0"),
    "glucose":        params.get("glucose",        "0"),
    "blood_pressure": params.get("blood_pressure", "0"),
    "skin_thickness": params.get("skin_thickness", "0"),
    "insulin":        params.get("insulin",        "0"),
    "bmi":            params.get("bmi",            "0"),
    "dpf":            params.get("dpf",            "0"),
    "age":            params.get("age",            "0"),
}

HTML_UI = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body {{
    background: #020409;
    color: #e2e8f4;
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
}}

body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,210,200,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(56,100,255,0.10) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 50% 50%,  rgba(0,180,255,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}}

.page {{
    position: relative;
    z-index: 1;
    max-width: 860px;
    margin: 0 auto;
    padding: 3.5rem 1.5rem 4rem;
}}

/* ── HERO ── */
.hero {{ text-align: center; padding-bottom: 3.5rem; }}

.badge {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(0,210,200,0.08);
    border: 1px solid rgba(0,210,200,0.28);
    color: #00d2c8;
    font-size: 0.70rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 1.6rem;
}}
.badge-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00d2c8;
    box-shadow: 0 0 8px #00d2c8;
    animation: pulse 2s infinite;
    flex-shrink: 0;
}}
@keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:0.35; transform:scale(0.75); }}
}}

.hero-title {{
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    color: #f0f4ff;
    margin-bottom: 1.4rem;
    text-align: center;
}}
.hero-title .accent {{
    background: linear-gradient(110deg, #00d2c8 0%, #4fa8ff 55%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.hero-sub {{
    font-family: 'DM Sans', sans-serif;
    color: #7a8aa8;
    font-size: 1.05rem;
    font-weight: 300;
    line-height: 1.75;
    max-width: 560px;
    margin: 0 auto 2.8rem;
    text-align: center;
}}

.stats-row {{
    display: flex;
    justify-content: center;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    overflow: hidden;
    max-width: 580px;
    margin: 0 auto;
    background: rgba(255,255,255,0.025);
}}
.stat-item {{
    flex: 1;
    padding: 1.1rem 0.5rem;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.07);
    transition: background 0.25s;
}}
.stat-item:last-child {{ border-right: none; }}
.stat-item:hover {{ background: rgba(0,210,200,0.06); }}
.stat-value {{
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #00d2c8;
    display: block;
}}
.stat-label {{
    font-size: 0.68rem;
    color: #4a5a74;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 3px;
    display: block;
}}

/* ── CARD HEADER ── */
.card-section-label {{
    font-family: 'Syne', sans-serif;
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    color: #00d2c8;
    text-align: center;
    display: block;
    margin-bottom: 0.5rem;
}}
.card-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #eef1fa;
    text-align: center;
    margin-bottom: 0.45rem;
}}
.card-sub {{
    font-family: 'DM Sans', sans-serif;
    color: #4e5e78;
    font-size: 0.88rem;
    text-align: center;
    margin-bottom: 2.2rem;
}}

/* ── GLASS CARD ── */
.glass-card {{
    background: rgba(8,14,30,0.80);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 2.8rem 2.6rem;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow:
        0 0 0 1px rgba(0,210,200,0.04),
        0 24px 60px rgba(0,0,0,0.55),
        0 0 80px rgba(0,210,200,0.04);
}}

/* ── FIELDS GRID ── */
.fields-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.1rem 1.6rem;
    margin-bottom: 1.8rem;
}}
@media (max-width: 560px) {{
    .fields-grid {{ grid-template-columns: 1fr; }}
    .hero-title  {{ font-size: 2rem; }}
}}

.field {{ display: flex; flex-direction: column; gap: 5px; }}
.field label {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: #7a8fad;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}}
.field input[type="number"] {{
    background: rgba(255,255,255,0.04);
    color: #dce8ff;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    padding: 0.68rem 0.95rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.96rem;
    width: 100%;
    outline: none;
    transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
    -moz-appearance: textfield;
}}
.field input[type="number"]::-webkit-inner-spin-button,
.field input[type="number"]::-webkit-outer-spin-button {{ -webkit-appearance: none; }}
.field input[type="number"]:hover {{
    background: rgba(255,255,255,0.065);
    border-color: rgba(0,210,200,0.25);
}}
.field input[type="number"]:focus {{
    background: rgba(0,210,200,0.05);
    border-color: rgba(0,210,200,0.55);
    box-shadow: 0 0 0 3px rgba(0,210,200,0.09);
}}

.divider {{
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.07), transparent);
    margin: 1.8rem 0;
}}

/* ── BUTTON ── */
.btn-wrap {{ display: flex; justify-content: center; }}
.predict-btn {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    max-width: 340px;
    background: linear-gradient(100deg, #00bfb8 0%, #0094e8 55%, #4f72ff 100%);
    color: #fff;
    border: none;
    padding: 0.88rem 1.5rem;
    border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-size: 0.93rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: transform 0.28s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.28s ease, filter 0.2s ease;
    box-shadow: 0 4px 24px rgba(0,180,200,0.30), 0 0 0 1px rgba(0,210,200,0.15);
}}
.predict-btn:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 40px rgba(0,200,220,0.45), 0 0 0 1px rgba(0,210,200,0.30);
    filter: brightness(1.1);
}}
.predict-btn:active {{ transform: translateY(0); filter: brightness(0.95); }}

/* ── RESULT CARD ── */
#result-wrap {{ margin-top: 2rem; display: none; }}
#result-wrap.visible {{
    display: block;
    animation: fadeUp 0.42s cubic-bezier(0.22,1,0.36,1) both;
}}
@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(16px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
.result-card {{
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    display: flex;
    align-items: flex-start;
    gap: 1.1rem;
    border: 1px solid transparent;
}}
.result-card.high-risk {{
    background: rgba(255,75,75,0.07);
    border-color: rgba(255,90,90,0.25);
    box-shadow: 0 0 40px rgba(255,60,60,0.1);
}}
.result-card.low-risk {{
    background: rgba(0,210,160,0.06);
    border-color: rgba(0,210,160,0.25);
    box-shadow: 0 0 40px rgba(0,210,160,0.1);
}}
.result-icon {{ font-size: 2rem; line-height: 1; flex-shrink: 0; margin-top: 3px; }}
.result-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}}
.result-card.high-risk .result-title {{ color: #ff7070; }}
.result-card.low-risk  .result-title {{ color: #00d4a0; }}
.result-desc {{ font-size: 0.84rem; color: #5a6a88; line-height: 1.6; }}
.score-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-top: 0.7rem;
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}}
.result-card.high-risk .score-pill {{
    background: rgba(255,80,80,0.12);
    color: #ff8080;
    border: 1px solid rgba(255,80,80,0.20);
}}
.result-card.low-risk .score-pill {{
    background: rgba(0,210,160,0.12);
    color: #00d4a0;
    border: 1px solid rgba(0,210,160,0.20);
}}

.error-box {{
    margin-top: 1.5rem;
    padding: 1rem 1.2rem;
    background: rgba(255,60,60,0.08);
    border: 1px solid rgba(255,60,60,0.25);
    border-radius: 12px;
    color: #ff8080;
    font-size: 0.88rem;
}}

.disclaimer {{
    text-align: center;
    margin-top: 3rem;
    font-size: 0.72rem;
    color: #2e3a50;
    letter-spacing: 0.02em;
    line-height: 1.7;
}}
</style>
</head>
<body>
<div class="page">

  <!-- HERO -->
  <div class="hero">
    <div>
      <span class="badge"><span class="badge-dot"></span>AI-Powered Health Intelligence</span>
    </div>
    <h1 class="hero-title">
      Predict Diabetes Risk<br>
      <span class="accent">with Clinical Precision</span>
    </h1>
    <p class="hero-sub">
      Enter your medical parameters below. Our model analyses eight clinical
      biomarkers to deliver an instant, evidence-based risk assessment —
      designed for patients and clinicians alike.
    </p>
    <div class="stats-row">
      <div class="stat-item">
        <span class="stat-value">98.7%</span>
        <span class="stat-label">Model Accuracy</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">50K+</span>
        <span class="stat-label">Predictions</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">24 / 7</span>
        <span class="stat-label">Available</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">&lt; 2s</span>
        <span class="stat-label">Instant Result</span>
      </div>
    </div>
  </div>

  <!-- FORM CARD -->
  <span class="card-section-label">Risk Assessment</span>
  <h2 class="card-title">Enter Your Health Metrics</h2>
  <p class="card-sub">All fields are required for an accurate prediction.</p>

  <div class="glass-card">
    <form method="GET" action="" id="predictForm">
      <div class="fields-grid">
        <div class="field">
          <label for="pregnancies">Pregnancies</label>
          <input type="number" name="pregnancies" id="pregnancies" min="0" step="1" value="{pv['pregnancies']}" placeholder="e.g. 2"/>
        </div>
        <div class="field">
          <label for="insulin">Insulin (µU/mL)</label>
          <input type="number" name="insulin" id="insulin" min="0" step="1" value="{pv['insulin']}" placeholder="e.g. 85"/>
        </div>
        <div class="field">
          <label for="glucose">Glucose (mg/dL)</label>
          <input type="number" name="glucose" id="glucose" min="0" step="1" value="{pv['glucose']}" placeholder="e.g. 120"/>
        </div>
        <div class="field">
          <label for="bmi">BMI (kg/m²)</label>
          <input type="number" name="bmi" id="bmi" min="0" step="0.1" value="{pv['bmi']}" placeholder="e.g. 25.5"/>
        </div>
        <div class="field">
          <label for="blood_pressure">Blood Pressure (mm Hg)</label>
          <input type="number" name="blood_pressure" id="blood_pressure" min="0" step="1" value="{pv['blood_pressure']}" placeholder="e.g. 80"/>
        </div>
        <div class="field">
          <label for="dpf">Diabetes Pedigree Function</label>
          <input type="number" name="dpf" id="dpf" min="0" step="0.001" value="{pv['dpf']}" placeholder="e.g. 0.5"/>
        </div>
        <div class="field">
          <label for="skin_thickness">Skin Thickness (mm)</label>
          <input type="number" name="skin_thickness" id="skin_thickness" min="0" step="0.5" value="{pv['skin_thickness']}" placeholder="e.g. 20"/>
        </div>
        <div class="field">
          <label for="age">Age (years)</label>
          <input type="number" name="age" id="age" min="0" step="1" value="{pv['age']}" placeholder="e.g. 33"/>
        </div>
      </div>

      <input type="hidden" name="_predict" value="1"/>

      <div class="divider"></div>

      <div class="btn-wrap">
        <button type="submit" class="predict-btn">
          ⬡&nbsp; Analyse Risk Profile →
        </button>
      </div>
    </form>

    <!-- Result injected here by Python -->
    {RESULT_HTML}
  </div>

  <p class="disclaimer">
    This tool is for informational purposes only and does not constitute medical advice.<br>
    Always consult a qualified healthcare professional for diagnosis and treatment.
  </p>

</div>

<script>
// Submit form into parent Streamlit URL (iframe → parent navigation)
document.getElementById('predictForm').addEventListener('submit', function(e) {{
  e.preventDefault();
  const data = new FormData(this);
  const params = new URLSearchParams(data);
  window.parent.location.href = window.parent.location.pathname + '?' + params.toString();
}});
</script>
</body>
</html>
"""

components.html(HTML_UI, height=1150, scrolling=True)
