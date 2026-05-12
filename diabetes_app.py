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
# GLOBAL CSS — forces dark theme on ALL wrappers
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #020409 !important;
    color: #e2e8f4 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Mesh background */
[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,210,200,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(56,100,255,0.10) 0%, transparent 55%);
    pointer-events: none;
    z-index: 0;
}

.block-container {
    padding: 3rem 1.5rem 4rem !important;
    max-width: 860px !important;
    position: relative;
    z-index: 1;
    background: transparent !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── HERO HTML elements ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(0,210,200,0.08);
    border: 1px solid rgba(0,210,200,0.28);
    color: #00d2c8;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.70rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
}
.badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00d2c8;
    box-shadow: 0 0 8px #00d2c8;
    animation: bpulse 2s infinite;
    flex-shrink: 0;
}
@keyframes bpulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.35; transform:scale(0.75); }
}

.hero-wrap  { text-align: center; padding-bottom: 2.5rem; }

.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.4rem, 5vw, 3.8rem) !important;
    font-weight: 800 !important;
    line-height: 1.08 !important;
    letter-spacing: -0.03em !important;
    color: #f0f4ff !important;
    -webkit-text-fill-color: #f0f4ff !important;
    margin: 1.2rem 0 !important;
    text-align: center !important;
}
.hero-title .accent {
    background: linear-gradient(110deg, #00d2c8 0%, #4fa8ff 55%, #a78bfa 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
.hero-sub {
    color: #7a8aa8 !important;
    font-size: 1.05rem !important;
    font-weight: 300 !important;
    line-height: 1.75 !important;
    max-width: 560px;
    margin: 0 auto 2.4rem !important;
    text-align: center !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Stats row */
.stats-row {
    display: flex;
    justify-content: center;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    overflow: hidden;
    max-width: 580px;
    margin: 0 auto;
    background: rgba(255,255,255,0.025);
}
.stat-item {
    flex: 1;
    padding: 1.1rem 0.5rem;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.07);
}
.stat-item:last-child { border-right: none; }
.stat-value {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.35rem;
    font-weight: 700;
    color: #00d2c8 !important;
    display: block;
}
.stat-label {
    font-size: 0.68rem;
    color: #4a5a74 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 3px;
    display: block;
}

/* ── FORM SECTION LABELS ── */
.form-sec-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    color: #00d2c8 !important;
    text-align: center;
    display: block;
    margin-bottom: 0.4rem;
}
.form-main-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.75rem;
    font-weight: 700;
    color: #eef1fa !important;
    text-align: center;
    margin-bottom: 0.4rem;
}
.form-sub-text {
    color: #4e5e78 !important;
    font-size: 0.88rem;
    text-align: center;
    margin-bottom: 1.5rem;
    font-family: 'DM Sans', sans-serif;
}

/* ── INPUT LABELS ── */
[data-testid="stNumberInput"] label,
label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #7a8fad !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* ── INPUT FIELDS ── */
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    background-color: rgba(255,255,255,0.04) !important;
    color: #dce8ff !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.96rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(0,210,200,0.55) !important;
    box-shadow: 0 0 0 3px rgba(0,210,200,0.09) !important;
    background: rgba(0,210,200,0.05) !important;
}

/* ── STEPPER BUTTONS ── */
[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #7a8fad !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(0,210,200,0.12) !important;
    color: #00d2c8 !important;
}

/* ── PREDICT BUTTON ── */
[data-testid="stFormSubmitButton"] button,
.stButton > button {
    width: 100% !important;
    background: linear-gradient(100deg, #00bfb8 0%, #0094e8 55%, #4f72ff 100%) !important;
    background-color: #00bfb8 !important;
    color: #fff !important;
    border: none !important;
    padding: 0.9rem 1.2rem !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.93rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 4px 24px rgba(0,180,200,0.30) !important;
    transition: all 0.28s cubic-bezier(0.34,1.56,0.64,1) !important;
    cursor: pointer !important;
}
[data-testid="stFormSubmitButton"] button:hover,
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 40px rgba(0,200,220,0.45) !important;
    filter: brightness(1.1) !important;
}

/* ── FORM BORDER HIDE ── */
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}

/* ── GLASS CARD ── */
.glass-card {
    background: rgba(8,14,30,0.80);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 2.4rem 2.2rem;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow:
        0 0 0 1px rgba(0,210,200,0.04),
        0 24px 60px rgba(0,0,0,0.55),
        0 0 80px rgba(0,210,200,0.04);
    margin-bottom: 1rem;
}

/* ── DIVIDER ── */
.form-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.07), transparent);
    margin: 1.5rem 0;
}

/* ── RESULT CARD ── */
.result-card {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    display: flex;
    align-items: flex-start;
    gap: 1.1rem;
    border: 1px solid transparent;
    margin-top: 1.5rem;
    animation: fadeUp 0.42s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-card.high-risk {
    background: rgba(255,75,75,0.07);
    border-color: rgba(255,90,90,0.25);
    box-shadow: 0 0 40px rgba(255,60,60,0.1);
}
.result-card.low-risk {
    background: rgba(0,210,160,0.06);
    border-color: rgba(0,210,160,0.25);
    box-shadow: 0 0 40px rgba(0,210,160,0.1);
}
.result-icon { font-size: 2rem; line-height: 1; flex-shrink: 0; margin-top: 3px; }
.result-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-card.high-risk .result-title { color: #ff7070 !important; }
.result-card.low-risk  .result-title { color: #00d4a0 !important; }
.result-desc { font-size: 0.84rem; color: #5a6a88 !important; line-height: 1.6; }
.score-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-top: 0.7rem;
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.result-card.high-risk .score-pill {
    background: rgba(255,80,80,0.12);
    color: #ff8080;
    border: 1px solid rgba(255,80,80,0.20);
}
.result-card.low-risk .score-pill {
    background: rgba(0,210,160,0.12);
    color: #00d4a0;
    border: 1px solid rgba(0,210,160,0.20);
}

/* ── DISCLAIMER ── */
.disclaimer {
    text-align: center;
    margin-top: 2.5rem;
    font-size: 0.72rem;
    color: #2e3a50 !important;
    letter-spacing: 0.02em;
    line-height: 1.7;
}

/* ── CENTER BUTTON WRAPPER ── */
[data-testid="stFormSubmitButton"] {
    display: flex !important;
    justify-content: center !important;
}
[data-testid="stFormSubmitButton"] button {
    max-width: 340px !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <span class="badge"><span class="badge-dot"></span>AI-Powered Health Intelligence</span>
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
      <span class="stat-value">77.28%</span>
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
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# FORM CARD
# ═══════════════════════════════════════════════
st.markdown("""
<span class="form-sec-label">Risk Assessment</span>
<h2 class="form-main-title">Enter Your Health Metrics</h2>
<p class="form-sub-text">All fields are required for an accurate prediction.</p>
""", unsafe_allow_html=True)


# ── Streamlit native form (reliable on Cloud) ──
with st.form("predict_form"):
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        pregnancies    = st.number_input("Pregnancies",               min_value=0.0, step=1.0,   format="%.0f")
        glucose        = st.number_input("Glucose (mg/dL)",           min_value=0.0, step=1.0,   format="%.0f")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)",    min_value=0.0, step=1.0,   format="%.0f")
        skin_thickness = st.number_input("Skin Thickness (mm)",       min_value=0.0, step=0.5,   format="%.1f")

    with col2:
        insulin        = st.number_input("Insulin (µU/mL)",           min_value=0.0, step=1.0,   format="%.0f")
        bmi            = st.number_input("BMI (kg/m²)",               min_value=0.0, step=0.1,   format="%.1f")
        dpf            = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.001, format="%.3f")
        age            = st.number_input("Age (years)",                min_value=0.0, step=1.0,   format="%.0f")

    st.markdown('<div class="form-divider"></div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("⬡  Analyse Risk Profile →")


# ═══════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════
if submitted:
    if not _model_ready:
        st.error(f"Model could not be loaded: {_load_err_msg}", icon="🚨")
    else:
        try:
            raw = {
                "Pregnancies"              : pregnancies,
                "Glucose"                  : glucose,
                "BloodPressure"            : blood_pressure,
                "SkinThickness"            : skin_thickness,
                "Insulin"                  : insulin,
                "BMI"                      : bmi,
                "DiabetesPedigreeFunction" : dpf,
                "Age"                      : age,
            }
            df         = pd.DataFrame([raw]).reindex(columns=feature_columns, fill_value=0)
            prediction = int(model.predict(df)[0])
            proba      = model.predict_proba(df)[0]
            risk_pct   = round(float(proba[1]) * 100, 1)
            confidence = round(float(proba[prediction]) * 100, 1)

            if prediction == 1:
                card_cls = "high-risk"
                icon     = "⚠️"
                rtitle   = "Elevated Diabetes Risk Detected"
                rdesc    = ("Your biomarker profile indicates a heightened risk for Type 2 diabetes. "
                            "We recommend consulting a healthcare professional for a comprehensive "
                            "evaluation and follow-up testing.")
            else:
                card_cls = "low-risk"
                icon     = "✓"
                rtitle   = "Low Diabetes Risk"
                rdesc    = ("Your current biomarker profile is within a healthy range. "
                            "Continue maintaining balanced nutrition, regular exercise, and routine "
                            "check-ups to preserve long-term metabolic health.")

            pill = f"● Diabetes Probability · {risk_pct}%  ·  Confidence {confidence}%"

            st.markdown(f"""
<div class="result-card {card_cls}">
  <div class="result-icon">{icon}</div>
  <div>
    <div class="result-title">{rtitle}</div>
    <div class="result-desc">{rdesc}</div>
    <span class="score-pill">{pill}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        except Exception as ex:
            st.error(f"Prediction error: {ex}", icon="🚨")

# ═══════════════════════════════════════════════
# DISCLAIMER
# ═══════════════════════════════════════════════
st.markdown("""
<p class="disclaimer">
  This tool is for informational purposes only and does not constitute medical advice.<br>
  Always consult a qualified healthcare professional for diagnosis and treatment.
</p>
""", unsafe_allow_html=True)
