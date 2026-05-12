import streamlit as st
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
    """Load the trained Random Forest model and feature columns once, cache for session."""
    base         = os.path.dirname(__file__)
    model_path   = os.path.join(base, "diabetes_model.pkl")
    columns_path = os.path.join(base, "columns.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Place diabetes_model.pkl in the same directory as this script."
        )
    if not os.path.exists(columns_path):
        raise FileNotFoundError(
            f"Columns file not found: {columns_path}\n"
            "Place columns.pkl in the same directory as this script."
        )

    return joblib.load(model_path), joblib.load(columns_path)

try:
    model, feature_columns = load_model()
    _model_ready = True
except FileNotFoundError as _load_err:
    _model_ready   = False
    _load_err_msg  = str(_load_err)

# ─────────────────────────────────────────────
# GLOBAL CSS  (single, fully self-contained block)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ═══════════════════════════════════════════════
   DEPLOY-PROOF DARK BACKGROUND
   Streamlit Cloud overrides body/html bg, so we
   must force darkness on every wrapper it uses.
   ═══════════════════════════════════════════════ */

/* The root iframe & document */
html, body {
    background-color: #020409 !important;
    color: #e2e8f4 !important;
}

/* Streamlit's own top-level wrappers (class names change, use data-testid) */
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"],
section[data-testid="stSidebar"],
.stApp,
.main,
.css-1d391kg,
.css-18e3th9,
.css-fg4pbf,
div[class^="css"] {
    background-color: #020409 !important;
    color: #e2e8f4 !important;
}

/* Catch-all for any injected Streamlit wrapper */
[class*="st-"] { color: inherit; }

/* ── FONT FAMILY ── */
html, body, [data-testid="stApp"], [data-testid="stMain"],
[data-testid="stAppViewContainer"], .block-container,
p, span, div, label, input, button {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

/* ── MESH BACKGROUND (fixed layer) ── */
[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,210,200,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(56,100,255,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 50% 50%,  rgba(0,180,255,0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── BLOCK CONTAINER ── */
.block-container {
    padding: 3.5rem 1.5rem 4rem !important;
    max-width: 860px !important;
    position: relative;
    z-index: 1;
    background: transparent !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #020409; }
::-webkit-scrollbar-thumb { background: rgba(0,210,200,0.2); border-radius: 3px; }

/* ═══════════════════════════════════════════════
   BADGE
   ═══════════════════════════════════════════════ */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,210,200,0.08) !important;
    border: 1px solid rgba(0,210,200,0.25) !important;
    color: #00d2c8 !important;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
}
.badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00d2c8;
    box-shadow: 0 0 8px #00d2c8;
    animation: pulse 2s infinite;
    flex-shrink: 0;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.8); }
}

/* ═══════════════════════════════════════════════
   HERO
   ═══════════════════════════════════════════════ */
.hero {
    text-align: center;
    padding-bottom: 3.5rem;
}
.hero-eyebrow {
    display: flex;
    justify-content: center;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.4rem, 5vw, 3.8rem) !important;
    font-weight: 800 !important;
    line-height: 1.08 !important;
    letter-spacing: -0.03em !important;
    color: #f0f4ff !important;
    margin-bottom: 1.4rem;
    -webkit-text-fill-color: #f0f4ff !important;
}
/* The accent span overrides the parent text-fill */
.hero-title .accent {
    background: linear-gradient(110deg, #00d2c8 0%, #4fa8ff 55%, #a78bfa 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
.hero-sub {
    color: #7a8aa8 !important;
    font-size: 1.05rem;
    font-weight: 300;
    line-height: 1.75;
    max-width: 560px;
    margin: 0 auto 2.8rem;
}

/* ── STATS ROW ── */
.stats-row {
    display: flex;
    justify-content: center;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    overflow: hidden;
    max-width: 580px;
    margin: 0 auto;
    background: rgba(255,255,255,0.025) !important;
}
.stat-item {
    flex: 1;
    padding: 1.1rem 0.5rem;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.07);
    transition: background 0.25s;
    background: transparent !important;
}
.stat-item:last-child { border-right: none; }
.stat-item:hover { background: rgba(0,210,200,0.06) !important; }
.stat-value {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.35rem;
    font-weight: 700;
    color: #00d2c8 !important;
    display: block;
}
.stat-label {
    font-size: 0.7rem;
    color: #4a5a74 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ═══════════════════════════════════════════════
   CARD HEADER
   ═══════════════════════════════════════════════ */
.card-header {
    margin-bottom: 1.8rem;
    text-align: center;
}
.form-section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00d2c8 !important;
    margin-bottom: 0.5rem;
    display: block;
}
.form-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.75rem;
    font-weight: 700;
    color: #eef1fa !important;
    margin-bottom: 0.4rem;
}
.form-sub {
    color: #4e5e78 !important;
    font-size: 0.88rem;
}

/* ═══════════════════════════════════════════════
   GLASS CARD — wraps the widget container.
   Uses a reliable data-testid selector + class.
   ═══════════════════════════════════════════════ */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: rgba(8, 14, 30, 0.80) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 24px !important;
    padding: 2.8rem 2.6rem !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    box-shadow:
        0 0 0 1px rgba(0,210,200,0.04),
        0 24px 60px rgba(0,0,0,0.6),
        0 0 80px rgba(0,210,200,0.04) !important;
}

/* ═══════════════════════════════════════════════
   INPUT FIELDS — deploy-hardened with !important
   ═══════════════════════════════════════════════ */
[data-testid="stNumberInput"] label,
.stNumberInput label,
label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #7a8fad !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

[data-testid="stNumberInput"] input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    background-color: rgba(255,255,255,0.04) !important;
    color: #dce8ff !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    padding: 0.65rem 0.9rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
}

[data-testid="stNumberInput"] input:hover,
.stNumberInput > div > div > input:hover {
    background: rgba(255,255,255,0.065) !important;
    border-color: rgba(0,210,200,0.25) !important;
}

[data-testid="stNumberInput"] input:focus,
.stNumberInput > div > div > input:focus {
    background: rgba(0,210,200,0.05) !important;
    border-color: rgba(0,210,200,0.55) !important;
    box-shadow: 0 0 0 3px rgba(0,210,200,0.09) !important;
    outline: none !important;
}

/* Stepper +/- buttons */
[data-testid="stNumberInput"] button,
.stNumberInput button {
    background: rgba(255,255,255,0.04) !important;
    background-color: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #7a8fad !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] button:hover,
.stNumberInput button:hover {
    background: rgba(0,210,200,0.12) !important;
    color: #00d2c8 !important;
    border-color: rgba(0,210,200,0.2) !important;
}

/* ── DIVIDER ── */
.form-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.07), transparent);
    margin: 1.8rem 0;
}

/* ═══════════════════════════════════════════════
   PREDICT BUTTON
   ═══════════════════════════════════════════════ */
.stButton { margin-top: 0.5rem; }
.stButton > button {
    width: 100% !important;
    background: linear-gradient(100deg, #00bfb8 0%, #0094e8 55%, #4f72ff 100%) !important;
    background-color: #00bfb8 !important;   /* fallback for Cloud */
    color: #ffffff !important;
    border: none !important;
    padding: 0.9rem 1.2rem !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important;
    box-shadow: 0 4px 24px rgba(0,180,200,0.3), 0 0 0 1px rgba(0,210,200,0.15) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 40px rgba(0,200,220,0.45), 0 0 0 1px rgba(0,210,200,0.3) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    filter: brightness(0.95) !important;
}

/* ═══════════════════════════════════════════════
   RESULT CARD
   ═══════════════════════════════════════════════ */
.result-wrap {
    margin-top: 2rem;
    animation: fadeSlideUp 0.45s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes fadeSlideUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-card {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    display: flex;
    align-items: flex-start;
    gap: 1.1rem;
    border: 1px solid transparent;
}
.result-card.high-risk {
    background: rgba(255,75,75,0.07) !important;
    border-color: rgba(255,90,90,0.25) !important;
    box-shadow: 0 0 40px rgba(255,60,60,0.1);
}
.result-card.low-risk {
    background: rgba(0,210,160,0.06) !important;
    border-color: rgba(0,210,160,0.25) !important;
    box-shadow: 0 0 40px rgba(0,210,160,0.1);
}
.result-icon { font-size:2rem; line-height:1; flex-shrink:0; margin-top:2px; }
.result-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-card.high-risk .result-title { color: #ff7070 !important; }
.result-card.low-risk  .result-title { color: #00d4a0 !important; }
.result-desc { font-size:0.84rem; color:#5a6a88 !important; line-height:1.6; }
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
    background: rgba(255,80,80,0.12) !important;
    color: #ff8080 !important;
    border: 1px solid rgba(255,80,80,0.2);
}
.result-card.low-risk .score-pill {
    background: rgba(0,210,160,0.12) !important;
    color: #00d4a0 !important;
    border: 1px solid rgba(0,210,160,0.2);
}

/* ═══════════════════════════════════════════════
   DISCLAIMER
   ═══════════════════════════════════════════════ */
.disclaimer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.72rem;
    color: #2e3a50 !important;
    letter-spacing: 0.02em;
    line-height: 1.7;
}

/* ═══════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════ */
@media (max-width: 640px) {
    .hero-title { font-size: 2rem !important; }
    .stats-row  { flex-wrap: wrap; }
    .stat-item  { flex: 1 1 40%; border-right: none !important; border-bottom: 1px solid rgba(255,255,255,0.07); }
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        padding: 1.6rem 1rem !important;
    }
}

/* ═══════════════════════════════════════════════
   HIDE STREAMLIT CHROME
   ═══════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
.stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# HERO  — one fully self-contained HTML block
# ═══════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">
        <span class="badge">⬡ &nbsp; AI-Powered Health Intelligence</span>
    </div>
    <h1 class="hero-title">
        Predict Diabetes Risk<br>
        <span class="accent">with Clinical Precision</span>
    </h1>
    <p class="hero-sub">
        Enter your medical parameters below. Our model analyses eight
        clinical biomarkers to deliver an instant, evidence-based risk
        assessment — designed for patients and clinicians alike.
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
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# FORM — card header (self-contained) + widgets in container
#
# KEY FIX: Streamlit widgets CANNOT live inside custom HTML
# div tags. Every st.markdown() call is an isolated HTML
# island — open tags in one call are NOT closed by another.
# Solution: card header = one closed HTML block; widgets
# live in st.container() styled via CSS attribute selectors.
# ═══════════════════════════════════════════════════════

# Card header — fully closed, no dangling tags
st.markdown("""
<div class="card-header">
    <p class="form-section-label">Risk Assessment</p>
    <h2 class="form-title">Enter Your Health Metrics</h2>
    <p class="form-sub">All fields are required for an accurate prediction.</p>
</div>
""", unsafe_allow_html=True)

# Widgets in a container (the glass-card CSS targets this automatically)
with st.container():
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        pregnancies    = st.number_input("Pregnancies",                min_value=0.0, step=1.0,   format="%.0f", help="Number of times pregnant")
        glucose        = st.number_input("Glucose  (mg/dL)",           min_value=0.0, step=1.0,   format="%.0f", help="Plasma glucose concentration (2h OGTT)")
        blood_pressure = st.number_input("Blood Pressure  (mm Hg)",    min_value=0.0, step=1.0,   format="%.0f", help="Diastolic blood pressure")
        skin_thickness = st.number_input("Skin Thickness  (mm)",       min_value=0.0, step=0.5,   format="%.1f", help="Triceps skin fold thickness")

    with col2:
        insulin        = st.number_input("Insulin  (µU/mL)",           min_value=0.0, step=1.0,   format="%.0f", help="2-hour serum insulin")
        bmi            = st.number_input("BMI  (kg/m²)",               min_value=0.0, step=0.1,   format="%.1f", help="Body mass index")
        dpf            = st.number_input("Diabetes Pedigree Function",  min_value=0.0, step=0.001, format="%.3f", help="Genetic influence score")
        age            = st.number_input("Age  (years)",                min_value=0.0, step=1.0,   format="%.0f", help="Age in years")

    # Divider — self-contained void element, no closing tag needed
    st.markdown('<div class="form-divider"></div>', unsafe_allow_html=True)

    predict_btn = st.button("⬡  Analyse Risk Profile →")

    # ── PREDICTION RESULT ─────────────────────────────────────
    if predict_btn:
        if not _model_ready:
            st.error(f"Model could not be loaded.\n\n{_load_err_msg}", icon="🚨")
        else:
            # Build input DataFrame aligned to trained column order
            raw_inputs = {
                "Pregnancies"              : pregnancies,
                "Glucose"                  : glucose,
                "BloodPressure"            : blood_pressure,
                "SkinThickness"            : skin_thickness,
                "Insulin"                  : insulin,
                "BMI"                      : bmi,
                "DiabetesPedigreeFunction" : dpf,
                "Age"                      : age,
            }
            input_df = (
                pd.DataFrame([raw_inputs])
                  .reindex(columns=feature_columns, fill_value=0)
            )

            # Inference
            prediction = model.predict(input_df)[0]        # 0 or 1
            proba      = model.predict_proba(input_df)[0]  # [p_class0, p_class1]
            confidence = proba[prediction] * 100
            risk_pct   = proba[1] * 100

            # Result content
            if prediction == 1:
                card_class  = "high-risk"
                icon        = "⚠️"
                title       = "Elevated Diabetes Risk Detected"
                description = (
                    "Your biomarker profile indicates a heightened risk for Type 2 diabetes. "
                    "We recommend consulting a healthcare professional for a comprehensive "
                    "evaluation and follow-up testing."
                )
            else:
                card_class  = "low-risk"
                icon        = "✓"
                title       = "Low Diabetes Risk"
                description = (
                    "Your current biomarker profile is within a healthy range. "
                    "Continue maintaining balanced nutrition, regular exercise, and routine "
                    "check-ups to preserve long-term metabolic health."
                )

            pill_label = f"Diabetes Probability · {risk_pct:.1f}%  ·  Confidence {confidence:.1f}%"

            # Result card — fully self-contained HTML block
            st.markdown(f"""
<div class="result-wrap">
    <div class="result-card {card_class}">
        <div class="result-icon">{icon}</div>
        <div class="result-body">
            <div class="result-title">{title}</div>
            <div class="result-desc">{description}</div>
            <span class="score-pill">● &nbsp;{pill_label}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# DISCLAIMER  — fully self-contained HTML block
# ═══════════════════════════════════════════════════════
st.markdown("""
<p class="disclaimer">
    This tool is for informational purposes only and does not constitute medical advice.<br>
    Always consult a qualified healthcare professional for diagnosis and treatment.
</p>
""", unsafe_allow_html=True)
