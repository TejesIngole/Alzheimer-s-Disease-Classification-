import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }

    /* Main background */
    .stApp { background: #0f1117; color: #e8e8e8; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #2a3047;
    }

    /* Cards */
    .card {
        background: #161b27;
        border: 1px solid #2a3047;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.2rem;
    }
    .section-header {
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #6b7fa3;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }

    /* Risk badge */
    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-high   { background:#4a0e0e; color:#ff6b6b; border:1px solid #ff6b6b44; }
    .badge-medium { background:#4a3800; color:#ffc742; border:1px solid #ffc74244; }
    .badge-low    { background:#0a3a1e; color:#4ccd8e; border:1px solid #4ccd8e44; }

    /* Metric boxes */
    .metric-box {
        background: #1e2536;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.72rem; color: #6b7fa3; text-transform: uppercase; letter-spacing: 0.1em; }

    /* Gauge bar */
    .gauge-bg {
        background: #2a3047;
        border-radius: 999px;
        height: 14px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }

    /* Footer */
    .footer { text-align:center; color:#3b4667; font-size:0.72rem; margin-top:3rem; }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Load model & scaler
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️  Could not load model files: {e}")

SELECTED_FEATURES = [
    'Age','Gender','Ethnicity','BMI','Smoking','AlcoholConsumption',
    'MMSE','ADL','FunctionalAssessment','MemoryComplaints',
    'Confusion','Disorientation','DifficultyCompletingTasks',
    'Forgetfulness','BehavioralProblems','PersonalityChanges',
    'SleepQuality','FamilyHistoryAlzheimers','Hypertension',
    'CardiovascularDisease','Diabetes','Depression','HeadInjury'
]

# ─────────────────────────────────────────────
#  Sidebar — input form
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Patient Data Entry")
    st.markdown("---")

    st.markdown("**👤 Demographics**")
    age        = st.slider("Age",        30, 90, 65)
    gender     = st.selectbox("Gender",  ["Male (0)", "Female (1)"])
    ethnicity  = st.selectbox("Ethnicity", ["Caucasian (0)","African American (1)","Asian (2)","Other (3)"])
    bmi        = st.slider("BMI",        15.0, 45.0, 25.0, 0.1)

    st.markdown("---")
    st.markdown("**🏃 Lifestyle**")
    smoking    = st.radio("Smoking",          ["No (0)", "Yes (1)"], horizontal=True)
    alcohol    = st.slider("Alcohol Consumption (units/week)", 0.0, 20.0, 5.0, 0.5)
    sleep      = st.slider("Sleep Quality (0-10)", 0.0, 10.0, 6.0, 0.5)

    st.markdown("---")
    st.markdown("**🩺 Medical History**")
    family_hx  = st.radio("Family History of Alzheimer's",  ["No (0)", "Yes (1)"], horizontal=True)
    hypertension = st.radio("Hypertension",   ["No (0)", "Yes (1)"], horizontal=True)
    cardio     = st.radio("Cardiovascular Disease", ["No (0)", "Yes (1)"], horizontal=True)
    diabetes   = st.radio("Diabetes",         ["No (0)", "Yes (1)"], horizontal=True)
    depression = st.radio("Depression",       ["No (0)", "Yes (1)"], horizontal=True)
    head_inj   = st.radio("Head Injury",      ["No (0)", "Yes (1)"], horizontal=True)

    st.markdown("---")
    st.markdown("**🧩 Cognitive Assessments**")
    mmse       = st.slider("MMSE Score (0–30)",          0,  30, 20)
    adl        = st.slider("ADL Score (0–10)",            0,  10,  6)
    func_ass   = st.slider("Functional Assessment (0–10)",0,  10,  6)
    mem_comp   = st.radio("Memory Complaints",           ["No (0)", "Yes (1)"], horizontal=True)
    confusion  = st.radio("Confusion",                   ["No (0)", "Yes (1)"], horizontal=True)
    disorien   = st.radio("Disorientation",              ["No (0)", "Yes (1)"], horizontal=True)
    diff_task  = st.radio("Difficulty Completing Tasks", ["No (0)", "Yes (1)"], horizontal=True)
    forgetful  = st.radio("Forgetfulness",               ["No (0)", "Yes (1)"], horizontal=True)
    behav_prob = st.radio("Behavioral Problems",         ["No (0)", "Yes (1)"], horizontal=True)
    pers_change= st.radio("Personality Changes",         ["No (0)", "Yes (1)"], horizontal=True)

    st.markdown("---")
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  Helper: extract numeric from radio/select
# ─────────────────────────────────────────────
def num(s):
    return int(s.split("(")[1].replace(")",""))

# ─────────────────────────────────────────────
#  Main area
# ─────────────────────────────────────────────
st.markdown("# 🧠 Alzheimer's Disease Risk Predictor")
st.markdown("*An XGBoost-powered clinical decision support tool*")
st.markdown("---")

# ── Info cards row ──
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="card">
        <div class="section-header">Model</div>
        <div style="font-size:1.3rem;font-weight:700;">XGBoost Classifier</div>
        <div style="color:#6b7fa3;font-size:0.85rem;margin-top:0.3rem;">Trained on 2,149 patient records</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="card">
        <div class="section-header">Features Used</div>
        <div style="font-size:1.3rem;font-weight:700;">23 Clinical Features</div>
        <div style="color:#6b7fa3;font-size:0.85rem;margin-top:0.3rem;">Demographics, lifestyle, cognitive & medical</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="card">
        <div class="section-header">Task</div>
        <div style="font-size:1.3rem;font-weight:700;">Binary Classification</div>
        <div style="color:#6b7fa3;font-size:0.85rem;margin-top:0.3rem;">Alzheimer's present vs. not detected</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────
if predict_btn and model_loaded:

    patient = {
        'Age': age, 'Gender': num(gender), 'Ethnicity': num(ethnicity),
        'BMI': bmi, 'Smoking': num(smoking), 'AlcoholConsumption': alcohol,
        'MMSE': mmse, 'ADL': adl, 'FunctionalAssessment': func_ass,
        'MemoryComplaints': num(mem_comp), 'Confusion': num(confusion),
        'Disorientation': num(disorien), 'DifficultyCompletingTasks': num(diff_task),
        'Forgetfulness': num(forgetful), 'BehavioralProblems': num(behav_prob),
        'PersonalityChanges': num(pers_change), 'SleepQuality': sleep,
        'FamilyHistoryAlzheimers': num(family_hx), 'Hypertension': num(hypertension),
        'CardiovascularDisease': num(cardio), 'Diabetes': num(diabetes),
        'Depression': num(depression), 'HeadInjury': num(head_inj)
    }

    df_input = pd.DataFrame([patient])[SELECTED_FEATURES]
    df_scaled = scaler.transform(df_input)
    pred  = int(model.predict(df_scaled)[0])
    proba = float(model.predict_proba(df_scaled)[0][1])
    healthy_prob = 1 - proba

    # Risk tier
    if proba >= 0.75:
        risk_label, badge_cls, risk_color = "Very High Risk", "badge-high", "#ff6b6b"
    elif proba >= 0.50:
        risk_label, badge_cls, risk_color = "Moderate Risk", "badge-medium", "#ffc742"
    elif proba >= 0.25:
        risk_label, badge_cls, risk_color = "Low-Moderate Risk", "badge-medium", "#ffc742"
    else:
        risk_label, badge_cls, risk_color = "Low Risk", "badge-low", "#4ccd8e"

    st.markdown("## 📊 Prediction Results")

    col_res, col_gauge = st.columns([1, 1])

    with col_res:
        diag_text = "Alzheimer's Detected" if pred == 1 else "No Alzheimer's Detected"
        diag_icon = "🔴" if pred == 1 else "🟢"

        st.markdown(f"""
        <div class="card">
            <div class="section-header">Diagnosis</div>
            <div style="font-size:1.6rem;font-weight:700;margin-bottom:0.6rem;">
                {diag_icon} {diag_text}
            </div>
            <span class="badge {badge_cls}">{risk_label}</span>
            <hr style="border-color:#2a3047;margin:1rem 0">
            <div style="display:flex;gap:1.5rem;">
                <div>
                    <div class="metric-value" style="color:{risk_color};">{proba:.1%}</div>
                    <div class="metric-label">Alzheimer's Probability</div>
                </div>
                <div>
                    <div class="metric-value" style="color:#4ccd8e;">{healthy_prob:.1%}</div>
                    <div class="metric-label">Healthy Probability</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_gauge:
        st.markdown(f"""
        <div class="card">
            <div class="section-header">Risk Gauge</div>
            <div style="margin-bottom:1rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#6b7fa3;">
                    <span>Low</span><span>Moderate</span><span>High</span>
                </div>
                <div class="gauge-bg">
                    <div class="gauge-fill" style="width:{proba*100:.1f}%;background:{risk_color};"></div>
                </div>
                <div style="text-align:center;font-size:1.4rem;font-weight:700;color:{risk_color};margin-top:0.5rem;">
                    {proba*100:.1f}%
                </div>
            </div>
            <hr style="border-color:#2a3047;">
            <div class="section-header" style="margin-top:0.8rem;">Clinical Recommendation</div>
            <div style="font-size:0.9rem;color:#b0bec5;">
        """, unsafe_allow_html=True)

        if proba > 0.75:
            st.markdown("🚨 **Immediate medical follow-up recommended.** Comprehensive neurological evaluation advised.")
        elif proba > 0.50:
            st.markdown("⚠️ **Monitoring and cognitive screening advised.** Schedule follow-up within 3 months.")
        elif proba > 0.25:
            st.markdown("📋 **Routine monitoring suggested.** Annual cognitive assessment recommended.")
        else:
            st.markdown("✅ **No immediate cognitive concern.** Continue healthy lifestyle practices.")

        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Feature contribution chart ──
    st.markdown("### 🔍 Input Feature Summary")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#161b27")
    ax.set_facecolor("#161b27")

    feat_vals = [patient[f] for f in SELECTED_FEATURES]
    colors_bar = [risk_color if v > 0 else "#4ccd8e" for v in feat_vals]

    bars = ax.barh(SELECTED_FEATURES, feat_vals, color=colors_bar, alpha=0.85, height=0.6)
    ax.set_xlabel("Feature Value", color="#6b7fa3")
    ax.set_title("Patient Feature Values", color="#e8e8e8", fontsize=13, pad=12)
    ax.tick_params(colors="#b0bec5", labelsize=8)
    ax.spines[:].set_color("#2a3047")
    ax.xaxis.label.set_color("#6b7fa3")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

elif predict_btn and not model_loaded:
    st.error("Model files not found. Please ensure `best_model.pkl` and `scaler.pkl` are in the same directory.")

else:
    # Default: show feature info
    st.markdown("### 📋 Feature Groups Used for Prediction")

    g1, g2, g3 = st.columns(3)
    groups = {
        "👤 Demographics & Lifestyle": ['Age','Gender','Ethnicity','BMI','Smoking','AlcoholConsumption','SleepQuality'],
        "🩺 Medical History": ['FamilyHistoryAlzheimers','Hypertension','CardiovascularDisease','Diabetes','Depression','HeadInjury'],
        "🧩 Cognitive & Functional": ['MMSE','ADL','FunctionalAssessment','MemoryComplaints','Confusion','Disorientation','DifficultyCompletingTasks','Forgetfulness','BehavioralProblems','PersonalityChanges'],
    }
    for col, (title, feats) in zip([g1, g2, g3], groups.items()):
        with col:
            items = "".join([f"<li style='margin:0.2rem 0;color:#b0bec5;font-size:0.85rem;'>{f}</li>" for f in feats])
            st.markdown(f"""
            <div class="card">
                <div class="section-header">{title}</div>
                <ul style="padding-left:1.2rem;margin:0;">{items}</ul>
            </div>""", unsafe_allow_html=True)

    st.info("👈  Fill in the patient details in the sidebar and click **Run Prediction** to get the diagnosis.")

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ⚠️ This tool is for educational/research purposes only. It is not a substitute for professional medical diagnosis.<br>
    Built with XGBoost · Streamlit · scikit-learn
</div>
""", unsafe_allow_html=True)