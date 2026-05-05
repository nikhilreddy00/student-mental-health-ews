# ============================================================
# CAPSTONE: Student Mental Health Early Warning System
# WEEK 3 — Part 2: Streamlit Dashboard
#
# HOW TO RUN IN COLAB:
#   !pip install streamlit pyngrok -q
#   !ngrok authtoken YOUR_TOKEN   (free at ngrok.com)
#   Run the launch cell at the bottom of this file
#
# OR locally:
#   pip install streamlit
#   streamlit run dashboard.py
# ============================================================

import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import shap
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Allow imports from the code/ directory
sys.path.insert(0, str(Path(__file__).parent))
from ai_agent import run_counselor_agent, generate_risk_narrative, get_student_profile
from multi_agent import run_multi_agent_workflow, get_student_context
from risk_monitor import compute_weekly_trajectories, generate_crossing_alert
from booking import (load_bookings, save_booking, update_booking_status,
                     generate_booking_id, compute_urgency, generate_advisor_briefing,
                     TIME_SLOTS, ADVISOR_TYPES)
from rag_kb import load_kb_documents, build_tfidf_index, retrieve_relevant_chunks, generate_rag_answer
from temporal_analysis import (compute_delay_table, generate_temporal_narrative,
                                build_enhanced_trajectory_chart)

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Mental Health Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1040 50%, #24243e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.9rem !important;
        padding: 0.3rem 0 !important;
        transition: color 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #a78bfa !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stSidebar"] caption,
    [data-testid="stSidebar"] .stCaption {
        color: #64748b !important;
        font-size: 0.75rem !important;
    }

    /* ── Page header banner ── */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
        padding: 1.6rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(79,70,229,0.35);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
    }
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -60%;
        right: 10%;
        width: 200px;
        height: 200px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .main-header h2 { font-weight: 700; letter-spacing: -0.02em; }
    .main-header p  { opacity: 0.88; font-size: 0.95rem; }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #ffffff, #f8faff);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 2px 12px rgba(79,70,229,0.08), 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid rgba(79,70,229,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79,70,229,0.14);
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1e1b4b !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b !important;
    }

    /* ── Risk tier metric cards ── */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8faff);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #4f46e5;
        margin-bottom: 1rem;
    }
    .risk-high   { border-left: 5px solid #ef4444 !important;
                   background: linear-gradient(145deg, #fff5f5, #fff1f2) !important; }
    .risk-medium { border-left: 5px solid #f59e0b !important;
                   background: linear-gradient(145deg, #fffbeb, #fef3c7) !important; }
    .risk-low    { border-left: 5px solid #10b981 !important;
                   background: linear-gradient(145deg, #f0fdf4, #dcfce7) !important; }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.4rem !important;
        transition: all 0.2s !important;
        border: none !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(79,70,229,0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79,70,229,0.5) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: white !important;
        color: #4f46e5 !important;
        border: 1.5px solid #c7d2fe !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: #eef2ff !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.6rem !important;
    }
    .stChatInput textarea {
        border-radius: 12px !important;
        border: 1.5px solid #c7d2fe !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stChatInput textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid #e0e7ff !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #4f46e5 !important;
    }

    /* ── Info / Warning / Error boxes ── */
    [data-testid="stInfo"] {
        background: linear-gradient(135deg, #eef2ff, #e0e7ff) !important;
        border: 1px solid #c7d2fe !important;
        border-radius: 12px !important;
        color: #3730a3 !important;
    }
    [data-testid="stWarning"] {
        border-radius: 12px !important;
    }
    [data-testid="stSuccess"] {
        border-radius: 12px !important;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #e0e7ff !important;
    }

    /* ── Selectbox / Multiselect — main area ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: 10px !important;
        border-color: #c7d2fe !important;
    }

    /* ── Sidebar selectbox: dark bg, readable text ── */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(167,139,250,0.5) !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #f1f5f9 !important;
        background: transparent !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #a78bfa !important;
    }
    /* ── Sidebar multiselect tags ── */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background: rgba(124,58,237,0.55) !important;
        border: 1px solid rgba(167,139,250,0.6) !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] span,
    [data-testid="stSidebar"] [data-baseweb="tag"] button {
        color: #f1f5f9 !important;
    }
    /* ── Sidebar multiselect input box ── */
    [data-testid="stSidebar"] [data-baseweb="input"] {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(167,139,250,0.5) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }
    /* ── Dropdown popup list (opens over main area) ── */
    [data-baseweb="popover"] ul {
        background: #1e1b4b !important;
        border: 1px solid rgba(167,139,250,0.4) !important;
        border-radius: 10px !important;
    }
    [data-baseweb="popover"] li {
        color: #e2e8f0 !important;
        border-radius: 6px !important;
    }
    [data-baseweb="popover"] li:hover {
        background: rgba(124,58,237,0.35) !important;
        color: #ffffff !important;
    }
    [data-baseweb="popover"] li[aria-selected="true"] {
        background: rgba(79,70,229,0.5) !important;
        color: #ffffff !important;
    }

    /* ── Slider ── */
    .stSlider [data-testid="stThumbValue"] {
        background: #4f46e5 !important;
    }
    .stSlider [role="slider"] {
        background: #4f46e5 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }

    /* ── Horizontal rule ── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #c7d2fe, transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Section subheaders ── */
    h3 {
        color: #1e1b4b !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    h2 { font-weight: 800 !important; letter-spacing: -0.02em; }

    /* ── Agent streaming boxes ── */
    .agent-box {
        background: linear-gradient(145deg, #fafafe, #f1f5ff);
        border: 1px solid #e0e7ff;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 0.6rem;
        font-size: 0.94rem;
        line-height: 1.7;
        min-height: 60px;
    }

    /* ── Footer ── */
    footer { visibility: hidden; }
    .footer-custom {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid #e0e7ff;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = Path("/content/data")
    if not base.exists():
        base = Path("data")  # local fallback

    explanations = pd.read_csv(base / "student_explanations.csv")
    master       = pd.read_csv(base / "master_dataset.csv")
    shap_vals    = pd.read_csv(base / "shap_values.csv")

    # Merge useful cols into explanations
    extra_cols = ['id_student', 'gender', 'age_band', 'region',
                  'imd_band', 'highest_education', 'num_of_prev_attempts',
                  'studied_credits', 'disability']
    available = [c for c in extra_cols if c in master.columns]
    explanations = explanations.merge(
        master[available], on='id_student', how='left'
    )
    return explanations, master, shap_vals

@st.cache_resource
def load_models():
    base = Path("/content/models")
    if not base.exists():
        base = Path("models")
    try:
        xgb = joblib.load(base / "xgb_model.pkl")
        scaler = joblib.load(base / "scaler.pkl")
        return xgb, scaler
    except:
        return None, None

@st.cache_resource
def load_kb_index():
    try:
        docs = load_kb_documents()
        vectorizer, tfidf_matrix, chunk_meta = build_tfidf_index(docs)
        return vectorizer, tfidf_matrix, chunk_meta, docs
    except Exception as e:
        return None, None, None, []

# ── LOAD ─────────────────────────────────────────────────────
try:
    df, master, shap_df = load_data()
    xgb_model, scaler   = load_models()
    data_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load data: {e}\nMake sure you ran Week 1, 2, and 3 scripts first.")
    data_loaded = False
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=60)
    st.title("Navigation")

    page = st.radio("Go to", [
        "📊 Dashboard Overview",
        "🔴 Alert Center",
        "👤 Student Profile",
        "📈 Model Performance",
        "🧠 SHAP Explainability",
        "⚖️ Fairness Analysis",
        "🤖 AI Counselor",
        "👥 Multi-Agent Planner",
        "📡 Live Risk Monitor",
        "📅 Book Appointment",
        "📋 Booking Queue",
        "💬 KB Assistant",
        "⏰ Early Warning Analysis",
    ])

    st.markdown("---")
    st.markdown("### Filters")

    modules = ["All"] + sorted(df['code_module'].dropna().unique().tolist())
    selected_module = st.selectbox("Module", modules)

    risk_filter = st.multiselect(
        "Risk Tier",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )

    st.markdown("---")
    st.caption("🎓 Capstone Project | Data Analytics MS")
    st.caption("OULAD Dataset | XGBoost Model")
    st.caption(f"AUC-ROC: **0.975** | Accuracy: **91%**")

# ── APPLY FILTERS ─────────────────────────────────────────────
filtered = df.copy()
if selected_module != "All":
    filtered = filtered[filtered['code_module'] == selected_module]
if risk_filter:
    filtered = filtered[filtered['risk_tier'].isin(risk_filter)]

# ════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>🎓 Student Mental Health Early Warning System</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Proactive identification of at-risk students using behavioral analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────
    total     = len(filtered)
    high      = (filtered['risk_tier'] == 'High').sum()
    medium    = (filtered['risk_tier'] == 'Medium').sum()
    low       = (filtered['risk_tier'] == 'Low').sum()
    avg_score = filtered['risk_score'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students",   f"{total:,}")
    c2.metric("🔴 High Risk",     f"{high:,}",   f"{high/total*100:.1f}%")
    c3.metric("🟡 Medium Risk",   f"{medium:,}", f"{medium/total*100:.1f}%")
    c4.metric("🟢 Low Risk",      f"{low:,}",    f"{low/total*100:.1f}%")
    c5.metric("Avg Risk Score",   f"{avg_score:.3f}")

    st.markdown("---")

    # ── Charts Row ────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1.5])

    with col1:
        st.subheader("Risk Tier Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        tier_counts = filtered['risk_tier'].value_counts()
        colors_map = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
        ordered = [t for t in ['High','Medium','Low'] if t in tier_counts.index]
        vals = [tier_counts[t] for t in ordered]
        clrs = [colors_map[t] for t in ordered]
        wedges, texts, autotexts = ax.pie(
            vals, labels=ordered, colors=clrs,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2}
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight('bold')
        ax.set_title("Students by Risk Tier", fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Risk Score Histogram")
        fig, ax = plt.subplots(figsize=(5, 4))
        high_s = filtered[filtered['risk_tier'] == 'High']['risk_score']
        med_s  = filtered[filtered['risk_tier'] == 'Medium']['risk_score']
        low_s  = filtered[filtered['risk_tier'] == 'Low']['risk_score']
        ax.hist(low_s,  bins=20, alpha=0.7, color='#2ecc71', label='Low',    density=True)
        ax.hist(med_s,  bins=20, alpha=0.7, color='#f39c12', label='Medium', density=True)
        ax.hist(high_s, bins=20, alpha=0.7, color='#e74c3c', label='High',   density=True)
        ax.axvline(0.33, color='gray',  linestyle=':', linewidth=1.5)
        ax.axvline(0.66, color='black', linestyle=':', linewidth=1.5)
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.set_title("Risk Score Distribution", fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col3:
        st.subheader("Top Reasons Students Are Flagged")
        reason_counts = filtered[filtered['risk_tier'] == 'High']['top_reason'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        reason_counts.sort_values().plot(kind='barh', ax=ax, color='#e74c3c',
                                          edgecolor='black', linewidth=0.5)
        ax.set_title("Most Common Primary Risk Driver\n(High-Risk Students)",
                     fontweight='bold')
        ax.set_xlabel("Number of Students")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Module-level risk breakdown ───────────────────────────
    st.markdown("---")
    st.subheader("📚 Risk Breakdown by Module")
    if 'code_module' in filtered.columns:
        module_risk = filtered.groupby('code_module').agg(
            total     = ('id_student', 'count'),
            high_risk = ('risk_tier', lambda x: (x == 'High').sum()),
            avg_score = ('risk_score', 'mean')
        ).reset_index()
        module_risk['high_risk_pct'] = (module_risk['high_risk'] / module_risk['total'] * 100).round(1)
        module_risk = module_risk.sort_values('high_risk_pct', ascending=False)

        fig, ax = plt.subplots(figsize=(14, 4))
        bars = ax.bar(module_risk['code_module'], module_risk['high_risk_pct'],
                      color=['#e74c3c' if v > 50 else '#f39c12' if v > 30 else '#2ecc71'
                             for v in module_risk['high_risk_pct']],
                      edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, module_risk['high_risk_pct']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
        ax.axhline(50, color='red',  linestyle='--', lw=1.5, alpha=0.6, label='50% threshold')
        ax.axhline(30, color='orange', linestyle='--', lw=1.5, alpha=0.6, label='30% threshold')
        ax.set_ylabel("% High-Risk Students")
        ax.set_title("High-Risk Student Rate by Module", fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════
# PAGE 2: ALERT CENTER
# ════════════════════════════════════════════════════════════
elif page == "🔴 Alert Center":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>🔴 Alert Center</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Students flagged for proactive counselor outreach
        </p>
    </div>
    """, unsafe_allow_html=True)

    high_risk = filtered[filtered['risk_tier'] == 'High'].copy()
    high_risk = high_risk.sort_values('risk_score', ascending=False)

    st.info(f"🔴 **{len(high_risk):,} students** require attention | "
            f"Showing top {min(50, len(high_risk))} alerts")

    # Alert cards
    st.markdown("### Priority Alerts")

    def risk_badge(score):
        if score >= 0.66:
            return f"🔴 HIGH ({score:.2f})"
        elif score >= 0.33:
            return f"🟡 MED ({score:.2f})"
        return f"🟢 LOW ({score:.2f})"

    # Display as styled table
    display_cols = ['id_student', 'risk_score', 'risk_tier',
                    'top_reason', 'reason_2', 'reason_3']
    available_cols = [c for c in display_cols if c in high_risk.columns]
    show_df = high_risk[available_cols].head(50).copy()
    show_df['risk_score'] = show_df['risk_score'].round(4)
    show_df.columns = [c.replace('_',' ').title() for c in show_df.columns]

    st.dataframe(
        show_df,
        use_container_width=True,
        height=400,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=1, format="%.3f"
            )
        }
    )

    # ── Risk factors breakdown ────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 What's Driving High-Risk Flags?")
        all_reasons = pd.concat([
            high_risk['top_reason'],
            high_risk['reason_2'],
            high_risk['reason_3']
        ]).value_counts().head(10)

        fig, ax = plt.subplots(figsize=(7, 5))
        all_reasons.sort_values().plot(kind='barh', ax=ax,
                                        color='#e74c3c', edgecolor='black', lw=0.5)
        ax.set_title("Risk Factors Across All High-Risk Students\n"
                     "(Combined top-3 SHAP reasons per student)", fontweight='bold')
        ax.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("📊 Behavioral Profile of High-Risk Students")
        if all(c in high_risk.columns for c in ['engagement_span','mean_score']):
            compare_features = ['engagement_span','mean_score','active_days',
                                 'submission_rate','engagement_decline']
            compare_features = [f for f in compare_features if f in high_risk.columns]

            high_avgs = high_risk[compare_features].mean()
            low_avgs  = filtered[filtered['risk_tier'] == 'Low'][compare_features].mean()

            compare_df = pd.DataFrame({
                'High Risk': high_avgs,
                'Low Risk':  low_avgs
            })

            fig, ax = plt.subplots(figsize=(7, 5))
            x = np.arange(len(compare_df))
            w = 0.35
            b1 = ax.bar(x - w/2, compare_df['High Risk'], w,
                        color='#e74c3c', label='High Risk', edgecolor='black', lw=0.5)
            b2 = ax.bar(x + w/2, compare_df['Low Risk'], w,
                        color='#2ecc71', label='Low Risk',  edgecolor='black', lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(compare_df.index, rotation=25, ha='right', fontsize=9)
            ax.set_title("Avg Feature Values: High-Risk vs Low-Risk", fontweight='bold')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Download alert list ───────────────────────────────────
    st.markdown("---")
    csv_bytes = high_risk[available_cols].head(100).to_csv(index=False).encode()
    st.download_button(
        label="📥 Download Alert List (CSV)",
        data=csv_bytes,
        file_name="high_risk_students_alert.csv",
        mime="text/csv"
    )

# ════════════════════════════════════════════════════════════
# PAGE 3: STUDENT PROFILE
# ════════════════════════════════════════════════════════════
elif page == "👤 Student Profile":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>👤 Individual Student Profile</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Detailed risk analysis and behavioral explanation for any student
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Student search
    student_ids = sorted(df['id_student'].unique().tolist())
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_id = st.selectbox(
            "Search student by ID",
            student_ids,
            help="Select a student to view their full risk profile"
        )
    with col2:
        quick_filter = st.selectbox("Or pick by tier", ["Any","High Risk","Medium Risk","Low Risk"])
        if quick_filter != "Any":
            tier_map = {"High Risk":"High","Medium Risk":"Medium","Low Risk":"Low"}
            candidates = df[df['risk_tier'] == tier_map[quick_filter]]['id_student'].tolist()
            if candidates:
                selected_id = st.selectbox("Pick from this tier", candidates)

    student = df[df['id_student'] == selected_id].iloc[0]
    risk_score = student['risk_score']
    risk_tier  = student.get('risk_tier', 'Unknown')

    # Risk score gauge
    col1, col2, col3 = st.columns([1.2, 1.5, 1.5])

    with col1:
        tier_color = {'High':'#e74c3c','Medium':'#f39c12','Low':'#2ecc71'}.get(risk_tier,'gray')
        tier_emoji = {'High':'🔴','Medium':'🟡','Low':'🟢'}.get(risk_tier,'⚪')

        st.markdown(f"""
        <div class='metric-card risk-{"high" if risk_tier=="High" else "medium" if risk_tier=="Medium" else "low"}'>
            <h3 style='margin:0; color:{tier_color}'>{tier_emoji} {risk_tier} Risk</h3>
            <h1 style='margin:0.2rem 0; color:{tier_color}'>{risk_score:.3f}</h1>
            <p style='margin:0; color:gray; font-size:0.85rem'>Risk Probability Score (0–1)</p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'projection': 'polar'})
        theta = np.linspace(0, np.pi, 100)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi)

        ax.barh(0.5, np.pi * 0.33, left=0,          height=0.3, color='#2ecc71', alpha=0.7)
        ax.barh(0.5, np.pi * 0.33, left=np.pi*0.33, height=0.3, color='#f39c12', alpha=0.7)
        ax.barh(0.5, np.pi * 0.34, left=np.pi*0.66, height=0.3, color='#e74c3c', alpha=0.7)

        needle_angle = np.pi * (1 - risk_score)
        ax.annotate('', xy=(needle_angle, 0.7), xytext=(needle_angle, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        ax.set_title(f"Risk Gauge\n{risk_score:.3f}", fontsize=11, fontweight='bold', pad=10)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("👤 Student Demographics")
        demo_fields = {
            'Student ID':   str(student.get('id_student','')),
            'Module':       str(student.get('code_module','')),
            'Gender':       str(student.get('gender','')),
            'Age Band':     str(student.get('age_band','')),
            'Region':       str(student.get('region','')),
            'Education':    str(student.get('highest_education','')),
            'IMD Band':     str(student.get('imd_band','')),
            'Disability':   str(student.get('disability','')),
            'Prev Attempts':str(student.get('num_of_prev_attempts','')),
        }
        for label, val in demo_fields.items():
            if val and val != 'nan':
                st.markdown(f"**{label}:** {val}")

    with col3:
        st.subheader("📊 Behavioral Metrics")
        behavioral_fields = {
            'Engagement Span (days)': student.get('engagement_span',''),
            'Mean Assessment Score':  student.get('mean_score',''),
            'Active Days':            student.get('active_days',''),
            'Submission Rate':        student.get('submission_rate',''),
            'Engagement Decline':     student.get('engagement_decline',''),
            'Dropout Modules':        student.get('dropout_modules',''),
        }
        for label, val in behavioral_fields.items():
            if val != '' and str(val) != 'nan':
                try:
                    st.markdown(f"**{label}:** {float(val):.2f}")
                except:
                    st.markdown(f"**{label}:** {val}")

    # SHAP explanation for this student
    st.markdown("---")
    st.subheader("🧠 Why Is This Student Flagged?")

    top_reason = student.get('top_reason','')
    r2 = student.get('reason_2','')
    r3 = student.get('reason_3','')

    if top_reason:
        col1, col2, col3 = st.columns(3)
        col1.error(f"**#1 Reason:**\n{top_reason.replace('_',' ').title()}")
        col2.warning(f"**#2 Reason:**\n{r2.replace('_',' ').title()}")
        col3.info(f"**#3 Reason:**\n{r3.replace('_',' ').title()}")

    # SHAP waterfall for this student
    student_idx = df[df['id_student'] == selected_id].index[0]
    if student_idx < len(shap_df):
        shap_vals = shap_df.iloc[student_idx]
        top8 = shap_vals.abs().sort_values(ascending=False).head(8)
        plot_vals = shap_vals[top8.index].sort_values()

        fig, ax = plt.subplots(figsize=(10, 5))
        bar_colors = ['#e74c3c' if v > 0 else '#3498db' for v in plot_vals.values]
        bars = ax.barh(range(len(plot_vals)), plot_vals.values,
                       color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(plot_vals)))
        ax.set_yticklabels([f.replace('_',' ').title() for f in plot_vals.index])
        ax.axvline(0, color='black', lw=1.5)
        ax.set_xlabel("SHAP Value  (🔴 Positive = increases risk | 🔵 Negative = reduces risk)")
        ax.set_title(f"Student {selected_id} — Feature Contributions to Risk Score",
                     fontweight='bold', fontsize=12)

        for bar, val in zip(bars, plot_vals.values):
            ax.text(val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center',
                    ha='left' if val >= 0 else 'right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # AI-generated narrative
    st.markdown("---")
    st.subheader("🤖 AI Risk Narrative")
    st.caption("Plain-English summary generated by Claude AI based on this student's behavioral data")
    if st.button("Generate AI Summary", key="narrative_btn"):
        with st.spinner("Claude is analyzing this student's risk profile..."):
            narrative = generate_risk_narrative(selected_id)
        st.info(narrative)

    # Suggested action
    st.markdown("---")
    st.subheader("💬 Suggested Counselor Action")
    if risk_tier == "High":
        st.error(f"""
        **Immediate outreach recommended** for Student {selected_id}

        This student shows a risk score of **{risk_score:.3f}**, primarily driven by
        **{top_reason.replace('_',' ') if top_reason else 'behavioral withdrawal'}**.

        Suggested approach: Reach out within 48 hours with a supportive, non-academic
        check-in. Ask about wellbeing, not just coursework.
        """)
    elif risk_tier == "Medium":
        st.warning(f"""
        **Monitor and consider proactive check-in** for Student {selected_id}

        Risk score: **{risk_score:.3f}**. Main concern:
        **{top_reason.replace('_',' ') if top_reason else 'declining engagement'}**.

        Suggested: Schedule a 15-min welfare check in the next 1-2 weeks.
        """)
    else:
        st.success(f"""
        Student {selected_id} appears to be **on track** (score: {risk_score:.3f}).
        No immediate action required. Continue routine monitoring.
        """)

# ════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>📈 Model Performance</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Validation metrics for the trained XGBoost model
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC",   "0.975", "↑ vs target 0.80")
    col2.metric("F1 Score",  "0.911")
    col3.metric("Precision", "0.936")
    col4.metric("Recall",    "0.887", "88.7% at-risk caught")

    st.markdown("---")
    st.subheader("Confusion Matrix Interpretation")

    col1, col2 = st.columns(2)
    with col1:
        cm_data = np.array([[2153, 155], [292, 2289]])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_data, annot=False, cmap='Blues', ax=ax,
                    xticklabels=['Not At-Risk','At-Risk'],
                    yticklabels=['Not At-Risk','At-Risk'], cbar=False)
        cm_pct = cm_data / cm_data.sum(axis=1, keepdims=True) * 100
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.35, f'{cm_data[i,j]:,}',
                        ha='center', fontsize=14, fontweight='bold',
                        color='white' if cm_pct[i,j] > 50 else 'black')
                ax.text(j+0.5, i+0.65, f'({cm_pct[i,j]:.1f}%)',
                        ha='center', fontsize=11,
                        color='white' if cm_pct[i,j] > 50 else 'black')
        ax.set_title("XGBoost — Test Set Confusion Matrix", fontweight='bold')
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("""
        **Reading the confusion matrix:**

        | Cell | Value | Meaning |
        |------|-------|---------|
        | Top-Left (TN) | 2,153 (93.3%) | Correctly identified as NOT at-risk |
        | Top-Right (FP) | 155 (6.7%) | Incorrectly flagged — counselor would check and find OK |
        | Bottom-Left (FN) | 292 (11.3%) | Missed at-risk students — most critical error |
        | Bottom-Right (TP) | 2,289 (88.7%) | Correctly identified at-risk students |

        **Why recall matters more here:**
        - A false negative (missing a struggling student) is worse than a false positive (unnecessary check-in)
        - Our 88.7% recall means we catch **~9 out of 10** at-risk students
        - The 155 false positives = minor extra workload for counselors

        **5-fold Cross-Validation stability:**
        - XGBoost: 0.9734 ± 0.0012 (very low variance = stable model)
        """)

    st.markdown("---")
    st.subheader("Feature Importance (XGBoost)")

    try:
        importances = pd.Series(
            xgb_model.feature_importances_,
            index=[
                'total_clicks', 'active_days', 'engagement_decline',
                'avg_daily_clicks', 'early_clicks', 'late_clicks',
                'engagement_span', 'clicks_per_day', 'active_day_rate',
                'mean_score', 'min_score', 'max_score', 'score_std',
                'num_assessments', 'num_unsubmitted', 'submission_rate',
                'score_trend', 'num_late',
                'num_modules_registered', 'avg_reg_delay', 'dropout_modules',
                'early_dropout_flag', 'num_of_prev_attempts', 'studied_credits',
            ][:len(xgb_model.feature_importances_)]
        ).sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind='barh', ax=ax, color='#e74c3c',
                         edgecolor='black', linewidth=0.5)
        ax.set_title("XGBoost Feature Importance — Top 15", fontweight='bold')
        ax.set_xlabel("Importance Score")
        ax.axvline(importances.mean(), color='navy', linestyle='--', lw=1.5,
                   label=f'Mean: {importances.mean():.4f}')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Feature importance chart unavailable: {e}")

# ════════════════════════════════════════════════════════════
# PAGE 5: SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainability":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>🧠 SHAP Explainability</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Why the model makes each prediction — transparency for counselors
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    **What is SHAP?** SHAP (SHapley Additive exPlanations) shows how much each feature
    contributed to a specific prediction. Red = pushes toward At-Risk. Blue = pushes away from risk.
    This makes the model transparent and trustworthy for counselors.
    """)

    plots_dir = Path(__file__).parent.parent / "plots"
    if not plots_dir.exists():
        plots_dir = Path("/content")

    col1, col2 = st.columns(2)
    for fname, title, col in [
        ("plot10_shap_summary.png", "SHAP Beeswarm — Global Feature Impact", col1),
        ("plot11_shap_bar.png",     "Mean |SHAP| — Feature Importance Ranking", col2),
    ]:
        fpath = plots_dir / fname
        if fpath.exists():
            col.subheader(title)
            col.image(str(fpath), use_container_width=True)
        else:
            col.warning(f"Run week3_shap.py to generate {fname}")

    st.markdown("---")
    fpath = plots_dir / "plot12_shap_dependence.png"
    if fpath.exists():
        st.subheader("SHAP Dependence Plots — Top 4 Features")
        st.image(str(fpath), use_container_width=True)

    fpath = plots_dir / "plot13_shap_individual.png"
    if fpath.exists():
        st.subheader("Individual Student SHAP Explanations (3 Case Studies)")
        st.image(str(fpath), use_container_width=True)

    fpath = plots_dir / "plot14_shap_tier_heatmap.png"
    if fpath.exists():
        st.subheader("Average SHAP Values by Risk Tier")
        st.image(str(fpath), use_container_width=True)

    st.markdown("---")
    st.subheader("📖 How to Read These Charts")
    st.markdown("""
    | Chart | How to read it |
    |-------|---------------|
    | **Beeswarm** | Each dot = one student. X position = SHAP impact. Color = feature value (red=high, blue=low). Features sorted by importance. |
    | **Bar chart** | Average absolute SHAP across all students. Higher = more important overall. |
    | **Dependence plot** | X = raw feature value, Y = SHAP impact. Curve shape shows if relationship is linear, threshold-based, etc. |
    | **Individual plot** | Red bars = this feature INCREASED this student's risk score. Blue = decreased it. |
    | **Tier heatmap** | How each risk group differs. High-risk students have very different SHAP patterns from low-risk. |
    """)

# ════════════════════════════════════════════════════════════
# PAGE 6: FAIRNESS ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "⚖️ Fairness Analysis":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>⚖️ Fairness & Bias Analysis</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Ensuring the model performs equitably across demographic groups
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    **Why fairness matters:** An AI system used for student welfare must not
    systematically over-flag or under-flag any demographic group. We check that
    the model's accuracy is consistent across gender, age, and socioeconomic status.
    """)

    def group_metrics(group_col):
    # df already has demographic columns from load_data merge — no re-merge needed
        if group_col not in df.columns:
            return pd.DataFrame()  # graceful empty return if column missing
        merged = df[[group_col, 'risk_score', 'risk_tier', 'is_at_risk']].copy()
        merged = merged.dropna(subset=[group_col, 'risk_score', 'is_at_risk'])

        rows = []
        for grp in merged[group_col].unique():
            sub = merged[merged[group_col] == grp]
            if len(sub) < 20:
                continue
            n       = len(sub)
            n_risk  = sub['is_at_risk'].sum()
            avg_sc  = sub['risk_score'].mean()
            high_rt = (sub['risk_tier'] == 'High').mean()
            rows.append({
                'Group': grp, 'Count': n,
                'Actual At-Risk': f"{n_risk/n*100:.1f}%",
                'Avg Risk Score': round(avg_sc, 3),
                'High-Risk Flag Rate': f"{high_rt*100:.1f}%"
            })
        return pd.DataFrame(rows).sort_values('Count', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("By Gender")
        gdf = group_metrics('gender')
        if not gdf.empty:
            st.dataframe(gdf, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5, 3))
            for gv, gc in [('M','#3498db'),('F','#e91e63')]:
                s = df[df['gender']==gv]['risk_score'].dropna()
                if len(s) > 0:
                    ax.hist(s, bins=20, alpha=0.6, color=gc, label=gv, density=True)
            ax.set_xlabel('Risk Score')
            ax.set_ylabel('Density')
            ax.set_title('Risk Score by Gender', fontweight='bold')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    with col2:
        st.subheader("By Age Band")
        adf = group_metrics('age_band')
        if not adf.empty:
            st.dataframe(adf, use_container_width=True)
            age_risk = df.groupby('age_band')['risk_score'].mean().sort_values()
            age_risk = df.groupby('age_band')['risk_score'].mean().sort_values()
            fig, ax = plt.subplots(figsize=(5, 3))
            age_risk.plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='black', lw=0.5)
            ax.set_title("Avg Risk Score by Age Band", fontweight='bold')
            ax.set_xlabel("Average Risk Score")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.subheader("By IMD Band (Deprivation Index — Socioeconomic Status)")
    idf = group_metrics('imd_band')
    if not idf.empty:
        st.dataframe(idf, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### ✅ Fairness Checklist

    | Check | Status | Notes |
    |-------|--------|-------|
    | Gender parity | ✅ Monitor | Should not over-flag either gender |
    | Age group equity | ✅ Monitor | Younger students genuinely higher risk (valid) |
    | Socioeconomic bias | ✅ Monitor | IMD band reflects real disadvantage, not model bias |
    | Disability group | ✅ Monitor | Feature used only for fairness analysis, not prediction |
    | Explainability | ✅ Done | SHAP shows counselor WHY each student is flagged |
    | Human-in-the-loop | ✅ Designed | Model alerts; counselor decides; student chooses to engage |

    **Key principle:** Disparities in outcomes may reflect real differences in support needs,
    not model bias. The test is whether the model's *false negative rate* (missing at-risk students)
    differs significantly by group — that would indicate unfair discrimination.
    """)

# ════════════════════════════════════════════════════════════
# PAGE 7: AI COUNSELOR AGENT
# ════════════════════════════════════════════════════════════
elif page == "🤖 AI Counselor":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>🤖 AI Counselor Assistant</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Ask questions about at-risk students in natural language — powered by Claude AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    **How to use:** Ask me anything about your students in plain English.

    Try: *"Who needs outreach this week?"* · *"What's driving student 12345's risk?"*
    · *"Show me high-risk female students in module AAA"* · *"Which students have the biggest engagement drop?"*
    """)

    # Initialize chat history
    if "counselor_messages" not in st.session_state:
        st.session_state.counselor_messages = []
    if "counselor_api_messages" not in st.session_state:
        st.session_state.counselor_api_messages = []

    # Display chat history
    for msg in st.session_state.counselor_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about your students...")
    if user_input:
        # Show user message
        st.session_state.counselor_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build API messages list
        st.session_state.counselor_api_messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing student data..."):
                try:
                    response_text, tool_calls, updated_msgs = run_counselor_agent(
                        st.session_state.counselor_api_messages
                    )
                    st.session_state.counselor_api_messages = updated_msgs + [
                        {"role": "assistant", "content": response_text}
                    ]
                except Exception as e:
                    response_text = f"⚠️ Error: {e}"

            # Show tool calls made (transparent AI reasoning)
            if tool_calls:
                for tc in tool_calls:
                    with st.expander(f"🔧 Tool used: `{tc['tool']}`", expanded=False):
                        st.json(tc["input"])

            st.markdown(response_text)

        st.session_state.counselor_messages.append(
            {"role": "assistant", "content": response_text}
        )

    # Clear button
    if st.session_state.counselor_messages:
        if st.button("🗑️ Clear conversation"):
            st.session_state.counselor_messages = []
            st.session_state.counselor_api_messages = []
            st.rerun()

# ════════════════════════════════════════════════════════════
# PAGE 8: MULTI-AGENT INTERVENTION PLANNER
# ════════════════════════════════════════════════════════════
elif page == "👥 Multi-Agent Planner":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>👥 Multi-Agent Intervention Planner</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Three specialized AI agents collaborate to create a personalized intervention plan
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **How it works:** Select a high-risk student and click **Generate Intervention Plan**.
    Watch three AI agents reason sequentially:
    1. 🔍 **Risk Analyst** — interprets behavioral signals
    2. 📋 **Intervention Advisor** — recommends specific strategies
    3. ✉️ **Counselor Note Writer** — drafts a personalized outreach message
    """)

    # Student selector — high risk only
    high_risk_ids = (
        df[df["risk_tier"] == "High"]
        .sort_values("risk_score", ascending=False)
        .drop_duplicates("id_student")["id_student"]
        .tolist()
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_student = st.selectbox(
            "Select a High-Risk Student",
            high_risk_ids,
            format_func=lambda sid: (
                f"Student {sid} — Risk: "
                f"{df[df['id_student']==sid]['risk_score'].max():.3f}"
            ),
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Generate Intervention Plan", type="primary")

    # Show student snapshot
    if selected_student:
        ctx = get_student_context(selected_student)
        if ctx:
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risk Score", f"{ctx['risk_score']:.3f}")
            c2.metric("Risk Tier", ctx["risk_tier"])
            c3.metric("Module", ctx["module"])
            c4.metric("Mean Score", f"{ctx['mean_score']:.1f}/100")

    st.markdown("---")

    if run_btn and selected_student:
        # Agent 1 section
        st.markdown("### 🔍 Agent 1: Risk Analyst")
        st.caption("Interpreting behavioral signals and producing a structured risk profile")
        agent1_box = st.empty()

        # Agent 2 section (shown after agent 1 completes)
        agent2_header = st.empty()
        agent2_caption = st.empty()
        agent2_box = st.empty()

        # Agent 3 section
        agent3_header = st.empty()
        agent3_caption = st.empty()
        agent3_box = st.empty()

        # Done message
        done_box = st.empty()

        agent1_text = ""
        agent2_text = ""
        agent3_text = ""

        for phase, payload in run_multi_agent_workflow(selected_student):
            if phase == "error":
                st.error(payload)
            elif phase == "agent1":
                agent1_text += payload
                agent1_box.markdown(agent1_text + "▌")
            elif phase == "agent2":
                if not agent2_text:
                    agent2_header.markdown("### 📋 Agent 2: Intervention Advisor")
                    agent2_caption.caption(
                        "Recommending 3 ranked intervention strategies based on risk profile"
                    )
                agent2_text += payload
                agent2_box.markdown(agent2_text + "▌")
            elif phase == "agent3":
                if not agent3_text:
                    agent3_header.markdown("### ✉️ Agent 3: Counselor Note Writer")
                    agent3_caption.caption(
                        "Drafting a warm, personalized outreach message"
                    )
                agent3_text += payload
                agent3_box.markdown(agent3_text + "▌")
            elif phase == "done":
                agent1_box.markdown(agent1_text)
                agent2_box.markdown(agent2_text)
                agent3_box.markdown(agent3_text)
                done_box.success(
                    "✅ Intervention plan complete. "
                    "Review the outreach message above before sending."
                )

# ════════════════════════════════════════════════════════════
# PAGE 9: LIVE RISK MONITOR
# ════════════════════════════════════════════════════════════
elif page == "📡 Live Risk Monitor":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>📡 Live Risk Monitor</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Week-by-week risk trajectory simulation — the smoke detector for student wellbeing
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    **Temporal change detection:** Risk scores are simulated week-by-week using actual VLE
    engagement data. When a student's weekly activity drops relative to their personal peak,
    their risk score rises. Drag the slider to replay the semester — watch for the moment
    students cross into High Risk territory.
    """)

    with st.spinner("⏳ Loading 10.6M VLE engagement records and computing risk trajectories... (cached after first run)"):
        try:
            trajectories = compute_weekly_trajectories(top_n=10)
        except Exception as e:
            st.error(f"Could not compute trajectories: {e}")
            st.stop()

    if not trajectories:
        st.warning("No trajectory data available.")
        st.stop()

    # Max weeks across all students
    all_weeks_flat = [w for t in trajectories.values() for w in t["weeks"]]
    max_week_global = max(all_weeks_flat) if all_weeks_flat else 40

    # Student selector
    student_options = list(trajectories.keys())
    selected_monitor_students = st.multiselect(
        "Select students to monitor (default: top 5 highest risk)",
        options=student_options,
        default=student_options[:5],
        format_func=lambda sid: (
            f"Student {sid} — Final Risk: {trajectories[sid]['final_risk']:.3f} "
            f"| {trajectories[sid]['module']} | {trajectories[sid]['top_reason'][:30]}"
        ),
    )

    if not selected_monitor_students:
        st.warning("Please select at least one student to monitor.")
        st.stop()

    # Week slider
    current_week = st.slider(
        "📅 Simulate to Week",
        min_value=1,
        max_value=max_week_global,
        value=max_week_global // 2,
        help="Drag to replay the semester week by week",
    )

    # Build Plotly figure
    fig = go.Figure()

    fired_alerts = []
    HIGH_THRESHOLD = 0.66

    for sid in selected_monitor_students:
        t = trajectories[sid]
        weeks = t["weeks"]
        scores = t["risk_scores"]

        # Clip to current_week
        visible_weeks = [w for w in weeks if w <= current_week]
        visible_scores = scores[:len(visible_weeks)]

        fig.add_trace(go.Scatter(
            x=visible_weeks,
            y=visible_scores,
            mode="lines+markers",
            name=f"Student {sid}",
            hovertemplate=(
                f"<b>Student {sid}</b><br>"
                "Week: %{x}<br>Risk Score: %{y:.3f}<extra></extra>"
            ),
            line=dict(width=2),
            marker=dict(size=5),
        ))

        # Check if threshold crossed — only record the FIRST crossing per student
        for i in range(1, len(visible_scores)):
            if visible_scores[i] >= HIGH_THRESHOLD and visible_scores[i - 1] < HIGH_THRESHOLD:
                fired_alerts.append({
                    "student_id": sid,
                    "week": visible_weeks[i],
                    "prev": visible_scores[i - 1],
                    "curr": visible_scores[i],
                    "top_reason": t["top_reason"],
                })
                break  # only first crossing per student

    # Add threshold line
    fig.add_hline(
        y=HIGH_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text="High Risk Threshold (0.66)",
        annotation_position="top left",
    )
    fig.add_hrect(y0=HIGH_THRESHOLD, y1=1.0, fillcolor="red", opacity=0.05, line_width=0)

    fig.update_layout(
        title=f"Risk Score Trajectories — Week 0 to Week {current_week}",
        xaxis_title="Course Week",
        yaxis_title="Simulated Risk Score",
        yaxis=dict(range=[0, 1]),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Fire Claude alerts for threshold crossings
    if fired_alerts:
        st.markdown("---")
        st.markdown("### 🚨 Threshold Crossing Alerts")
        for alert_info in fired_alerts:
            col1, col2 = st.columns([3, 1])
            with col1:
                alert_key = f"alert_{alert_info['student_id']}_{alert_info['week']}"
                if alert_key not in st.session_state:
                    with st.spinner(f"Claude generating alert for Student {alert_info['student_id']}..."):
                        try:
                            alert_text = generate_crossing_alert(
                                alert_info["student_id"],
                                alert_info["week"],
                                alert_info["prev"],
                                alert_info["curr"],
                                alert_info["top_reason"],
                            )
                            st.session_state[alert_key] = alert_text
                        except Exception as e:
                            st.session_state[alert_key] = (
                                f"⚠️ ALERT: Student {alert_info['student_id']} crossed into "
                                f"High Risk at Week {alert_info['week']} "
                                f"(score: {alert_info['prev']:.2f} → {alert_info['curr']:.2f}). "
                                f"Primary concern: {alert_info['top_reason']}."
                            )
                st.warning(st.session_state[alert_key])
            with col2:
                st.metric("Risk Jump", f"{alert_info['prev']:.2f} → {alert_info['curr']:.2f}")

    # Summary table
    st.markdown("---")
    st.markdown(f"### 📊 Risk Status at Week {current_week}")
    summary_rows = []
    for sid in selected_monitor_students:
        t = trajectories[sid]
        visible = [s for w, s in zip(t["weeks"], t["risk_scores"]) if w <= current_week]
        if visible:
            curr_score = visible[-1]
            tier = "🔴 High" if curr_score >= 0.66 else "🟡 Medium" if curr_score >= 0.33 else "🟢 Low"
            summary_rows.append({
                "Student ID": sid,
                "Module": t["module"],
                "Risk Score": round(curr_score, 3),
                "Tier": tier,
                "Top Concern": t["top_reason"],
            })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE 10: BOOK APPOINTMENT
# ════════════════════════════════════════════════════════════
elif page == "📅 Book Appointment":
    import datetime

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>📅 Appointment Booking System</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Book at-risk students with the right advisor — with an AI-generated pre-meeting briefing
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Student selector ──────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col2:
        include_medium = st.checkbox("Include Medium Risk", value=False)

    tiers = ["High", "Medium"] if include_medium else ["High"]
    eligible = (
        df[df["risk_tier"].isin(tiers)]
        .sort_values("risk_score", ascending=False)
        .drop_duplicates("id_student")
    )

    with col1:
        booking_student_id = st.selectbox(
            "Select Student to Book",
            eligible["id_student"].tolist(),
            format_func=lambda sid: (
                f"Student {sid} — Risk: "
                f"{eligible[eligible['id_student']==sid]['risk_score'].values[0]:.3f} "
                f"({eligible[eligible['id_student']==sid]['risk_tier'].values[0]})"
            ),
        )

    if booking_student_id:
        brow = eligible[eligible["id_student"] == booking_student_id].iloc[0]
        b_risk = float(brow["risk_score"])
        b_tier = str(brow["risk_tier"])
        b_urgency = compute_urgency(b_risk)
        b_module = str(brow.get("code_module", ""))

        # Student snapshot
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", f"{b_risk:.3f}")
        c2.metric("Risk Tier", b_tier)
        c3.metric("Module", b_module)
        urgency_color = {"Immediate": "🔴", "Soon": "🟡", "Routine": "🟢"}.get(b_urgency, "⚪")
        c4.metric("Urgency", f"{urgency_color} {b_urgency}")

        top_reason = str(brow.get("top_reason", "")).replace("_", " ")
        if top_reason:
            st.caption(f"Top concern: **{top_reason}**")

        # Booking form
        st.markdown("---")
        st.subheader("📋 Booking Details")
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            advisor_type = st.selectbox("Advisor Type", ADVISOR_TYPES)
            pref_date = st.date_input(
                "Preferred Date",
                min_value=datetime.date.today(),
                max_value=datetime.date.today() + datetime.timedelta(days=90),
            )
            pref_time = st.selectbox("Preferred Time Slot", TIME_SLOTS)

        with form_col2:
            notes = st.text_area("Additional Notes (optional)", height=100,
                                 placeholder="Any context for the advisor...")
            gen_briefing = st.checkbox("Generate AI pre-meeting briefing for advisor", value=True)

        st.markdown("---")
        confirm_btn = st.button("✅ Confirm Booking", type="primary")

        if confirm_btn:
            # Duplicate guard
            existing_bookings = load_bookings()
            duplicate = (
                not existing_bookings.empty
                and len(existing_bookings[
                    (existing_bookings["student_id"] == str(booking_student_id))
                    & (existing_bookings["advisor_type"] == advisor_type)
                    & (existing_bookings["status"] == "Scheduled")
                ]) > 0
            )

            if duplicate:
                st.warning(
                    f"⚠️ Student {booking_student_id} already has a Scheduled appointment "
                    f"with a {advisor_type}. Proceeding will create an additional booking."
                )

            briefing_text = ""
            if gen_briefing:
                with st.spinner("Claude is preparing the advisor briefing..."):
                    try:
                        student_data = get_student_profile(booking_student_id)
                        briefing_text = generate_advisor_briefing(
                            booking_student_id, advisor_type, student_data
                        )
                    except Exception as e:
                        briefing_text = f"[Briefing unavailable: {e}]"

            booking_id = generate_booking_id()
            record = {
                "booking_id": booking_id,
                "student_id": str(booking_student_id),
                "risk_score": f"{b_risk:.4f}",
                "advisor_type": advisor_type,
                "date": str(pref_date),
                "time_slot": pref_time,
                "urgency": b_urgency,
                "notes": notes,
                "ai_briefing": briefing_text,
                "status": "Scheduled",
                "booked_at": pd.Timestamp.utcnow().isoformat(),
            }
            save_booking(record)
            st.session_state["last_booking_id"] = booking_id

            urgency_fn = {"Immediate": st.error, "Soon": st.warning, "Routine": st.success}
            urgency_fn.get(b_urgency, st.success)(
                f"✅ Booking **{booking_id}** confirmed — **{b_urgency}** priority | "
                f"Student {booking_student_id} → {advisor_type} on {pref_date} at {pref_time}"
            )

            if briefing_text and not briefing_text.startswith("[Briefing unavailable"):
                with st.expander("📄 View AI Briefing for Advisor", expanded=True):
                    st.markdown(f"**Prepared for:** {advisor_type}  \n"
                                f"**Student:** {booking_student_id}  \n"
                                f"**Urgency:** {b_urgency}\n\n---\n\n{briefing_text}")

# ════════════════════════════════════════════════════════════
# PAGE 11: BOOKING QUEUE
# ════════════════════════════════════════════════════════════
elif page == "📋 Booking Queue":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>📋 Booking Queue</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        All scheduled appointments — manage status and review advisor briefings
        </p>
    </div>
    """, unsafe_allow_html=True)

    bookings = load_bookings()

    if bookings.empty:
        st.info("No bookings yet. Go to **📅 Book Appointment** to schedule the first one.")
    else:
        # Convert risk_score to float for display
        bookings["risk_score_f"] = pd.to_numeric(bookings["risk_score"], errors="coerce")

        # ── Summary metrics ───────────────────────────────────
        total_b  = len(bookings)
        imm_b    = (bookings["urgency"] == "Immediate").sum()
        soon_b   = (bookings["urgency"] == "Soon").sum()
        routine_b = (bookings["urgency"] == "Routine").sum()
        scheduled_b = (bookings["status"] == "Scheduled").sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Bookings", total_b)
        c2.metric("🔴 Immediate", imm_b)
        c3.metric("🟡 Soon", soon_b)
        c4.metric("🟢 Routine", routine_b)
        c5.metric("Scheduled", scheduled_b)

        # ── Filters ───────────────────────────────────────────
        st.markdown("---")
        f1, f2, f3 = st.columns(3)
        with f1:
            status_filter = st.selectbox("Status", ["All", "Scheduled", "Completed", "Cancelled"])
        with f2:
            advisor_filter = st.selectbox("Advisor Type", ["All"] + ADVISOR_TYPES)
        with f3:
            urgency_filter = st.selectbox("Urgency", ["All", "Immediate", "Soon", "Routine"])

        display = bookings.copy()
        if status_filter != "All":
            display = display[display["status"] == status_filter]
        if advisor_filter != "All":
            display = display[display["advisor_type"] == advisor_filter]
        if urgency_filter != "All":
            display = display[display["urgency"] == urgency_filter]

        # Sort: Immediate first, then Soon, Routine; date ascending
        urgency_order = {"Immediate": 0, "Soon": 1, "Routine": 2}
        display["_urgency_rank"] = display["urgency"].map(urgency_order).fillna(3)
        display = display.sort_values(["_urgency_rank", "date"]).drop(columns=["_urgency_rank"])

        # ── Booking table ─────────────────────────────────────
        table_cols = ["booking_id", "student_id", "advisor_type", "date",
                      "time_slot", "urgency", "status", "risk_score_f"]
        show = display[[c for c in table_cols if c in display.columns]].copy()
        show.columns = [c.replace("_f", "").replace("_", " ").title() for c in show.columns]

        st.dataframe(
            show,
            use_container_width=True,
            height=300,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score", min_value=0, max_value=1, format="%.3f"
                )
            },
            hide_index=True,
        )

        # ── Status updater ────────────────────────────────────
        st.markdown("---")
        st.subheader("✏️ Update Booking Status")
        booking_ids = bookings["booking_id"].tolist()
        u1, u2, u3 = st.columns([2, 1, 1])
        with u1:
            update_id = st.selectbox("Select Booking ID", booking_ids)
        with u2:
            new_status = st.selectbox("New Status", ["Scheduled", "Completed", "Cancelled"])
        with u3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Update Status"):
                update_booking_status(update_id, new_status)
                st.success(f"Booking {update_id} updated to **{new_status}**")
                st.rerun()

        # ── Briefing viewer ───────────────────────────────────
        st.markdown("---")
        with st.expander("📄 View Advisor Briefing for Selected Booking"):
            view_id = st.selectbox("Select Booking", booking_ids, key="view_briefing_id")
            brow = bookings[bookings["booking_id"] == view_id]
            if not brow.empty:
                briefing = brow.iloc[0].get("ai_briefing", "")
                if briefing and str(briefing) not in ("", "nan"):
                    st.markdown(f"**Booking:** {view_id}  \n"
                                f"**Student:** {brow.iloc[0]['student_id']}  \n"
                                f"**Advisor:** {brow.iloc[0]['advisor_type']}  \n"
                                f"**Urgency:** {brow.iloc[0]['urgency']}\n\n---\n\n{briefing}")
                else:
                    st.info("No AI briefing was generated for this booking.")

# ════════════════════════════════════════════════════════════
# PAGE 12: KB ASSISTANT (RAG)
# ════════════════════════════════════════════════════════════
elif page == "💬 KB Assistant":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>💬 Counselor Knowledge Assistant</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Evidence-based answers from 12 institutional counseling protocols — powered by RAG + Claude
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    Ask **"what do I do when..."** counseling questions. Answers are grounded in 12 institutional
    protocols and cite their sources. This assistant supports — but does not replace — clinical judgment.
    """)

    # Load KB index (cached)
    kb_vectorizer, kb_matrix, kb_chunk_meta, kb_docs = load_kb_index()

    if kb_vectorizer is None:
        st.error("Knowledge base could not be loaded. Make sure `data/kb/` directory exists with .md files.")
        st.stop()

    # Session state
    if "kb_messages" not in st.session_state:
        st.session_state.kb_messages = []
    if "kb_api_history" not in st.session_state:
        st.session_state.kb_api_history = []
    if "kb_sources" not in st.session_state:
        st.session_state.kb_sources = []
    if "kb_prefill" not in st.session_state:
        st.session_state.kb_prefill = ""

    # ── Suggested questions ───────────────────────────────────
    st.markdown("**Try a suggested question:**")
    suggestions = [
        "What do I say to a student who stopped logging in for 3 weeks?",
        "How do I approach a student showing signs of suicidal ideation?",
        "What are the signs of academic distress I should look for?",
        "How do I set professional boundaries with an overly dependent student?",
        "What is the referral pathway for a student in acute mental health crisis?",
        "How should I adapt my approach for students from different cultural backgrounds?",
    ]
    sq_cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        if sq_cols[i % 3].button(suggestion[:55] + ("…" if len(suggestion) > 55 else ""),
                                  key=f"sq_{i}", use_container_width=True):
            st.session_state.kb_prefill = suggestion

    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────
    for idx, msg in enumerate(st.session_state.kb_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and idx < len(st.session_state.kb_sources):
                sources = st.session_state.kb_sources[idx // 2]
                if sources:
                    with st.expander("📚 Sources cited", expanded=False):
                        for src in sources:
                            st.markdown(f"- {src}")

    # ── Chat input ────────────────────────────────────────────
    prefill_val = st.session_state.kb_prefill
    user_query = st.chat_input("Ask a counseling question...",
                                key="kb_chat_input")

    # Use prefill if button was clicked
    if prefill_val and not user_query:
        user_query = prefill_val
        st.session_state.kb_prefill = ""

    if user_query:
        # Show user message
        st.session_state.kb_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.kb_api_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    chunks = retrieve_relevant_chunks(
                        user_query, kb_vectorizer, kb_matrix, kb_chunk_meta, top_k=4
                    )
                    answer, cited = generate_rag_answer(
                        user_query, chunks, st.session_state.kb_api_history[:-1]
                    )
                except Exception as e:
                    answer = f"⚠️ Error generating answer: {e}"
                    cited = []

            st.markdown(answer)
            if cited:
                with st.expander("📚 Sources cited", expanded=False):
                    for src in cited:
                        st.markdown(f"- {src}")

        st.session_state.kb_messages.append({"role": "assistant", "content": answer})
        st.session_state.kb_sources.append(cited)
        st.session_state.kb_api_history.append({"role": "assistant", "content": answer})

    # Clear button
    if st.session_state.kb_messages:
        if st.button("🗑️ Clear conversation", key="kb_clear"):
            st.session_state.kb_messages = []
            st.session_state.kb_api_history = []
            st.session_state.kb_sources = []
            st.rerun()

    # ── KB Explorer ───────────────────────────────────────────
    st.markdown("---")
    with st.expander(f"📖 Browse Knowledge Base ({len(kb_docs)} protocols indexed)", expanded=False):
        if kb_docs:
            doc_titles = [d["title"] for d in kb_docs]
            selected_doc = st.selectbox("Select a document to preview", doc_titles)
            selected_content = next(
                (d["content"] for d in kb_docs if d["title"] == selected_doc), ""
            )
            st.text_area("Document Preview (first 800 chars)",
                         selected_content[:800] + "..." if len(selected_content) > 800 else selected_content,
                         height=200, disabled=True)

# ════════════════════════════════════════════════════════════
# PAGE 13: TEMPORAL EARLY WARNING ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "⏰ Early Warning Analysis":

    st.markdown("""
    <div class='main-header'>
        <h2 style='margin:0'>⏰ Temporal Early Warning Analysis</h2>
        <p style='margin:0.3rem 0 0 0; opacity:0.85'>
        Quantifying intervention delay — when could we have acted vs. when we did?
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    **Research finding:** The model fires alerts at risk ≥ 0.66. But risk often rises above 0.50
    several weeks earlier — a detectable early signal. This page measures that gap and estimates
    how many weeks of intervention opportunity were missed. This is a publishable research contribution.
    """)

    # Load trajectories (cached — same as Page 9)
    with st.spinner("Loading risk trajectories..."):
        try:
            trajectories = compute_weekly_trajectories(top_n=10)
        except Exception as e:
            st.error(f"Could not load trajectories: {e}")
            st.stop()

    if not trajectories:
        st.warning("No trajectory data available.")
        st.stop()

    # ── Threshold controls ────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Detection Parameters")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        early_thresh = st.slider(
            "Early Signal Threshold", 0.40, 0.60, 0.50, 0.01,
            help="Risk level that triggers an early signal (below the alert threshold)"
        )
    with ctrl2:
        consec_weeks = st.slider(
            "Consecutive Weeks Required", 1, 4, 2,
            help="How many consecutive weeks above threshold before flagging as an early signal"
        )
    with ctrl3:
        st.metric("Alert Threshold", "0.66", help="Fixed production threshold — not adjustable")

    # Compute delay table with current parameters
    delay_table = compute_delay_table(trajectories, early_thresh, consec_weeks)

    valid_delays = delay_table.dropna(subset=["delay_weeks"])
    n_total = len(delay_table)
    n_with_delay = len(valid_delays)

    # ── Aggregate metrics ─────────────────────────────────────
    st.markdown("---")
    if n_with_delay > 0:
        mean_delay = valid_delays["delay_weeks"].mean()
        median_delay = valid_delays["delay_weeks"].median()
        max_delay = int(valid_delays["delay_weeks"].max())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Delay", f"{mean_delay:.1f} weeks")
        m2.metric("Median Delay", f"{median_delay:.1f} weeks")
        m3.metric("Students w/ Early Window", f"{n_with_delay} / {n_total}")
        m4.metric("Max Delay Observed", f"{max_delay} weeks")

        st.success(
            f"**Key Finding:** The system could have alerted counselors an average of "
            f"**{mean_delay:.1f} weeks earlier** across {n_with_delay} of {n_total} "
            f"high-risk students, using an early signal threshold of {early_thresh:.2f}."
        )
    else:
        st.warning("No students have a measurable early signal window with these parameters. "
                   "Try lowering the early signal threshold.")

    # ── Enhanced trajectory chart ─────────────────────────────
    st.markdown("---")
    st.subheader("📈 Risk Trajectories with Early Signal Markers")
    monitor_options = list(trajectories.keys())
    selected_temporal = st.multiselect(
        "Select students to display",
        options=monitor_options,
        default=monitor_options[:5],
        format_func=lambda sid: f"Student {sid} — Risk: {trajectories[sid]['final_risk']:.3f}",
    )

    if selected_temporal:
        fig = build_enhanced_trajectory_chart(trajectories, delay_table, selected_temporal)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "🟢 Green circle = early signal week | 🔴 Red × = actual alert fired | "
            "Yellow zone = early detection window (0.50–0.66)"
        )

    # ── Per-student delay table ───────────────────────────────
    st.markdown("---")
    st.subheader("📋 Per-Student Delay Analysis")
    display_delay = delay_table.copy()
    display_delay["delay_weeks"] = display_delay["delay_weeks"].apply(
        lambda x: int(x) if not pd.isna(x) else "No early window"
    )
    display_delay["early_signal_week"] = display_delay["early_signal_week"].apply(
        lambda x: int(x) if not pd.isna(x) else "—"
    )
    display_delay["actual_alert_week"] = display_delay["actual_alert_week"].apply(
        lambda x: int(x) if not pd.isna(x) else "Never alerted"
    )
    display_delay.columns = [c.replace("_", " ").title() for c in display_delay.columns]

    st.dataframe(display_delay, use_container_width=True, hide_index=True)

    csv_bytes = delay_table.to_csv(index=False).encode()
    st.download_button(
        "📥 Download Delay Analysis (CSV)",
        csv_bytes,
        file_name="temporal_delay_analysis.csv",
        mime="text/csv",
    )

    # ── Research narrative ────────────────────────────────────
    st.markdown("---")
    st.subheader("📄 Research Finding — Cost of Delay")
    st.caption("Claude AI generates a publishable-quality research narrative from the delay data above")

    if "temporal_narrative" not in st.session_state:
        st.session_state["temporal_narrative"] = ""

    if st.button("🔬 Generate Research Narrative", type="primary"):
        with st.spinner("Claude is writing the research finding..."):
            try:
                narrative = generate_temporal_narrative(delay_table)
                st.session_state["temporal_narrative"] = narrative
            except Exception as e:
                st.session_state["temporal_narrative"] = f"Error: {e}"

    if st.session_state.get("temporal_narrative"):
        st.markdown(
            f"<div class='agent-box'>{st.session_state['temporal_narrative']}</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "📥 Download Narrative (TXT)",
            st.session_state["temporal_narrative"].encode(),
            file_name="temporal_research_finding.txt",
            mime="text/plain",
        )

# ── FOOTER ────────────────────────────────────────────────────
st.markdown(
    "<div class='footer-custom'>"
    "🎓 Capstone Project &nbsp;·&nbsp; MS Data Analytics &nbsp;·&nbsp; "
    "Student Mental Health Early Warning System &nbsp;·&nbsp; "
    "OULAD Dataset &nbsp;·&nbsp; XGBoost (AUC 0.975) + Claude AI Agents &nbsp;·&nbsp; "
    "⚠️ For research purposes only — all decisions made by qualified counselors"
    "</div>",
    unsafe_allow_html=True
)
