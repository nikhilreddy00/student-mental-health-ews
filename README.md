<div align="center">

# 🎓 Student Mental Health Early Warning System

### *AI-powered early intervention — identifying struggling students weeks before crisis hits*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Claude](https://img.shields.io/badge/Claude-Sonnet%204.6-D97706?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.975-006400?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8B5CF6?style=for-the-badge)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

> **The gap between when a student starts struggling and when the university notices averages 4–6 weeks.**  
> This system closes that gap — detecting behavioral distress signals in real-time and surfacing them  
> to counselors with AI-generated explanations, intervention plans, and personalized outreach drafts.

<br/>

---

</div>

## ✨ What This System Does

| Without This System | With This System |
|---|---|
| Counselors learn about struggling students after failed exams or withdrawals | Counselors are alerted **3–5 weeks earlier** via behavioral pattern detection |
| No prioritization — all students treated equally | Students ranked by risk score; counselors focus limited time on highest-need cases |
| Counselors must manually dig through records before outreach | AI generates a full case packet: risk profile → intervention plan → personalized draft message |
| Institutional knowledge buried in manuals | Counselors ask questions in plain English; RAG system surfaces relevant guidance instantly |
| No historical view of how risk evolved | Week-by-week risk trajectory shows exactly *when* and *why* a student started declining |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              OULAD Dataset  (32,593 students · 7 modules)        │
│   studentInfo · studentVle · studentAssessment · Registrations   │
└────────────────────────┬─────────────────────────────────────────┘
                         │  Feature Engineering
                         ▼
              ┌─────────────────────┐
              │   master_dataset    │   ← Behavioral + academic features
              │   (32,593 rows)     │
              └──────────┬──────────┘
                         │  ML Training
          ┌──────────────┼───────────────┐
          ▼              ▼               ▼
   Logistic       Random Forest     XGBoost ★
   Regression                     AUC 0.975
                         │
                         │  SHAP Explainability
                         ▼
              ┌─────────────────────┐
              │  scored_students    │   ← Risk scores + tiers
              │  student_explanations│  ← Top-3 behavioral drivers
              └──────────┬──────────┘
                         │
     ┌───────────────────┼────────────────────────┐
     ▼                   ▼                         ▼
 AI Agent          Multi-Agent              RAG Knowledge
 (Tool Use)        Workflow                 Base (TF-IDF)
 Counselor Chat    Risk→Interventions       Counseling Docs
                   →Outreach Draft
                         │
                         ▼
          ┌──────────────────────────────┐
          │    Streamlit Dashboard        │
          │  7 Tabs · Real-time Streaming │
          └──────────────────────────────┘
```

---

## 🚀 Key Features

### 🤖 AI Counselor Chat Agent
Natural language interface powered by **Claude Sonnet 4-6 with tool use**. Counselors ask questions like:

> *"Show me the top 5 high-risk female students in module AAA"*  
> *"Why is student 30268 flagged? What should I do?"*  
> *"Which students in the 35-55 age group need immediate outreach?"*

Claude dynamically calls structured data tools, retrieves live results, and responds with actionable, compassionate guidance.

---

### 🧠 3-Agent Case Analysis Pipeline
For any student, a sequential multi-agent workflow generates a complete case packet via **streaming output**:

```
Agent 1: Risk Analyst          →  Structured behavioral risk profile
         ↓ (feeds output)
Agent 2: Intervention Advisor  →  3 ranked interventions (48h / 1 week / 4 weeks)
         ↓ (feeds strategy)
Agent 3: Outreach Writer       →  Warm, human, non-algorithmic student message
```

Each agent has a specialized system context. Output streams in real-time so counselors see the analysis build live.

---

### 📊 Temporal Early Warning Analysis
Quantifies exactly how many weeks earlier a tiered alert system would catch each student:

```
Early Signal Threshold  ──────●─────────────────────────────────────
(risk > 0.50, 2wk run)        │                                     
                              │◄─── 4-week intervention window ─────►│
Standard Alert Threshold ─────┼──────────────────────●──────────────
(risk ≥ 0.66)                 Wk 6                   Wk 10
```

Claude generates a publishable 3-paragraph research finding from the delay data in ACM/IEEE style.

---

### 📚 RAG Knowledge Base
12 institutional counseling guidance documents (crisis protocol, trauma-informed care, referral pathways, and more) made searchable via TF-IDF retrieval + Claude grounded Q&A. Every answer is cited to its source document.

---

### 📅 Smart Appointment Booking
Risk-aware scheduling with automatic urgency assignment:

| Risk Score | Urgency | Counselor Guidance |
|---|---|---|
| ≥ 0.80 | 🔴 Immediate | Contact within 24 hours |
| 0.66–0.79 | 🟡 Soon | Schedule within 48 hours |
| < 0.66 | 🟢 Routine | Monitor and check-in |

Every booking generates a confidential **AI pre-meeting briefing** tailored to the advisor type (Class Advisor / Therapist / Career Advisor / Professional Advisor).

---

## 📈 Model Performance

| Model | AUC-ROC | Accuracy | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.89 | 82% | 0.77 |
| Random Forest | 0.96 | 88% | 0.85 |
| **XGBoost ★** | **0.975** | **91%** | **0.88** |

**Top behavioral predictors (SHAP-ranked):**
1. `engagement_span` — How long the student stayed active in the course
2. `engagement_decline` — Week-over-week engagement deterioration slope
3. `dropout_modules` — Prior withdrawal history across modules
4. `mean_score` — Average assessment score
5. `active_days` — Consistency of VLE presence

**Temporal findings:** Adding a 0.50 amber threshold delivers **3–5 additional weeks** of intervention opportunity for ~80% of high-risk students.

---

## 🗂️ Project Structure

```
Capstone_MentalHealth/
│
├── code/
│   ├── dashboard.py          # Streamlit app — main entry point
│   ├── ai_agent.py           # Claude tool-use counselor chat agent
│   ├── multi_agent.py        # 3-agent streaming case analysis workflow
│   ├── risk_monitor.py       # Weekly trajectory simulation + crossing alerts
│   ├── temporal_analysis.py  # Intervention delay quantification
│   ├── booking.py            # Appointment booking + AI briefings
│   └── rag_kb.py             # TF-IDF retrieval + grounded knowledge Q&A
│
├── data/
│   ├── scored_students.csv       # ML predictions + risk tiers (32,593 students)
│   ├── student_explanations.csv  # SHAP top-3 behavioral drivers per student
│   ├── shap_values.csv           # Full SHAP value matrix
│   ├── master_dataset.csv        # Engineered feature matrix
│   ├── [raw OULAD CSVs]          # studentInfo, studentAssessment, etc.
│   └── kb/                       # 12 counseling guidance documents
│
├── models/
│   ├── xgb_model.pkl         # Primary XGBoost model (AUC 0.975)
│   ├── rf_model.pkl          # Random Forest model
│   ├── lr_model.pkl          # Logistic Regression baseline
│   └── scaler.pkl            # StandardScaler for feature normalization
│
├── plots/                    # 14 generated visualization outputs
│   ├── plot1_outcomes_demographics.png
│   ├── plot5_roc_pr_curves.png
│   ├── plot10_shap_summary.png
│   └── ...
│
├── requirements.txt
├── PROJECT_DOCUMENTATION.md  # Full technical documentation (18 sections)
└── Student_Mental_Health_EarlyWarningSystem.ipynb
```

---

## ⚡ Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/uvnikhil/student-mental-health-ews.git
cd student-mental-health-ews
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
# Create .env file in project root
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

Get your API key at [console.anthropic.com](https://console.anthropic.com).

### 3. Download the OULAD Dataset

The raw VLE interaction log (`studentVle.csv`) is too large for GitHub (433MB). Download it from the [Open University OULAD page](https://analyse.kmi.open.ac.uk/open_dataset) and place it in `data/`.

### 4. Run the Notebook (First Time Only)

Open `Student_Mental_Health_EarlyWarningSystem.ipynb` and run all cells to:
- Engineer features and build `master_dataset.csv`
- Train and save all models
- Generate SHAP explanations
- Create all 14 visualization plots

### 5. Launch the Dashboard

```bash
streamlit run code/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔬 Dataset

**OULAD — Open University Learning Analytics Dataset**  
Published by The Open University (UK) for academic research.

| Attribute | Value |
|---|---|
| Total student-module records | **32,593** |
| Modules (courses) | 7 (AAA–GGG) |
| At-risk students | ~4,100 (12.6%) |
| Behavioral features | VLE clicks · active days · engagement span · decline slope |
| Academic features | Assessment scores · submission rates · late submissions |
| Demographic features | Gender · Age band · IMD deprivation band |

> Note: Demographic features are used for context display only — not as ML predictors — to prevent discriminatory risk scoring.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| ML Model | XGBoost | Best AUC; tree-based = exact SHAP support |
| Explainability | SHAP TreeExplainer | Exact Shapley values per student |
| LLM | Claude Sonnet 4-6 | Tool use + streaming; ideal speed/capability balance |
| AI Framework | Anthropic Python SDK | Native tool use, streaming, multi-turn conversation |
| Dashboard | Streamlit | Python-native; handles state, chat, plots with zero HTML |
| Charts | Plotly + Matplotlib | Interactive trajectories + static SHAP visualizations |
| Retrieval | TF-IDF (scikit-learn) | Deterministic, fast, no embedding API needed |
| Environment | python-dotenv | Keeps API keys out of source code |

---

## 🧭 Dashboard Tabs

| Tab | What It Does |
|---|---|
| **Overview** | Cohort metrics, demographic breakdown, filterable student table |
| **AI Counselor Chat** | Multi-turn chat with Claude; tool call transparency panel |
| **Multi-Agent Case** | 3-agent streaming workflow: risk → interventions → outreach |
| **Risk Monitor** | Week-by-week trajectory playback with crossing alerts |
| **Temporal Analysis** | Intervention delay quantification + Plotly chart + research narrative |
| **Appointment Booking** | Risk-aware scheduling + AI pre-meeting briefings |
| **Knowledge Base** | TF-IDF RAG over 12 counseling guidance documents |

---

## ⚖️ Ethics and Design Principles

- **Counselor in the loop** — The system advises; humans decide. No autonomous action is taken.
- **Behavioral signals only** — Demographics are excluded from ML features. Risk is predicted from behavior, not identity.
- **No clinical diagnoses** — All output uses behavioral language. Specialist referral is always recommended for clinical concerns.
- **Privacy-by-design** — AI briefings for counselors never expose algorithmic language to students.
- **Explainability first** — Every risk flag comes with a ranked explanation of *why*. Black-box alerts are not acceptable in a wellbeing context.

---

## 📄 Documentation

Full technical documentation covering all 18 phases of the project — from dataset selection and feature engineering to SHAP explainability, agent architecture, and ethical design — is available in [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).

---

## 👥 Team

| Name | Role |
|---|---|
| **Nikhil Kumar Reddy** | Lead Developer — ML pipeline, AI agents, dashboard |
| **Nithin Sarva** | Contributor — Data analysis, model evaluation |
| **Jyothika Priyanka** | Contributor — Research, documentation, evaluation |

---

## 🙌 Acknowledgements

- **Dataset:** [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) — Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics Dataset. *Scientific Data, 4*, 170171.
- **LLM:** [Anthropic Claude](https://anthropic.com) — Claude Sonnet 4-6
- **Explainability:** [SHAP](https://github.com/slundberg/shap) — Lundberg & Lee (2017)

---

<div align="center">

**Built as a Capstone Project · 2026**

*Helping universities find struggling students before it's too late.*

</div>
