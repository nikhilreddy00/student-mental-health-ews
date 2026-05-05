# Student Mental Health Early Warning System
## Complete Project Documentation

---

## Table of Contents

1. [Project Overview and Motivation](#1-project-overview-and-motivation)
2. [Problem Statement](#2-problem-statement)
3. [Dataset: OULAD](#3-dataset-oulad)
4. [System Architecture](#4-system-architecture)
5. [Phase 1 — Data Engineering and Feature Construction](#5-phase-1--data-engineering-and-feature-construction)
6. [Phase 2 — Machine Learning Pipeline](#6-phase-2--machine-learning-pipeline)
7. [Phase 3 — Explainability with SHAP](#7-phase-3--explainability-with-shap)
8. [Phase 4 — AI Agent Layer (Claude Sonnet)](#8-phase-4--ai-agent-layer-claude-sonnet)
9. [Phase 5 — Multi-Agent Workflow](#9-phase-5--multi-agent-workflow)
10. [Phase 6 — Risk Monitor and Temporal Analysis](#10-phase-6--risk-monitor-and-temporal-analysis)
11. [Phase 7 — Appointment Booking System](#11-phase-7--appointment-booking-system)
12. [Phase 8 — RAG Knowledge Base](#12-phase-8--rag-knowledge-base)
13. [Phase 9 — Streamlit Dashboard](#13-phase-9--streamlit-dashboard)
14. [Technology Stack Justification](#14-technology-stack-justification)
15. [Ethical Considerations](#15-ethical-considerations)
16. [Results and Model Performance](#16-results-and-model-performance)
17. [Directory Structure](#17-directory-structure)
18. [How to Run the System](#18-how-to-run-the-system)

---

## 1. Project Overview and Motivation

This project is a **capstone-level AI system** that detects university students at risk of academic dropout or mental health decline — before crisis hits. The system combines classical machine learning, explainable AI, and large language model (LLM) agents to give university counselors an actionable, data-driven toolkit.

The core problem in student wellbeing programs is **latency**: counselors only learn about a struggling student after visible failure — a missed deadline, a failed exam, or a withdrawal notification. By then, the best intervention window has already passed. This system moves the identification point weeks earlier, when behavioral signals in the Virtual Learning Environment (VLE) first diverge from healthy engagement patterns.

The system is not designed to replace counselors. It is designed to make them more effective by surfacing the right students at the right time, explaining why each student was flagged, and preparing counselors with data-backed talking points before they make contact.

---

## 2. Problem Statement

Universities worldwide face a silent mental health crisis:
- 1 in 3 students experiences significant psychological distress during their studies.
- Dropout rates are strongly correlated with unaddressed mental health struggles.
- The gap between when a student begins disengaging and when the institution notices averages **4–6 weeks** in manual systems.
- Counseling departments are under-resourced relative to need, making it critical that their limited outreach capacity is directed at the highest-priority students first.

**What we needed to build:**
- A model that predicts which students are at risk, using behavioral signals already captured in existing university systems.
- An explainability layer so counselors understand *why* a student was flagged, not just *that* they were flagged.
- An AI assistant layer so counselors can query the data in natural language and get actionable briefings without needing data science expertise.
- A temporal analysis module that quantifies exactly how many weeks earlier this system could have intervened compared to conventional threshold-based alerts.

---

## 3. Dataset: OULAD

**Dataset:** Open University Learning Analytics Dataset (OULAD)
**Source:** The Open University (UK), publicly released for academic research
**Scale:** 32,593 student-module enrollment records across 7 courses (modules) and multiple presentation years

### Why OULAD?

OULAD was chosen over alternatives for several concrete reasons:

1. **Rich behavioral signals.** It contains Virtual Learning Environment (VLE) interaction logs — click counts by activity type and date — which serve as a behavioral proxy for student engagement. Most mental health datasets lack this granularity.

2. **Ground truth labels.** It includes final outcomes (`Pass`, `Fail`, `Withdrawn`, `Distinction`), allowing supervised learning. Students who `Withdrawn` or `Fail` are treated as the at-risk population.

3. **Demographic coverage.** It includes gender, age band, and Index of Multiple Deprivation (IMD) band — an established UK socioeconomic deprivation metric — enabling equity analysis across subgroups.

4. **Real-world scale.** 32,593 records is large enough for meaningful model training but small enough to run in a standard laptop environment, making the project reproducible.

5. **Temporal structure.** VLE logs are timestamped with `date` (days relative to course start), enabling week-by-week behavioral trajectory reconstruction.

### Key Data Files

| File | Description |
|---|---|
| `studentInfo.csv` | Demographics, module, presentation, final result |
| `studentVle.csv` | Daily VLE click logs per student per activity |
| `studentAssessment.csv` | Assessment scores and submission dates |
| `studentRegistration.csv` | Registration and unregistration dates |
| `assessments.csv` | Assessment metadata (type, weight, due date) |
| `vle.csv` | VLE activity metadata |
| `courses.csv` | Course (module) metadata |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OULAD Raw Data (7 CSV files)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    Feature Engineering
                             │
               ┌─────────────▼─────────────┐
               │    master_dataset.csv      │
               │  (32,593 rows × features)  │
               └─────────────┬─────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    ML Training Pipeline      │
              │  Logistic Regression         │
              │  Random Forest               │
              │  XGBoost  ← Primary Model    │
              └──────────────┬──────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │         SHAP Explainability          │
          │  per-student risk drivers ranked     │
          └──────────────────┬──────────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │           scored_students.csv                  │
     │   + student_explanations.csv + shap_values.csv │
     └───────────────────────┬───────────────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │            AI Agent Layer (Claude)             │
     │                                                │
     │  ai_agent.py    → Counselor Chat Agent         │
     │  multi_agent.py → 3-Agent Workflow              │
     │  risk_monitor.py→ Temporal Trajectories        │
     │  temporal_analysis.py → Delay Quantification  │
     │  booking.py     → Appointment System           │
     │  rag_kb.py      → Knowledge Base Q&A           │
     └───────────────────────┬───────────────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │          Streamlit Dashboard (dashboard.py)    │
     │  7 tabs: Overview · Chat · Multi-Agent ·       │
     │  Monitor · Temporal · Booking · Knowledge Base  │
     └───────────────────────────────────────────────┘
```

---

## 5. Phase 1 — Data Engineering and Feature Construction

### What We Did

Raw OULAD tables are relational. No single file contains everything needed for prediction. The data engineering phase joins all tables and computes behavioral features that serve as inputs to the ML models.

### Feature Construction

All features are computed per student-module enrollment (a student taking multiple modules generates one row per module).

#### VLE Engagement Features

| Feature | Computation | Why It Matters |
|---|---|---|
| `total_clicks` | Sum of all VLE interactions across the course | Proxy for overall engagement volume |
| `active_days` | Count of unique days with at least one VLE click | Measures consistency of engagement, not just total volume |
| `engagement_span` | Max VLE date minus min VLE date for this student | How long the student stayed active in the course |
| `engagement_decline` | Slope of a linear regression on weekly clicks over time | Detects gradual disengagement even when total clicks look acceptable |
| `peak_week_clicks` | Maximum single-week click count | Establishes the student's personal engagement ceiling |

**Why engagement_decline specifically?** A student who had 500 clicks in week 1 but 10 clicks by week 8 is more at risk than a student who consistently had 100 clicks per week. The linear slope captures this deterioration pattern that raw totals obscure.

#### Assessment Performance Features

| Feature | Computation | Why It Matters |
|---|---|---|
| `mean_score` | Average score across all submitted assessments | Baseline academic performance |
| `score_std` | Standard deviation of assessment scores | High variance = inconsistent performance, potential crisis events |
| `submission_rate` | Fraction of assessments actually submitted | Unsubmitted work is a stronger distress signal than a low score |
| `late_submission_rate` | Fraction submitted after the due date | Lateness pattern correlates with life disruption |

#### Registration and Dropout Features

| Feature | Computation | Why It Matters |
|---|---|---|
| `dropout_modules` | Count of other modules where this student withdrew | Prior withdrawal pattern predicts future withdrawal |
| `days_until_unregistration` | Days from course start to withdrawal (if applicable) | Early withdrawal is more acute than late withdrawal |

#### Demographic Features (Contextual, Not Predictive)

`gender`, `age_band`, and `imd_band` are included in the final output for counselor context and equity analysis, but were kept out of the primary ML features to avoid building a discriminatory model. Risk should be predicted from behavior, not identity.

### Target Variable

`is_at_risk` is a binary label:
- **1 (At Risk):** Final result is `Withdrawn` or `Fail`
- **0 (Not At Risk):** Final result is `Pass` or `Distinction`

This is a deliberate simplification. In reality, a student might pass while struggling, or withdraw for legitimate reasons. But the behavioral signals that precede withdrawal and failure are real and consistent enough to justify this framing for an early intervention system.

### Output

The master dataset (`data/master_dataset.csv`) contains 32,593 rows and all engineered features, ready for ML training.

---

## 6. Phase 2 — Machine Learning Pipeline

### Model Selection Strategy

Three models were trained and compared:

| Model | Why Included |
|---|---|
| **Logistic Regression** | Baseline; maximally interpretable; strong when features are linearly separable |
| **Random Forest** | Strong out-of-box nonlinear model; robust to outliers; useful for feature importance comparison |
| **XGBoost** | State-of-art gradient boosting; handles class imbalance well; best AUC in practice; directly compatible with SHAP |

**XGBoost was selected as the primary model** because it achieved the highest AUC (0.975) and produced the most calibrated risk probabilities — which are essential since the system uses continuous risk scores (not just binary labels) to tier students.

### Training Setup

**Train/test split:** 80/20 stratified split (stratified to preserve the class imbalance ratio in both splits).

**Class imbalance:** The dataset is imbalanced — most students pass. We used `scale_pos_weight` in XGBoost to weight the minority (at-risk) class more heavily during training. This is critical for a system where false negatives (missing a struggling student) are far more costly than false positives (flagging a student who is fine).

**Feature scaling:** StandardScaler was applied. All feature values are standardized to zero mean and unit variance before training and saved as `models/scaler.pkl`. The same scaler is applied during inference to maintain consistency.

### Saved Artifacts

| File | Purpose |
|---|---|
| `models/xgb_model.pkl` | Primary prediction model |
| `models/rf_model.pkl` | Comparison model |
| `models/lr_model.pkl` | Baseline model |
| `models/scaler.pkl` | Feature scaler for consistent inference |

### Risk Scoring

After training, XGBoost's `predict_proba` method produces a continuous score between 0 and 1 for each student. This score is stored in `scored_students.csv` as `risk_score`.

**Risk tiers are assigned by threshold:**

| Tier | Threshold | Counselor Interpretation |
|---|---|---|
| High | ≥ 0.66 | Immediate outreach needed |
| Medium | 0.40 – 0.65 | Monitor closely; schedule proactive check-in |
| Low | < 0.40 | Routine; no immediate action required |

The 0.66 threshold was chosen to balance precision and recall at the point where intervention is cost-effective for the counseling team. It is configurable.

---

## 7. Phase 3 — Explainability with SHAP

### Why Explainability is Non-Negotiable

A counselor who receives a list of student IDs with risk scores and no explanation is not meaningfully helped. They need to know:
- **Why** is this student flagged?
- **Which specific behavior** is most concerning?
- **How urgent** is the concern?

Without answers to these questions, the system is a black box that counselors will not trust or use effectively. Explainability is not a nice-to-have — it is load-bearing for adoption.

### What SHAP Does

SHAP (SHapley Additive exPlanations) is a game-theoretic framework that assigns each feature a contribution score (SHAP value) for each individual prediction. The SHAP value for feature `f` on student `s` represents: "how much did feature `f` push student `s`'s risk score above or below the baseline?"

SHAP values are:
- **Signed:** Positive = pushes toward high risk; Negative = pushes toward low risk
- **Additive:** The sum of all SHAP values equals the difference between the model's prediction and the expected prediction
- **Local:** Specific to each student, not global averages

### How We Used SHAP

1. We computed SHAP values using `TreeExplainer` (the efficient exact SHAP algorithm for tree-based models like XGBoost).
2. For each student, we ranked their features by absolute SHAP magnitude.
3. We stored the top 3 risk drivers per student as `top_reason`, `reason_2`, `reason_3` — human-readable feature names.
4. We stored the SHAP magnitude of the primary driver as `top_shap` — a numeric measure of how strongly that factor drives the risk.

### Output Files

`data/student_explanations.csv` contains one row per student-module with:
- `risk_score`, `risk_tier`
- `top_reason`, `reason_2`, `reason_3` — the three most impactful behavioral signals
- `top_shap` — the SHAP magnitude of the primary driver
- Raw behavioral metrics (`engagement_span`, `mean_score`, etc.) for display in the dashboard

`data/shap_values.csv` contains the full SHAP value matrix for all features and all students.

### Visualization

The dashboard includes SHAP visualizations:
- **SHAP Summary Plot:** All features ranked by mean absolute SHAP value across the dataset — shows which features matter most globally
- **SHAP Bar Plot:** Per-feature contribution for the selected student
- **SHAP Dependence Plot:** How `engagement_span` relates to risk score across all students
- **Tier Heatmap:** How SHAP contributions vary by risk tier

---

## 8. Phase 4 — AI Agent Layer (Claude Sonnet)

**File:** `code/ai_agent.py`

### Design Philosophy

The AI agent layer is built on **Anthropic's Claude Sonnet 4-6** using the Anthropic Python SDK. It implements the **tool use (function calling)** pattern, which allows Claude to decide which structured data functions to call based on a counselor's natural language question.

This design was chosen over a simple RAG approach because counselors ask compound questions ("Show me all high-risk female students in module AAA") that require dynamic query composition, not just document retrieval.

### Tools Available to the Agent

The agent has four tools registered:

#### `query_high_risk_students(limit, tier)`
Returns the top N students sorted by risk score, optionally filtered by risk tier. Used when a counselor asks "Who needs help most right now?" or "Give me the top 10 high-risk students."

#### `get_student_profile(student_id)`
Returns a complete profile for a specific student: demographics, module, behavioral metrics, and risk details. Used when a counselor says "Tell me about student 12345."

#### `get_shap_explanation(student_id)`
Returns the top 3 behavioral risk drivers with their SHAP magnitudes. Used when a counselor asks "Why is student 12345 flagged?" or "What's driving the risk for this student?"

#### `search_students(gender, age_band, module, imd_band, tier, limit)`
Filters students by any combination of demographic and risk criteria. Used for queries like "Show me at-risk female students over 35 in module BBB."

### How Tool Use Works

1. The counselor types a natural language question in the dashboard chat interface.
2. The question, along with prior conversation history, is sent to Claude with the four tool definitions.
3. Claude decides whether to call a tool (and which one) or answer directly from context.
4. If Claude calls a tool, the Python function executes against the in-memory CSV data and returns JSON results.
5. Claude receives the JSON result and formulates a natural language response.
6. This loop continues until Claude's `stop_reason` is `end_turn` (no more tools needed).
7. The final text response is displayed in the dashboard.

### System Prompt

The system prompt instructs Claude to:
- Be concise and actionable (counselors are busy)
- Always explain WHY a student is at risk using specific behavioral data
- Suggest practical next steps tied to actual risk drivers
- Be compassionate — these are real students
- Never make clinical diagnoses
- Format student lists as numbered lists with key statistics

### Risk Narrative Generation

Beyond the interactive chat agent, `generate_risk_narrative(student_id)` generates a standalone 3-sentence counselor-facing summary:
- Sentence 1: State the risk tier and most concerning metric
- Sentence 2: Describe the behavioral pattern with concrete numbers
- Sentence 3: Recommend the single most appropriate immediate counselor action

This is used in the dashboard to auto-generate a quick briefing without requiring the counselor to ask a question.

---

## 9. Phase 5 — Multi-Agent Workflow

**File:** `code/multi_agent.py`

### Why Multiple Agents?

A single Claude call can answer a question. But generating a comprehensive student case packet — risk profile, ranked interventions, and a personalized outreach message — from a single prompt tends to produce generic output. Each task requires a different cognitive posture:

- **Risk analysis** requires objective, clinical language and precise quantification
- **Intervention planning** requires knowledge of support tiers and practical constraints
- **Outreach writing** requires warmth, sensitivity, and deliberate avoidance of data-science language

Separating these into three specialized agents produces better outputs because each agent's system context is optimized for its specific task. The output of Agent 1 becomes context for Agent 2, and Agent 2's strategy informs Agent 3's tone.

### Agent Architecture

#### Agent 1: Risk Analyst
- **Input:** Raw behavioral data (engagement metrics, SHAP drivers)
- **Task:** Produce a structured 4-5 sentence risk profile covering urgency, behavioral signals, and recommended intervention timeline (Immediate/Soon/Monitor)
- **Why separate:** A risk analyst needs to be direct and clinical. Mixing this with empathetic student communication would dilute both.

#### Agent 2: Intervention Advisor
- **Input:** Agent 1's risk profile + student context
- **Task:** Produce 3 ranked interventions (Immediate within 48h / Short-term within 1 week / Ongoing monitor 4 weeks), each with a rationale tied to this student's specific behavioral pattern
- **Why separate:** Intervention planning requires different knowledge — support pathways, escalation protocols, advisor types — than risk quantification.

#### Agent 3: Counselor Note Writer
- **Input:** Agent 2's intervention strategy (first 200 chars, for tone guidance)
- **Task:** Write a 4-5 sentence warm outreach message to send to the student
- **Critical constraint:** Must NOT mention AI, algorithms, risk scores, data analysis, or surveillance. The message must feel genuinely human.
- **Why separate:** This agent needs to write in a completely different register — empathetic, non-alarming, inviting. Exposing it to the full clinical language of Agents 1 and 2 would contaminate its output.

### Streaming

All three agents use Claude's **streaming API** (`client.messages.stream`). The dashboard displays each agent's output in real-time as tokens arrive, giving counselors visibility into the analysis process and making the system feel responsive rather than frozen during a ~6-8 second generation.

The generator pattern (`yield ("agent1", chunk)`) allows the dashboard to update incrementally without blocking the UI thread.

---

## 10. Phase 6 — Risk Monitor and Temporal Analysis

**Files:** `code/risk_monitor.py`, `code/temporal_analysis.py`

### Risk Monitor: Weekly Trajectory Simulation

**File:** `code/risk_monitor.py`

Because OULAD is a historical dataset, we do not have real-time streaming data. Instead, we reconstruct week-by-week behavioral trajectories for the top N high-risk students using their actual VLE engagement logs.

#### Trajectory Algorithm

For each student, for each course week:
1. Compute actual VLE click count that week from `studentVle.csv`
2. Compute `relative_engagement = clicks_this_week / personal_peak_clicks`
3. Simulate risk score: `0.08 + (final_risk - 0.08) × ((1 - relative_engagement) ^ 0.6)`

This formula has two key properties:
- When relative engagement = 1.0 (at personal peak), simulated risk is near baseline (0.08)
- When relative engagement = 0.0 (no engagement), simulated risk equals the student's final XGBoost-predicted risk score

The `^ 0.6` exponent creates a **concave trajectory** — risk rises steeply as engagement first drops, then more gradually. This reflects the real-world pattern where the first significant drop in engagement is the most alarming signal.

#### Crossing Alert Generation

When a student's simulated risk crosses the 0.66 High threshold during trajectory playback, `generate_crossing_alert()` calls Claude to produce a 2-sentence urgent counselor alert. The prompt includes the student ID, course week, the exact score transition, and the primary behavioral concern.

### Temporal Analysis: Quantifying the Intervention Window

**File:** `code/temporal_analysis.py`

This module answers a specific research question: **"How many weeks before the standard alert would a lower-threshold early warning system have flagged this student?"**

#### Methodology

For each student trajectory:

1. **Early Signal Week:** First week where risk score exceeds 0.50 for 2 or more consecutive weeks (sustained, not a spike)
2. **Actual Alert Week:** First week where risk score crosses the standard 0.66 threshold
3. **Delay:** Actual Alert Week − Early Signal Week

#### Interpretation

If a student's early signal appears at week 6 and the standard alert fires at week 10, the delay is 4 weeks. This means a system monitoring the 0.50 threshold with a 2-week sustain requirement would have flagged this student 4 weeks earlier — when lighter-touch intervention (a check-in email) might have been sufficient, rather than urgent referral.

#### Temporal Narrative Generation

`generate_temporal_narrative()` takes the delay table and calls Claude with a structured academic writing prompt. Claude produces a 3-paragraph research finding in ACM/IEEE conference style:
- Paragraph 1: Quantitative finding with specific delay numbers
- Paragraph 2: What behavioral intervention opportunities existed during the delay window
- Paragraph 3: Recommendation for a tiered alert system (amber at 0.50, red at 0.66)

This narrative is directly exportable for academic publications or institutional reports.

#### Trajectory Visualization

`build_enhanced_trajectory_chart()` uses Plotly to render:
- Individual student risk trajectories over course weeks
- Yellow shaded "Early Detection Zone" (0.50–0.66)
- Red shaded "Alert Zone" (>0.66)
- Green markers at the early signal week
- Red X markers at the actual alert week
- Delay annotations (e.g., "4wk delay") between the two markers

---

## 11. Phase 7 — Appointment Booking System

**File:** `code/booking.py`

### Purpose

Identifying at-risk students is only useful if it triggers a human action. The booking system closes the loop between detection and intervention by allowing counselors to directly schedule a meeting with a student from within the dashboard, without switching to a separate calendar system.

### Urgency-Aware Booking

Urgency is automatically computed from risk score:

| Risk Score | Urgency Level |
|---|---|
| ≥ 0.80 | Immediate |
| 0.66 – 0.79 | Soon |
| < 0.66 | Routine |

This urgency level is stored with the booking and displayed in the booking management table, allowing counselors to prioritize their scheduled meetings by urgency rather than calendar order.

### Advisor Type Matching

Four advisor types are available: Class Advisor, Professional Advisor, Career Advisor, Therapist. The system stores the advisor type with each booking to support routing to the appropriate professional based on the nature of the risk signals.

### AI Pre-Meeting Briefing

When a booking is confirmed, `generate_advisor_briefing(student_id, advisor_type, student_data)` calls Claude to produce a 4-sentence confidential pre-meeting briefing for the specific advisor type. The briefing covers:
1. Why this student was flagged and what behavioral pattern the data shows
2. The two most important things the advisor should listen for
3. One specific opening question that builds trust without alarming the student
4. Any boundary or referral consideration relevant to this advisor type

**Critical privacy constraint:** The briefing explicitly excludes any mention of AI, algorithms, risk scores, or data analysis. It is framed as clinical support notes, not data science output.

### Booking Persistence

Bookings are stored in `data/bookings.csv` with the full schema:
`booking_id, student_id, risk_score, advisor_type, date, time_slot, urgency, notes, ai_briefing, status, booked_at`

Status transitions: `Scheduled` → `Completed` or `Cancelled`, managed from the booking management tab in the dashboard.

---

## 12. Phase 8 — RAG Knowledge Base

**File:** `code/rag_kb.py`

### Purpose

Counselors frequently need guidance on *how* to handle a specific situation they have identified. The knowledge base bridges the gap between detection (knowing a student is at risk) and response (knowing what to do about it). Rather than requiring counselors to search through institutional manuals, the RAG system surfaces the most relevant guidance on demand.

### Knowledge Base Content

12 institutional-quality counseling guidance documents covering:

| Document | Coverage |
|---|---|
| `01_crisis_protocol.md` | Suicide/self-harm emergency response, step-by-step |
| `02_engagement_dropout.md` | Academic disengagement patterns and re-engagement strategies |
| `03_first_contact_scripts.md` | Sample scripts for initiating first counselor contact |
| `04_academic_distress_signs.md` | Recognizing academic distress signals |
| `05_referral_pathways.md` | When and how to refer to clinical services |
| `06_trauma_informed_care.md` | Trauma-sensitive counseling principles |
| `07_boundary_setting.md` | Maintaining appropriate professional boundaries |
| `08_online_learning_challenges.md` | Unique challenges of distance/online learners |
| `09_cultural_considerations.md` | Culturally responsive counseling practices |
| `10_anxiety_depression_support.md` | Supporting students with anxiety and depression |
| `11_grief_loss_protocol.md` | Grief and bereavement support guidance |
| `12_academic_misconduct_support.md` | Supporting students facing academic misconduct |

### Retrieval Architecture

#### Step 1: Document Chunking

Each `.md` file is split on double newlines into semantic chunks. Chunks shorter than 20 words are discarded (too brief to be meaningful context). The remaining chunks form the retrieval corpus.

#### Step 2: TF-IDF Index

A TF-IDF (Term Frequency–Inverse Document Frequency) index is built over all chunks using scikit-learn's `TfidfVectorizer`:
- `ngram_range=(1,2)`: Unigrams and bigrams — captures phrases like "suicidal ideation" not just "suicidal"
- `max_df=0.85`: Ignores terms appearing in more than 85% of chunks (too common to discriminate)
- `sublinear_tf=True`: Logarithmic term frequency dampening to prevent high-frequency terms from dominating

#### Step 3: Query Retrieval

When a counselor asks a question, it is vectorized using the same TF-IDF vectorizer. Cosine similarity is computed between the query vector and all chunk vectors. The top 4 chunks are selected, with a constraint of at most 2 chunks per document (to avoid over-representing a single document and ensure breadth of coverage).

#### Step 4: Grounded Answer Generation

Retrieved chunks are assembled into a context block and passed to Claude with a strict system prompt:
- Ground every answer in the retrieved excerpts
- Cite sources using `[Source: Document Title]` notation
- If excerpts don't cover the question, flag this before answering from general practice
- Be specific and actionable
- Keep answers concise (150–300 words) using bullet points for steps

### Why TF-IDF Over Embeddings?

For this application, TF-IDF was deliberately chosen over a vector embedding approach (e.g., OpenAI embeddings or sentence-transformers) for three reasons:

1. **No external dependency.** TF-IDF runs entirely on CPU with scikit-learn. Embedding-based retrieval would require either a remote API call (latency, cost, privacy) or a local embedding model (setup complexity).

2. **Keyword precision.** Counseling queries tend to use specific, consistent terminology ("suicidal ideation", "crisis protocol", "referral pathway") that TF-IDF handles well. The marginal benefit of semantic similarity is lower than in open-domain search.

3. **Full reproducibility.** The TF-IDF index is deterministic, serializable, and requires no GPU or special hardware. Any university IT department can deploy and maintain this system.

---

## 13. Phase 9 — Streamlit Dashboard

**File:** `code/dashboard.py`

### Why Streamlit?

Streamlit was chosen as the frontend framework because:
- It renders Python code as a web app with zero HTML/JS required
- It natively supports chat message components, data tables, plots, and forms
- It has a session state mechanism for maintaining conversation history and cached data across user interactions
- It is deployable on any server that can run Python, including university-managed infrastructure

### Dashboard Tabs

The dashboard is organized into 7 navigation sections accessible from the left sidebar:

#### Tab 1: Overview and Cohort Analysis
- Summary metrics: total students, percentage high/medium/low risk, average risk score
- Demographic breakdown by risk tier
- Module-level risk distribution
- Student filter table with download capability

#### Tab 2: AI Counselor Chat
- Multi-turn chat interface powered by the Claude Sonnet agent (`ai_agent.py`)
- Conversation history persisted in `st.session_state`
- Tool call transparency: expanding panel shows which tools Claude called and with what parameters
- Auto-generated risk narrative for any student ID entered

#### Tab 3: Multi-Agent Case Analysis
- Student ID input triggers the full 3-agent workflow (`multi_agent.py`)
- Three panels render in real-time via streaming: Risk Profile, Intervention Plan, Outreach Draft
- Designed to produce a complete case packet in ~8 seconds

#### Tab 4: Risk Monitor
- Top N high-risk student selector (configurable)
- Animated week-by-week trajectory playback
- Crossing alert display when a student's trajectory hits the 0.66 threshold
- Claude-generated alert text per crossing event

#### Tab 5: Temporal Analysis
- Multi-student early detection analysis
- Plotly chart with early signal / alert markers and delay annotations
- Claude-generated research narrative quantifying the intervention window gain

#### Tab 6: Appointment Booking
- Student risk lookup by ID before booking
- Advisor type and time slot selection
- Urgency auto-assignment from risk score
- Booking confirmation with AI pre-meeting briefing
- Booking management table with status updates

#### Tab 7: Knowledge Base
- Full-text chat interface for counseling guidance queries
- Retrieved source citations displayed below each answer
- Chat history maintained within session

### Session State Architecture

The dashboard uses `st.session_state` as an in-session store for:
- `chat_messages`: Full conversation history for the counselor chat agent
- `kb_history`: Conversation history for the knowledge base chat
- `trajectories_cache`: Computed weekly trajectories (expensive to recompute)
- `kb_index_cache`: TF-IDF index (built once per session)
- `delay_table_cache`: Temporal analysis results

Caching computed artifacts in session state avoids recomputing the trajectory simulation and TF-IDF index on every Streamlit rerun (which occurs after every user interaction).

---

## 14. Technology Stack Justification

| Technology | Role | Justification |
|---|---|---|
| **Python 3.x** | Primary language | Universal in data science; required for the Anthropic SDK |
| **pandas** | Data manipulation | Standard; required for OULAD CSV handling and feature engineering |
| **scikit-learn** | ML utilities + TF-IDF | StandardScaler, TF-IDF vectorizer, cosine similarity; lightweight and stable |
| **XGBoost** | Primary ML model | Best AUC in benchmarks; tree-based, so directly compatible with TreeExplainer SHAP |
| **SHAP** | Model explainability | Gold standard for tree-based model explanations; produces exact Shapley values, not approximations |
| **Anthropic SDK** | LLM integration | Required for Claude Sonnet 4-6; native tool use and streaming support |
| **Claude Sonnet 4-6** | LLM backbone | `claude-sonnet-4-6` — best balance of capability, speed, and cost for this use case; supports tool use and streaming |
| **Streamlit** | Dashboard frontend | Python-native web app framework; handles state, widgets, and plotting without HTML/JS |
| **Plotly** | Interactive charts | Interactive trajectory charts with hover, zoom, and annotation support |
| **matplotlib + seaborn** | Static plots | SHAP visualizations, confusion matrices, feature importance charts |
| **python-dotenv** | Environment management | Keeps API keys out of source code |
| **joblib** | Model serialization | Standard for scikit-learn pipeline persistence |
| **uuid** | Booking ID generation | Unique booking identifiers without a database |

### Why Claude Sonnet 4-6 Specifically?

`claude-sonnet-4-6` was selected over other model tiers because:
- It supports **tool use** (function calling) natively — required for the counselor chat agent
- It supports **streaming** — required for the multi-agent real-time display
- It has sufficient reasoning capability for clinical briefing generation and academic writing tasks
- It is fast enough for dashboard interactions that feel responsive (sub-5 second per turn)
- It is more cost-effective than Opus for tasks that do not require the highest reasoning ceiling

---

## 15. Ethical Considerations

### Transparency and Non-Deception

The system is transparent about its nature to the counselor at every level:
- Risk scores and SHAP drivers are always displayed alongside AI-generated text
- The AI counselor agent identifies itself as an assistant, not a clinician
- Pre-meeting briefings explicitly state they come from behavioral data, not diagnosis

However, outreach messages to students deliberately do not mention AI or data analysis — not to deceive, but because disclosing algorithmic surveillance in a wellbeing outreach message would cause harm (anxiety, distrust, disengagement). The ethical standard is different for the counselor-facing and student-facing outputs.

### Non-Discrimination

Demographic features (`gender`, `age_band`, `imd_band`) are displayed for context but are intentionally excluded from the primary ML features used for risk prediction. Risk scores are computed from behavioral signals only. This design choice prevents the model from learning to associate deprivation or gender with risk in ways that could encode or amplify existing institutional biases.

### Counselor in the Loop

The system never takes autonomous action. Every alert, intervention plan, and booking requires a counselor to review and confirm. The AI layer advises; the human decides. This is the appropriate design for a mental health system where the cost of errors — both false positives (unwarranted contact) and false negatives (missed crises) — is high.

### Data Privacy

All student data in this project uses the anonymized OULAD dataset (student IDs are already pseudonymized, no names or contact information). In a production deployment:
- All data would remain on university-controlled infrastructure
- API calls to Claude would be made under a data processing agreement
- Student records would be access-controlled to authorized counselors only
- Audit logs of all AI queries and generated outputs would be retained

### Clinical Disclaimer

The system explicitly does not make clinical diagnoses. All outputs use behavioral language ("engagement decline", "assessment submission gap") rather than clinical language ("depression", "anxiety disorder"). The AI system prompt and all agent prompts contain explicit instructions to recommend specialist referral rather than offering clinical conclusions.

---

## 16. Results and Model Performance

### Classification Metrics

| Model | AUC-ROC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | ~0.89 | ~0.82 | ~0.74 | ~0.81 | ~0.77 |
| Random Forest | ~0.96 | ~0.88 | ~0.83 | ~0.87 | ~0.85 |
| **XGBoost** | **0.975** | **~0.91** | **~0.87** | **~0.89** | **~0.88** |

### Risk Tier Distribution (32,593 students)

| Tier | Approx. Count | Percentage |
|---|---|---|
| High Risk | ~4,100 | ~12.6% |
| Medium Risk | ~7,200 | ~22.1% |
| Low Risk | ~21,300 | ~65.3% |

### Temporal Analysis Results

From the 10 highest-risk student trajectories analyzed:
- Average delay between early signal (0.50 sustained) and standard alert (0.66): **3–5 weeks**
- Maximum observed delay: **6+ weeks**
- Students with no early signal window (risk rises too steeply for tiered detection to help): ~20%

This means a tiered alert system adding an amber 0.50 threshold would deliver an average of **3–5 additional weeks of intervention opportunity** for approximately 80% of high-risk students.

### Top SHAP Risk Drivers (Global, Across All Students)

1. `engagement_span` — How long the student stayed active in the course
2. `engagement_decline` — Rate of engagement deterioration over the semester
3. `dropout_modules` — Prior withdrawal history
4. `mean_score` — Average assessment score
5. `active_days` — Consistency of VLE engagement

These rankings are consistent with the academic literature on student attrition: sustained low engagement and declining behavioral presence are stronger predictors of dropout than any single demographic factor.

---

## 17. Directory Structure

```
Capstone_MentalHealth/
├── .env                              # ANTHROPIC_API_KEY (not committed)
├── requirements.txt                  # Python dependencies
│
├── code/
│   ├── dashboard.py                  # Streamlit main app (entry point)
│   ├── ai_agent.py                   # Claude tool-use counselor agent
│   ├── multi_agent.py                # 3-agent streaming workflow
│   ├── risk_monitor.py               # Weekly trajectory simulation + alerts
│   ├── temporal_analysis.py          # Early detection delay quantification
│   ├── booking.py                    # Appointment booking + AI briefings
│   └── rag_kb.py                     # TF-IDF retrieval + grounded Q&A
│
├── data/
│   ├── studentInfo.csv               # Raw OULAD: demographics + outcomes
│   ├── studentVle.csv                # Raw OULAD: daily VLE click logs
│   ├── studentAssessment.csv         # Raw OULAD: assessment scores
│   ├── studentRegistration.csv       # Raw OULAD: registration events
│   ├── assessments.csv               # Raw OULAD: assessment metadata
│   ├── vle.csv                       # Raw OULAD: VLE activity metadata
│   ├── courses.csv                   # Raw OULAD: course (module) metadata
│   ├── master_dataset.csv            # Engineered feature matrix (32,593 rows)
│   ├── scored_students.csv           # ML predictions + risk tiers
│   ├── student_explanations.csv      # SHAP top-3 drivers per student
│   ├── shap_values.csv               # Full SHAP value matrix
│   ├── bookings.csv                  # Appointment booking records
│   └── kb/                           # Counseling knowledge base
│       ├── 01_crisis_protocol.md
│       ├── 02_engagement_dropout.md
│       ├── 03_first_contact_scripts.md
│       ├── 04_academic_distress_signs.md
│       ├── 05_referral_pathways.md
│       ├── 06_trauma_informed_care.md
│       ├── 07_boundary_setting.md
│       ├── 08_online_learning_challenges.md
│       ├── 09_cultural_considerations.md
│       ├── 10_anxiety_depression_support.md
│       ├── 11_grief_loss_protocol.md
│       └── 12_academic_misconduct_support.md
│
├── models/
│   ├── xgb_model.pkl                 # Trained XGBoost model
│   ├── rf_model.pkl                  # Trained Random Forest model
│   ├── lr_model.pkl                  # Trained Logistic Regression model
│   └── scaler.pkl                    # StandardScaler for feature normalization
│
├── plots/                            # Generated visualization outputs
│   ├── plot1_outcomes_demographics.png
│   ├── plot2_registration_patterns.png
│   ├── plot3_behavioral_signals.png
│   ├── plot4_correlation_heatmap.png
│   ├── plot5_roc_pr_curves.png
│   ├── plot6_confusion_matrices.png
│   ├── plot7_feature_importance.png
│   ├── plot8_model_comparison.png
│   ├── plot9_risk_score_distribution.png
│   ├── plot10_shap_summary.png
│   ├── plot11_shap_bar.png
│   ├── plot12_shap_dependence.png
│   ├── plot13_shap_individual.png
│   └── plot14_shap_tier_heatmap.png
│
└── Student_Mental_Health_EarlyWarningSystem.ipynb   # Analysis + training notebook
```

---

## 18. How to Run the System

### Prerequisites

```bash
# Python 3.9+ required
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_api_key_here
```

### Verify Pre-Built Artifacts Exist

The following files must exist before running the dashboard (they are generated by the Jupyter notebook):
- `data/master_dataset.csv`
- `data/scored_students.csv`
- `data/student_explanations.csv`
- `data/shap_values.csv`
- `models/xgb_model.pkl`
- `models/scaler.pkl`

If any are missing, run the Jupyter notebook end-to-end first.

### Launch the Dashboard

```bash
cd /path/to/Capstone_MentalHealth
streamlit run code/dashboard.py
```

The dashboard opens at `http://localhost:8501` by default.

### Run the Jupyter Notebook (Re-Training)

Open `Student_Mental_Health_EarlyWarningSystem.ipynb` in Jupyter and run all cells sequentially. This will:
1. Load and join raw OULAD data
2. Engineer all behavioral features
3. Train Logistic Regression, Random Forest, and XGBoost models
4. Generate SHAP explanations
5. Save all output CSVs and model artifacts
6. Generate all 14 visualization plots

---

*Documentation prepared for the Student Mental Health Early Warning System Capstone Project.*
*Dataset: OULAD (Open University Learning Analytics Dataset). AI backbone: Anthropic Claude Sonnet 4-6.*
