"""
Student Mental Health Early Warning System — FastAPI Backend
Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Or:  python ../../code/dashboard.py
"""
import sys
import json
import time
import threading
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Path setup: add code/ dir so we can import existing modules ───────────────
REPO_ROOT = Path(__file__).parent.parent.parent
CODE_DIR = REPO_ROOT / "code"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
PLOTS_DIR = REPO_ROOT / "plots"

sys.path.insert(0, str(CODE_DIR))

import pandas as pd
import joblib

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from ai_agent import run_counselor_agent, generate_risk_narrative, get_student_profile as _get_profile
from multi_agent import run_multi_agent_workflow
from risk_monitor import compute_weekly_trajectories, generate_crossing_alert
from booking import (
    load_bookings, save_booking, update_booking_status, generate_booking_id,
    compute_urgency, generate_advisor_briefing, TIME_SLOTS, ADVISOR_TYPES
)
from rag_kb import load_kb_documents, build_tfidf_index, retrieve_relevant_chunks, generate_rag_answer
from temporal_analysis import compute_delay_table, generate_temporal_narrative, compute_early_signal_week, compute_actual_alert_week

# ── Global state ──────────────────────────────────────────────────────────────
_bookings_lock = threading.Lock()

# ── Lifespan: load heavy data once at startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import pandas as pd

    # Load core CSVs
    expl = pd.read_csv(DATA_DIR / "student_explanations.csv")
    scored = pd.read_csv(DATA_DIR / "scored_students.csv")
    master = pd.read_csv(DATA_DIR / "master_dataset.csv")

    # Merge demographics from master into explanations (same logic as dashboard.py)
    demo_cols = ["id_student", "gender", "age_band", "region", "imd_band",
                 "highest_education", "num_of_prev_attempts", "studied_credits",
                 "disability", "final_result"]
    demo_cols = [c for c in demo_cols if c in master.columns]
    df = expl.merge(master[demo_cols].drop_duplicates("id_student"), on="id_student", how="left")

    # Load SHAP values
    shap_df = pd.read_csv(DATA_DIR / "shap_values.csv")

    app.state.df = df
    app.state.scored = scored
    app.state.master = master
    app.state.shap_df = shap_df

    # Load XGBoost model for feature importances
    try:
        app.state.xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")
    except Exception:
        app.state.xgb_model = None

    # Compute weekly trajectories in background thread (heavy: 433MB VLE file)
    try:
        traj = await asyncio.to_thread(compute_weekly_trajectories, 10)
        app.state.trajectories = traj
    except Exception as e:
        print(f"Warning: Could not compute trajectories: {e}")
        app.state.trajectories = {}

    # Build KB index
    try:
        docs = load_kb_documents()
        vectorizer, tfidf_matrix, chunk_meta = build_tfidf_index(docs)
        app.state.kb_docs = docs
        app.state.kb_vectorizer = vectorizer
        app.state.kb_matrix = tfidf_matrix
        app.state.kb_chunk_meta = chunk_meta
    except Exception as e:
        print(f"Warning: Could not build KB index: {e}")
        app.state.kb_docs = []
        app.state.kb_vectorizer = None
        app.state.kb_matrix = None
        app.state.kb_chunk_meta = []

    yield  # app runs here


app = FastAPI(title="Mental Health EWS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_df(df: pd.DataFrame, module: str, tiers: list[str]) -> pd.DataFrame:
    if module and module != "All":
        df = df[df["code_module"] == module]
    if tiers:
        df = df[df["risk_tier"].isin(tiers)]
    return df


def _np_safe(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


def _row_to_dict(row) -> dict:
    return {k: _np_safe(v) for k, v in row.items()}


TIER_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}

MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]

FEATURE_NAMES = [
    "total_clicks", "active_days", "engagement_span", "engagement_decline",
    "peak_week_clicks", "avg_daily_clicks", "early_clicks", "late_clicks",
    "mean_score", "min_score", "max_score", "score_std", "num_assessments",
    "num_unsubmitted", "submission_rate", "score_trend", "num_late",
    "dropout_modules", "early_dropout_flag", "num_modules_registered",
    "avg_reg_delay", "num_of_prev_attempts", "studied_credits",
]

COUNSELOR_ACTIONS = {
    "High": "Initiate immediate outreach — email or phone call within 24 hours. Consider booking an urgent appointment with a professional advisor or therapist.",
    "Medium": "Schedule a check-in within the week. Review their module engagement and recent assessment results before the meeting.",
    "Low": "Monitor progress. Flag for follow-up if risk score increases or assessment results decline.",
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[dict]

class KBChatRequest(BaseModel):
    query: str
    history: list[dict] = []

class BookingRequest(BaseModel):
    student_id: int
    advisor_type: str
    date: str
    time_slot: str
    notes: str = ""
    generate_briefing: bool = True

class BookingStatusUpdate(BaseModel):
    status: str

class RiskAlertRequest(BaseModel):
    student_id: int
    week: int
    prev_score: float
    curr_score: float
    top_reason: str

class TemporalNarrativeRequest(BaseModel):
    early_threshold: float = 0.50
    consecutive_weeks: int = 2


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/metadata")
def get_metadata():
    return {
        "modules": ["All"] + MODULES,
        "advisor_types": ADVISOR_TYPES,
        "time_slots": TIME_SLOTS,
        "risk_tiers": ["High", "Medium", "Low"],
    }


@app.get("/api/overview")
def get_overview(
    module: str = "All",
    tiers: list[str] = Query(default=["High", "Medium", "Low"]),
):
    df = app.state.df
    filtered = _filter_df(df, module, tiers)

    total = len(filtered)
    tier_counts = filtered["risk_tier"].value_counts()
    high = int(tier_counts.get("High", 0))
    medium = int(tier_counts.get("Medium", 0))
    low = int(tier_counts.get("Low", 0))
    avg_risk = float(filtered["risk_score"].mean()) if len(filtered) else 0.0

    # Tier distribution for pie chart
    tier_dist = [
        {"tier": t, "count": int(tier_counts.get(t, 0))}
        for t in ["High", "Medium", "Low"]
    ]

    # Risk histogram (20 bins per tier)
    histogram = {}
    for tier in ["High", "Medium", "Low"]:
        tier_scores = filtered[filtered["risk_tier"] == tier]["risk_score"].dropna().values
        if len(tier_scores):
            counts, edges = np.histogram(tier_scores, bins=20, range=(0, 1))
            histogram[tier] = [
                {"bin_start": round(float(edges[i]), 3), "bin_end": round(float(edges[i+1]), 3), "count": int(counts[i])}
                for i in range(len(counts))
            ]
        else:
            histogram[tier] = []

    # Top reasons (from high-risk students)
    high_df = filtered[filtered["risk_tier"] == "High"]
    all_reasons = pd.concat([
        high_df["top_reason"], high_df["reason_2"], high_df["reason_3"]
    ]).dropna().str.replace("_", " ")
    top_reasons = [
        {"reason": str(r), "count": int(c)}
        for r, c in all_reasons.value_counts().head(8).items()
    ]

    # Module breakdown
    module_breakdown = []
    for mod in MODULES:
        mdf = df[df["code_module"] == mod]
        if len(mdf) == 0:
            continue
        module_breakdown.append({
            "module": mod,
            "total": int(len(mdf)),
            "high_risk": int((mdf["risk_tier"] == "High").sum()),
            "high_risk_pct": round(float((mdf["risk_tier"] == "High").mean() * 100), 1),
            "avg_score": round(float(mdf["risk_score"].mean()), 3),
        })

    return {
        "total": total,
        "high": high, "medium": medium, "low": low,
        "avg_risk_score": round(avg_risk, 4),
        "high_pct": round(high / total * 100, 1) if total else 0,
        "medium_pct": round(medium / total * 100, 1) if total else 0,
        "low_pct": round(low / total * 100, 1) if total else 0,
        "tier_distribution": tier_dist,
        "risk_histogram": histogram,
        "top_reasons": top_reasons,
        "module_breakdown": module_breakdown,
    }


@app.get("/api/alerts")
def get_alerts(
    module: str = "All",
    tiers: list[str] = Query(default=["High"]),
    page: int = 1,
    page_size: int = 50,
):
    df = _filter_df(app.state.df, module, tiers)
    df = df.sort_values("risk_score", ascending=False).drop_duplicates("id_student")
    total = len(df)

    page_df = df.iloc[(page - 1) * page_size: page * page_size]

    students = []
    for _, row in page_df.iterrows():
        students.append({
            "id_student": int(row["id_student"]),
            "risk_score": round(float(row["risk_score"]), 4),
            "risk_tier": str(row.get("risk_tier", "")),
            "top_reason": str(row.get("top_reason", "")).replace("_", " "),
            "reason_2": str(row.get("reason_2", "")).replace("_", " "),
            "reason_3": str(row.get("reason_3", "")).replace("_", " "),
            "code_module": str(row.get("code_module", "")),
        })

    # Aggregate risk factors across top 200 high-risk students
    top_200 = df.head(200)
    all_reasons = pd.concat([
        top_200["top_reason"], top_200["reason_2"], top_200["reason_3"]
    ]).dropna().str.replace("_", " ")
    risk_factors = [
        {"factor": str(r), "count": int(c)}
        for r, c in all_reasons.value_counts().head(12).items()
    ]

    # Behavioral comparison high vs low risk
    full_df = app.state.df
    feat_cols = ["engagement_span", "mean_score", "active_days", "submission_rate", "engagement_decline"]
    feat_cols = [c for c in feat_cols if c in full_df.columns]
    high_avgs = [round(float(full_df[full_df["risk_tier"] == "High"][c].mean()), 2) for c in feat_cols]
    low_avgs = [round(float(full_df[full_df["risk_tier"] == "Low"][c].mean()), 2) for c in feat_cols]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "students": students,
        "risk_factors": risk_factors,
        "behavioral_comparison": {
            "features": feat_cols,
            "high_risk_avgs": high_avgs,
            "low_risk_avgs": low_avgs,
        },
    }


@app.get("/api/high-risk-ids")
def get_high_risk_ids(include_medium: bool = False):
    df = app.state.df
    tiers = ["High", "Medium"] if include_medium else ["High"]
    filtered = df[df["risk_tier"].isin(tiers)]
    ids = (
        filtered.sort_values("risk_score", ascending=False)
        .drop_duplicates("id_student")["id_student"]
        .astype(int)
        .tolist()
    )
    return {"ids": ids}


@app.get("/api/students")
def search_students(
    q: Optional[str] = None,
    gender: Optional[str] = None,
    age_band: Optional[str] = None,
    module: Optional[str] = None,
    imd_band: Optional[str] = None,
    tier: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "risk_score",
    sort_dir: str = "desc",
):
    df = app.state.df.copy()
    df = df.sort_values("risk_score", ascending=False).drop_duplicates("id_student")

    if q:
        try:
            sid = int(q)
            df = df[df["id_student"] == sid]
        except ValueError:
            pass

    if gender:
        df = df[df["gender"].str.upper() == gender.upper()]
    if age_band:
        df = df[df["age_band"].str.contains(age_band, case=False, na=False)]
    if module and module != "All":
        df = df[df["code_module"].str.upper() == module.upper()]
    if imd_band:
        df = df[df["imd_band"].str.contains(imd_band, case=False, na=False)]
    if tier:
        df = df[df["risk_tier"] == tier]

    total = len(df)
    ascending = sort_dir == "asc"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    page_df = df.iloc[(page - 1) * page_size: page * page_size]

    students = []
    for _, row in page_df.iterrows():
        students.append({
            "id_student": int(row["id_student"]),
            "code_module": str(row.get("code_module", "")),
            "gender": str(row.get("gender", "")),
            "age_band": str(row.get("age_band", "")),
            "risk_score": round(float(row["risk_score"]), 4),
            "risk_tier": str(row.get("risk_tier", "")),
            "top_reason": str(row.get("top_reason", "")).replace("_", " "),
        })

    return {"total": total, "students": students, "page": page, "page_size": page_size}


@app.get("/api/students/{student_id}")
def get_student(student_id: int):
    df = app.state.df
    shap_df = app.state.shap_df

    rows = df[df["id_student"] == student_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")

    row = rows.loc[rows["risk_score"].idxmax()]
    risk_tier = str(row.get("risk_tier", ""))

    # SHAP waterfall
    shap_rows = shap_df[shap_df["id_student"] == student_id] if "id_student" in shap_df.columns else pd.DataFrame()
    shap_waterfall = []
    if not shap_rows.empty:
        sr = shap_rows.iloc[0]
        feat_shap = {
            col: float(sr[col])
            for col in shap_df.columns
            if col != "id_student" and not pd.isna(sr[col])
        }
        sorted_feats = sorted(feat_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        shap_waterfall = [
            {
                "feature": k.replace("_", " "),
                "shap_value": round(v, 4),
                "direction": "positive" if v > 0 else "negative",
            }
            for k, v in sorted_feats
        ]

    return {
        "student_id": student_id,
        "code_module": str(row.get("code_module", "")),
        "risk_score": round(float(row["risk_score"]), 4),
        "risk_tier": risk_tier,
        "gender": str(row.get("gender", "")),
        "age_band": str(row.get("age_band", "")),
        "region": str(row.get("region", "")),
        "imd_band": str(row.get("imd_band", "")),
        "highest_education": str(row.get("highest_education", "")),
        "disability": str(row.get("disability", "")),
        "num_of_prev_attempts": int(row.get("num_of_prev_attempts", 0) or 0),
        "studied_credits": int(row.get("studied_credits", 0) or 0),
        "top_reason": str(row.get("top_reason", "")).replace("_", " "),
        "reason_2": str(row.get("reason_2", "")).replace("_", " "),
        "reason_3": str(row.get("reason_3", "")).replace("_", " "),
        "engagement_span": float(row.get("engagement_span", 0) or 0),
        "mean_score": float(row.get("mean_score", 0) or 0),
        "active_days": float(row.get("active_days", 0) or 0),
        "submission_rate": float(row.get("submission_rate", 0) or 0),
        "engagement_decline": float(row.get("engagement_decline", 0) or 0),
        "dropout_modules": int(row.get("dropout_modules", 0) or 0),
        "shap_waterfall": shap_waterfall,
        "suggested_action": {
            "tier": risk_tier,
            "message": COUNSELOR_ACTIONS.get(risk_tier, ""),
        },
    }


@app.post("/api/students/{student_id}/narrative")
def get_narrative(student_id: int):
    try:
        narrative = generate_risk_narrative(student_id)
        return {"narrative": narrative}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-performance")
def get_model_performance():
    xgb = app.state.xgb_model
    feature_importances = []

    if xgb is not None and hasattr(xgb, "feature_importances_"):
        importances = xgb.feature_importances_
        names = FEATURE_NAMES[:len(importances)]
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        feature_importances = [
            {"feature": k.replace("_", " "), "importance": round(float(v), 5)}
            for k, v in pairs[:15]
        ]

    return {
        "metrics": {
            "auc_roc": 0.975,
            "f1": 0.911,
            "precision": 0.936,
            "recall": 0.887,
            "accuracy": 0.91,
        },
        "confusion_matrix": [[2153, 155], [292, 2289]],
        "confusion_matrix_pct": [[93.3, 6.7], [11.3, 88.7]],
        "cv_score": 0.9734,
        "cv_std": 0.0012,
        "feature_importances": feature_importances,
        "labels": {
            "rows": ["Actually Not At-Risk", "Actually At-Risk"],
            "cols": ["Predicted Not At-Risk", "Predicted At-Risk"],
        },
    }


@app.get("/api/shap")
def get_shap():
    plot_defs = [
        ("beeswarm", "SHAP Beeswarm — Global Feature Impact", "plot10_shap_summary.png"),
        ("bar", "Mean |SHAP| — Feature Importance Ranking", "plot11_shap_bar.png"),
        ("dependence", "SHAP Dependence Plots", "plot12_shap_dependence.png"),
        ("individual", "Individual Student SHAP Explanations", "plot13_shap_individual.png"),
        ("tier_heatmap", "Average SHAP Values by Risk Tier", "plot14_shap_tier_heatmap.png"),
    ]
    plots = [
        {
            "id": pid,
            "title": title,
            "url": f"/plots/{fname}",
            "exists": (PLOTS_DIR / fname).exists(),
        }
        for pid, title, fname in plot_defs
    ]
    return {"plots": plots}


@app.get("/api/fairness")
def get_fairness(
    module: str = "All",
    tiers: list[str] = Query(default=["High", "Medium", "Low"]),
):
    df = _filter_df(app.state.df, module, tiers)

    def group_stats(col):
        if col not in df.columns:
            return []
        grp = df.groupby(col).agg(
            count=("id_student", "count"),
            avg_risk_score=("risk_score", "mean"),
            actual_at_risk=("is_at_risk", "mean") if "is_at_risk" in df.columns else ("risk_score", "count"),
            high_risk_flag_rate=("risk_tier", lambda x: (x == "High").mean()),
        ).reset_index()
        result = []
        for _, row in grp.iterrows():
            result.append({
                "group": str(row[col]),
                "count": int(row["count"]),
                "avg_risk_score": round(float(row["avg_risk_score"]), 4),
                "high_risk_flag_rate_pct": round(float(row["high_risk_flag_rate"]) * 100, 1),
            })
        return sorted(result, key=lambda x: x["avg_risk_score"], reverse=True)

    # Gender histogram
    gender_hist = {}
    if "gender" in df.columns:
        for g in df["gender"].dropna().unique():
            scores = df[df["gender"] == g]["risk_score"].dropna().values
            if len(scores):
                counts, edges = np.histogram(scores, bins=15, range=(0, 1))
                gender_hist[str(g)] = [
                    {"bin_start": round(float(edges[i]), 3), "count": int(counts[i])}
                    for i in range(len(counts))
                ]

    # Age band avg scores
    age_band_avgs = []
    if "age_band" in df.columns:
        for ab, grp_df in df.groupby("age_band"):
            age_band_avgs.append({
                "age_band": str(ab),
                "avg_risk_score": round(float(grp_df["risk_score"].mean()), 4),
            })

    return {
        "gender": group_stats("gender"),
        "age_band": group_stats("age_band"),
        "imd_band": group_stats("imd_band"),
        "gender_histogram": gender_hist,
        "age_band_avg_scores": sorted(age_band_avgs, key=lambda x: x["avg_risk_score"], reverse=True),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        text, tool_calls, _ = run_counselor_agent(req.messages)
        return {"response": text, "tool_calls": tool_calls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/multi-agent/{student_id}")
def multi_agent_stream(student_id: int):
    def generator():
        last_ping = time.time()
        try:
            for phase, payload in run_multi_agent_workflow(student_id):
                if time.time() - last_ping > 15:
                    yield f"data: {json.dumps({'phase': 'ping', 'payload': ''})}\n\n"
                    last_ping = time.time()
                event = json.dumps({"phase": phase, "payload": payload}, default=str)
                yield f"data: {event}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'phase': 'error', 'payload': str(e)})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/risk-monitor")
def get_risk_monitor():
    traj = app.state.trajectories
    students = []
    max_week = 0
    for sid, t in traj.items():
        if t.get("weeks"):
            max_week = max(max_week, max(t["weeks"]))
        students.append({
            "student_id": int(sid),
            "weeks": t.get("weeks", []),
            "risk_scores": [round(float(s), 4) for s in t.get("risk_scores", [])],
            "final_risk": round(float(t.get("final_risk", 0)), 4),
            "top_reason": str(t.get("top_reason", "")).replace("_", " "),
            "module": str(t.get("module", "")),
        })
    return {"students": students, "max_week": max_week, "threshold": 0.66}


@app.post("/api/risk-alert")
def risk_alert_stream(req: RiskAlertRequest):
    def generator():
        try:
            from anthropic import Anthropic
            client = Anthropic()
            prompt = (
                f"Write a 2-sentence urgent counselor alert for Student {req.student_id}. "
                f"Their risk score just crossed the alert threshold in week {req.week}: "
                f"{req.prev_score:.3f} → {req.curr_score:.3f}. "
                f"Primary concern: {req.top_reason.replace('_', ' ')}. "
                "Be specific and action-oriented."
            )
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/bookings")
def create_booking(req: BookingRequest):
    with _bookings_lock:
        existing = load_bookings()
        dup = existing[
            (existing["student_id"] == str(req.student_id)) &
            (existing["status"] == "Scheduled")
        ]
        duplicate_warning = len(dup) > 0

        profile = _get_profile(req.student_id)
        risk_score = float(profile.get("risk_score", 0.5))
        urgency = compute_urgency(risk_score)

        ai_briefing = ""
        if req.generate_briefing:
            try:
                ai_briefing = generate_advisor_briefing(req.student_id, req.advisor_type, profile)
            except Exception:
                ai_briefing = "Briefing generation unavailable."

        booking_id = generate_booking_id()
        record = {
            "booking_id": booking_id,
            "student_id": str(req.student_id),
            "risk_score": str(round(risk_score, 4)),
            "advisor_type": req.advisor_type,
            "date": req.date,
            "time_slot": req.time_slot,
            "urgency": urgency,
            "notes": req.notes,
            "ai_briefing": ai_briefing,
            "status": "Scheduled",
            "booked_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_booking(record)

    return {
        "booking_id": booking_id,
        "urgency": urgency,
        "student_risk_score": risk_score,
        "ai_briefing": ai_briefing,
        "duplicate_warning": duplicate_warning,
    }


@app.get("/api/bookings")
def list_bookings(
    status: str = "All",
    advisor_type: str = "All",
    urgency: str = "All",
):
    df = load_bookings()
    if status != "All":
        df = df[df["status"] == status]
    if advisor_type != "All":
        df = df[df["advisor_type"] == advisor_type]
    if urgency != "All":
        df = df[df["urgency"] == urgency]

    total = len(df)
    immediate = int((df["urgency"] == "Immediate").sum())
    soon = int((df["urgency"] == "Soon").sum())
    routine = int((df["urgency"] == "Routine").sum())
    scheduled = int((df["status"] == "Scheduled").sum())

    bookings = []
    for _, row in df.iterrows():
        bookings.append({
            "booking_id": str(row.get("booking_id", "")),
            "student_id": str(row.get("student_id", "")),
            "risk_score": float(row.get("risk_score", 0) or 0),
            "advisor_type": str(row.get("advisor_type", "")),
            "date": str(row.get("date", "")),
            "time_slot": str(row.get("time_slot", "")),
            "urgency": str(row.get("urgency", "")),
            "status": str(row.get("status", "")),
            "booked_at": str(row.get("booked_at", "")),
        })

    return {
        "total": total,
        "immediate": immediate,
        "soon": soon,
        "routine": routine,
        "scheduled": scheduled,
        "bookings": bookings,
        "advisor_types": ADVISOR_TYPES,
        "time_slots": TIME_SLOTS,
    }


@app.patch("/api/bookings/{booking_id}")
def patch_booking(booking_id: str, update: BookingStatusUpdate):
    with _bookings_lock:
        update_booking_status(booking_id, update.status)
    return {"success": True, "booking_id": booking_id, "new_status": update.status}


@app.get("/api/bookings/{booking_id}/briefing")
def get_briefing(booking_id: str):
    df = load_bookings()
    row = df[df["booking_id"] == booking_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Booking not found")
    r = row.iloc[0]
    return {
        "booking_id": booking_id,
        "student_id": str(r.get("student_id", "")),
        "advisor_type": str(r.get("advisor_type", "")),
        "urgency": str(r.get("urgency", "")),
        "ai_briefing": str(r.get("ai_briefing", "")),
    }


@app.post("/api/kb-chat")
def kb_chat(req: KBChatRequest):
    if app.state.kb_vectorizer is None:
        raise HTTPException(status_code=503, detail="KB index not available")
    try:
        chunks = retrieve_relevant_chunks(
            req.query,
            app.state.kb_vectorizer,
            app.state.kb_matrix,
            app.state.kb_chunk_meta,
            top_k=4,
        )
        answer, sources = generate_rag_answer(req.query, chunks, req.history)
        return {"answer": answer, "sources": sources, "chunks_used": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kb-documents")
def get_kb_documents():
    docs = app.state.kb_docs
    return {
        "documents": [
            {
                "title": d["title"],
                "filename": d["filename"],
                "chunk_count": len(d.get("chunks", [])),
                "preview": d.get("content", "")[:300],
            }
            for d in docs
        ]
    }


@app.get("/api/temporal-analysis")
def get_temporal_analysis(
    early_threshold: float = 0.50,
    consecutive_weeks: int = 2,
):
    traj = app.state.trajectories
    if not traj:
        raise HTTPException(status_code=503, detail="Trajectories not computed yet")

    delay_df = compute_delay_table(traj, early_threshold, consecutive_weeks)
    valid = delay_df.dropna(subset=["delay_weeks"])

    metrics = {
        "mean_delay": round(float(valid["delay_weeks"].mean()), 1) if len(valid) else 0,
        "median_delay": round(float(valid["delay_weeks"].median()), 1) if len(valid) else 0,
        "max_delay": int(valid["delay_weeks"].max()) if len(valid) else 0,
        "n_total": len(delay_df),
        "n_with_delay": len(valid),
    }

    students = []
    for _, row in delay_df.iterrows():
        students.append({
            "student_id": int(row["student_id"]),
            "module": str(row.get("module", "")),
            "top_reason": str(row.get("top_reason", "")).replace("_", " "),
            "final_risk": round(float(row.get("final_risk", 0)), 4),
            "early_signal_week": int(row["early_signal_week"]) if pd.notna(row.get("early_signal_week")) else None,
            "actual_alert_week": int(row["actual_alert_week"]) if pd.notna(row.get("actual_alert_week")) else None,
            "delay_weeks": int(row["delay_weeks"]) if pd.notna(row.get("delay_weeks")) else None,
        })

    trajectories = []
    for sid, t in traj.items():
        trajectories.append({
            "student_id": int(sid),
            "weeks": t.get("weeks", []),
            "risk_scores": [round(float(s), 4) for s in t.get("risk_scores", [])],
            "final_risk": round(float(t.get("final_risk", 0)), 4),
            "top_reason": str(t.get("top_reason", "")).replace("_", " "),
            "module": str(t.get("module", "")),
        })

    return {"metrics": metrics, "students": students, "trajectories": trajectories}


@app.post("/api/temporal-analysis/narrative")
def temporal_narrative_stream(req: TemporalNarrativeRequest):
    def generator():
        try:
            traj = app.state.trajectories
            delay_df = compute_delay_table(traj, req.early_threshold, req.consecutive_weeks)

            from anthropic import Anthropic
            client = Anthropic()

            valid = delay_df.dropna(subset=["delay_weeks"])
            if len(valid) == 0:
                yield f"data: {json.dumps({'chunk': 'Insufficient data for narrative.'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            mean_delay = valid["delay_weeks"].mean()
            median_delay = valid["delay_weeks"].median()
            n_total = len(delay_df)
            n_with = len(valid)

            prompt = (
                f"Write a 3-paragraph academic research finding (ACM/IEEE style) about this early warning system result:\n\n"
                f"- Dataset: {n_total} university students tracked over multiple weeks\n"
                f"- {n_with} students had both an early signal (risk > {req.early_threshold}) and an alert (risk >= 0.66)\n"
                f"- Mean intervention window gained: {mean_delay:.1f} weeks\n"
                f"- Median intervention window: {median_delay:.1f} weeks\n"
                f"- Early signal threshold: {req.early_threshold} | Alert threshold: 0.66\n\n"
                "Paragraph 1: Describe the finding and its significance for early intervention.\n"
                "Paragraph 2: Interpret what this intervention window means practically for counselors.\n"
                "Paragraph 3: Discuss implications and limitations for deploying such systems."
            )

            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
