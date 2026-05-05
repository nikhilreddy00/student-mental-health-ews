import os
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-sonnet-4-6"
client = Anthropic()

BASE = Path(__file__).parent.parent / "data"

_expl_cache = None
_scored_cache = None


def _explanations():
    global _expl_cache
    if _expl_cache is None:
        _expl_cache = pd.read_csv(BASE / "student_explanations.csv")
    return _expl_cache


def _scored():
    global _scored_cache
    if _scored_cache is None:
        _scored_cache = pd.read_csv(BASE / "scored_students.csv")
    return _scored_cache


def get_student_context(student_id):
    """Fetch all data needed for the multi-agent workflow."""
    expl = _explanations()
    scored = _scored()

    e_rows = expl[expl["id_student"] == student_id]
    s_rows = scored[scored["id_student"] == student_id]

    if e_rows.empty:
        return None

    e = e_rows.loc[e_rows["risk_score"].idxmax()]
    s = s_rows.loc[s_rows["risk_score"].idxmax()] if not s_rows.empty else None

    return {
        "student_id": int(student_id),
        "module": str(e.get("code_module", "Unknown")),
        "gender": str(s.get("gender", "Unknown")) if s is not None else "Unknown",
        "age_band": str(s.get("age_band", "Unknown")) if s is not None else "Unknown",
        "risk_score": round(float(e["risk_score"]), 4),
        "risk_tier": str(e["risk_tier"]),
        "top_reason": str(e.get("top_reason", "")).replace("_", " "),
        "reason_2": str(e.get("reason_2", "")).replace("_", " "),
        "reason_3": str(e.get("reason_3", "")).replace("_", " "),
        "engagement_span": float(e.get("engagement_span", 0) or 0),
        "mean_score": float(e.get("mean_score", 0) or 0),
        "dropout_modules": int(e.get("dropout_modules", 0) or 0),
        "active_days": float(e.get("active_days", 0) or 0),
        "engagement_decline": float(e.get("engagement_decline", 0) or 0),
    }


def run_multi_agent_workflow(student_id):
    """
    Generator yielding (phase, payload) tuples:
      ('start', student_context)
      ('agent1', text_chunk)
      ('agent2', text_chunk)
      ('agent3', text_chunk)
      ('done', {'profile', 'interventions', 'message'})
      ('error', message_string)
    """
    ctx = get_student_context(student_id)
    if ctx is None:
        yield ("error", f"Student {student_id} not found in dataset.")
        return

    yield ("start", ctx)

    # ── Agent 1: Risk Analyst ─────────────────────────────────────────────────
    agent1_prompt = f"""You are a student risk analyst. Analyze this student's behavioral data
and produce a clear, structured risk profile for a university counselor.

Student Data:
- Student ID: {ctx['student_id']} | Module: {ctx['module']}
- Risk Score: {ctx['risk_score']:.3f} ({ctx['risk_tier']} Risk)
- Engagement Span: {ctx['engagement_span']:.0f} days active in the course
- Mean Assessment Score: {ctx['mean_score']:.1f}/100
- Modules Dropped Out Of: {ctx['dropout_modules']}
- Active Days: {ctx['active_days']:.0f}
- Engagement Decline Rate: {ctx['engagement_decline']:.3f} (higher = more decline)

SHAP Risk Drivers (what the model says matters most):
1. Primary: {ctx['top_reason']}
2. Secondary: {ctx['reason_2']}
3. Tertiary: {ctx['reason_3']}

Produce a structured risk profile (4-5 sentences) covering:
1. Overall risk assessment and urgency level
2. The most concerning behavioral signals and why they matter
3. What this behavioral pattern typically indicates about student wellbeing
4. Your recommended urgency level for counselor intervention (Immediate/Soon/Monitor)"""

    agent1_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": agent1_prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            agent1_text += chunk
            yield ("agent1", chunk)

    # ── Agent 2: Intervention Advisor ─────────────────────────────────────────
    agent2_prompt = f"""You are a university student support specialist. Based on this student risk
profile, recommend 3 specific, ranked intervention strategies for the counseling team.

Risk Profile:
{agent1_text}

Student Context:
- Module: {ctx['module']} | Risk Tier: {ctx['risk_tier']}
- Primary concern: {ctx['top_reason']}
- Mean score: {ctx['mean_score']:.1f}/100 | Dropped modules: {ctx['dropout_modules']}

Provide exactly 3 interventions ranked by urgency. For each, explain WHY it addresses
this student's specific behavioral pattern:

**Intervention 1 — Immediate (within 48 hours):** [specific action + rationale]
**Intervention 2 — Short-term (within 1 week):** [specific action + rationale]
**Intervention 3 — Ongoing (monitor for 4 weeks):** [specific action + rationale]"""

    agent2_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": agent2_prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            agent2_text += chunk
            yield ("agent2", chunk)

    # ── Agent 3: Counselor Note Writer ────────────────────────────────────────
    agent3_prompt = f"""You are a compassionate university counselor writing a personal outreach
message to a student who may be struggling academically and emotionally.

For reference only (do NOT reveal this to the student):
- Risk concern: {ctx['top_reason']}
- Recommended approach: {agent2_text[:200]}

Write a brief, warm outreach message (4-5 sentences) that:
- Feels personal and genuine, NOT formulaic or bureaucratic
- Does NOT mention AI, algorithms, risk scores, data analysis, or surveillance
- Opens a natural conversation without alarming or pressuring the student
- Offers a low-barrier way to connect (a brief call, coffee chat, or quick reply)
- Comes from genuine care — not obligation

Write only the message itself, nothing else."""

    agent3_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=250,
        messages=[{"role": "user", "content": agent3_prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            agent3_text += chunk
            yield ("agent3", chunk)

    yield ("done", {
        "profile": agent1_text,
        "interventions": agent2_text,
        "message": agent3_text,
    })
