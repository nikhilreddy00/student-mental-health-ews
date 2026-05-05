import os
import json
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-sonnet-4-6"
client = Anthropic()

BASE = Path(__file__).parent.parent / "data"

# ── Data loading (module-level cache) ────────────────────────────────────────

_explanations_cache = None
_scored_cache = None

def _explanations():
    global _explanations_cache
    if _explanations_cache is None:
        _explanations_cache = pd.read_csv(BASE / "student_explanations.csv")
    return _explanations_cache

def _scored():
    global _scored_cache
    if _scored_cache is None:
        _scored_cache = pd.read_csv(BASE / "scored_students.csv")
    return _scored_cache

def _best_row(df, student_id):
    """Return the highest-risk row for a student (handles multi-module students)."""
    rows = df[df["id_student"] == student_id]
    if rows.empty:
        return None
    return rows.loc[rows["risk_score"].idxmax()]

# ── Tool implementations ──────────────────────────────────────────────────────

def query_high_risk_students(limit=5, tier="High"):
    df = _explanations().copy()
    if tier != "All":
        df = df[df["risk_tier"] == tier]
    df = df.sort_values("risk_score", ascending=False).drop_duplicates("id_student").head(limit)
    cols = ["id_student", "code_module", "risk_score", "risk_tier",
            "top_reason", "reason_2", "reason_3"]
    return df[[c for c in cols if c in df.columns]].to_dict("records")


def get_student_profile(student_id):
    e = _best_row(_explanations(), student_id)
    s = _best_row(_scored(), student_id)
    if e is None:
        return {"error": f"Student {student_id} not found"}
    result = {
        "student_id": int(student_id),
        "module": str(e.get("code_module", "")),
        "risk_score": round(float(e["risk_score"]), 4),
        "risk_tier": str(e["risk_tier"]),
        "top_reason": str(e.get("top_reason", "")).replace("_", " "),
        "reason_2": str(e.get("reason_2", "")).replace("_", " "),
        "reason_3": str(e.get("reason_3", "")).replace("_", " "),
        "engagement_span_days": float(e.get("engagement_span", 0) or 0),
        "mean_score": float(e.get("mean_score", 0) or 0),
        "dropout_modules": int(e.get("dropout_modules", 0) or 0),
        "active_days": float(e.get("active_days", 0) or 0),
        "engagement_decline": float(e.get("engagement_decline", 0) or 0),
    }
    if s is not None:
        result.update({
            "gender": str(s.get("gender", "")),
            "age_band": str(s.get("age_band", "")),
            "imd_band": str(s.get("imd_band", "")),
            "final_result": str(s.get("final_result", "")),
        })
    return result


def get_shap_explanation(student_id):
    e = _best_row(_explanations(), student_id)
    if e is None:
        return {"error": f"Student {student_id} not found"}
    return {
        "student_id": int(student_id),
        "risk_score": round(float(e["risk_score"]), 4),
        "risk_tier": str(e["risk_tier"]),
        "primary_driver": str(e.get("top_reason", "")).replace("_", " "),
        "primary_shap_magnitude": round(float(e.get("top_shap", 0) or 0), 4),
        "secondary_driver": str(e.get("reason_2", "")).replace("_", " "),
        "tertiary_driver": str(e.get("reason_3", "")).replace("_", " "),
        "engagement_span_days": float(e.get("engagement_span", 0) or 0),
        "mean_score": float(e.get("mean_score", 0) or 0),
        "dropout_modules": int(e.get("dropout_modules", 0) or 0),
        "active_days": float(e.get("active_days", 0) or 0),
        "engagement_decline": float(e.get("engagement_decline", 0) or 0),
    }


def search_students(gender=None, age_band=None, module=None, imd_band=None,
                    tier=None, limit=10):
    scored = _scored().copy()
    expl = _explanations()[["id_student", "top_reason"]].drop_duplicates("id_student")
    df = scored.merge(expl, on="id_student", how="left")
    df = df.sort_values("risk_score", ascending=False).drop_duplicates("id_student")

    if gender:
        df = df[df["gender"].str.upper() == gender.upper()]
    if age_band:
        df = df[df["age_band"].str.contains(age_band, case=False, na=False)]
    if module:
        df = df[df["code_module"].str.upper() == module.upper()]
    if imd_band:
        df = df[df["imd_band"].str.contains(imd_band, case=False, na=False)]
    if tier:
        df = df[df["risk_tier"] == tier]

    df = df.head(int(limit) if limit else 10)
    cols = ["id_student", "code_module", "gender", "age_band",
            "risk_score", "risk_tier", "top_reason"]
    return df[[c for c in cols if c in df.columns]].to_dict("records")


# ── Tool definitions for Claude ───────────────────────────────────────────────

TOOLS = [
    {
        "name": "query_high_risk_students",
        "description": (
            "Get top N at-risk students sorted by risk score. Use when counselors ask "
            "who needs help, who to prioritize, or which students need outreach."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer",
                          "description": "Number of students to return (default 5)"},
                "tier": {"type": "string",
                         "enum": ["High", "Medium", "Low", "All"],
                         "description": "Risk tier to filter by (default High)"},
            },
        },
    },
    {
        "name": "get_student_profile",
        "description": (
            "Get full profile for a specific student: demographics, risk score, "
            "module, behavioral metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "integer", "description": "The student ID number"},
            },
            "required": ["student_id"],
        },
    },
    {
        "name": "get_shap_explanation",
        "description": (
            "Get the top 3 behavioral risk drivers for a student showing exactly "
            "why they were flagged by the model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "integer", "description": "The student ID number"},
            },
            "required": ["student_id"],
        },
    },
    {
        "name": "search_students",
        "description": (
            "Filter and search students by demographics or module. "
            "Use when a counselor asks about a specific group of students."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gender": {"type": "string", "description": "M or F"},
                "age_band": {"type": "string",
                             "description": "Age band string e.g. '35-55' or '0-35'"},
                "module": {"type": "string",
                           "description": "Module code e.g. AAA, BBB, CCC"},
                "imd_band": {"type": "string",
                             "description": "Deprivation band e.g. '10-20%'"},
                "tier": {"type": "string", "enum": ["High", "Medium", "Low"]},
                "limit": {"type": "integer",
                          "description": "Number of results (default 10)"},
            },
        },
    },
]

TOOL_MAP = {
    "query_high_risk_students": query_high_risk_students,
    "get_student_profile": get_student_profile,
    "get_shap_explanation": get_shap_explanation,
    "search_students": search_students,
}

SYSTEM_PROMPT = """You are an AI counselor assistant for a university student mental health early warning system.
You help counselors identify and understand at-risk students using behavioral analytics data from 32,593 students.

The system analyzes VLE engagement (clicks, active days, engagement span), assessment performance
(scores, submission rates), and module registration patterns to predict which students may be struggling.

When answering:
- Be concise and actionable — counselors are busy
- Always explain WHY a student is at risk using specific behavioral data
- Suggest practical next steps tied to the actual risk drivers
- Be compassionate — these are real students who may be struggling
- Never make clinical diagnoses — you support counselors, not replace them
- When listing students, format as a numbered list with key stats"""


def run_counselor_agent(messages):
    """
    Run one turn of the counselor agent with tool use.
    Returns (response_text, tool_calls_made, updated_messages).
    """
    tool_calls_made = []

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []

        for block in tool_use_blocks:
            tool_calls_made.append({"tool": block.name, "input": block.input})
            try:
                result = TOOL_MAP[block.name](**block.input)
            except Exception as exc:
                result = {"error": str(exc)}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })

        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return text, tool_calls_made, messages


# ── Natural language narrative ────────────────────────────────────────────────

def generate_risk_narrative(student_id):
    """Generate a 3-sentence plain-English risk summary for counselors."""
    profile = get_student_profile(student_id)
    if "error" in profile:
        return f"Could not generate narrative: {profile['error']}"

    prompt = f"""Write a 3-sentence counselor-facing risk summary for this student.
Be factual, specific, and actionable. Avoid jargon.

Student data:
- Module: {profile.get('module', 'Unknown')}
- Gender: {profile.get('gender', 'Unknown')} | Age Band: {profile.get('age_band', 'Unknown')}
- Risk Score: {profile.get('risk_score', 0):.3f} ({profile.get('risk_tier', 'Unknown')} Risk)
- Engagement Span: {profile.get('engagement_span_days', 0):.0f} days active in course
- Mean Assessment Score: {profile.get('mean_score', 0):.1f}/100
- Modules Dropped Out Of: {profile.get('dropout_modules', 0)}
- Active Days: {profile.get('active_days', 0):.0f}
- Engagement Decline Rate: {profile.get('engagement_decline', 0):.3f}
- Primary risk driver: {profile.get('top_reason', 'Unknown')}
- Secondary driver: {profile.get('reason_2', 'Unknown')}
- Tertiary driver: {profile.get('reason_3', 'Unknown')}

Write exactly 3 sentences:
1. State the risk tier and score with the most concerning specific metric.
2. Describe the behavioral pattern driving this risk with concrete numbers.
3. Recommend the single most appropriate immediate counselor action."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
