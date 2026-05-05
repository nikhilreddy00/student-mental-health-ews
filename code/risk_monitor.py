import os
import pandas as pd
import numpy as np
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-sonnet-4-6"
BASE = Path(__file__).parent.parent / "data"


def compute_weekly_trajectories(top_n=10):
    """
    Compute simulated week-by-week risk trajectories for the top N high-risk students.

    Uses actual VLE engagement data to drive the trajectory: when a student's
    weekly engagement drops relative to their personal peak, their simulated
    risk score rises. The trajectory is calibrated to reach the student's
    final XGBoost risk score when engagement reaches zero.

    Returns:
        dict: student_id -> {
            'weeks': [int],
            'risk_scores': [float],
            'final_risk': float,
            'top_reason': str,
            'module': str
        }
    """
    scored = pd.read_csv(BASE / "scored_students.csv")
    expl = pd.read_csv(BASE / "student_explanations.csv")
    vle = pd.read_csv(BASE / "studentVle.csv")

    # Top N unique high-risk students by score
    top_students = (
        scored[scored["risk_tier"] == "High"]
        .sort_values("risk_score", ascending=False)
        .drop_duplicates("id_student")
        .head(top_n)["id_student"]
        .tolist()
    )

    # Filter VLE to these students, course period only (date >= 0)
    vle = vle[vle["id_student"].isin(top_students) & (vle["date"] >= 0)].copy()
    vle["week"] = (vle["date"] // 7).astype(int)
    weekly = vle.groupby(["id_student", "week"])["sum_click"].sum().reset_index()

    result = {}
    for sid in top_students:
        final_risk = (
            scored[scored["id_student"] == sid]["risk_score"].max()
        )
        e_rows = expl[expl["id_student"] == sid]
        top_reason = (
            str(e_rows.loc[e_rows["risk_score"].idxmax(), "top_reason"]).replace("_", " ")
            if not e_rows.empty else ""
        )
        module = (
            str(e_rows.loc[e_rows["risk_score"].idxmax(), "code_module"])
            if not e_rows.empty else ""
        )

        s = weekly[weekly["id_student"] == sid].sort_values("week")
        if s.empty:
            continue

        max_week = int(s["week"].max())
        clicks_by_week = dict(zip(s["week"], s["sum_click"]))
        max_clicks = float(s["sum_click"].max())

        all_weeks = list(range(0, max_week + 1))
        risk_scores = []

        for w in all_weeks:
            clicks = float(clicks_by_week.get(w, 0))
            # Relative engagement: 1.0 = at personal peak, 0.0 = no engagement
            relative = clicks / max_clicks if max_clicks > 0 else 0.0
            # Risk rises smoothly as engagement falls, calibrated to final_risk
            simulated = 0.08 + (final_risk - 0.08) * ((1.0 - relative) ** 0.6)
            risk_scores.append(round(min(0.99, max(0.02, simulated)), 4))

        result[sid] = {
            "weeks": all_weeks,
            "risk_scores": risk_scores,
            "final_risk": round(float(final_risk), 4),
            "top_reason": top_reason,
            "module": module,
        }

    return result


def generate_crossing_alert(student_id, week, prev_score, curr_score, top_reason):
    """Call Claude to generate a 2-sentence counselor alert when risk crosses 0.66."""
    client = Anthropic()
    prompt = (
        f"A student just crossed into HIGH RISK status in the early warning system.\n\n"
        f"Student ID: {student_id}\n"
        f"Course week: {week}\n"
        f"Risk score: {prev_score:.2f} → {curr_score:.2f}\n"
        f"Primary behavioral concern: {top_reason}\n\n"
        f"Write a 2-sentence urgent but calm alert for the counselor dashboard. "
        f"Be specific about what changed this week. Start with '⚠️ ALERT:'"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
