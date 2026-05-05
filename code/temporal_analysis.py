import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-sonnet-4-6"


def compute_early_signal_week(
    weeks: list,
    risk_scores: list,
    early_threshold: float = 0.50,
    consecutive: int = 2,
):
    """Return first week of a sustained run above early_threshold, or None."""
    run_length = 0
    run_start_idx = None
    for i, score in enumerate(risk_scores):
        if score > early_threshold:
            if run_length == 0:
                run_start_idx = i
            run_length += 1
            if run_length >= consecutive:
                return weeks[run_start_idx]
        else:
            run_length = 0
            run_start_idx = None
    return None


def compute_actual_alert_week(
    weeks: list,
    risk_scores: list,
    alert_threshold: float = 0.66,
):
    """Return first week where risk_score >= alert_threshold, or None."""
    for w, s in zip(weeks, risk_scores):
        if s >= alert_threshold:
            return w
    return None


def compute_delay_table(
    trajectories: dict,
    early_threshold: float = 0.50,
    consecutive: int = 2,
) -> pd.DataFrame:
    """Build per-student delay analysis table."""
    rows = []
    for sid, t in trajectories.items():
        weeks = t["weeks"]
        scores = t["risk_scores"]

        if len(weeks) < 4:
            continue

        early_week = compute_early_signal_week(weeks, scores, early_threshold, consecutive)
        actual_week = compute_actual_alert_week(weeks, scores)

        if early_week is not None and actual_week is not None:
            delay = actual_week - early_week
            delay = max(0, delay)
        else:
            delay = float("nan")

        rows.append({
            "student_id": sid,
            "module": t.get("module", ""),
            "top_reason": t.get("top_reason", ""),
            "final_risk": t.get("final_risk", 0),
            "early_signal_week": early_week,
            "actual_alert_week": actual_week,
            "delay_weeks": delay,
        })

    return pd.DataFrame(rows)


def generate_temporal_narrative(delay_table: pd.DataFrame) -> str:
    """Generate a 3-paragraph publishable research finding about intervention delay."""
    client = Anthropic()

    valid = delay_table.dropna(subset=["delay_weeks"])
    n_total = len(delay_table)
    n_with_delay = len(valid)

    if n_with_delay == 0:
        return "Insufficient data to generate a temporal narrative."

    mean_delay = valid["delay_weeks"].mean()
    median_delay = valid["delay_weeks"].median()
    max_delay_row = valid.loc[valid["delay_weeks"].idxmax()]
    max_delay = int(max_delay_row["delay_weeks"])
    max_delay_student = int(max_delay_row["student_id"])

    n_no_window = (delay_table["early_signal_week"].isna()).sum()

    # Format a summary table
    table_str = "\n".join(
        f"  Student {int(r.student_id)} | Early: Wk {int(r.early_signal_week) if not pd.isna(r.early_signal_week) else 'N/A'} "
        f"| Alert: Wk {int(r.actual_alert_week) if not pd.isna(r.actual_alert_week) else 'N/A'} "
        f"| Delay: {int(r.delay_weeks) if not pd.isna(r.delay_weeks) else 'N/A'} wks "
        f"| Signal: {r.top_reason}"
        for _, r in delay_table.iterrows()
    )

    # Common top reasons
    top_reasons = (
        valid["top_reason"].value_counts().head(3).index.tolist()
        if not valid.empty else []
    )
    top_reasons_str = ", ".join(top_reasons) if top_reasons else "engagement decline"

    prompt = (
        f"Below is data from a temporal analysis of an XGBoost early warning system "
        f"(AUC 0.975) trained on the UK Open University OULAD dataset (32,593 students).\n\n"
        f"Production alert threshold: risk_score >= 0.66\n"
        f"Early signal threshold tested: risk_score > 0.50 sustained for 2+ consecutive weeks\n\n"
        f"Per-student results:\n{table_str}\n\n"
        f"Aggregate statistics:\n"
        f"- Students analyzed: {n_total}\n"
        f"- Students with measurable early-to-alert delay: {n_with_delay}\n"
        f"- Mean delay: {mean_delay:.1f} weeks\n"
        f"- Median delay: {median_delay:.1f} weeks\n"
        f"- Maximum delay: {max_delay} weeks (Student {max_delay_student})\n"
        f"- Students with no early signal window: {n_no_window}\n"
        f"- Most common primary behavioral signals in delay period: {top_reasons_str}\n\n"
        f"Write a 3-paragraph research finding in ACM/IEEE conference style:\n\n"
        f"Paragraph 1 (2-3 sentences): State the quantitative finding about intervention delay "
        f"with specific numbers. Be precise.\n\n"
        f"Paragraph 2 (3-4 sentences): Describe what behavioral intervention opportunities "
        f"were visible but unacted upon during the delay period. Reference the behavioral signals "
        f"({top_reasons_str}) specifically. State what a counselor could have done at the early "
        f"signal week that would not have been possible once the threshold was crossed.\n\n"
        f"Paragraph 3 (2-3 sentences): Recommend a system design change — specifically a tiered "
        f"alert system with an amber tier at 0.50 — and estimate the potential improvement in "
        f"early identification. Be concrete about the tradeoff (more alerts vs. earlier detection)."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=(
            "You are an academic researcher writing a findings section for a conference paper "
            "on AI-assisted student mental health early warning systems. Your writing is precise, "
            "evidence-based, and uses specific numbers from the data. Avoid hedging."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def build_enhanced_trajectory_chart(
    trajectories: dict,
    delay_table: pd.DataFrame,
    selected_students: list,
) -> go.Figure:
    """Build Plotly chart with early/alert zone shading and per-student delay annotations."""
    fig = go.Figure()

    # Background zones
    fig.add_hrect(y0=0.50, y1=0.66, fillcolor="rgba(251,191,36,0.12)",
                  line_width=0, annotation_text="Early Detection Zone",
                  annotation_position="left", annotation_font_size=11)
    fig.add_hrect(y0=0.66, y1=1.0, fillcolor="rgba(239,68,68,0.07)", line_width=0)

    # Threshold lines
    fig.add_hline(y=0.50, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                  annotation_text="Early Signal (0.50)", annotation_position="top right",
                  annotation_font_color="#f59e0b")
    fig.add_hline(y=0.66, line_dash="dash", line_color="#ef4444", line_width=2,
                  annotation_text="Alert Threshold (0.66)", annotation_position="top left",
                  annotation_font_color="#ef4444")

    colors = ["#6366f1", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
              "#3b82f6", "#10b981", "#f59e0b", "#a855f7", "#ef4444"]

    delay_idx = delay_table.set_index("student_id") if not delay_table.empty else pd.DataFrame()

    for i, sid in enumerate(selected_students):
        t = trajectories[sid]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=t["weeks"],
            y=t["risk_scores"],
            mode="lines+markers",
            name=f"Student {sid}",
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color),
            hovertemplate=f"<b>Student {sid}</b><br>Week: %{{x}}<br>Risk: %{{y:.3f}}<extra></extra>",
        ))

        # Add early signal and actual alert markers
        if sid in delay_idx.index:
            row = delay_idx.loc[sid]
            early_w = row.get("early_signal_week")
            actual_w = row.get("actual_alert_week")

            if pd.notna(early_w):
                early_w = int(early_w)
                weeks_list = t["weeks"]
                scores_list = t["risk_scores"]
                if early_w in weeks_list:
                    idx_e = weeks_list.index(early_w)
                    early_score = scores_list[idx_e]
                    fig.add_trace(go.Scatter(
                        x=[early_w], y=[early_score],
                        mode="markers+text",
                        marker=dict(symbol="circle", size=14, color="#10b981",
                                    line=dict(color="white", width=2)),
                        text=["Early"],
                        textposition="top center",
                        textfont=dict(size=9, color="#10b981"),
                        showlegend=False,
                        hovertemplate=f"<b>Student {sid} — Early Signal</b><br>Week {early_w}<extra></extra>",
                    ))

            if pd.notna(actual_w):
                actual_w = int(actual_w)
                fig.add_trace(go.Scatter(
                    x=[actual_w], y=[0.66],
                    mode="markers+text",
                    marker=dict(symbol="x", size=14, color="#ef4444",
                                line=dict(color="#ef4444", width=2)),
                    text=["Alert"],
                    textposition="top center",
                    textfont=dict(size=9, color="#ef4444"),
                    showlegend=False,
                    hovertemplate=f"<b>Student {sid} — Alert Fired</b><br>Week {actual_w}<extra></extra>",
                ))

            if pd.notna(early_w) and pd.notna(actual_w):
                delay = int(actual_w) - int(early_w)
                if delay > 0:
                    mid_x = (int(early_w) + int(actual_w)) / 2
                    fig.add_annotation(
                        x=mid_x, y=0.58,
                        text=f"{delay}wk delay",
                        showarrow=False,
                        font=dict(size=9, color="#6b7280"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#e5e7eb",
                        borderwidth=1,
                        borderpad=3,
                    )

    fig.update_layout(
        title="Temporal Early Warning — Detection Delay per Student",
        xaxis_title="Course Week",
        yaxis_title="Simulated Risk Score",
        yaxis=dict(range=[0, 1.05]),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="rgba(250,250,255,1)",
        paper_bgcolor="rgba(255,255,255,1)",
    )

    return fig
