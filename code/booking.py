import uuid
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-sonnet-4-6"
BASE = Path(__file__).parent.parent / "data"
BOOKINGS_CSV = BASE / "bookings.csv"
BOOKING_SCHEMA = [
    "booking_id", "student_id", "risk_score", "advisor_type",
    "date", "time_slot", "urgency", "notes", "ai_briefing", "status", "booked_at",
]

TIME_SLOTS = [
    "09:00–09:30", "09:30–10:00", "10:00–10:30", "11:00–11:30",
    "14:00–14:30", "14:30–15:00", "15:00–15:30", "16:00–16:30",
]

ADVISOR_TYPES = ["Class Advisor", "Professional Advisor", "Career Advisor", "Therapist"]


def load_bookings() -> pd.DataFrame:
    if BOOKINGS_CSV.exists():
        try:
            df = pd.read_csv(BOOKINGS_CSV, dtype=str)
            for col in BOOKING_SCHEMA:
                if col not in df.columns:
                    df[col] = ""
            return df[BOOKING_SCHEMA]
        except Exception:
            pass
    return pd.DataFrame(columns=BOOKING_SCHEMA)


def save_booking(record: dict) -> None:
    existing = load_bookings()
    new_row = pd.DataFrame([{col: record.get(col, "") for col in BOOKING_SCHEMA}])
    combined = pd.concat([existing, new_row], ignore_index=True)
    BOOKINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(BOOKINGS_CSV, index=False)


def update_booking_status(booking_id: str, new_status: str) -> None:
    df = load_bookings()
    df.loc[df["booking_id"] == booking_id, "status"] = new_status
    df.to_csv(BOOKINGS_CSV, index=False)


def generate_booking_id() -> str:
    return uuid.uuid4().hex[:8].upper()


def compute_urgency(risk_score: float) -> str:
    if risk_score >= 0.80:
        return "Immediate"
    if risk_score >= 0.66:
        return "Soon"
    return "Routine"


def generate_advisor_briefing(student_id: int, advisor_type: str, student_data: dict) -> str:
    client = Anthropic()

    risk_score = student_data.get("risk_score", 0)
    urgency = compute_urgency(float(risk_score))

    prompt = (
        f"Prepare a pre-meeting briefing for a {advisor_type} who is about to meet "
        f"Student {student_id}.\n\n"
        f"Student behavioral data (CONFIDENTIAL — for advisor use only):\n"
        f"- Risk Score: {float(risk_score):.3f} | Urgency: {urgency}\n"
        f"- Module: {student_data.get('module', 'Unknown')} | "
        f"Gender: {student_data.get('gender', 'Unknown')} | "
        f"Age Band: {student_data.get('age_band', 'Unknown')}\n"
        f"- Engagement Span: {float(student_data.get('engagement_span_days', 0)):.0f} days active\n"
        f"- Mean Assessment Score: {float(student_data.get('mean_score', 0)):.1f}/100\n"
        f"- Modules Dropped: {student_data.get('dropout_modules', 0)}\n"
        f"- Active Days: {float(student_data.get('active_days', 0)):.0f}\n"
        f"- Engagement Decline Rate: {float(student_data.get('engagement_decline', 0)):.3f}\n"
        f"- Primary behavioral signal: {student_data.get('top_reason', 'Unknown')}\n"
        f"- Secondary signal: {student_data.get('reason_2', 'Unknown')}\n"
        f"- Tertiary signal: {student_data.get('reason_3', 'Unknown')}\n\n"
        f"Write a 4-sentence pre-meeting briefing covering:\n"
        f"1. Why this student was flagged and what behavioral pattern the data shows.\n"
        f"2. The two most important things the {advisor_type} should listen for.\n"
        f"3. One specific opening question that builds trust without alarming the student.\n"
        f"4. Any boundary or referral consideration relevant to this advisor type.\n\n"
        f"Do NOT mention AI, algorithms, risk scores, or data analysis in the briefing. "
        f"This briefing is for the advisor's eyes only."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=350,
        system=(
            "You are a clinical support coordinator preparing concise pre-meeting briefings "
            "for university advisors. Your briefings are factual, warm, and focused on "
            "behavioral signals — never diagnoses. You write in 4 clear sentences."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
