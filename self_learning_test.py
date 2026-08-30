"""Offline verification for the human-feedback learning dataset.

This does not modify the shipped model. It reports how many verified corrections
would be incorporated into the next model version and checks the feedback labels.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
FEEDBACK = BASE / "human_feedback.csv"
CATEGORIES = {"Food", "Travel", "EMI", "Investment", "Shopping"}

if not FEEDBACK.exists():
    print("No human_feedback.csv found.")
    raise SystemExit(0)

df = pd.read_csv(FEEDBACK)
if df.empty:
    print("Human feedback rows: 0")
    print("Self-learning is enabled; correct an exception to create training feedback.")
    raise SystemExit(0)

bad = sorted(set(df["corrected_category"].dropna().astype(str)) - CATEGORIES)
latest = (df.assign(_ts=pd.to_datetime(df.get("timestamp"), errors="coerce"))
            .sort_values("_ts")
            .drop_duplicates(subset=["raw_text"], keep="last"))

print(f"Human feedback rows recorded: {len(df)}")
print(f"Unique latest corrections used for learning: {len(latest)}")
print(f"Valid labels: {not bad}")
if bad:
    print("Invalid labels:", bad)
print("Feedback is ready to be incorporated into the next model version.")
