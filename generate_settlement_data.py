"""
Generates a synthetic settlement dataset for the Q&A agent.
Uses the trained classifier to assign categories, and adds realistic
fields needed for settlement Q&A: date, status, counterparty, transaction_id.
"""
import pandas as pd
import numpy as np
import joblib
import re
import random
from datetime import datetime, timedelta

random.seed(7)
np.random.seed(7)

# ---------- Load trained classifier ----------
model = joblib.load("/home/claude/finance_classifier_model.joblib")
vectorizer = joblib.load("/home/claude/finance_vectorizer.joblib")

def clean_text(t):
    t = t.lower()
    t = re.sub(r'ref:\w+', '', t)
    t = re.sub(r'inr\s*[\d.]+', '', t)
    t = re.sub(r'[^a-z\s/]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ---------- Load base transaction text templates ----------
df = pd.read_csv("/home/claude/trainable_dataset.csv")

# Sample 70 records to build the settlement batch (buildathon wants 50+)
sample = df.sample(n=70, random_state=7).reset_index(drop=True)

# ---------- Build settlement-specific fields ----------
statuses = ["Settled", "Pending", "Failed"]
status_weights = [0.75, 0.18, 0.07]  # mostly settled, some pending/failed - realistic

start_date = datetime(2026, 7, 1)
records = []
for i, row in sample.iterrows():
    txn_id = f"TXN{100000+i}"
    date = start_date + timedelta(days=random.randint(0, 45))
    status = np.random.choice(statuses, p=status_weights)

    # extract amount from original text
    amt_match = re.search(r'INR\s*([\d.]+)', row['Transaction_Text'])
    amount = float(amt_match.group(1)) if amt_match else round(random.uniform(200, 50000), 2)

    # extract merchant/counterparty (text before " | Ref")
    counterparty = row['Transaction_Text'].split(' | Ref')[0].split('|')[0].strip()

    # predict category + confidence using OUR trained model
    cleaned = clean_text(row['Transaction_Text'])
    X = vectorizer.transform([cleaned])
    pred_category = model.predict(X)[0]
    confidence = model.predict_proba(X).max()

    records.append({
        "transaction_id": txn_id,
        "date": date.strftime("%Y-%m-%d"),
        "counterparty": counterparty,
        "amount": amount,
        "category": pred_category,
        "category_confidence": round(confidence, 4),
        "status": status,
        "raw_text": row['Transaction_Text']
    })

settlement_df = pd.DataFrame(records)
settlement_df.to_csv("/home/claude/settlement_batch.csv", index=False)

print(f"Generated settlement batch: {len(settlement_df)} records")
print()
print("Status distribution:")
print(settlement_df['status'].value_counts())
print()
print("Category distribution:")
print(settlement_df['category'].value_counts())
print()
print("Sample records:")
print(settlement_df.head(5).to_string(index=False))
print()
print(f"Date range: {settlement_df['date'].min()} to {settlement_df['date'].max()}")
print(f"Total amount: INR {settlement_df['amount'].sum():,.2f}")
