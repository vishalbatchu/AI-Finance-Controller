"""
Trains a transaction category classifier on the trainable dataset.

Pipeline: TF-IDF (text -> numbers) + Logistic Regression (the actual "brain")
Also builds a confidence-based exceptions layer, and tests it against the
held-out ambiguous set to prove it can flag uncertain cases honestly.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import re

# ---------- 1. Load data ----------
df = pd.read_csv("/home/claude/trainable_dataset.csv")

# ---------- 2. Clean text a little ----------
def clean_text(t):
    t = t.lower()
    t = re.sub(r'ref:\w+', '', t)          # remove reference codes (not useful signal)
    t = re.sub(r'inr\s*[\d.]+', '', t)     # remove amounts (not useful signal for category)
    t = re.sub(r'[^a-z\s/]', ' ', t)       # keep letters, spaces, slashes
    t = re.sub(r'\s+', ' ', t).strip()
    return t

df['clean_text'] = df['Transaction_Text'].apply(clean_text)

# ---------- 3. Split: 70% train, 15% val, 15% test (stratified so all labels stay balanced) ----------
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['Label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['Label'], random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---------- 4. Vectorize text (TF-IDF: turns words into meaningful numbers) ----------
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_val = vectorizer.transform(val_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

y_train = train_df['Label']
y_val = val_df['Label']
y_test = test_df['Label']

# ---------- 5. Train the model (the actual "brain") ----------
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# ---------- 6. Evaluate on validation set ----------
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"\nValidation accuracy: {val_acc:.4f}")

# ---------- 7. Evaluate on TEST set (the real, honest number) ----------
test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)
test_acc = accuracy_score(y_test, test_preds)

print(f"\n{'='*50}")
print(f"TEST SET ACCURACY (honest, held-out): {test_acc:.4f}")
print(f"{'='*50}\n")

print("Classification report (per category):")
print(classification_report(y_test, test_preds))

print("Confusion matrix (rows=actual, cols=predicted):")
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, test_preds, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# ---------- 8. Confidence + exceptions layer ----------
CONFIDENCE_THRESHOLD = 0.60

max_probs = test_probs.max(axis=1)
low_confidence_mask = max_probs < CONFIDENCE_THRESHOLD

n_exceptions = low_confidence_mask.sum()
match_rate = 1 - (n_exceptions / len(test_df))

print(f"\n{'='*50}")
print(f"EXCEPTIONS REPORT (confidence threshold = {CONFIDENCE_THRESHOLD})")
print(f"{'='*50}")
print(f"Total test records: {len(test_df)}")
print(f"Confidently matched: {len(test_df) - n_exceptions}")
print(f"Flagged as exceptions (low confidence): {n_exceptions}")
print(f"Match rate: {match_rate:.2%}")

# ---------- 9. THE REAL TEST: does it flag the genuinely ambiguous records? ----------
print(f"\n{'='*50}")
print("AMBIGUOUS HOLD-OUT TEST (records it was never trained on)")
print(f"{'='*50}")

ambig_df = pd.read_csv("/home/claude/ambiguous_holdout.csv")
ambig_df['clean_text'] = ambig_df['Transaction_Text'].apply(clean_text)
X_ambig = vectorizer.transform(ambig_df['clean_text'])
ambig_preds = model.predict(X_ambig)
ambig_probs = model.predict_proba(X_ambig)
ambig_max_probs = ambig_probs.max(axis=1)

for i, row in ambig_df.iterrows():
    pred = ambig_preds[i]
    conf = ambig_max_probs[i]
    flag = "EXCEPTION (low confidence)" if conf < CONFIDENCE_THRESHOLD else "confidently matched"
    print(f"\nText: {row['Transaction_Text'][:60]}")
    print(f"  Predicted: {pred} | Confidence: {conf:.2%} | Status: {flag}")

ambig_exception_count = (ambig_max_probs < CONFIDENCE_THRESHOLD).sum()
print(f"\n{ambig_exception_count}/{len(ambig_df)} ambiguous records correctly flagged as exceptions")

# ---------- 10. Save the trained model ----------
joblib.dump(model, "/home/claude/finance_classifier_model.joblib")
joblib.dump(vectorizer, "/home/claude/finance_vectorizer.joblib")
print("\nModel and vectorizer saved.")
