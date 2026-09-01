from pathlib import Path
import io
import json
import os
import pickle
from datetime import datetime, timezone
import threading

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from settlement_qa import SettlementQA
from classifier_utils import clean_transaction_text as clean_text
from db_store import Store

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "settlement_batch.csv"
MODEL_FILE = BASE_DIR / "finance_classifier_model.joblib"
VECTORIZER_FILE = BASE_DIR / "finance_vectorizer.joblib"
FEEDBACK_FILE = BASE_DIR / "human_feedback.csv"
AUDIT_FILE = BASE_DIR / "audit_log.jsonl"
STRESS_FILE = BASE_DIR / "transaction_classifier_stress_results.csv"
TRAIN_FILE = BASE_DIR / "trainable_dataset.csv"
MODEL_META_FILE = BASE_DIR / "model_metadata.json"

RETRAIN_LOCK = threading.Lock()

CONFIDENCE_THRESHOLD = 0.60
ALLOWED_CATEGORIES = ["Food", "Travel", "EMI", "Investment", "Shopping"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["MAX_UPLOAD_ROWS"] = 5000

store = Store(BASE_DIR)


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    # Browser fetch() expects JSON from API routes. Flask's default 500 page is HTML,
    # which caused: "Unexpected token '<', '<!doctype...' is not valid JSON".
    if request.path.startswith("/api/"):
        log_event("API_ERROR", path=request.path, error=str(exc))
        return jsonify({"error": f"Server error in {request.path}: {exc}"}), 500
    raise exc


def _load_live_model():
    latest = store.model_latest()
    if latest and latest.get("model_blob") and latest.get("vectorizer_blob"):
        try:
            return pickle.loads(bytes(latest["model_blob BYTEA"])), pickle.loads(bytes(latest["vectorizer_blob BYTEA"]))
        except Exception:
            pass
    return joblib.load(MODEL_FILE), joblib.load(VECTORIZER_FILE)


model, vectorizer = _load_live_model()

# Persist the baseline model artifact once when a database is configured. This means
# future Render restarts can load the model from PostgreSQL instead of relying only
# on the ephemeral service filesystem.
try:
    if store.model_latest() is None:
        baseline_meta = {
            "version": "baseline",
            "trained_at": None,
            "training_rows": 0,
            "human_feedback_rows": 0,
            "latest_feedback_rows": 0,
            "learning_mode": "Automatic human-feedback learning enabled",
        }
        store.save_model("baseline", "1970-01-01T00:00:00+00:00", baseline_meta, pickle.dumps(model), pickle.dumps(vectorizer))
except Exception:
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_model_metadata():
    default = {
        "version": "baseline",
        "trained_at": None,
        "training_rows": 0,
        "human_feedback_rows": 0,
        "learning_mode": "Human corrections are automatically incorporated into the next model version.",
    }
    latest = store.model_latest()
    if latest:
        try:
            default.update(json.loads(latest.get("metadata") or "{}"))
            default["version"] = latest.get("version") or default["version"]
            return default
        except Exception:
            pass
    if not MODEL_META_FILE.exists():
        return default
    try:
        data = json.loads(MODEL_META_FILE.read_text(encoding="utf-8"))
        default.update(data)
    except Exception:
        pass
    return default


def save_model_metadata(**updates):
    meta = load_model_metadata()
    meta.update(updates)
    MODEL_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if not MODEL_META_FILE.exists():
    MODEL_META_FILE.write_text(json.dumps({
        "version": "baseline",
        "trained_at": None,
        "training_rows": 0,
        "human_feedback_rows": 0,
        "latest_feedback_rows": 0,
        "learning_mode": "Automatic human-feedback learning enabled",
    }, indent=2), encoding="utf-8")


def log_event(event, **details):
    record = {"timestamp": now_iso(), "event": event, **details}
    try:
        with AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
    try:
        store.audit(record)
    except Exception:
        pass
    return record


def load_data():
    if store.transaction_count() > 0:
        df = store.load_transactions()
    else:
        df = pd.read_csv(DATA_FILE)
        if "original_category" not in df.columns:
            df["original_category"] = df["category"]
        if "classification_source" not in df.columns:
            df["classification_source"] = "Model"
        try:
            store.replace_transactions(df)
        except Exception:
            pass
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


transactions = load_data()

# One-time migration of any existing demo feedback into the persistent store.
try:
    if store.feedback_df().empty and FEEDBACK_FILE.exists():
        existing_feedback = pd.read_csv(FEEDBACK_FILE)
        for row in existing_feedback.fillna("").to_dict(orient="records"):
            store.append_feedback(row)
except Exception:
    pass


def classify_texts(texts):
    cleaned = [clean_text(x) for x in texts]
    X = vectorizer.transform(cleaned)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    results = []
    for pred, probs in zip(predictions, probabilities):
        ranked = sorted(zip(classes, probs), key=lambda item: item[1], reverse=True)
        confidence = float(ranked[0][1])
        second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        review_required = confidence < CONFIDENCE_THRESHOLD or (confidence - second) < 0.10
        signal = "LOW_SIGNAL" if confidence < 0.40 else ("HUMAN_REVIEW" if review_required else "AUTO_RESOLVE")
        results.append({
            "category": str(pred),
            "category_confidence": round(confidence, 4),
            "review_required": review_required,
            "decision": signal,
            "top_alternatives": [
                {"category": str(cat), "confidence": round(float(prob), 4)}
                for cat, prob in ranked[:3]
            ],
            "margin": round(confidence - second, 4),
        })
    return results


def reason_for_exception(row, prediction=None):
    confidence = float(row["category_confidence"])
    margin = float((prediction or {}).get("margin", 1.0))
    if confidence < 0.40:
        return "Very low model confidence; the narration does not strongly indicate one category."
    if margin < 0.10:
        return "The top category is close to the next-best category, so the prediction needs review."
    return "Model confidence is below the 60% auto-resolution threshold."


def serialize_row(row):
    return {
        "transaction_id": str(row["transaction_id"]),
        "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d") if pd.notna(row["date"]) else "",
        "counterparty": str(row.get("counterparty", "")),
        "amount": round(float(row.get("amount", 0)), 2),
        "category": str(row["category"]),
        "category_confidence": round(float(row["category_confidence"]), 4),
        "status": str(row.get("status", "Pending")),
        "classification_source": str(row.get("classification_source", "Model")),
        "original_category": str(row.get("original_category", row["category"])),
        "raw_text": str(row.get("raw_text", row.get("counterparty", ""))),
        "review_required": float(row.get("category_confidence", 0)) < CONFIDENCE_THRESHOLD,
        "decision": "HUMAN_REVIEW" if float(row.get("category_confidence", 0)) < CONFIDENCE_THRESHOLD else "AUTO_RESOLVE",
    }


def get_exceptions_df():
    df = transactions.copy()
    return df[pd.to_numeric(df["category_confidence"], errors="coerce") < CONFIDENCE_THRESHOLD].copy()


def exception_items():
    df = get_exceptions_df()
    rows = []
    for _, row in df.iterrows():
        prediction = None
        try:
            prediction = classify_texts([row.get("raw_text", row.get("counterparty", ""))])[0]
        except Exception:
            prediction = None
        item = serialize_row(row)
        item["confidence_pct"] = round(float(row["category_confidence"]) * 100, 1)
        item["reason"] = reason_for_exception(row, prediction)
        item["alternatives"] = (prediction or {}).get("top_alternatives", [])
        item["review_status"] = "OPEN"
        rows.append(item)
    return rows


def read_audit(limit=30):
    try:
        db_events = store.read_audit(limit)
        if db_events:
            return db_events
    except Exception:
        pass
    if not AUDIT_FILE.exists():
        return []
    lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    output = []
    for line in reversed(lines):
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return output


def feedback_df():
    try:
        db = store.feedback_df()
        if not db.empty:
            return db
    except Exception:
        pass
    if not FEEDBACK_FILE.exists():
        return pd.DataFrame(columns=[
            "timestamp", "transaction_id", "raw_text", "original_category",
            "model_confidence", "corrected_category", "source"
        ])
    try:
        return pd.read_csv(FEEDBACK_FILE)
    except Exception:
        return pd.DataFrame()


def stress_metrics():
    if not STRESS_FILE.exists():
        return {"accuracy": None, "tested": 0, "correct": 0, "exceptions": 0, "by_category": {}}
    try:
        df = pd.read_csv(STRESS_FILE)
        correct = int((df["Expected"].astype(str) == df["Predicted"].astype(str)).sum())
        total = len(df)
        by_category = {}
        for category, group in df.groupby("Expected"):
            by_category[str(category)] = {
                "tested": int(len(group)),
                "accuracy": round(float((group["Expected"] == group["Predicted"]).mean()), 4),
            }
        return {
            "accuracy": round(correct / total, 4) if total else None,
            "tested": total,
            "correct": correct,
            "exceptions": int(df.get("Exception", pd.Series(dtype=bool)).sum()) if "Exception" in df else 0,
            "by_category": by_category,
        }
    except Exception:
        return {"accuracy": None, "tested": 0, "correct": 0, "exceptions": 0, "by_category": {}}


def train_with_feedback():
    """Retrain the classifier using the original corpus plus verified human corrections.

    Human corrections are given a higher sample weight so the model learns from them
    without throwing away the broad baseline training data. If the same narration has
    been corrected multiple times, the latest verified label wins.
    """
    global model, vectorizer
    with RETRAIN_LOCK:
        base = pd.read_csv(TRAIN_FILE)[["Transaction_Text", "Label"]]
        noisy_path = BASE_DIR / "noisy_labelable.csv"
        noisy = pd.read_csv(noisy_path)[["Transaction_Text", "Label"]] if noisy_path.exists() else pd.DataFrame(columns=["Transaction_Text", "Label"])

        aug = {
            "Food": ["swiggy food order", "zomato restaurant payment", "mcdonalds meal", "kfc dinner", "pos kfc food purchase", "dominos takeaway", "blinkit grocery order", "bigbasket groceries", "restaurant card purchase", "upi food delivery", "grocery store purchase", "dinner bill paid", "food order payment"],
            "Travel": ["uber cab ride", "ola ride fare", "rapido bike taxi", "irctc rail ticket", "redbus ticket", "indigo flight ticket", "air india airfare", "makemytrip hotel", "oyo hotel stay", "airport transfer", "taxi fare", "fuel station payment", "flight booking payment"],
            "EMI": ["home loan installment", "car loan installment", "bike loan installment", "credit card emi", "bajaj finserv installment", "loan auto debit", "monthly loan installment", "mortgage payment", "education loan installment", "nach loan installment", "bnpl installment", "paylater installment", "loan repayment debit"],
            "Investment": ["zerodha equity purchase", "groww mutual fund sip", "fd booking", "mutual fund investment", "sip debit", "shares purchase", "equity delivery", "etf purchase", "demat investment", "nps contribution", "ppf contribution", "bonds purchase", "stock market investment"],
            "Shopping": ["amazon online order", "flipkart purchase", "myntra clothing order", "ajio fashion purchase", "meesho order", "croma electronics purchase", "titan purchase", "tanishq jewellery purchase", "ikea furniture purchase", "decathlon shopping", "electronics purchase", "clothing purchase", "online shopping payment"]
        }
        augdf = pd.DataFrame([(text, cat) for cat, texts in aug.items() for text in texts], columns=["Transaction_Text", "Label"])

        fb = feedback_df()
        extra = pd.DataFrame(columns=["Transaction_Text", "Label"])
        if not fb.empty:
            fb = fb.dropna(subset=["raw_text", "corrected_category"]).copy()
            fb["timestamp"] = pd.to_datetime(fb.get("timestamp"), errors="coerce")
            fb = fb.sort_values("timestamp").drop_duplicates(subset=["raw_text"], keep="last")
            extra = fb.rename(columns={"raw_text": "Transaction_Text", "corrected_category": "Label"})[["Transaction_Text", "Label"]]

        combined = pd.concat([base, noisy, augdf, extra], ignore_index=True).dropna(subset=["Transaction_Text", "Label"])
        combined["Transaction_Text"] = combined["Transaction_Text"].astype(str)
        combined["Label"] = combined["Label"].astype(str)
        combined = combined.drop_duplicates(subset=["Transaction_Text", "Label"]).reset_index(drop=True)
        combined["clean_text"] = combined["Transaction_Text"].map(clean_text)

        vectorizer_new = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=12000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=18000, sublinear_tf=True)),
        ])
        X = vectorizer_new.fit_transform(combined["clean_text"])

        # Human-verified examples receive extra influence while baseline examples
        # retain the majority of the training signal.
        human_texts = set(extra["Transaction_Text"].astype(str)) if not extra.empty else set()
        sample_weights = np.ones(len(combined), dtype=float)
        if human_texts:
            sample_weights[combined["Transaction_Text"].isin(human_texts).to_numpy()] = 3.0

        model_new = LogisticRegression(max_iter=2500, C=2.0, class_weight="balanced")
        model_new.fit(X, combined["Label"], sample_weight=sample_weights)

        # Only swap the live model after the complete new model has trained.
        joblib.dump(model_new, MODEL_FILE)
        joblib.dump(vectorizer_new, VECTORIZER_FILE)
        model = model_new
        vectorizer = vectorizer_new

        version = f"learned-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        meta = save_model_metadata(
            version=version,
            trained_at=now_iso(),
            training_rows=int(len(combined)),
            human_feedback_rows=int(len(fb)) if not fb.empty else 0,
            latest_feedback_rows=int(len(extra)),
            learning_mode="Automatic human-feedback learning enabled",
        )
        try:
            store.save_model(version, meta.get("trained_at"), meta, pickle.dumps(model_new), pickle.dumps(vectorizer_new))
        except Exception as exc:
            log_event("MODEL_DATABASE_SAVE_FAILED", error=str(exc), model_version=version)
        return int(len(combined)), int(len(extra)), meta


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/settlement_batch")
def settlement_batch():
    return jsonify([serialize_row(row) for _, row in transactions.iterrows()])


@app.get("/api/summary")
def summary():
    df = transactions.copy()
    total = len(df)
    confidence = pd.to_numeric(df["category_confidence"], errors="coerce").fillna(0)
    exceptions = int((confidence < CONFIDENCE_THRESHOLD).sum())
    manual = int((df.get("classification_source", pd.Series(index=df.index, data="Model")) == "Manual review").sum())
    auto_resolved = max(total - exceptions, 0)
    feedback = feedback_df()
    stress = stress_metrics()
    model_meta = load_model_metadata()

    avg_conf_by_category = {}
    for category, group in df.groupby("category"):
        avg_conf_by_category[str(category)] = round(float(pd.to_numeric(group["category_confidence"], errors="coerce").mean()), 4)

    return jsonify({
        "total_transactions": total,
        "total_amount": round(float(pd.to_numeric(df["amount"], errors="coerce").fillna(0).sum()), 2),
        "match_rate": round(((total - exceptions) / total) if total else 1.0, 4),
        "exceptions_count": exceptions,
        "auto_resolved": auto_resolved,
        "manual_resolved": manual,
        "unresolved": exceptions,
        "feedback_count": int(len(feedback)),
        "status_counts": {str(k): int(v) for k, v in df["status"].value_counts().to_dict().items()},
        "category_counts": {str(k): int(v) for k, v in df["category"].value_counts().to_dict().items()},
        "avg_confidence_by_category": avg_conf_by_category,
        "model_performance": stress,
        "model_learning": {
            "version": model_meta.get("version"),
            "trained_at": model_meta.get("trained_at"),
            "training_rows": model_meta.get("training_rows", 0),
            "human_feedback_rows": model_meta.get("human_feedback_rows", 0),
            "learning_mode": model_meta.get("learning_mode"),
        },
    })


@app.get("/api/exceptions")
def exceptions():
    return jsonify(exception_items())


@app.get("/api/activity")
def activity():
    return jsonify(read_audit(40))


@app.get("/api/feedback")
def feedback():
    df = feedback_df()
    if df.empty:
        return jsonify([])
    return jsonify(df.tail(30).iloc[::-1].fillna("").to_dict(orient="records"))


@app.post("/api/transactions/<transaction_id>/category")
def update_category(transaction_id):
    global transactions
    try:
        payload = request.get_json(silent=True) or {}
        new_category = str(payload.get("category", "")).strip()
        if new_category not in ALLOWED_CATEGORIES:
            return jsonify({"error": f"Invalid category. Choose one of: {', '.join(ALLOWED_CATEGORIES)}"}), 400

        mask = transactions["transaction_id"].astype(str).str.upper() == str(transaction_id).upper()
        if not mask.any():
            return jsonify({"error": f"Transaction {transaction_id} not found."}), 404

        idx = transactions.index[mask][0]
        old_category = str(transactions.at[idx, "category"])
        confidence = float(transactions.at[idx, "category_confidence"])
        raw_text = str(transactions.loc[idx].get("raw_text", transactions.loc[idx].get("counterparty", "")))

        if "original_category" not in transactions.columns:
            transactions["original_category"] = transactions["category"]
        if "classification_source" not in transactions.columns:
            transactions["classification_source"] = "Model"

        # Persist the human decision first. This makes the correction durable even
        # if model retraining temporarily fails.
        transactions.at[idx, "category"] = new_category
        transactions.at[idx, "category_confidence"] = 1.0
        transactions.at[idx, "classification_source"] = "Manual review"
        transactions.to_csv(DATA_FILE, index=False)

        feedback_row = pd.DataFrame([{
            "timestamp": now_iso(),
            "transaction_id": str(transaction_id),
            "raw_text": raw_text,
            "original_category": old_category,
            "model_confidence": confidence,
            "corrected_category": new_category,
            "source": "finance_admin",
        }])
        header = not FEEDBACK_FILE.exists()
        feedback_row.to_csv(FEEDBACK_FILE, mode="a", index=False, header=header)
        try:
            store.append_feedback(feedback_row.iloc[0].to_dict())
        except Exception as exc:
            log_event("FEEDBACK_DATABASE_SAVE_FAILED", transaction_id=str(transaction_id), error=str(exc))
        log_event(
            "HUMAN_CORRECTION",
            transaction_id=str(transaction_id),
            original_category=old_category,
            corrected_category=new_category,
            model_confidence=round(confidence, 4),
        )

        # Automatic self-learning. A training failure must NOT turn a successful
        # human correction into a browser 500 error. The correction is already saved
        # and can be learned on the next run.
        learning_error = None
        learning = None
        try:
            training_rows, feedback_rows, learning = train_with_feedback()
            log_event(
                "MODEL_LEARNED_FROM_HUMAN",
                transaction_id=str(transaction_id),
                corrected_category=new_category,
                training_rows=training_rows,
                feedback_rows=feedback_rows,
                model_version=learning.get("version"),
            )
        except Exception as exc:
            learning_error = str(exc)
            log_event("MODEL_LEARNING_FAILED", transaction_id=str(transaction_id), error=learning_error)

        updated = serialize_row(transactions.loc[idx])
        updated.update({
            "previous_category": old_category,
            "message": f"{transaction_id} changed from {old_category} to {new_category}.",
            "review_status": "RESOLVED",
            "learning_status": "LEARNED" if learning else "SAVED_FOR_LEARNING",
            "model_version": (learning or {}).get("version"),
            "learning_error": learning_error,
        })
        return jsonify(updated)
    except Exception as exc:
        log_event("HUMAN_CORRECTION_FAILED", transaction_id=str(transaction_id), error=str(exc))
        return jsonify({"error": f"Could not save this correction: {exc}"}), 500


@app.post("/api/qa")
def qa():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"answer": "Please enter a question about the settlement batch."}), 400
    log_event("QA_QUERY", question=question)
    answer = SettlementQA(transactions.copy()).answer(question)
    return jsonify({"answer": answer})


@app.post("/api/upload")
def upload():
    global transactions
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "Please choose a CSV file."}), 400
    try:
        incoming = pd.read_csv(io.BytesIO(uploaded.read()))
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV: {exc}"}), 400
    if incoming.empty:
        return jsonify({"error": "The CSV is empty."}), 400
    if len(incoming) > app.config["MAX_UPLOAD_ROWS"]:
        return jsonify({"error": f"CSV contains {len(incoming)} rows; maximum allowed is {app.config['MAX_UPLOAD_ROWS']}."}), 400

    text_col = next((c for c in ["Transaction_Text", "transaction_text", "raw_text", "description", "narration"] if c in incoming.columns), None)
    if text_col is None:
        return jsonify({"error": "CSV must contain Transaction_Text, raw_text, description, or narration."}), 400

    texts = incoming[text_col].fillna("").astype(str)
    predictions = classify_texts(texts)
    default_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    ids = incoming["transaction_id"] if "transaction_id" in incoming else pd.Series([""] * len(incoming), index=incoming.index)
    dates = incoming["date"] if "date" in incoming else pd.Series([default_date] * len(incoming), index=incoming.index)
    counterparties = incoming["counterparty"] if "counterparty" in incoming else texts
    amounts = incoming["amount"] if "amount" in incoming else pd.Series([0.0] * len(incoming), index=incoming.index)
    statuses = incoming["status"] if "status" in incoming else pd.Series(["Pending"] * len(incoming), index=incoming.index)

    result = pd.DataFrame({
        "transaction_id": [str(x).strip() if str(x).strip() else f"UPL{100000 + i}" for i, x in enumerate(ids)],
        "date": pd.to_datetime(dates, errors="coerce").fillna(pd.Timestamp.today()).dt.strftime("%Y-%m-%d"),
        "counterparty": counterparties.fillna("").astype(str),
        "amount": pd.to_numeric(amounts, errors="coerce").fillna(0.0),
        "category": [x["category"] for x in predictions],
        "category_confidence": [x["category_confidence"] for x in predictions],
        "status": statuses.fillna("Pending").astype(str),
        "raw_text": texts,
        "original_category": [x["category"] for x in predictions],
        "classification_source": ["Model"] * len(predictions),
    })
    transactions = result
    transactions.to_csv(DATA_FILE, index=False)
    try:
        store.replace_transactions(transactions)
    except Exception as exc:
        return jsonify({"error": f"Transaction batch could not be persisted safely: {exc}"}), 500
    log_event("BATCH_UPLOADED", filename=uploaded.filename, transactions=len(result), exceptions=int((result["category_confidence"] < CONFIDENCE_THRESHOLD).sum()))
    log_event("CLASSIFICATION_COMPLETED", transactions=len(result), auto_resolved=int((result["category_confidence"] >= CONFIDENCE_THRESHOLD).sum()))
    return jsonify({"transactions": [serialize_row(row) for _, row in transactions.iterrows()], "count": len(transactions)})


@app.post("/api/retrain")
def retrain():
    feedback = feedback_df()
    if feedback.empty:
        return jsonify({"error": "No human corrections are available for retraining yet."}), 400
    try:
        total, added, meta = train_with_feedback()
        log_event("MODEL_RETRAINED", training_rows=total, human_feedback_rows=added, model_version=meta.get("version"))
        return jsonify({"message": "Model retrained with human feedback.", "training_rows": total, "feedback_rows": added, "model_version": meta.get("version")})
    except Exception as exc:
        return jsonify({"error": f"Retraining failed: {exc}"}), 500


@app.get("/api/feedback/export")
def export_feedback():
    if not FEEDBACK_FILE.exists():
        return jsonify({"error": "No human feedback exists yet."}), 404
    return send_file(FEEDBACK_FILE, as_attachment=True, download_name="human_feedback.csv")


@app.get("/api/storage")
def storage():
    """Expose safe storage diagnostics without revealing credentials."""
    try:
        return jsonify(store.storage_info())
    except Exception as exc:
        return jsonify({"persistent": False, "error": str(exc)}), 500


@app.get("/api/health")
def health():
    meta = load_model_metadata()
    return jsonify({"ok": True, "transactions": len(transactions), "exceptions": len(get_exceptions_df()), "model_version": meta.get("version"), "human_feedback_rows": meta.get("human_feedback_rows", 0), "storage": store.backend, "database_configured": store.is_configured})


if __name__ == "__main__":
    log_event("APP_STARTED", transactions=len(transactions), exceptions=len(get_exceptions_df()))
    app.run(debug=False, host="127.0.0.1", port=5000)
