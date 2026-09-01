# AI Finance Controller - Razorpay AI Buildathon Track 04

A Flask-based, human-in-the-loop finance-ops agent for a 50+ transaction synthetic settlement batch.

## Pitch

Finance controllers do not only need faster classification; they need a verified loop. This project classifies settlement transactions, auto-resolves confident records, exposes uncertain records as exceptions, captures human corrections, persists the audit trail, and retrains from verified feedback.

Built for Razorpay Buildathon Track 04: AI Finance Controller.

## Demo evidence

- Bundled settlement batch: 70 synthetic transactions.
- Exception threshold: predictions below 60% confidence require human review.
- Bundled demo batch currently flags 1 exception out of 70 records.
- Included stress artifact: `transaction_classifier_stress_results.csv`.
- Current stress artifact result: 100/100 correct synthetic cases.
- Production storage: Render PostgreSQL when `DATABASE_URL` is configured.
- Local storage fallback: SQLite.

## Core loop

1. Upload a transaction batch.
2. Normalize messy bank narrations and classify them with a robust TF-IDF + Logistic Regression model.
3. Measure confidence and auto-resolve high-confidence transactions.
4. Route low-confidence transactions to an honest exception queue.
5. Let a finance admin override the category when they can identify the correct one.
6. Persist the correction as human feedback and an audit event.
7. Ask natural-language questions against the current batch.
8. Export human feedback or retrain the classifier with verified corrections.
9. Report match rate, status/category mix, model stress-test performance, agent activity, and review outcomes.

## Live health check

After deployment, open:

```text
https://YOUR_RENDER_SERVICE_URL.onrender.com/api/health
```

Expected result:

```json
{
  "ok": true,
  "database_configured": true
}
```

The exact `storage` value depends on the active backend. On Render it should indicate PostgreSQL; locally it should indicate SQLite fallback.

## Dashboard

The front page intentionally shows only the transaction table. The **DASHBOARD** button reveals five operating sections:

- **Summary** — match rate, batch totals, statuses, AI/manual resolution counts, agent activity, model performance, and feedback readiness.
- **Exceptions** — low-confidence transactions, reason, alternative prediction, admin category override, audit trail, feedback export, and retraining.
- **Ask the Controller** — data-grounded finance Q&A with suggested questions.
- **Category Breakdown** — current classification mix.
- **Upload & Re-run** — classify a fresh CSV and refresh the complete workflow.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## CSV upload

Required transaction-text field: `Transaction_Text` (also accepts `raw_text`, `description`, or `narration`).

Optional fields: `transaction_id`, `date`, `counterparty`, `amount`, `status`.

## Human-in-the-loop

Manual category corrections are written to `human_feedback.csv`. Events are appended to `audit_log.jsonl`. The **RETRAIN WITH FEEDBACK** action adds verified corrections to the training data and rebuilds the word + character TF-IDF classifier.

## Evaluation

`transaction_classifier_stress_results.csv` contains 100 unseen-style synthetic transaction cases covering Food, Travel, EMI, Investment, and Shopping. The dashboard reports the measured result from that file separately from the live batch match rate.

The live **match rate** is the percentage of current-batch transactions at or above the 60% confidence threshold; it is not presented as ground-truth accuracy.

## Persistent storage (production-ready prototype)

The app now includes a storage abstraction in `db_store.py`.
- If `DATABASE_URL` is set, the application uses PostgreSQL for transactions, human feedback, audit events, and versioned model artifacts.
- If `DATABASE_URL` is not set, local development automatically falls back to SQLite.
- Existing CSV files remain useful as demo/import artifacts and are used to bootstrap an empty database.
- Human corrections are persisted before model learning, so a training failure does not erase the admin decision.

For Render, create a PostgreSQL database and set its connection string as the `DATABASE_URL` environment variable on the web service. No source-code secret is required.

### ML safety improvements

The classifier exposes a human-review decision for low-confidence predictions and close top-two probability margins. It also caps CSV uploads at 5,000 rows to keep the demo service responsive. The reported stress-test metrics are evaluation results on the project's synthetic holdout sets, not claims of universal production accuracy.
