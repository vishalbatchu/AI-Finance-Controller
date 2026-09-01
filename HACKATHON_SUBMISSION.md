# Razorpay Buildathon Submission Guide

## One-line pitch

AI Finance Controller is a human-in-the-loop finance-ops agent that classifies settlement transactions, auto-resolves high-confidence records, flags exceptions, accepts finance-admin corrections, learns from feedback, and answers batch questions.

## Track fit

Track 04: AI Finance Controller

The project closes one finance-ops loop across a synthetic settlement batch:

1. Ingest settlement transactions.
2. Normalize messy transaction narration.
3. Predict finance category with an ML classifier.
4. Auto-resolve confident predictions.
5. Route low-confidence records to an exception queue.
6. Let a human reviewer correct categories.
7. Persist corrections and audit events.
8. Retrain from verified feedback.
9. Report batch status through dashboard metrics and Q&A.

## Demo evidence

- Synthetic settlement batch: 70 records in `settlement_batch.csv`
- Current exception rule: confidence below 60%
- Current demo exceptions: 1 out of 70 records in the bundled batch
- Stress result artifact: `transaction_classifier_stress_results.csv`
- Stress result currently included: 100/100 correct synthetic cases
- Production persistence: Render PostgreSQL via `DATABASE_URL`
- Local fallback: SQLite when `DATABASE_URL` is not configured

## What to show in the video

Keep the demo tight and practical:

1. Open the live Render app and show the transaction-first home screen.
2. Open Dashboard and point to total transactions, match rate, exceptions, and model status.
3. Open Exceptions and show that uncertain records are not hidden.
4. Correct one transaction category and show the learning/audit response.
5. Ask a batch question in Q&A, for example: "Which category has the highest total amount?"
6. Upload a CSV batch and show the system reclassifies the batch.
7. Open `/api/health` briefly to show PostgreSQL is configured.

## Suggested 3-minute narration

AI Finance Controller solves a finance-operations bottleneck: settlement review still needs verification, not just prediction. The system processes a synthetic batch of 70 transactions, classifies each one into finance categories, and only auto-resolves records when model confidence is high enough.

The important part is the exception loop. Low-confidence records are pushed into review with a reason and alternative predictions. A finance admin can correct the category, and that decision is persisted as human feedback with an audit event. The model can then retrain using verified corrections, so the workflow improves without losing human control.

The deployed version uses PostgreSQL on Render through `DATABASE_URL`, with SQLite as a local fallback. The dashboard reports throughput, match rate, unresolved exceptions, status/category mix, model stress-test evidence, and activity history. The Q&A panel lets the reviewer ask data-grounded questions about the current settlement batch.

This is built as a working finance-ops loop: batch in, predictions out, exceptions reviewed, feedback saved, learning updated, and metrics reported honestly.

## Final checklist before submission

- Live Render link works.
- `/api/health` shows `ok: true`.
- `/api/health` shows `database_configured: true`.
- GitHub repository is public.
- README has local setup and deployment notes.
- Demo video shows the live app, not only screenshots.
- Submission mentions synthetic data and does not claim real bank production accuracy.
- Any reused/generated code or AI assistance is disclosed if the form asks.
