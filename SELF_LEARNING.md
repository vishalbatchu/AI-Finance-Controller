# Human-Feedback Self-Learning

The AI Finance Controller now uses **human-in-the-loop self-learning**.

## What happens when an admin corrects an exception?

1. The original AI prediction and confidence are recorded.
2. The admin chooses the correct category.
3. The correction is appended to `human_feedback.csv`.
4. The application immediately retrains a fresh model using:
   - the original training dataset,
   - noisy bank-narration examples,
   - robust category/merchant augmentation,
   - all verified human corrections.
5. Human-verified examples receive extra training weight so the model learns from them more strongly without discarding the baseline knowledge.
6. The new model and vectorizer replace the live model only after training succeeds.
7. A new model version is recorded in `model_metadata.json` and the audit log.
8. Future uploads use the newly learned model automatically.

## Why this is safer than training only on corrections

The system does **not** throw away the original training data. Human feedback is an additional verified signal. If the same narration is corrected multiple times, the latest verified correction is used for learning.

## Demo flow

`AI predicts → low confidence → admin corrects → feedback saved → model learns immediately → next transaction benefits from the new model`

The **RETRAIN NOW WITH FEEDBACK** button remains available for an explicit full retraining run, but normal admin corrections already trigger learning automatically.
