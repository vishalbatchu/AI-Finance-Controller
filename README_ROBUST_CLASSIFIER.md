# Robust transaction classification update

The classifier has been retrained to handle real-world bank narration styles instead of only the original clean merchant phrases.

## What was added

- Word + character TF-IDF features so partial merchant names, abbreviations, separators, and noisy narrations can still match.
- Additional labeled noisy banking examples (`noisy_labelable.csv`) used during training.
- 130+ hand-crafted paraphrases and real-world merchant/narration variants used during training.
- Normalization for common narration formats: UPI, POS, NEFT, IMPS, NACH, ECS, UTR-like numbers, EMI, BNPL, FD, MF, PPF, NPS, etc.
- A 100-case stress test covering five categories: Food, Travel, EMI, Investment, Shopping.

## Stress-test result

`transaction_classifier_stress_test.py` evaluates 100 unseen-style transaction narrations.

Latest result: **100/100 correctly classified (100% accuracy)**.

Some correct predictions remain below the 60% confidence threshold. Those are intentionally surfaced as exceptions because a finance controller should review uncertain transactions rather than receive a falsely confident answer.

## Important

The Flask app now uses the exact same text normalization used during training. This is important: training and upload-time classification must preprocess transaction text consistently.
