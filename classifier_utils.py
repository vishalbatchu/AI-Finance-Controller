import re

def clean_transaction_text(text):
    """Normalize bank narrations while preserving merchant/category clues."""
    t = str(text).lower()
    replacements = {
        'upi-': 'upi ', 'upi/': 'upi ', 'pos ': 'pos ', 'neft dr': 'neft ',
        'imps/': 'imps ', 'nach dr': 'nach ', 'ecs return': 'ecs ', 'txn:': 'txn ',
        'a/c': ' account ', 'autodebit': 'auto debit',
    }
    for a, b in replacements.items():
        t = t.replace(a, b)
    for a, b in [
        (r'\bfd\b', 'fixed deposit'), (r'\bmf\b', 'mutual fund'),
        (r'\bcc\b', 'credit card'), (r'\bppf\b', 'public provident fund'),
        (r'\bnps\b', 'pension investment'), (r'\bbnpl\b', 'buy now pay later'),
        (r'\bemi\b', 'installment'), (r'\bpos\b', 'point of sale'),
    ]:
        t = re.sub(a, b, t)
    t = re.sub(r'ref[:/\-]?\w+', ' ', t)
    t = re.sub(r'(?:amount|amt)\s*[:=]?\s*(?:inr|rs|₹)?\s*[\d,]+(?:\.\d+)?', ' ', t)
    t = re.sub(r'\binr\s*[\d,]+(?:\.\d+)?', ' ', t)
    t = re.sub(r'\brs\.?\s*[\d,]+(?:\.\d+)?', ' ', t)
    t = re.sub(r'\b\d{5,}\b', ' ', t)
    t = re.sub(r'[^a-z0-9\s/]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
