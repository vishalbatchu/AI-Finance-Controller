import re
import calendar
import difflib
from datetime import datetime

import pandas as pd


class SettlementQA:
    """Data-grounded natural-language Q&A for the finance settlement batch.

    The engine is intentionally deterministic: every numeric answer is computed
    from the current DataFrame rather than hallucinated. It supports synonyms,
    combined filters, date/amount ranges, ranking, percentages and exception
    analysis.
    """

    CATEGORY_ALIASES = {
        "Food": [
            "food", "dining", "restaurant", "restaurants", "meal", "meals",
            "eat", "eating", "cafe", "café", "coffee", "pizza", "swiggy",
            "zomato", "dominos", "mcdonald", "groceries", "grocery",
        ],
        "Travel": [
            "travel", "trip", "transport", "transportation", "flight", "flights",
            "train", "trains", "railway", "bus", "hotel", "hotels", "cab",
            "cabs", "taxi", "ride", "rides", "uber", "ola", "irctc", "indigo",
            "makemytrip", "redbus", "petrol", "fuel",
        ],
        "EMI": [
            "emi", "installment", "instalment", "loan", "loans", "repayment",
            "repayments", "mortgage", "financing", "finance payment",
        ],
        "Investment": [
            "investment", "investments", "invest", "investing", "invested", "sipped", "sip", "mutual fund",
            "mutual funds", "mf", "etf", "stock", "stocks",
            "equity", "fixed deposit", "fd", "deposit", "ppf", "zerodha", "groww",
        ],
        "Shopping": [
            "shopping", "shop", "purchase", "purchases", "bought", "buy", "retail",
            "ecommerce", "e-commerce", "online order", "electronics", "clothing",
            "clothes", "furniture", "amazon", "flipkart", "ikea", "myntra",
            "reliance digital",
        ],
    }

    STATUS_ALIASES = {
        "Settled": ["settled", "settle", "successful", "success", "completed", "complete", "cleared", "processed"],
        "Pending": ["pending", "awaiting", "waiting", "processing", "in progress", "not settled", "unsettled"],
        "Failed": ["failed", "failure", "declined", "rejected", "unsuccessful", "error", "errored", "did not go through"],
    }

    def __init__(self, data, confidence_threshold=0.60):
        self.df = data.copy()
        self.confidence_threshold = confidence_threshold
        self.df["date"] = pd.to_datetime(self.df.get("date"), errors="coerce")
        self.df["amount"] = pd.to_numeric(self.df.get("amount"), errors="coerce").fillna(0.0)
        self.df["category_confidence"] = pd.to_numeric(
            self.df.get("category_confidence"), errors="coerce"
        ).fillna(0.0)
        self.categories = [str(x) for x in self.df["category"].dropna().unique()]
        self.statuses = [str(x) for x in self.df["status"].dropna().unique()]
        self.counterparties = [str(x) for x in self.df["counterparty"].dropna().unique()]

    @staticmethod
    def _normalize(text):
        text = str(text).lower().replace("₹", " inr ")
        text = text.replace("rs.", " inr ").replace("rs", " inr ")
        text = re.sub(r"[’'`\"]", "", text)
        text = re.sub(r"[^a-z0-9:/.,%\-\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _money(value):
        return f"INR {float(value):,.2f}"

    @staticmethod
    def _percent(value):
        return f"{float(value) * 100:.1f}%"

    def _find_alias(self, q, aliases):
        # Longest aliases first so "mutual fund" wins over "fund"-like terms.
        hits = []
        for canonical, words in aliases.items():
            for word in sorted(words, key=len, reverse=True):
                if re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", q):
                    hits.append((len(word), canonical))
                    break
        return max(hits)[1] if hits else None

    def _extract_category(self, q):
        direct = self._find_alias(q, self.CATEGORY_ALIASES)
        if direct:
            return direct
        for cat in self.categories:
            if re.search(rf"\b{re.escape(cat.lower())}\b", q):
                return cat
        return None

    def _extract_status(self, q):
        direct = self._find_alias(q, self.STATUS_ALIASES)
        if direct:
            return direct
        for status in self.statuses:
            if re.search(rf"\b{re.escape(status.lower())}\b", q):
                return status
        return None

    def _parse_number(self, token):
        try:
            return float(str(token).replace(",", ""))
        except ValueError:
            return None

    def _extract_amount_filter(self, q):
        # between INR 5,000 and 20,000 / between 5000 and 20000
        m = re.search(r"between\s+(?:inr\s*)?([\d,]+(?:\.\d+)?)\s+(?:and|to)\s+(?:inr\s*)?([\d,]+(?:\.\d+)?)", q)
        if m:
            a, b = self._parse_number(m.group(1)), self._parse_number(m.group(2))
            if a is not None and b is not None:
                return ("between", min(a, b), max(a, b))

        patterns = [
            ("gte", r"(?:above|over|more than|greater than|at least|minimum of)\s+(?:inr\s*)?([\d,]+(?:\.\d+)?)"),
            ("lte", r"(?:below|under|less than|smaller than|at most|maximum of)\s+(?:inr\s*)?([\d,]+(?:\.\d+)?)"),
            ("eq", r"(?:exactly|equal to|equals?)\s+(?:inr\s*)?([\d,]+(?:\.\d+)?)"),
        ]
        for kind, pattern in patterns:
            m = re.search(pattern, q)
            if m:
                value = self._parse_number(m.group(1))
                if value is not None:
                    return (kind, value)
        return None

    def _extract_dates(self, q):
        months = {name: i for i, name in enumerate(calendar.month_name) if name}
        months.update({name: i for i, name in enumerate(calendar.month_abbr) if name})

        exact_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
        if len(exact_dates) >= 2:
            return ("between", pd.Timestamp(exact_dates[0]), pd.Timestamp(exact_dates[1]))
        if exact_dates:
            if re.search(r"(?:after|since|on or after)\s+" + re.escape(exact_dates[0]), q):
                return ("after", pd.Timestamp(exact_dates[0]))
            if re.search(r"(?:before|until|up to|on or before)\s+" + re.escape(exact_dates[0]), q):
                return ("before", pd.Timestamp(exact_dates[0]))
            return ("exact", pd.Timestamp(exact_dates[0]))

        # "from July 1 to July 31" / "between July 1 and July 31"
        m = re.search(
            r"(?:from|between)\s+([a-z]+)\s+(\d{1,2})\s+(?:to|and)\s+([a-z]+)\s+(\d{1,2})",
            q,
        )
        if m:
            try:
                year = int(self.df["date"].dt.year.max())
                a = pd.Timestamp(year=year, month=months[m.group(1).title()], day=int(m.group(2)))
                b = pd.Timestamp(year=year, month=months[m.group(3).title()], day=int(m.group(4)))
                return ("between", min(a, b), max(a, b))
            except Exception:
                pass

        # "July 2026" or simply "in July"
        for name, month_no in months.items():
            if re.search(rf"\b{re.escape(name.lower())}\b", q):
                year_match = re.search(rf"{re.escape(name.lower())}\s+(20\d{{2}})", q)
                year = int(year_match.group(1)) if year_match else int(self.df["date"].dt.year.max())
                return ("month", year, month_no)

        if "this month" in q or "current month" in q:
            latest = self.df["date"].max()
            return ("month", latest.year, latest.month)
        if "last month" in q or "previous month" in q:
            latest = self.df["date"].max()
            prev = latest - pd.DateOffset(months=1)
            return ("month", prev.year, prev.month)

        return None

    def _extract_counterparty(self, q):
        # Only treat a merchant/counterparty as a filter when the question
        # explicitly points at a merchant, or contains a distinctive brand
        # name. This prevents generic words such as "payment" or "investment"
        # from accidentally becoming merchant filters.
        stopwords = {
            "order", "payment", "purchase", "booking", "ticket", "fare",
            "investment", "investments", "fixed", "deposit", "mutual",
            "fund", "meal", "dining", "restaurant", "transaction",
            "transactions", "amount", "money", "total", "spend", "spent",
            "shopping", "online", "clothing", "electronics", "furniture",
            "loan", "home", "car", "personal", "finance", "repayment",
            "emi", "ride", "travel", "flight", "train", "bus", "hotel",
            "pizza", "coffee", "dinner", "grocery", "groceries", "fuel", "paid", "amount",
        }
        merchant_hint = any(x in q for x in [
            "at ", "from ", "with ", "merchant ", "vendor ", "counterparty ",
            "paid to ", "pay to ", "purchase at ", "bought from ",
        ])

        for name in sorted(self.counterparties, key=len, reverse=True):
            compact = self._normalize(name)
            words = [w for w in re.findall(r"[a-z0-9]+", compact) if len(w) >= 4 and w not in stopwords]
            if not words:
                continue
            distinctive_hits = [w for w in words if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", q)]
            if merchant_hint and distinctive_hits:
                return name
            # Brand/entity-only questions such as "How much did IKEA cost?"
            # should also work without an explicit "at/from" phrase.
            if any(w in {"ikea", "amazon", "flipkart", "myntra", "dominos", "swiggy", "zomato", "groww", "zerodha", "indigo", "irctc", "ola", "uber", "sbi", "hdfc", "bajaj", "big", "bazaar", "makemytrip", "redbus", "reliance"} for w in distinctive_hits):
                return name
        return None

    def _extract_top_n(self, q, default=10):
        m = re.search(r"\b(?:top|bottom|first|last|largest|biggest|highest|smallest|lowest|cheapest)\s+(\d+)\b", q)
        if not m:
            m = re.search(r"\b(\d+)\s+(?:top|bottom|largest|biggest|highest|smallest|lowest|cheapest)\b", q)
        return int(m.group(1)) if m else default

    @staticmethod
    def _contains_any(q, phrases):
        return any(
            re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", q)
            for phrase in phrases
        )

    def _apply_filters(self, q, include_status=True):
        filtered = self.df.copy()
        filters = []

        category = self._extract_category(q)
        status = self._extract_status(q)
        date_info = self._extract_dates(q)
        amount_info = self._extract_amount_filter(q)
        counterparty = self._extract_counterparty(q)

        if category:
            filtered = filtered[filtered["category"].astype(str).str.lower() == category.lower()]
            filters.append(category)
        if status and include_status:
            filtered = filtered[filtered["status"].astype(str).str.lower() == status.lower()]
            filters.append(status)
        if date_info:
            kind = date_info[0]
            if kind == "exact":
                filtered = filtered[filtered["date"].dt.normalize() == date_info[1].normalize()]
                filters.append(date_info[1].strftime("%Y-%m-%d"))
            elif kind == "after":
                filtered = filtered[filtered["date"] >= date_info[1]]
                filters.append(f"date >= {date_info[1].date()}")
            elif kind == "before":
                filtered = filtered[filtered["date"] <= date_info[1]]
                filters.append(f"date <= {date_info[1].date()}")
            elif kind == "between":
                filtered = filtered[(filtered["date"] >= date_info[1]) & (filtered["date"] <= date_info[2])]
                filters.append(f"{date_info[1].date()} to {date_info[2].date()}")
            elif kind == "month":
                filtered = filtered[(filtered["date"].dt.year == date_info[1]) & (filtered["date"].dt.month == date_info[2])]
                filters.append(pd.Timestamp(year=date_info[1], month=date_info[2], day=1).strftime("%B %Y"))
        if amount_info:
            kind = amount_info[0]
            if kind == "between":
                filtered = filtered[(filtered["amount"] >= amount_info[1]) & (filtered["amount"] <= amount_info[2])]
                filters.append(f"amount {self._money(amount_info[1])}–{self._money(amount_info[2])}")
            elif kind == "gte":
                filtered = filtered[filtered["amount"] >= amount_info[1]]
                filters.append(f"amount >= {self._money(amount_info[1])}")
            elif kind == "lte":
                filtered = filtered[filtered["amount"] <= amount_info[1]]
                filters.append(f"amount <= {self._money(amount_info[1])}")
            elif kind == "eq":
                filtered = filtered[filtered["amount"].round(2) == round(amount_info[1], 2)]
                filters.append(f"amount = {self._money(amount_info[1])}")
        if counterparty:
            # Match the meaningful merchant/entity token across the batch so
            # variants such as "Dominos pizza order" and "upi/dominos..."
            # are treated as the same merchant when the user asks for Dominos.
            stopwords = {
                "order", "payment", "purchase", "booking", "ticket", "fare",
                "investment", "investments", "fixed", "deposit", "mutual", "fund",
                "meal", "dining", "restaurant", "transaction", "transactions",
                "amount", "money", "total", "spend", "spent", "shopping", "online",
                "clothing", "electronics", "furniture", "loan", "home", "car",
                "personal", "finance", "repayment", "emi", "ride", "travel",
                "flight", "train", "bus", "hotel", "pizza", "coffee", "dinner",
                "grocery", "groceries", "fuel", "paid",
            }
            words = [w for w in re.findall(r"[a-z0-9]+", self._normalize(counterparty)) if len(w) >= 4 and w not in stopwords]
            key = words[0] if words else self._normalize(counterparty)
            filtered = filtered[filtered["counterparty"].astype(str).str.lower().str.contains(re.escape(key), regex=True, na=False)]
            filters.append(counterparty if key == self._normalize(counterparty) else key)

        return filtered, filters

    def _format_rows(self, rows):
        parts = []
        for _, r in rows.iterrows():
            date = pd.to_datetime(r["date"]).strftime("%Y-%m-%d") if pd.notna(r["date"]) else "unknown date"
            parts.append(
                f"{r['transaction_id']} | {date} | {r['counterparty']} | "
                f"{self._money(r['amount'])} | {r['category']} | {r['status']}"
            )
        return "\n".join(parts)

    def _filter_description(self, filters):
        return ", ".join(filters) if filters else "all transactions"

    def answer(self, question):
        q = self._normalize(question)
        if not q:
            return "Please enter a question about the settlement batch."

        # Common help / meta questions.
        if q in {"hi", "hello", "hey", "help", "what can you do", "what can you answer"}:
            return (
                "I can analyze this settlement batch. Try questions about totals, counts, "
                "average/median, highest/lowest, category or status breakdowns, date ranges, "
                "amount ranges, exceptions, match rate, or a transaction ID."
            )

        # Transaction ID lookup should have highest priority.
        txn_match = re.search(r"\btxn\s*\d+\b", q)
        if txn_match:
            txn_id = re.sub(r"\s+", "", txn_match.group(0)).upper()
            row = self.df[self.df["transaction_id"].astype(str).str.upper() == txn_id]
            if row.empty:
                return f"No transaction found with ID {txn_id}."
            r = row.iloc[0]
            return (
                f"{r['transaction_id']}: {r['counterparty']} | {self._money(r['amount'])} | "
                f"{pd.to_datetime(r['date']).strftime('%Y-%m-%d')} | {r['category']} | {r['status']} | "
                f"confidence {self._percent(r['category_confidence'])}"
            )

        filtered, filters = self._apply_filters(q)
        desc = self._filter_description(filters)

        # Exception / model quality questions.
        if any(x in q for x in ["exception", "exceptions", "low confidence", "low-confidence", "needs review", "need review", "flagged"]):
            ex = self.df[self.df["category_confidence"] < self.confidence_threshold]
            if filters:
                # Re-apply user filters to exceptions while retaining the exception threshold.
                ex, _ = self._apply_filters(q)
                ex = ex[ex["category_confidence"] < self.confidence_threshold]
            return (
                f"There are {len(ex)} exception(s) below the {self._percent(self.confidence_threshold)} confidence threshold, "
                f"with total value {self._money(ex['amount'].sum())}."
            )

        if any(x in q for x in ["match rate", "confidence rate", "classification accuracy"]):
            rate = (self.df["category_confidence"] >= self.confidence_threshold).mean() if len(self.df) else 0
            return f"Current model match rate at the {self._percent(self.confidence_threshold)} threshold: {self._percent(rate)} ({int((self.df['category_confidence'] >= self.confidence_threshold).sum())}/{len(self.df)} transactions)."

        # Category/status breakdowns.
        if any(x in q for x in ["breakdown", "distribution", "split", "by category", "per category"]) and "status" not in q:
            grouped = filtered.groupby("category").agg(count=("amount", "size"), amount=("amount", "sum")).sort_values("amount", ascending=False)
            if grouped.empty:
                return f"No matching transactions for {desc}."
            lines = [f"{idx}: {int(row['count'])} transaction(s), {self._money(row['amount'])}" for idx, row in grouped.iterrows()]
            return "Category breakdown:\n" + "\n".join(lines)

        if "status" in q and any(x in q for x in ["breakdown", "distribution", "split", "by status", "each status"]):
            grouped = filtered.groupby("status").agg(count=("amount", "size"), amount=("amount", "sum")).sort_values("count", ascending=False)
            if grouped.empty:
                return f"No matching transactions for {desc}."
            lines = [f"{idx}: {int(row['count'])} transaction(s), {self._money(row['amount'])}" for idx, row in grouped.iterrows()]
            return "Status breakdown:\n" + "\n".join(lines)

        # Percentage / rate questions.
        if any(x in q for x in ["percentage", "percent", "%", "share", "proportion", "rate", "ratio"]):
            # A rate question such as "what percentage is pending?" means
            # pending / all relevant transactions, not pending / pending.
            rate_base, rate_filters = self._apply_filters(q, include_status=False)
            rate_desc = self._filter_description(rate_filters)
            if any(x in q for x in ["settled", "successful", "success", "completed", "completion"]):
                numerator = int((rate_base["status"].str.lower() == "settled").sum())
                denominator = len(rate_base)
                return f"Settled rate: {self._percent(numerator / denominator if denominator else 0)} ({numerator}/{denominator}) for {rate_desc}."
            if any(x in q for x in ["pending", "awaiting", "processing"]):
                numerator = int((rate_base["status"].str.lower() == "pending").sum())
                denominator = len(rate_base)
                return f"Pending rate: {self._percent(numerator / denominator if denominator else 0)} ({numerator}/{denominator}) for {rate_desc}."
            if any(x in q for x in ["failed", "failure", "declined", "rejected"]):
                numerator = int((rate_base["status"].str.lower() == "failed").sum())
                denominator = len(rate_base)
                return f"Failed rate: {self._percent(numerator / denominator if denominator else 0)} ({numerator}/{denominator}) for {rate_desc}."
            if self._extract_category(q):
                numerator = len(filtered)
                denominator = len(self.df)
                return f"{self._extract_category(q)} represents {self._percent(numerator / denominator if denominator else 0)} of all transactions ({numerator}/{denominator})."

        # Median / average.
        if any(x in q for x in ["median", "middle amount"]):
            if filtered.empty:
                return f"No matching transactions for {desc}."
            return f"Median amount: {self._money(filtered['amount'].median())} across {len(filtered)} transaction(s)."

        if any(x in q for x in ["average", "avg", "mean"]):
            if filtered.empty:
                return f"No matching transactions for {desc}."
            return f"Average amount: {self._money(filtered['amount'].mean())} across {len(filtered)} transaction(s)."

        # Ranking / extremes.
        top_words = ["highest", "largest", "biggest", "maximum", "max", "most expensive", "top"]
        low_words = ["lowest", "smallest", "minimum", "min", "least expensive", "cheapest", "bottom"]
        if self._contains_any(q, top_words):
            if filtered.empty:
                return f"No matching transactions for {desc}."
            n = self._extract_top_n(q, 1)
            rows = filtered.nlargest(n, "amount")
            return (f"Top {len(rows)} by amount for {desc}:\n" + self._format_rows(rows)) if n > 1 else f"Highest: {self._format_rows(rows)}"
        if self._contains_any(q, low_words):
            if filtered.empty:
                return f"No matching transactions for {desc}."
            n = self._extract_top_n(q, 1)
            rows = filtered.nsmallest(n, "amount")
            return (f"Bottom {len(rows)} by amount for {desc}:\n" + self._format_rows(rows)) if n > 1 else f"Lowest: {self._format_rows(rows)}"

        # Totals: broad finance wording.
        if self._contains_any(q, ["total", "sum", "how much", "spend", "spent", "expenditure", "value", "amount spent", "amount paid", "money", "what did we pay", "what did we spend", "cost"]):
            return f"Total amount: {self._money(filtered['amount'].sum())} across {len(filtered)} transaction(s) for {desc}."

        # Counts.
        if any(x in q for x in ["how many", "count", "number of", "number", "transactions are there"]):
            return f"Count: {len(filtered)} transaction(s) for {desc}."

        # Listing / show / details.
        if any(x in q for x in ["list", "show", "which", "give me", "display", "find", "what are", "what transactions"]) or (("transaction" in q or "transactions" in q or "payment" in q or "payments" in q or "record" in q or "records" in q) and not self._contains_any(q, ["how many", "count", "number of", "total", "sum", "average", "median", "percentage", "percent", "rate", "share", "proportion", "match rate"])):
            if filtered.empty:
                return f"No matching transactions for {desc}."
            n = self._extract_top_n(q, 15)
            rows = filtered.sort_values("date", ascending=False).head(n)
            return f"Found {len(filtered)} matching transaction(s). Showing {len(rows)}:\n{self._format_rows(rows)}"

        return (
            "I couldn't confidently determine the request. I can answer questions such as: "
            "'How much did we spend on dining?', 'show failed payments', 'what percentage is pending?', "
            "'top 5 travel expenses', 'transactions above INR 20,000', 'July totals', "
            "'category breakdown', 'how many need review?', or 'details for TXN100003'."
        )
