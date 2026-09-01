"""Persistent storage layer for AI Finance Controller.

Uses DATABASE_URL when provided (Render PostgreSQL in production) and falls back to
local SQLite for development. CSV files remain useful as import/export artifacts,
but application state is persisted in the database when DATABASE_URL is configured.
"""
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import LargeBinary


def _normalize_url(url: str) -> str:
    if not url:
        return "sqlite:///finance_controller.db"
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


class Store:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        configured = os.getenv("DATABASE_URL", "").strip()
        self.is_configured = bool(configured)
        if configured:
            url = _normalize_url(configured)
        else:
            url = f"sqlite:///{(base_dir / 'finance_controller.db').as_posix()}"
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        self.backend = "PostgreSQL" if configured else "SQLite (local fallback)"
        self.init_schema()

    def init_schema(self):
        ddl = [
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id VARCHAR(255) PRIMARY KEY,
                date VARCHAR(64), counterparty TEXT, amount DOUBLE PRECISION,
                category VARCHAR(64), category_confidence DOUBLE PRECISION,
                status VARCHAR(64), raw_text TEXT, original_category VARCHAR(64),
                classification_source VARCHAR(64), updated_at VARCHAR(64)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS human_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp VARCHAR(64), transaction_id VARCHAR(255), raw_text TEXT,
                original_category VARCHAR(64), model_confidence DOUBLE PRECISION,
                corrected_category VARCHAR(64), source VARCHAR(128)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp VARCHAR(64), event VARCHAR(128), details TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                version VARCHAR(128) PRIMARY KEY, trained_at VARCHAR(64),
                training_rows INTEGER, human_feedback_rows INTEGER,
                latest_feedback_rows INTEGER, learning_mode TEXT,
                model_blob BLOB, vectorizer_blob BLOB, metadata TEXT
            )
            """
        ]
        # PostgreSQL doesn't support SQLite AUTOINCREMENT. Use identity-like BIGINT
        # tables by creating without the sqlite-specific clause when needed.
        if self.is_configured:
            ddl[1] = ddl[1].replace("id INTEGER PRIMARY KEY AUTOINCREMENT", "id BIGSERIAL PRIMARY KEY")
            ddl[2] = ddl[2].replace("id INTEGER PRIMARY KEY AUTOINCREMENT", "id BIGSERIAL PRIMARY KEY")
        with self.engine.begin() as conn:
            for statement in ddl:
                conn.execute(text(statement))

    @staticmethod
    def _df_rows(df: pd.DataFrame):
        out = df.copy()
        required = ["transaction_id", "date", "counterparty", "amount", "category",
                    "category_confidence", "status", "raw_text", "original_category",
                    "classification_source"]
        for col in required:
            if col not in out.columns:
                if col == "date": out[col] = ""
                elif col == "amount" or col == "category_confidence": out[col] = 0.0
                elif col == "category": out[col] = ""
                elif col == "status": out[col] = "Pending"
                elif col == "raw_text": out[col] = out.get("counterparty", "")
                elif col == "original_category": out[col] = out.get("category", "")
                elif col == "classification_source": out[col] = "Model"
                else: out[col] = ""
        out = out[required].copy()
        out["transaction_id"] = out["transaction_id"].astype(str)
        out["date"] = out["date"].astype(str)
        out["counterparty"] = out["counterparty"].fillna("").astype(str)
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
        out["category"] = out["category"].fillna("").astype(str)
        out["category_confidence"] = pd.to_numeric(out["category_confidence"], errors="coerce").fillna(0.0)
        out["status"] = out["status"].fillna("Pending").astype(str)
        out["raw_text"] = out["raw_text"].fillna("").astype(str)
        out["original_category"] = out["original_category"].fillna("").astype(str)
        out["classification_source"] = out["classification_source"].fillna("Model").astype(str)
        return out

    def transaction_count(self):
        with self.engine.connect() as conn:
            return int(conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar() or 0)

    def replace_transactions(self, df: pd.DataFrame):
        rows = self._df_rows(df)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM transactions"))
            if not rows.empty:
                stmt = text("""
                    INSERT INTO transactions
                    (transaction_id,date,counterparty,amount,category,category_confidence,status,raw_text,original_category,classification_source,updated_at)
                    VALUES (:transaction_id,:date,:counterparty,:amount,:category,:category_confidence,:status,:raw_text,:original_category,:classification_source,:updated_at)
                """)
                for row in rows.to_dict(orient="records"):
                    row["updated_at"] = now
                    conn.execute(stmt, row)

    def load_transactions(self) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text("SELECT transaction_id,date,counterparty,amount,category,category_confidence,status,raw_text,original_category,classification_source FROM transactions ORDER BY transaction_id"), conn)

    def update_transaction(self, transaction_id, **fields):
        allowed = {"category", "category_confidence", "classification_source", "original_category", "status", "raw_text", "counterparty", "amount", "date"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        assignments = ", ".join(f"{k} = :{k}" for k in fields)
        fields["transaction_id"] = str(transaction_id)
        with self.engine.begin() as conn:
            conn.execute(text(f"UPDATE transactions SET {assignments} WHERE transaction_id = :transaction_id"), fields)

    def feedback_df(self):
        with self.engine.connect() as conn:
            return pd.read_sql(text("SELECT timestamp,transaction_id,raw_text,original_category,model_confidence,corrected_category,source FROM human_feedback ORDER BY id"), conn)

    def append_feedback(self, row: dict):
        stmt = text("""
            INSERT INTO human_feedback(timestamp,transaction_id,raw_text,original_category,model_confidence,corrected_category,source)
            VALUES (:timestamp,:transaction_id,:raw_text,:original_category,:model_confidence,:corrected_category,:source)
        """)
        with self.engine.begin() as conn:
            conn.execute(stmt, row)

    def audit(self, record: dict):
        details = {k: v for k, v in record.items() if k not in {"timestamp", "event"}}
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO audit_events(timestamp,event,details) VALUES (:timestamp,:event,:details)"), {
                "timestamp": record.get("timestamp"), "event": record.get("event"),
                "details": json.dumps(details, ensure_ascii=False, default=str)
            })

    def read_audit(self, limit=40):
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT timestamp,event,details FROM audit_events ORDER BY id DESC LIMIT :limit"), {"limit": int(limit)}).mappings().all()
        out = []
        for r in rows:
            item = {"timestamp": r["timestamp"], "event": r["event"]}
            try: item.update(json.loads(r["details"] or "{}"))
            except Exception: pass
            out.append(item)
        return out

    def model_latest(self):
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT version,trained_at,training_rows,human_feedback_rows,latest_feedback_rows,learning_mode,model_blob,vectorizer_blob,metadata FROM model_versions ORDER BY trained_at DESC LIMIT 1")).mappings().first()
            return dict(row) if row else None

    def save_model(self, version, trained_at, metadata, model_blob, vectorizer_blob):
        # SQLite and PostgreSQL both accept the parameterized insert below.
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO model_versions(version,trained_at,training_rows,human_feedback_rows,latest_feedback_rows,learning_mode,model_blob,vectorizer_blob,metadata)
                VALUES (:version,:trained_at,:training_rows,:human_feedback_rows,:latest_feedback_rows,:learning_mode,:model_blob,:vectorizer_blob,:metadata)
            """), {
                "version": version, "trained_at": trained_at,
                "training_rows": int(metadata.get("training_rows", 0)),
                "human_feedback_rows": int(metadata.get("human_feedback_rows", 0)),
                "latest_feedback_rows": int(metadata.get("latest_feedback_rows", 0)),
                "learning_mode": metadata.get("learning_mode", ""),
                "model_blob": model_blob, "vectorizer_blob": vectorizer_blob,
                "metadata": json.dumps(metadata, ensure_ascii=False)
            })

    def storage_info(self):
        return {"backend": self.backend, "persistent": True, "database_configured": self.is_configured,
                "transactions": self.transaction_count(),
                "human_feedback": int(self.feedback_df().shape[0])}
