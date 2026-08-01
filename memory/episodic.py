from core.common import *
from core.models import *
from core.database import Database
from core.logger import record_event
from config.settings import _plain
from memory.semantic import SemanticMemory, deterministic_embedding

class EpisodicMemory:
    def __init__(self, database: Database, config: typing.Any = None):
        self.db = database
        self.config = config
        self.semantic = SemanticMemory(database, self.config)

    @staticmethod
    def _event_fields(event: typing.Any) -> typing.Tuple[dict, float, str, str, int, typing.Optional[typing.List[float]]]:
        if isinstance(event, dict):
            payload = event.get("payload", {})
            timestamp = event.get("timestamp")
            task_id = event.get("task_id", uuid.UUID(int=0))
            event_type = event.get("type", EventType.ERROR.value)
            tokens = int(event.get("tokens_used", 0) or 0)
            embedding = event.get("embedding")
        else:
            payload = getattr(event, "payload", {})
            timestamp = getattr(event, "timestamp", None)
            task_id = getattr(event, "task_id", uuid.UUID(int=0))
            event_type = getattr(event, "type", EventType.ERROR.value)
            tokens = int(getattr(event, "tokens_used", 0) or 0)
            embedding = getattr(event, "embedding", None)
        if isinstance(timestamp, str):
            timestamp = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        ts = timestamp.timestamp() if isinstance(timestamp, dt.datetime) else time.time()
        type_text = event_type.value if isinstance(event_type, enum.Enum) else str(event_type)
        return dict(payload) if isinstance(payload, dict) else {"value": payload}, ts, str(task_id), type_text, tokens, embedding

    def append_event(self, event: typing.Any) -> int:
        payload, ts, task_id, type_text, tokens, embedding = self._event_fields(event)
        payload_text = stable_json_dumps(payload)
        vector = embedding or deterministic_embedding(payload_text, self.config.embedding_dim)
        utility = max(0.0, min(1.0, 0.25 + min(0.75, max(0, tokens) / 10000.0)))
        conn = self.db.connect()
        try:
            with self.db.transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO events(ts,task_id,type,payload_json,tokens_used,emb,utility_score,last_access_ts) VALUES(?,?,?,?,?,?,?,?)",
                    (ts, task_id, type_text, payload_text, tokens, pack_vector(vector), utility, time.time()),
                )
                row_id = int(cursor.lastrowid)
        finally:
            conn.close()
        memory_size_bytes.labels(tier="episodic").set(self._size_bytes())
        return row_id

    def _size_bytes(self) -> int:
        conn = self.db.connect()
        try:
            row = conn.execute("SELECT COALESCE(SUM(length(payload_json)+length(emb)),0) AS n FROM events").fetchone()
            return int(row["n"] or 0)
        finally:
            conn.close()

    def count(self) -> int:
        conn = self.db.connect()
        try:
            return int(conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"] or 0)
        finally:
            conn.close()

    def retrieve(self, task_id: typing.Any, query: str, limit: int = 50) -> typing.List[typing.Tuple[float, sqlite3.Row]]:
        qvec = deterministic_embedding(query, self.config.embedding_dim)
        conn = self.db.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM events WHERE task_id IN (?,?) OR task_id IS NULL ORDER BY ts DESC LIMIT 2000",
                (str(task_id), str(uuid.UUID(int=0))),
            ).fetchall()
            results = []
            now = time.time()
            half_life = max(3600.0, self.config.retention_hours * 3600.0)
            for row in rows:
                age = max(0.0, now - float(row["ts"] or now))
                recency = math.exp(-math.log(2) * age / half_life)
                access_boost = math.log1p(int(row["access_count"] or 0)) / 10.0
                richness = min(1.0, estimate_tokens(row["payload_json"]) / 1000.0)
                similarity = cosine_similarity(qvec, unpack_vector(row["emb"]))
                score = similarity * 0.65 + recency * 0.2 + access_boost * 0.1 + richness * 0.05
                results.append((score, row))
            results.sort(key=lambda item: (item[0], int(item[1]["id"])), reverse=True)
            return results[:max(0, int(limit))]
        finally:
            conn.close()

    def mark_access(self, event_ids: typing.Iterable[int]) -> None:
        ids = sorted({int(item) for item in event_ids})
        if not ids:
            return
        accessed_at = time.time()
        conn = self.db.connect()
        try:
            with self.db.transaction(conn):
                conn.executemany(
                    "UPDATE events SET access_count=access_count+1,last_access_ts=? WHERE id=?",
                    [(accessed_at, item_id) for item_id in ids],
                )
        finally:
            conn.close()

    async def summarize_old_events(self, summarizer: typing.Any, max_len: int) -> int:
        event_count = self.count()
        cutoff = time.time() - self.config.retention_hours * 3600.0
        conn = self.db.connect()
        try:
            rows = conn.execute("SELECT * FROM events WHERE ts<? ORDER BY ts ASC LIMIT 200", (cutoff,)).fetchall()
            if not rows and event_count > self.config.episodic_summarize_threshold:
                compact_count = min(200, max(1, event_count - self.config.episodic_summarize_threshold))
                rows = conn.execute("SELECT * FROM events ORDER BY ts ASC LIMIT ?", (compact_count,)).fetchall()
        finally:
            conn.close()
        if not rows:
            return 0
        texts = [str(row["payload_json"]) for row in rows]
        summaries = await summarizer.summarize(texts, max_len)
        if len(summaries) != len(rows):
            raise PermanentError("summarizer must return one summary per event")
        semantic_ids = []
        try:
            for row, summary_text in zip(rows, summaries):
                if summary_text:
                    semantic_ids.append(self.semantic.add(row["task_id"] or uuid.UUID(int=0), summary_text, float(row["utility_score"] or 0.5)))
            conn = self.db.connect()
            try:
                with self.db.transaction(conn):
                    for row in rows:
                        conn.execute("INSERT OR IGNORE INTO event_archive SELECT * FROM events WHERE id=?", (row["id"],))
                        conn.execute("DELETE FROM events WHERE id=?", (row["id"],))
            finally:
                conn.close()
        except Exception:
            if semantic_ids:
                cleanup = self.db.connect()
                try:
                    with self.db.transaction(cleanup):
                        for semantic_id in semantic_ids:
                            cleanup.execute("DELETE FROM semantic WHERE id=?", (semantic_id,))
                finally:
                    cleanup.close()
                self.semantic._rebuild()
            raise
        memory_size_bytes.labels(tier="episodic").set(self._size_bytes())
        return len(rows)

    def admit_with_regret_gate(self, event: typing.Any, expected_utility: float) -> bool:
        return float(expected_utility) >= self.config.admission_utility_threshold

