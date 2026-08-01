from core.common import *
from core.models import *
from core.database import Database
from core.logger import record_event
from config.settings import _plain
from memory.semantic import SemanticMemory, deterministic_embedding

class EpisodicMemory:
    def __init__(self, database: Database, config: typing.Any = None):
        if database is None:
            raise ValueError("database must not be None")
        if config is None:
            raise ValueError("config must not be None")
        self.db = database
        self.config = config
        self.semantic = SemanticMemory(database, self.config)

    @staticmethod
    def _event_fields(event: typing.Any) -> typing.Tuple[dict, float, str, str, int, typing.Optional[typing.List[float]]]:
        if event is None:
            raise ValueError("event must not be None")
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
        if isinstance(timestamp, dt.datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
            ts = timestamp.timestamp()
        elif timestamp is None:
            ts = time.time()
        else:
            ts = float(timestamp)
        if not math.isfinite(ts):
            raise ValueError("event timestamp must be finite")
        if task_id is None:
            task_id = uuid.UUID(int=0)
        if event_type is None:
            event_type = EventType.ERROR.value
        type_text = event_type.value if isinstance(event_type, enum.Enum) else str(event_type)
        normalized_payload = dict(payload) if isinstance(payload, dict) else {"value": payload}
        normalized_embedding = None if embedding is None else [float(value) for value in embedding]
        return normalized_payload, ts, str(task_id), type_text, tokens, normalized_embedding

    def append_event(self, event: typing.Any) -> int:
        payload, ts, task_id, type_text, tokens, embedding = self._event_fields(event)
        payload_text = stable_json_dumps(payload)
        vector = deterministic_embedding(payload_text, self.config.embedding_dim) if embedding is None else embedding
        if len(vector) != int(self.config.embedding_dim):
            raise ValueError(
                f"event embedding dimension {len(vector)} does not match configured dimension {self.config.embedding_dim}"
            )
        if any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("event embedding must contain only finite values")
        utility = max(0.0, min(1.0, 0.25 + min(0.75, max(0, tokens) / 10000.0)))
        conn = self.db.connect()
        try:
            with self.db.transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO events(ts,task_id,type,payload_json,tokens_used,emb,utility_score,last_access_ts) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        ts,
                        task_id,
                        type_text,
                        payload_text,
                        tokens,
                        pack_vector(vector),
                        utility,
                        time.time(),
                    ),
                )
                row_id = int(cursor.lastrowid)
        finally:
            conn.close()
        memory_size_bytes.labels(tier="episodic").set(self._size_bytes())
        return row_id

    def _size_bytes(self) -> int:
        conn = self.db.connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(COALESCE(length(payload_json),0)+COALESCE(length(emb),0)),0) AS n FROM events"
            ).fetchone()
            return int(row["n"] or 0)
        finally:
            conn.close()

    def count(self) -> int:
        conn = self.db.connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
            return int(row["n"] or 0)
        finally:
            conn.close()

    def retrieve(self, task_id: typing.Any, query: str, limit: int = 50) -> typing.List[typing.Tuple[float, sqlite3.Row]]:
        normalized_limit = max(0, int(limit))
        if normalized_limit == 0:
            return []
        if query is None:
            raise ValueError("query must not be None")
        normalized_task_id = uuid.UUID(int=0) if task_id is None else task_id
        qvec = deterministic_embedding(str(query), self.config.embedding_dim)
        conn = self.db.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM events WHERE task_id IN (?,?) OR task_id IS NULL ORDER BY ts DESC LIMIT 2000",
                (str(normalized_task_id), str(uuid.UUID(int=0))),
            ).fetchall()
            results = []
            now = time.time()
            half_life = max(3600.0, float(self.config.retention_hours) * 3600.0)
            for row in rows:
                row_timestamp = float(row["ts"]) if row["ts"] is not None else now
                age = max(0.0, now - row_timestamp)
                recency = math.exp(-math.log(2.0) * age / half_life)
                access_boost = math.log1p(max(0, int(row["access_count"] or 0))) / 10.0
                payload_json = row["payload_json"] or ""
                richness = min(1.0, max(0.0, estimate_tokens(payload_json) / 1000.0))
                stored_vector = unpack_vector(row["emb"])
                if len(stored_vector) != len(qvec):
                    continue
                similarity = cosine_similarity(qvec, stored_vector)
                if not math.isfinite(similarity):
                    similarity = 0.0
                score = similarity * 0.65 + recency * 0.2 + access_boost * 0.1 + richness * 0.05
                results.append((score, row))
            results.sort(key=lambda item: (item[0], int(item[1]["id"])), reverse=True)
            return results[:normalized_limit]
        finally:
            conn.close()

    def mark_access(self, event_ids: typing.Iterable[int]) -> None:
        if event_ids is None:
            return
        ids = sorted({int(item) for item in event_ids})
        if not ids:
            return
        accessed_at = time.time()
        conn = self.db.connect()
        try:
            with self.db.transaction(conn):
                conn.executemany(
                    "UPDATE events SET access_count=COALESCE(access_count,0)+1,last_access_ts=? WHERE id=?",
                    [(accessed_at, item_id) for item_id in ids],
                )
        finally:
            conn.close()

    async def summarize_old_events(self, summarizer: typing.Any, max_len: int) -> int:
        if summarizer is None:
            raise ValueError("summarizer must not be None")
        normalized_max_len = int(max_len)
        if normalized_max_len <= 0:
            raise ValueError("max_len must be greater than zero")
        event_count = self.count()
        cutoff = time.time() - float(self.config.retention_hours) * 3600.0
        conn = self.db.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM events WHERE ts<? ORDER BY ts ASC,id ASC LIMIT 200",
                (cutoff,),
            ).fetchall()
            if not rows and event_count > int(self.config.episodic_summarize_threshold):
                compact_count = min(
                    200,
                    max(1, event_count - int(self.config.episodic_summarize_threshold)),
                )
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY ts ASC,id ASC LIMIT ?",
                    (compact_count,),
                ).fetchall()
        finally:
            conn.close()
        if not rows:
            return 0
        texts = [str(row["payload_json"] or "") for row in rows]
        summaries = list(await summarizer.summarize(texts, normalized_max_len))
        if len(summaries) != len(rows):
            raise PermanentError("summarizer must return one summary per event")
        semantic_ids = []
        archived_count = 0
        try:
            for row, summary_text in zip(rows, summaries):
                if summary_text is not None and str(summary_text).strip():
                    semantic_id = self.semantic.add(
                        row["task_id"] or uuid.UUID(int=0),
                        str(summary_text),
                        float(row["utility_score"] or 0.5),
                    )
                    semantic_ids.append(semantic_id)
            conn = self.db.connect()
            try:
                with self.db.transaction(conn):
                    for row in rows:
                        cursor = conn.execute(
                            "INSERT OR IGNORE INTO event_archive SELECT * FROM events WHERE id=?",
                            (row["id"],),
                        )
                        if cursor.rowcount > 0:
                            delete_cursor = conn.execute(
                                "DELETE FROM events WHERE id=?",
                                (row["id"],),
                            )
                            archived_count += max(0, int(delete_cursor.rowcount))
            finally:
                conn.close()
        except Exception:
            if semantic_ids:
                cleanup = self.db.connect()
                try:
                    with self.db.transaction(cleanup):
                        cleanup.executemany(
                            "DELETE FROM semantic WHERE id=?",
                            [(semantic_id,) for semantic_id in semantic_ids],
                        )
                finally:
                    cleanup.close()
                self.semantic._rebuild()
            raise
        memory_size_bytes.labels(tier="episodic").set(self._size_bytes())
        return archived_count

    def admit_with_regret_gate(self, event: typing.Any, expected_utility: float) -> bool:
        utility = float(expected_utility)
        if not math.isfinite(utility):
            return False
        return utility >= float(self.config.admission_utility_threshold)
