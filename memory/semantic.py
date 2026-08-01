from core.common import *
from core.models import *
from core.database import Database
from config.settings import SEMANTIC_INDEX_PATH

def deterministic_embedding(text: str, dimension: int = 384) -> typing.List[float]:
    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    values = [0.0] * dimension
    tokens = re.findall(r"[\w'-]+", str(text).casefold(), flags=re.UNICODE)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16, person=b"agent-embed-v1").digest()
        slot = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        weight = 1.0 + math.log1p(len(token))
        values[slot] += sign * weight
    if _HAS_NUMPY:
        arr = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        return (arr / norm).tolist() if norm else arr.tolist()
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values

def pack_vector(vector: typing.List[float]) -> bytes:
    if any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("vector contains a non-finite value")
    return struct.pack(f"<{len(vector)}f", *vector)

def unpack_vector(blob: bytes) -> typing.List[float]:
    if not blob:
        return []
    if len(blob) % 4:
        raise SchemaValidationError("vector blob length is not divisible by four")
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))

def cosine_similarity(a: typing.List[float], b: typing.List[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError("vectors must have equal dimensions")
    if _HAS_NUMPY:
        na = np.asarray(a, dtype=np.float32)
        nb = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(na) * np.linalg.norm(nb))
        return 0.0 if denom == 0.0 else float(np.dot(na, nb) / denom)
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if not norm_a or not norm_b else dot / (norm_a * norm_b)

class SemanticMemory:
    def __init__(self, database: Database, config: typing.Any = None):
        self.db = database
        self.config = config
        self._vectors: typing.List[typing.List[float]] = []
        self._ids: typing.List[int] = []
        self._task_ids: typing.List[str] = []
        self._alpha: typing.Dict[int, float] = {}
        self._beta: typing.Dict[int, float] = {}
        self._lock = threading.RLock()
        self._rebuild()

    def _persist_index(self) -> None:
        with self._lock:
            payload = {
                "version": 1,
                "dimension": self.config.embedding_dim,
                "ids": self._ids,
                "task_ids": self._task_ids,
                "vectors": self._vectors,
            }
        atomic_write(SEMANTIC_INDEX_PATH, stable_json_dumps(payload).encode("utf-8"))

    def _rebuild(self) -> None:
        conn = self.db.connect()
        try:
            rows = conn.execute("SELECT id,task_id,text,emb,alpha,beta FROM semantic ORDER BY id").fetchall()
            vectors: typing.List[typing.List[float]] = []
            repairs: typing.List[typing.Tuple[bytes, int]] = []
            for row in rows:
                try:
                    vector = unpack_vector(row["emb"])
                    if len(vector) != self.config.embedding_dim or any(not math.isfinite(float(value)) for value in vector):
                        raise ValueError("stored vector is incompatible with the configured dimension")
                except Exception:
                    vector = deterministic_embedding(str(row["text"]), self.config.embedding_dim)
                    repairs.append((pack_vector(vector), int(row["id"])))
                vectors.append(vector)
            if repairs:
                with self.db.transaction(conn):
                    conn.executemany("UPDATE semantic SET emb=? WHERE id=?", repairs)
        finally:
            conn.close()
        with self._lock:
            self._ids = [int(row["id"]) for row in rows]
            self._task_ids = [str(row["task_id"] or "") for row in rows]
            self._vectors = vectors
            self._alpha = {int(row["id"]): float(row["alpha"] or 1.0) for row in rows}
            self._beta = {int(row["id"]): float(row["beta"] or 1.0) for row in rows}
        self._persist_index()

    def add(self, task_id: typing.Any, text: str, utility_score: float = 0.5) -> int:
        vector = deterministic_embedding(text, self.config.embedding_dim)
        task_text = str(task_id)
        conn = self.db.connect()
        try:
            with self.db.transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO semantic(ts,task_id,text,emb,utility_score,alpha,beta) VALUES(?,?,?,?,?,1.0,1.0)",
                    (time.time(), task_text, str(text), pack_vector(vector), max(0.0, min(1.0, float(utility_score)))),
                )
                item_id = int(cursor.lastrowid)
        finally:
            conn.close()
        with self._lock:
            self._ids.append(item_id)
            self._task_ids.append(task_text)
            self._vectors.append(vector)
            self._alpha[item_id] = 1.0
            self._beta[item_id] = 1.0
        self._persist_index()
        memory_size_bytes.labels(tier="semantic").set(len(self._ids) * self.config.embedding_dim * 4)
        return item_id

    def search(self, query: str, limit: int = 20, task_id: typing.Optional[typing.Any] = None) -> typing.List[int]:
        limit = max(0, int(limit))
        if not limit:
            return []
        vector = deterministic_embedding(query, self.config.embedding_dim)
        task_filter = None if task_id is None else str(task_id)
        with self._lock:
            candidates = [
                (item_id, item_task, item_vector)
                for item_id, item_task, item_vector in zip(self._ids, self._task_ids, self._vectors)
                if task_filter is None or item_task in {task_filter, str(uuid.UUID(int=0)), ""}
            ]
        ranked = sorted(
            ((cosine_similarity(vector, item_vector), item_id) for item_id, _, item_vector in candidates),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        return [item_id for _, item_id in ranked[:limit]]

    def update_thompson(self, item_id: int, reward: float) -> None:
        reward = max(0.0, min(1.0, float(reward)))
        with self._lock:
            if item_id not in self._alpha:
                return
            self._alpha[item_id] += reward
            self._beta[item_id] += 1.0 - reward
            alpha = self._alpha[item_id]
            beta = self._beta[item_id]
        conn = self.db.connect()
        try:
            conn.execute("UPDATE semantic SET alpha=?,beta=? WHERE id=?", (alpha, beta, item_id))
        finally:
            conn.close()

    def thompson_evict(self, evict_count: int = 10, protect_high_utility: float = 0.8) -> int:
        evict_count = max(0, int(evict_count))
        with self._lock:
            ids = list(self._ids)
            alpha = dict(self._alpha)
            beta = dict(self._beta)
        if not evict_count or len(ids) <= evict_count:
            return 0
        conn = self.db.connect()
        try:
            rows = conn.execute(
                f"SELECT id,utility_score FROM semantic WHERE id IN ({','.join('?' for _ in ids)})",
                ids,
            ).fetchall()
            utility = {int(row["id"]): float(row["utility_score"] or 0.5) for row in rows}
            scores = [
                (random.betavariate(max(0.01, alpha.get(item_id, 1.0)), max(0.01, beta.get(item_id, 1.0))), item_id)
                for item_id in ids
                if utility.get(item_id, 0.5) < protect_high_utility
            ]
            scores.sort()
            to_remove = [item_id for _, item_id in scores[:evict_count]]
            with self.db.transaction(conn):
                for item_id in to_remove:
                    conn.execute("DELETE FROM semantic WHERE id=?", (item_id,))
        finally:
            conn.close()
        if to_remove:
            self._rebuild()
        return len(to_remove)

