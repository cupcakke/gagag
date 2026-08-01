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
        array = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        return (array / norm).tolist() if norm else array.tolist()
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values

def pack_vector(vector: typing.List[float]) -> bytes:
    values = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("vector contains a non-finite value")
    try:
        return struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as error:
        raise ValueError("vector contains a value outside the float32 range") from error

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
    if any(not math.isfinite(float(value)) for value in a) or any(not math.isfinite(float(value)) for value in b):
        raise ValueError("vectors must contain only finite values")
    if _HAS_NUMPY:
        array_a = np.asarray(a, dtype=np.float32)
        array_b = np.asarray(b, dtype=np.float32)
        denominator = float(np.linalg.norm(array_a) * np.linalg.norm(array_b))
        return 0.0 if denominator == 0.0 else float(np.dot(array_a, array_b) / denominator)
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if norm_a == 0.0 or norm_b == 0.0 else dot_product / (norm_a * norm_b)

class SemanticMemory:
    def __init__(self, database: Database, config: typing.Any = None):
        self.db = database
        self.config = config
        configured_dimension = getattr(config, "embedding_dim", 384) if config is not None else 384
        self.embedding_dim = int(configured_dimension)
        if self.embedding_dim <= 0:
            raise ValueError("embedding dimension must be positive")
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
                "dimension": self.embedding_dim,
                "ids": list(self._ids),
                "task_ids": list(self._task_ids),
                "vectors": [list(vector) for vector in self._vectors],
            }
            atomic_write(SEMANTIC_INDEX_PATH, stable_json_dumps(payload).encode("utf-8"))

    def _rebuild(self) -> None:
        with self._lock:
            conn = self.db.connect()
            try:
                rows = conn.execute(
                    "SELECT id,task_id,text,emb,alpha,beta FROM semantic ORDER BY id"
                ).fetchall()
                vectors: typing.List[typing.List[float]] = []
                repairs: typing.List[typing.Tuple[bytes, int]] = []
                for row in rows:
                    try:
                        vector = unpack_vector(row["emb"])
                        if len(vector) != self.embedding_dim:
                            raise ValueError("stored vector is incompatible with the configured dimension")
                        if any(not math.isfinite(float(value)) for value in vector):
                            raise ValueError("stored vector contains a non-finite value")
                    except (TypeError, ValueError, SchemaValidationError, struct.error):
                        vector = deterministic_embedding(str(row["text"]), self.embedding_dim)
                        repairs.append((pack_vector(vector), int(row["id"])))
                    vectors.append(vector)
                if repairs:
                    with self.db.transaction(conn):
                        conn.executemany("UPDATE semantic SET emb=? WHERE id=?", repairs)
            finally:
                conn.close()
            self._ids = [int(row["id"]) for row in rows]
            self._task_ids = [str(row["task_id"] or "") for row in rows]
            self._vectors = vectors
            self._alpha = {}
            self._beta = {}
            for row in rows:
                item_id = int(row["id"])
                alpha = float(row["alpha"]) if row["alpha"] is not None else 1.0
                beta = float(row["beta"]) if row["beta"] is not None else 1.0
                self._alpha[item_id] = alpha if math.isfinite(alpha) and alpha > 0.0 else 1.0
                self._beta[item_id] = beta if math.isfinite(beta) and beta > 0.0 else 1.0
            self._persist_index()
            memory_size_bytes.labels(tier="semantic").set(
                len(self._ids) * self.embedding_dim * 4
            )

    def add(self, task_id: typing.Any, text: str, utility_score: float = 0.5) -> int:
        text_value = str(text)
        task_text = str(task_id)
        utility = float(utility_score)
        if not math.isfinite(utility):
            raise ValueError("utility score must be finite")
        utility = max(0.0, min(1.0, utility))
        vector = deterministic_embedding(text_value, self.embedding_dim)
        with self._lock:
            conn = self.db.connect()
            try:
                with self.db.transaction(conn):
                    cursor = conn.execute(
                        "INSERT INTO semantic(ts,task_id,text,emb,utility_score,alpha,beta) VALUES(?,?,?,?,?,1.0,1.0)",
                        (
                            time.time(),
                            task_text,
                            text_value,
                            pack_vector(vector),
                            utility,
                        ),
                    )
                    if cursor.lastrowid is None:
                        raise RuntimeError("database did not return an identifier for the semantic memory")
                    item_id = int(cursor.lastrowid)
            finally:
                conn.close()
            self._ids.append(item_id)
            self._task_ids.append(task_text)
            self._vectors.append(vector)
            self._alpha[item_id] = 1.0
            self._beta[item_id] = 1.0
            self._persist_index()
            memory_size_bytes.labels(tier="semantic").set(
                len(self._ids) * self.embedding_dim * 4
            )
            return item_id

    def search(
        self,
        query: str,
        limit: int = 20,
        task_id: typing.Optional[typing.Any] = None,
    ) -> typing.List[int]:
        limit = max(0, int(limit))
        if limit == 0:
            return []
        vector = deterministic_embedding(str(query), self.embedding_dim)
        task_filter = None if task_id is None else str(task_id)
        global_task_id = str(uuid.UUID(int=0))
        with self._lock:
            candidates = [
                (item_id, item_vector)
                for item_id, item_task, item_vector in zip(
                    self._ids,
                    self._task_ids,
                    self._vectors,
                )
                if task_filter is None
                or item_task == task_filter
                or item_task == global_task_id
                or item_task == ""
            ]
        ranked = sorted(
            (
                (cosine_similarity(vector, item_vector), item_id)
                for item_id, item_vector in candidates
            ),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        return [item_id for _, item_id in ranked[:limit]]

    def update_thompson(self, item_id: int, reward: float) -> None:
        item_id = int(item_id)
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite")
        reward_value = max(0.0, min(1.0, reward_value))
        with self._lock:
            if item_id not in self._alpha:
                return
            alpha = self._alpha[item_id] + reward_value
            beta = self._beta[item_id] + 1.0 - reward_value
            conn = self.db.connect()
            try:
                with self.db.transaction(conn):
                    cursor = conn.execute(
                        "UPDATE semantic SET alpha=?,beta=? WHERE id=?",
                        (alpha, beta, item_id),
                    )
                    if cursor.rowcount == 0:
                        return
            finally:
                conn.close()
            self._alpha[item_id] = alpha
            self._beta[item_id] = beta

    def thompson_evict(
        self,
        evict_count: int = 10,
        protect_high_utility: float = 0.8,
    ) -> int:
        evict_count = max(0, int(evict_count))
        protection_threshold = float(protect_high_utility)
        if not math.isfinite(protection_threshold):
            raise ValueError("utility protection threshold must be finite")
        if evict_count == 0:
            return 0
        with self._lock:
            ids = list(self._ids)
            if not ids:
                return 0
            alpha = dict(self._alpha)
            beta = dict(self._beta)
            placeholders = ",".join("?" for _ in ids)
            conn = self.db.connect()
            to_remove: typing.List[int] = []
            try:
                rows = conn.execute(
                    f"SELECT id,utility_score FROM semantic WHERE id IN ({placeholders})",
                    ids,
                ).fetchall()
                utility: typing.Dict[int, float] = {}
                for row in rows:
                    item_id = int(row["id"])
                    value = float(row["utility_score"]) if row["utility_score"] is not None else 0.5
                    utility[item_id] = value if math.isfinite(value) else 0.5
                scores = [
                    (
                        random.betavariate(
                            max(0.01, alpha.get(item_id, 1.0)),
                            max(0.01, beta.get(item_id, 1.0)),
                        ),
                        item_id,
                    )
                    for item_id in ids
                    if utility.get(item_id, 0.5) < protection_threshold
                ]
                scores.sort(key=lambda item: (item[0], item[1]))
                to_remove = [
                    item_id
                    for _, item_id in scores[:min(evict_count, len(scores))]
                ]
                if to_remove:
                    with self.db.transaction(conn):
                        conn.executemany(
                            "DELETE FROM semantic WHERE id=?",
                            [(item_id,) for item_id in to_remove],
                        )
            finally:
                conn.close()
            if to_remove:
                removed_ids = set(to_remove)
                retained = [
                    (item_id, task_id, vector)
                    for item_id, task_id, vector in zip(
                        self._ids,
                        self._task_ids,
                        self._vectors,
                    )
                    if item_id not in removed_ids
                ]
                self._ids = [item_id for item_id, _, _ in retained]
                self._task_ids = [task_id for _, task_id, _ in retained]
                self._vectors = [vector for _, _, vector in retained]
                for item_id in to_remove:
                    self._alpha.pop(item_id, None)
                    self._beta.pop(item_id, None)
                self._persist_index()
                memory_size_bytes.labels(tier="semantic").set(
                    len(self._ids) * self.embedding_dim * 4
                )
            return len(to_remove)
