from core.common import *
from core.models import *
from core.database import Database
from memory.working import WorkingMemory
from memory.semantic import SemanticMemory
from memory.episodic import EpisodicMemory
from llm.clients import SummarizerClient

class ProceduralMemory:
    def __init__(self, database: Database):
        self.db = database

    def register(self, name: str, definition: typing.Any, version: str = "1") -> None:
        if not name or len(name) > 256:
            raise ValueError("invalid skill name")
        conn = self.db.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO skills(name,version,definition_json,updated_at) VALUES(?,?,?,?)",
                (name, version, stable_json_dumps(definition), time.time()),
            )
        finally:
            conn.close()

    def invoke_by_name(self, name: str) -> typing.Optional[dict]:
        conn = self.db.connect()
        try:
            row = conn.execute("SELECT definition_json FROM skills WHERE name=?", (name,)).fetchone()
            return json.loads(row["definition_json"]) if row else None
        finally:
            conn.close()

    def capture_from_reflection(self, reflection: typing.Any) -> typing.Optional[str]:
        if not isinstance(reflection, dict) or not reflection.get("skill"):
            return None
        skill_name = str(reflection["skill"])
        self.register(skill_name, reflection, str(reflection.get("version", "1")))
        return skill_name

class ContextBlock:
    text: str
    selected_ids: typing.List[int]
    selection_log: typing.List[typing.Dict[str, typing.Any]]
    tokens_used: int

class MemoryManager:
    def __init__(self, database: Database, config: typing.Any = None, summarizer: typing.Any = None):
        self.config = config
        self.working = WorkingMemory(self.config.working_tokens)
        self._working_by_task: typing.Dict[str, WorkingMemory] = {}
        self._working_lock = threading.RLock()
        self.episodic = EpisodicMemory(database, self.config)
        self.semantic = self.episodic.semantic
        self.procedural = ProceduralMemory(database)
        self.summarizer = summarizer

    def working_for(self, task_id: typing.Any) -> WorkingMemory:
        key = str(task_id)
        with self._working_lock:
            memory = self._working_by_task.get(key)
            if memory is None:
                memory = WorkingMemory(self.config.working_tokens)
                self._working_by_task[key] = memory
            return memory

    def append(self, event: typing.Any) -> typing.Optional[int]:
        payload, _, task_id, type_text, tokens, _ = self.episodic._event_fields(event)
        expected_utility = max(0.0, min(1.0, 0.25 + min(0.75, max(0, tokens) / 10000.0)))
        always_admit = type_text in {
            EventType.USER_INPUT.value,
            EventType.MODEL_OUTPUT.value,
            EventType.TOOL_RESULT.value,
            EventType.LIVELOCK_DETECTED.value,
            EventType.TIMEOUT.value,
            EventType.ERROR.value,
            EventType.CHECKPOINT.value,
            EventType.RESTART.value,
            EventType.SELF_CHECK.value,
        }
        if always_admit or self.episodic.admit_with_regret_gate(event, expected_utility):
            row_id = self.episodic.append_event(event)
            payload_text = stable_json_dumps(payload)
            self.working_for(task_id).push(payload_text, {"event_id": row_id})
            return row_id
        return None

    def get_context(self, task_id: typing.Any, query: str, token_budget: typing.Optional[int] = None) -> ContextBlock:
        total_budget = self.config.working_tokens if token_budget is None else int(token_budget)
        if total_budget <= 0:
            raise ValueError("context token budget must be positive")
        budget = TokenBudget(total_budget)
        pieces: typing.List[typing.Tuple[str, str, typing.Optional[int], typing.Optional[float]]] = []
        working_text = self.working_for(task_id).render()
        if working_text:
            pieces.append(("working", working_text, None, None))
        for score, row in self.episodic.retrieve(task_id, query, 100):
            pieces.append(("episodic", str(row["payload_json"]), int(row["id"]), float(score)))
        semantic_ids = self.semantic.search(query, 20, task_id=task_id)
        if semantic_ids:
            conn = self.episodic.db.connect()
            try:
                placeholders = ",".join("?" for _ in semantic_ids)
                rows = conn.execute(f"SELECT id,text FROM semantic WHERE id IN ({placeholders})", semantic_ids).fetchall()
                by_id = {int(row["id"]): str(row["text"]) for row in rows}
                for semantic_id in semantic_ids:
                    if semantic_id in by_id:
                        pieces.append(("semantic", by_id[semantic_id], semantic_id, None))
            finally:
                conn.close()
        text_parts = []
        selected_ids = []
        episodic_accessed: typing.List[int] = []
        logs = []
        for tier, text, item_id, score in pieces:
            prefix = f"[{tier}] "
            prefix_tokens = estimate_tokens(prefix)
            if prefix_tokens > budget.remaining:
                break
            budget.consume(prefix_tokens)
            fitted = budget.fit(text)
            if not fitted:
                continue
            text_parts.append(prefix + fitted)
            if item_id is not None:
                selected_ids.append(item_id)
                logs.append({"id": item_id, "tier": tier, "score": score})
                if tier == "semantic":
                    self.semantic.update_thompson(item_id, 1.0)
                elif tier == "episodic":
                    episodic_accessed.append(item_id)
            if budget.remaining <= 0:
                break
        self.episodic.mark_access(episodic_accessed)
        memory_size_bytes.labels(tier="working").set(len(working_text.encode("utf-8")))
        return ContextBlock("\n".join(text_parts), selected_ids, logs, budget.used)

