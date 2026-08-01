from core.common import *
from core.models import *
from core.database import Database
from memory.working import WorkingMemory
from memory.semantic import SemanticMemory
from memory.episodic import EpisodicMemory
from llm.clients import SummarizerClient

import json
import threading
import time
import typing


class ProceduralMemory:
    def __init__(self, database: Database):
        if database is None:
            raise ValueError("database is required")
        self.db = database

    def register(self, name: str, definition: typing.Any, version: str = "1") -> None:
        if not isinstance(name, str) or not name or len(name) > 256:
            raise ValueError("invalid skill name")
        if not isinstance(version, str) or not version:
            raise ValueError("invalid skill version")
        definition_json = stable_json_dumps(definition)
        conn = self.db.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO skills(name,version,definition_json,updated_at) VALUES(?,?,?,?)",
                (name, version, definition_json, time.time()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def invoke_by_name(self, name: str) -> typing.Optional[dict]:
        if not isinstance(name, str) or not name or len(name) > 256:
            raise ValueError("invalid skill name")
        conn = self.db.connect()
        try:
            row = conn.execute(
                "SELECT definition_json FROM skills WHERE name=?",
                (name,),
            ).fetchone()
            if row is None:
                return None
            definition = json.loads(str(row["definition_json"]))
            if not isinstance(definition, dict):
                raise ValueError("stored skill definition must be a JSON object")
            return definition
        finally:
            conn.close()

    def capture_from_reflection(self, reflection: typing.Any) -> typing.Optional[str]:
        if not isinstance(reflection, dict):
            return None
        skill = reflection.get("skill")
        if skill is None:
            return None
        skill_name = str(skill)
        if not skill_name:
            return None
        version = str(reflection.get("version", "1"))
        self.register(skill_name, reflection, version)
        return skill_name


class ContextBlock:
    __slots__ = ("text", "selected_ids", "selection_log", "tokens_used")

    def __init__(
        self,
        text: str,
        selected_ids: typing.List[int],
        selection_log: typing.List[typing.Dict[str, typing.Any]],
        tokens_used: int,
    ):
        self.text = text
        self.selected_ids = selected_ids
        self.selection_log = selection_log
        self.tokens_used = tokens_used


class MemoryManager:
    def __init__(
        self,
        database: Database,
        config: typing.Any = None,
        summarizer: typing.Any = None,
    ):
        if database is None:
            raise ValueError("database is required")
        if config is None:
            raise ValueError("config is required")
        if not hasattr(config, "working_tokens"):
            raise ValueError("config must define working_tokens")
        working_tokens = int(config.working_tokens)
        if working_tokens <= 0:
            raise ValueError("working_tokens must be positive")
        self.config = config
        self.working = WorkingMemory(working_tokens)
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
                memory = WorkingMemory(int(self.config.working_tokens))
                self._working_by_task[key] = memory
            return memory

    def append(self, event: typing.Any) -> typing.Optional[int]:
        payload, _, task_id, type_text, tokens, _ = self.episodic._event_fields(event)
        token_count = max(0, int(tokens))
        expected_utility = max(
            0.0,
            min(1.0, 0.25 + min(0.75, token_count / 10000.0)),
        )
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
        if not always_admit and not self.episodic.admit_with_regret_gate(
            event,
            expected_utility,
        ):
            return None
        row_id = self.episodic.append_event(event)
        payload_text = stable_json_dumps(payload)
        self.working_for(task_id).push(payload_text, {"event_id": row_id})
        return row_id

    def get_context(
        self,
        task_id: typing.Any,
        query: str,
        token_budget: typing.Optional[int] = None,
    ) -> ContextBlock:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        total_budget = (
            int(self.config.working_tokens)
            if token_budget is None
            else int(token_budget)
        )
        if total_budget <= 0:
            raise ValueError("context token budget must be positive")
        budget = TokenBudget(total_budget)
        pieces: typing.List[
            typing.Tuple[
                str,
                str,
                typing.Optional[int],
                typing.Optional[float],
            ]
        ] = []
        working_text = self.working_for(task_id).render()
        if working_text:
            pieces.append(("working", working_text, None, None))
        episodic_results = self.episodic.retrieve(task_id, query, 100)
        for score, row in episodic_results:
            pieces.append(
                (
                    "episodic",
                    str(row["payload_json"]),
                    int(row["id"]),
                    float(score),
                )
            )
        semantic_ids = [
            int(semantic_id)
            for semantic_id in self.semantic.search(query, 20, task_id=task_id)
        ]
        if semantic_ids:
            conn = self.episodic.db.connect()
            try:
                placeholders = ",".join("?" for _ in semantic_ids)
                rows = conn.execute(
                    f"SELECT id,text FROM semantic WHERE id IN ({placeholders})",
                    tuple(semantic_ids),
                ).fetchall()
                by_id = {
                    int(row["id"]): str(row["text"])
                    for row in rows
                }
                for semantic_id in semantic_ids:
                    semantic_text = by_id.get(semantic_id)
                    if semantic_text is not None:
                        pieces.append(
                            (
                                "semantic",
                                semantic_text,
                                semantic_id,
                                None,
                            )
                        )
            finally:
                conn.close()
        text_parts: typing.List[str] = []
        selected_ids: typing.List[int] = []
        episodic_accessed: typing.List[int] = []
        logs: typing.List[typing.Dict[str, typing.Any]] = []
        for tier, text, item_id, score in pieces:
            if not text:
                continue
            prefix = f"[{tier}] "
            prefix_tokens = estimate_tokens(prefix)
            if prefix_tokens >= budget.remaining:
                break
            budget.consume(prefix_tokens)
            fitted = budget.fit(text)
            if not fitted:
                break
            text_parts.append(prefix + fitted)
            if item_id is not None:
                selected_ids.append(item_id)
                logs.append(
                    {
                        "id": item_id,
                        "tier": tier,
                        "score": score,
                    }
                )
                if tier == "semantic":
                    self.semantic.update_thompson(item_id, 1.0)
                elif tier == "episodic":
                    episodic_accessed.append(item_id)
            if budget.remaining <= 0:
                break
        if episodic_accessed:
            self.episodic.mark_access(episodic_accessed)
        memory_size_bytes.labels(tier="working").set(
            len(working_text.encode("utf-8"))
        )
        return ContextBlock(
            "\n".join(text_parts),
            selected_ids,
            logs,
            budget.used,
        )
