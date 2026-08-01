from core.common import *
from core.models import *
from core.database import Database
from core.logger import record_event, LOGGER
from core.metrics import *
from llm.parsers import TokenBudget
from llm.clients import ModelClient, SummarizerClient, RequestyModel
from memory.manager import MemoryManager
from execution.scheduler import Scheduler
from execution.sandbox import ToolSandbox
from agent.checkpoint import CheckpointManager

class AgentLoop:
    def __init__(
        self,
        database: Database,
        scheduler: Scheduler,
        memory: MemoryManager,
        sandbox: ToolSandbox,
        model: ModelClient,
        summarizer: SummarizerClient,
        checkpoint: CheckpointManager,
        config: typing.Any,
    ):
        self.db = database
        self.scheduler = scheduler
        self.memory = memory
        self.sandbox = sandbox
        self.model = model
        self.summarizer = summarizer
        self.checkpoint = checkpoint
        self.config = config
        self.stop_event = threading.Event()
        self.iteration_id = 0
        self.last_heartbeat = time.time()
        self.last_checkpoint = 0.0
        self._sig_window: deque = deque(maxlen=500)
        self._running_tasks: typing.Set[uuid.UUID] = set()
        self._running_lock = threading.RLock()
        self._saved_checkpoint: typing.Optional[typing.Any] = None
        self._restored_checkpoint_tasks: typing.Set[str] = set()

    def request_stop(self) -> None:
        self.stop_event.set()
        self.scheduler._signal_wake()

    def _emit(self, task_id: typing.Any, kind: EventType, payload: dict, tokens: int = 0) -> None:
        event = record_event(kind, task_id, payload, tokens, self.iteration_id)
        if getattr(_EVENT_SINK, "__self__", None) is not self.memory:
            self.memory.append(event)

    def _normalize_action(self, call: dict) -> str:
        return stable_json_dumps({"name": call.get("name", ""), "arguments": call.get("arguments", {})})

    def _check_livelock(self, task_id: typing.Any, signature: str, window_s: float, threshold: int) -> bool:
        now = time.time()
        cutoff = now - window_s
        self._sig_window.append((now, str(task_id), signature))
        return sum(1 for ts, tid, sig in self._sig_window if ts >= cutoff and tid == str(task_id) and sig == signature) >= threshold

    def _early_abort_probe(self, step: int, signatures: typing.List[str], token_counts: typing.List[int]) -> bool:
        if step < 3:
            return False
        if len(token_counts) >= 3:
            recent = token_counts[-3:]
            if recent[-1] > recent[-2] > recent[-3] and sum(recent) > int(self.config.model.max_completion_tokens * 2.5):
                return True
        return False

    @staticmethod
    def _annotation_schema(annotation: typing.Any) -> dict:
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if annotation in {str, pathlib.Path, uuid.UUID, dt.datetime, dt.date}:
            return {"type": "string"}
        if annotation is bool:
            return {"type": "boolean"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation in {dict, typing.Dict} or origin in {dict, typing.Dict}:
            value_schema = AgentLoop._annotation_schema(args[1]) if len(args) == 2 and args[1] is not typing.Any else {}
            return {"type": "object", "additionalProperties": value_schema or True}
        if annotation in {list, typing.List, tuple, typing.Tuple, set, typing.Set} or origin in {list, typing.List, tuple, typing.Tuple, set, typing.Set}:
            item = AgentLoop._annotation_schema(args[0]) if args else {}
            return {"type": "array", "items": item}
        if origin in {typing.Union, types.UnionType}:
            variants = [item for item in args if item is not type(None)]
            permits_null = len(variants) != len(args)
            if len(variants) == 1:
                schema = AgentLoop._annotation_schema(variants[0])
                if permits_null and "type" in schema and isinstance(schema["type"], str):
                    schema["type"] = [schema["type"], "null"]
                return schema
            return {"anyOf": [AgentLoop._annotation_schema(item) for item in variants] + ([{"type": "null"}] if permits_null else [])}
        if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            return {"type": "string", "enum": [item.value for item in annotation]}
        return {}

    def _tool_schema(self) -> typing.List[dict]:
        schemas: typing.List[dict] = []
        for spec, _ in self.sandbox.tools.values():
            input_schema = spec.input_schema
            if _HAS_PYDANTIC and hasattr(input_schema, "model_json_schema"):
                parameters = input_schema.model_json_schema()
            elif dataclasses.is_dataclass(input_schema):
                properties: dict = {}
                required: typing.List[str] = []
                for field in dataclasses.fields(input_schema):
                    properties[field.name] = self._annotation_schema(field.type)
                    if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
                        required.append(field.name)
                parameters = {"type": "object", "properties": properties, "additionalProperties": False}
                if required:
                    parameters["required"] = required
            else:
                parameters = {"type": "object", "properties": {}, "additionalProperties": False}
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": f"Execute the {spec.name} capability.",
                        "strict": True,
                        "parameters": self._strict_schema(parameters),
                    },
                }
            )
        return schemas

    def _strict_schema(self, schema: typing.Any) -> typing.Any:
        if isinstance(schema, dict):
            result = {key: self._strict_schema(value) for key, value in schema.items()}
            if result.get("type") == "object" or "properties" in result:
                if "additionalProperties" not in result:
                    result["additionalProperties"] = False
                properties = result.get("properties")
                if isinstance(properties, dict) and "required" in result:
                    result["required"] = [name for name in result["required"] if name in properties]
            return result
        if isinstance(schema, list):
            return [self._strict_schema(item) for item in schema]
        return schema

    def _select_model_name(self, payload: dict) -> str:
        task_text = stable_json_dumps(payload).lower()
        default_model = self.config.model.controller or self.config.model.name
        attachments = payload.get("attachments", [])
        images = payload.get("images", [])
        if images or any(isinstance(item, dict) and item.get("kind") in {"image", "video"} for item in attachments):
            return self.config.model.multimodal or default_model
        if any(term in task_text for term in ("pdf", "long document", "hosszú dokumentum", "melléklet", "attachment")):
            return self.config.model.long_context or default_model
        if any(term in task_text for term in ("explicit sexual", "adult content", "felnőtt tartalom", "sensitive content", "érzékeny tartalom")):
            return self.config.model.sensitive or default_model
        if any(term in task_text for term in ("code", "python", "nix", "flake", "debug", "implement", "kód", "programoz", "javít")):
            return self.config.model.code or default_model
        if not default_model:
            if isinstance(self.model, RequestyModel):
                raise PermanentError("REQUESTY_MODEL or APP_MODEL__CONTROLLER must be configured")
            return type(self.model).__name__
        return default_model

    def _todo_text(self, task: typing.Any) -> typing.Tuple[pathlib.Path, str]:
        task_directory = TODO_PATH.parent / "tasks"
        task_directory.mkdir(parents=True, exist_ok=True)
        task_path = task_directory / f"{task.id}.md"
        payload_text = stable_json_dumps(task.payload)
        actions = ["Validate the task payload and dependencies"]
        if task.payload.get("attachments") or task.payload.get("images"):
            actions.append("Inspect and validate every supplied attachment")
        if any(term in payload_text.lower() for term in ("code", "python", "debug", "fix", "implement")):
            actions.append("Inspect, execute, and verify the relevant code paths")
        actions.extend(["Execute only required authorized tools", "Verify outputs against the requested result"])
        content = "\n".join([f"Task {task.id}", payload_text, *[f"{index + 1}. {action}" for index, action in enumerate(actions)]]) + "\n"
        atomic_write(task_path, content.encode("utf-8"))
        return task_path, content[:16000]

    async def run(self) -> None:
        self.scheduler.bind_loop()
        self._saved_checkpoint = self.checkpoint.load_latest()
        background_stop = asyncio.Event()
        workers = max(1, min(32, int(os.environ.get("APP_AGENT_WORKERS", "4"))))
        background = [
            asyncio.create_task(self._summarize_loop(background_stop)),
            asyncio.create_task(self._eviction_loop(background_stop)),
            asyncio.create_task(self.scheduler.aging_loop(background_stop)),
            asyncio.create_task(self.scheduler.schedule_loop(background_stop)),
            asyncio.create_task(self.checkpoint.flush_worker(background_stop)),
            asyncio.create_task(self._heartbeat_loop(background_stop)),
        ]
        worker_tasks = [asyncio.create_task(self._worker(index)) for index in range(workers)]
        try:
            await asyncio.gather(*worker_tasks)
        finally:
            background_stop.set()
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)

    async def _worker(self, worker_index: int) -> None:
        while not self.stop_event.is_set():
            task = self.scheduler.dequeue_next()
            if task is None:
                try:
                    await asyncio.wait_for(self.scheduler.wake.wait(), timeout=0.5)
                    self.scheduler.wake.clear()
                except asyncio.TimeoutError:
                    self.last_heartbeat = time.time()
                continue
            with self._running_lock:
                self._running_tasks.add(task.id)
            try:
                await self._process_task(task)
            except asyncio.CancelledError:
                self.scheduler.requeue_with_backoff(task.id)
                raise
            except Exception as exc:
                errors_total.labels(type=type(exc).__name__).inc()
                self.scheduler.mark_failed(task.id)
                self._emit(task.id, EventType.ERROR, {"error": str(exc), "traceback": traceback.format_exc(), "worker": worker_index})
            finally:
                with self._running_lock:
                    self._running_tasks.discard(task.id)

    async def _heartbeat_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            self.last_heartbeat = time.time()
            heartbeats_total.inc()
            record_event(EventType.HEARTBEAT, uuid.UUID(int=0), {"iteration": self.iteration_id, "running": len(self.running_tasks())})
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.config.supervisor.heartbeat_interval_s)
            except asyncio.TimeoutError:
                continue

    def running_tasks(self) -> typing.List[uuid.UUID]:
        with self._running_lock:
            return list(self._running_tasks)

    async def _summarize_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                try:
                    await self.memory.episodic.summarize_old_events(self.summarizer, self.config.summarizer.max_summary_len)
                except Exception as exc:
                    LOGGER.error(f"summarize loop failed: {exc}", extra={"component": "memory"})

    async def _eviction_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.config.memory.thompson_eviction_interval_s)
            except asyncio.TimeoutError:
                try:
                    self.memory.semantic.thompson_evict(evict_count=20)
                except Exception as exc:
                    LOGGER.error(f"eviction loop failed: {exc}", extra={"component": "memory"})

    def _restore_checkpoint(self, task: typing.Any) -> int:
        task_id = str(task.id)
        saved = self._saved_checkpoint
        if saved is None or task_id in self._restored_checkpoint_tasks:
            return 0
        if task_id not in {str(item) for item in saved.in_flight_tasks}:
            return 0
        queue_state = dict(getattr(saved, "queue_state", {}) or {})
        tasks = dict(queue_state.get("tasks", {}) or {})
        task_state = tasks.get(task_id)
        if not isinstance(task_state, dict):
            if str(queue_state.get("task_id", "")) != task_id:
                return 0
            task_state = {
                "step": int(getattr(saved, "step_counter", 0) or 0),
                "working_memory": queue_state.get("working_memory", []),
            }
        snapshot = task_state.get("working_memory", [])
        if isinstance(snapshot, list):
            self.memory.working_for(task.id).restore(snapshot)
        self._restored_checkpoint_tasks.add(task_id)
        return max(0, int(task_state.get("step", 0) or 0))

    async def _checkpoint_task(self, task: typing.Any, step: int) -> None:
        working = self.memory.working_for(task.id)
        await self.checkpoint.checkpoint_task(
            task.id,
            step,
            working.snapshot(),
            self.scheduler.list_in_flight(),
            self.memory.episodic.count(),
        )
        self.last_checkpoint = time.time()

    async def _inner_process(
        self,
        task: typing.Any,
        started_mono: float,
        step: int,
        signatures: typing.List[str],
        token_counts: typing.List[int],
        resume_step: int,
    ) -> None:
        if time.monotonic() - started_mono >= self.config.agent.wall_timeout_s:
            self._emit(task.id, EventType.TIMEOUT, {"reason": "wall_timeout", "elapsed_s": time.monotonic() - started_mono})
            self.scheduler.update_status(task.id, TaskStatus.TIMED_OUT)
            return
        await self._process_task(task)

    async def _process_task(self, task: typing.Any) -> None:
        started_mono = time.monotonic()
        signatures: typing.List[str] = []
        token_counts: typing.List[int] = []
        resume_step = self._restore_checkpoint(task)
        step = 0
        while step < self.config.agent.max_steps:
            if self.stop_event.is_set():
                self.scheduler.requeue_with_backoff(task.id)
                return
            elapsed = time.monotonic() - started_mono
            if elapsed >= self.config.agent.wall_timeout_s:
                self._emit(task.id, EventType.TIMEOUT, {"reason": "wall_timeout", "elapsed_s": elapsed})
                self.scheduler.update_status(task.id, TaskStatus.TIMED_OUT)
                return
            step += 1
            self.iteration_id += 1
            loop_iterations_total.inc()
            if step <= resume_step:
                continue
            context_block = self.memory.get_context(task.id, stable_json_dumps(task.payload), self.config.memory.working_tokens)
            todo_path, todo_text = self._todo_text(task)
            model_name = self._select_model_name(task.payload)
            prompt = (
                f"Task: {stable_json_dumps(task.payload)}\n"
                f"Todo file path: {todo_path}\n"
                f"Todo content:\n{todo_text}\n"
                f"Selected model: {model_name}\n"
                f"Context:\n{context_block.text}"
            )
            model_calls_total.inc()
            try:
                remaining = max(0.001, self.config.agent.wall_timeout_s - elapsed)
                async with TimeoutBudget(min(self.config.model.request_timeout_s, remaining), TaskTimeoutError):
                    active_model = self.model.for_model(model_name) if isinstance(self.model, RequestyModel) else self.model
                    action, usage = await active_model.generate(prompt, self._tool_schema(), self.config.model.max_completion_tokens)
            except TaskTimeoutError:
                self._emit(task.id, EventType.TIMEOUT, {"reason": "model_timeout", "step": step})
                self.scheduler.requeue_with_backoff(task.id)
                return
            except PermanentError as exc:
                errors_total.labels(type=type(exc).__name__).inc()
                self._emit(task.id, EventType.ERROR, {"error": str(exc), "step": step, "permanent": True})
                self.scheduler.mark_failed(task.id)
                return
            except Exception as exc:
                errors_total.labels(type=type(exc).__name__).inc()
                self._emit(task.id, EventType.ERROR, {"error": str(exc), "step": step, "transient": True})
                self.scheduler.requeue_with_backoff(task.id)
                return
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            token_counts.append(total_tokens)
            self._emit(task.id, EventType.MODEL_OUTPUT, {"action": _plain(action), "step": step}, total_tokens)
            tool_calls = list(getattr(action, "tool_calls", []) or [])
            final_answer = getattr(action, "final_answer", None)
            declared_intent = str(getattr(action, "declared_intent", "") or "")
            if not tool_calls and not final_answer:
                self._emit(task.id, EventType.ERROR, {"reason": "model_returned_no_action", "step": step})
                self.scheduler.mark_failed(task.id)
                return
            for call in tool_calls:
                if not isinstance(call, dict):
                    self._emit(task.id, EventType.ERROR, {"reason": "invalid_tool_call", "step": step})
                    self.scheduler.mark_failed(task.id)
                    return
                call_name = str(call.get("name", "")).strip()
                call_args = call.get("arguments", {})
                if not call_name or not isinstance(call_args, dict):
                    self._emit(task.id, EventType.ERROR, {"reason": "invalid_tool_call", "step": step})
                    self.scheduler.mark_failed(task.id)
                    return
                signature = self._normalize_action(call)
                signatures.append(signature)
                if self._check_livelock(task.id, signature, self.config.agent.livelock_window_s, self.config.agent.livelock_repeat_threshold):
                    livelocks_total.inc()
                    self._emit(task.id, EventType.LIVELOCK_DETECTED, {"signature": signature, "step": step})
                    await self._checkpoint_task(task, step)
                    self.scheduler.requeue_with_backoff(task.id)
                    return
                self._emit(task.id, EventType.TOOL_CALL, {"name": call_name, "arguments": call_args, "step": step})
                try:
                    result_data = await self.sandbox.execute(task.id, declared_intent, call_name, call_args)
                except (CircuitOpenError, ToolTimeoutError, TransientError) as exc:
                    errors_total.labels(type=type(exc).__name__).inc()
                    self._emit(task.id, EventType.ERROR, {"error": str(exc), "tool": call_name, "step": step, "transient": True})
                    self.scheduler.requeue_with_backoff(task.id)
                    return
                except (PermanentError, SchemaValidationError) as exc:
                    errors_total.labels(type=type(exc).__name__).inc()
                    self._emit(task.id, EventType.ERROR, {"error": str(exc), "tool": call_name, "step": step, "permanent": True})
                    self.scheduler.mark_failed(task.id)
                    return
                self._emit(task.id, EventType.TOOL_RESULT, {"name": call_name, "result": result_data, "step": step})
                self.memory.working_for(task.id).push(stable_json_dumps(result_data))
            if final_answer:
                self.memory.working_for(task.id).push(str(final_answer))
                self._emit(task.id, EventType.REFLECTION, {"answer": str(final_answer), "step": step}, total_tokens)
                self.scheduler.mark_done(task.id)
                return
            if self._early_abort_probe(step, signatures, token_counts):
                self._emit(task.id, EventType.SELF_CHECK, {"reason": "early_abort_probe", "step": step})
                await self._checkpoint_task(task, step)
                self.scheduler.requeue_with_backoff(task.id)
                return
            if step % 5 == 0 or time.time() - self.last_checkpoint > 30.0:
                await self._checkpoint_task(task, step)
        self._emit(task.id, EventType.TIMEOUT, {"reason": "step_cap", "max_steps": self.config.agent.max_steps})
        self.scheduler.update_status(task.id, TaskStatus.TIMED_OUT)

