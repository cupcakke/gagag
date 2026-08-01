from core.common import *
from config.settings import _mc, _plain

class EventType(str, enum.Enum):
    USER_INPUT = "user_input"
    MODEL_OUTPUT = "model_output"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    HEARTBEAT = "heartbeat"
    REFLECTION = "reflection"
    LIVELOCK_DETECTED = "livelock_detected"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_HALF_OPEN = "circuit_half_open"
    CIRCUIT_CLOSED = "circuit_closed"
    RESTART = "restart"
    SELF_CHECK = "self_check"


class TaskStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ABORTED = "aborted"


class BreakerState(str, enum.Enum):
    CLOSED = "Closed"
    OPEN = "Open"
    HALF_OPEN = "HalfOpen"


def _coerce_annotation(value, annotation):
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        for option in args:
            if option is type(None):
                continue
            try:
                return _coerce_annotation(value, option)
            except Exception:
                continue
        return value
    if origin in {list, typing.List}:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list")
        item_type = args[0] if args else typing.Any
        return [_coerce_annotation(item, item_type) for item in value]
    if origin in {dict, typing.Dict}:
        if not isinstance(value, dict):
            raise TypeError("expected an object")
        key_type = args[0] if args else typing.Any
        value_type = args[1] if len(args) > 1 else typing.Any
        return {_coerce_annotation(k, key_type): _coerce_annotation(v, value_type) for k, v in value.items()}
    if annotation in {typing.Any, None}:
        return value
    if annotation is uuid.UUID:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    if annotation is dt.datetime:
        if isinstance(value, dt.datetime):
            parsed = value
        else:
            parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return value if isinstance(value, annotation) else annotation(value)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise TypeError("expected a boolean")
    if annotation in {str, int, float}:
        return annotation(value)
    if inspect.isclass(annotation) and dataclasses.is_dataclass(annotation):
        return model_validate(annotation, value)
    return value


def model_validate(cls, data):
    if _HAS_PYDANTIC and hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    if dataclasses.is_dataclass(cls):
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__} input must be an object")
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(data) - set(fields)
        if unknown:
            raise TypeError(f"unknown fields for {cls.__name__}: {sorted(unknown)}")
        values = {}
        for name, field in fields.items():
            if name in data:
                values[name] = _coerce_annotation(data[name], field.type)
        return cls(**values)
    if not isinstance(data, dict):
        raise TypeError("validation input must be an object")
    return cls(**data)


if _HAS_PYDANTIC:
    class Event(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        id: int = 0
        timestamp: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
        type: EventType
        task_id: uuid.UUID
        payload: dict[str, typing.Any] = Field(default_factory=dict)
        tokens_used: int = 0
        embedding: typing.Optional[list[float]] = None

    class Task(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        id: uuid.UUID = Field(default_factory=uuid.uuid4)
        priority: int = 0
        deadline: typing.Optional[dt.datetime] = None
        dependencies: list[uuid.UUID] = Field(default_factory=list)
        payload: dict[str, typing.Any] = Field(default_factory=dict)
        created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
        updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
        status: TaskStatus = TaskStatus.QUEUED
        checkpoints: list[int] = Field(default_factory=list)

    class ModelAction(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        tool_calls: list[dict[str, typing.Any]] = Field(default_factory=list)
        final_answer: typing.Optional[str] = None
        declared_intent: str = ""
        confidence: float = 0.0
        rationale: str = ""

    class Heartbeat(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        iteration_id: int
        last_step_ts: float
        loop_state_digest: str

    class CheckpointState(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        working_memory_ptr: int = 0
        episodic_cursor: int = 0
        in_flight_tasks: list[uuid.UUID] = Field(default_factory=list)
        queue_state: dict[str, typing.Any] = Field(default_factory=dict)
        step_counter: int = 0
        version: int = 1
        crc32: int = 0

    class CircuitBreakerState(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        state: BreakerState = BreakerState.CLOSED
        failure_count: int = 0
        last_failure_ts: float = 0.0
        cooldown_until: float = 0.0

    class UsageStats(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    class ToolSpec(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        name: str
        version: str = "1"
        input_schema: typing.Any
        output_schema: typing.Any
        timeout_s: float = 30.0
        resource_limits: dict[str, typing.Any] = Field(default_factory=dict)
        side_effects: bool = False
        idempotency_key_fn: typing.Any = None
        fallback_tool: typing.Optional[str] = None

else:
    @dataclasses.dataclass
    class Event:
        type: EventType
        task_id: uuid.UUID
        id: int = 0
        timestamp: dt.datetime = dataclasses.field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
        payload: dict = dataclasses.field(default_factory=dict)
        tokens_used: int = 0
        embedding: typing.Optional[list] = None

    @dataclasses.dataclass
    class Task:
        id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
        priority: int = 0
        deadline: typing.Optional[dt.datetime] = None
        dependencies: list = dataclasses.field(default_factory=list)
        payload: dict = dataclasses.field(default_factory=dict)
        created_at: dt.datetime = dataclasses.field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
        updated_at: dt.datetime = dataclasses.field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
        status: TaskStatus = TaskStatus.QUEUED
        checkpoints: list = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class ModelAction:
        tool_calls: list = dataclasses.field(default_factory=list)
        final_answer: typing.Optional[str] = None
        declared_intent: str = ""
        confidence: float = 0.0
        rationale: str = ""

    @dataclasses.dataclass
    class Heartbeat:
        iteration_id: int = 0
        last_step_ts: float = 0.0
        loop_state_digest: str = ""

    @dataclasses.dataclass
    class CheckpointState:
        working_memory_ptr: int = 0
        episodic_cursor: int = 0
        in_flight_tasks: list = dataclasses.field(default_factory=list)
        queue_state: dict = dataclasses.field(default_factory=dict)
        step_counter: int = 0
        version: int = 1
        crc32: int = 0

    @dataclasses.dataclass
    class CircuitBreakerState:
        state: BreakerState = BreakerState.CLOSED
        failure_count: int = 0
        last_failure_ts: float = 0.0
        cooldown_until: float = 0.0

    @dataclasses.dataclass
    class UsageStats:
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    @dataclasses.dataclass
    class ToolSpec:
        name: str = ""
        version: str = "1"
        input_schema: typing.Any = None
        output_schema: typing.Any = None
        timeout_s: float = 30.0
        resource_limits: dict = dataclasses.field(default_factory=dict)
        side_effects: bool = False
        idempotency_key_fn: typing.Any = None
        fallback_tool: typing.Optional[str] = None


class AgentError(Exception):
    pass


class ContextWindowExceededError(AgentError):
    pass


class MaxRetriesExceeded(AgentError):
    pass


class AgentTimeoutError(AgentError):
    pass


class ToolTimeoutError(AgentTimeoutError):
    pass


class TaskTimeoutError(AgentTimeoutError):
    pass


class TransientError(AgentError):
    pass


class PermanentError(AgentError):
    pass


class LivelockDetected(AgentError):
    pass


class IntentMismatchError(AgentError):
    pass


class DuplicateActionError(AgentError):
    pass


class CircuitOpenError(AgentError):
    pass


class SchemaValidationError(AgentError):
    pass


class InstaVMError(AgentError):
    def __init__(
        self,
        operation_id: str,
        status_code: int,
        message: str,
        request_id: str = "",
        retryable: bool = False,
    ):
        self.operation_id = operation_id
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable
        super().__init__(f"{operation_id}: HTTP {status_code}: {message}")

def stable_json_dumps(value):
    def default(obj):
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, bytes):
            return {"__bytes__": base64.b64encode(obj).decode("ascii")}
        if isinstance(obj, enum.Enum):
            return obj.value
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, set):
            return sorted(obj, key=lambda item: stable_json_dumps(_plain(item)))
        raise TypeError(type(obj).__name__)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=default,
        allow_nan=False,
    )

def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF

def atomic_write(path: pathlib.Path, data: bytes) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    replaced = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, str(path))
        replaced = True
        if os.name == "posix":
            with contextlib.suppress(OSError, AttributeError):
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                dir_fd = os.open(str(path.parent), flags)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
    except Exception:
        if not replaced:
            with contextlib.suppress(FileNotFoundError, OSError):
                os.unlink(tmp_name)
        raise

def exponential_backoff_with_jitter(
    attempt: int,
    base: float,
    factor: float,
    max_delay: float,
    retry_after: typing.Optional[float] = None,
    max_attempts: typing.Optional[int] = None,
    rng: typing.Optional[random.Random] = None,
) -> float:
    attempt = int(attempt)
    base = float(base)
    factor = float(factor)
    max_delay = float(max_delay)
    if attempt < 0:
        raise ValueError("attempt must be nonnegative")
    if base < 0 or max_delay < 0 or factor < 1:
        raise ValueError("invalid backoff configuration")
    if max_attempts is not None and attempt >= int(max_attempts):
        raise MaxRetriesExceeded(f"attempt {attempt} exceeds max {max_attempts}")
    upper = min(max_delay, base * (factor ** attempt))
    minimum = 0.0
    if retry_after is not None:
        retry_after_value = float(retry_after)
        if not math.isfinite(retry_after_value) or retry_after_value < 0:
            raise ValueError("retry_after must be a nonnegative finite number")
        minimum = retry_after_value
        upper = max(upper, minimum)
    generator = rng or random
    return minimum + generator.random() * max(0.0, upper - minimum)

class TimeoutBudget:
    def __init__(self, seconds: float, error_type: type = AgentTimeoutError):
        self.seconds = float(seconds)
        if self.seconds <= 0:
            raise ValueError("timeout budget must be positive")
        self.error_type = error_type
        self._task: typing.Optional[asyncio.Task] = None
        self._handle: typing.Optional[asyncio.TimerHandle] = None
        self._expired = False

    async def __aenter__(self) -> "TimeoutBudget":
        self._task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        self._handle = loop.call_later(self.seconds, self._cancel)
        return self

    def _cancel(self) -> None:
        self._expired = True
        if self._task is not None:
            self._task.cancel()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if self._handle is not None:
            self._handle.cancel()
        if exc_type is asyncio.CancelledError and self._expired:
            uncancel = getattr(self._task, "uncancel", None)
            if callable(uncancel):
                uncancel()
            raise self.error_type(f"operation exceeded {self.seconds}s budget") from None
        return False

def make_idempotency_key(task_id: typing.Any, intent: str, normalized_action: typing.Any) -> str:
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, "autonomous-agent-v1")
    payload = f"{task_id}:{intent}:{stable_json_dumps(normalized_action)}"
    return str(uuid.uuid5(namespace, payload))

