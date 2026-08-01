from core.common import *
from core.models import EventType
from core.metrics import *

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "task_id": getattr(record, "task_id", None),
            "step": getattr(record, "step", None),
            "action": getattr(record, "action", None),
            "outcome": getattr(record, "outcome", None),
            "tokens_used": getattr(record, "tokens_used", 0),
            "latency_ms": getattr(record, "latency_ms", None),
            "circuit_state": getattr(record, "circuit_state", None),
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return stable_json_dumps(data)


LOGGER = logging.getLogger("autonomous-agent")
if not LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(JsonFormatter())
    LOGGER.addHandler(_handler)
LOGGER.setLevel(getattr(logging, os.environ.get("APP_OBSERVABILITY__LOG_LEVEL", "INFO").upper(), logging.INFO))
LOGGER.propagate = False

_WS_CLIENTS: typing.Set[typing.Tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
_WS_CLIENTS_LOCK = threading.Lock()
_EVENT_SINK: typing.Optional[typing.Callable[[typing.Any], typing.Any]] = None


def _queue_event(queue_obj: asyncio.Queue, event_data: dict) -> None:
    if queue_obj.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue_obj.get_nowait()
    with contextlib.suppress(asyncio.QueueFull):
        queue_obj.put_nowait(event_data.copy())


def record_event(
    event_type: typing.Any,
    task_id: typing.Any,
    payload: typing.Optional[dict] = None,
    tokens_used: int = 0,
    step: typing.Optional[int] = None,
    outcome: typing.Optional[str] = None,
) -> typing.Any:
    sanitized_payload = redact_instavm_value(payload or {})
    type_str = event_type.value if isinstance(event_type, EventType) else str(event_type)
    LOGGER.info(
        type_str,
        extra={
            "component": "event",
            "task_id": str(task_id),
            "step": step,
            "action": sanitized_payload.get("action"),
            "outcome": outcome,
            "tokens_used": tokens_used,
        },
    )
    event_data = {
        "id": 0,
        "type": type_str,
        "task_id": str(task_id),
        "payload": sanitized_payload,
        "tokens_used": int(tokens_used),
        "step": step,
        "outcome": outcome,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with _WS_CLIENTS_LOCK:
        clients = list(_WS_CLIENTS)
    dead = []
    for loop, queue_obj in clients:
        try:
            loop.call_soon_threadsafe(_queue_event, queue_obj, event_data)
        except RuntimeError:
            dead.append((loop, queue_obj))
    if dead:
        with _WS_CLIENTS_LOCK:
            for client in dead:
                _WS_CLIENTS.discard(client)
    try:
        event_obj = model_validate(Event, {
            "id": 0,
            "type": type_str,
            "task_id": str(task_id),
            "payload": sanitized_payload,
            "tokens_used": int(tokens_used),
        })
    except Exception:
        event_obj = event_data
    if _EVENT_SINK is not None:
        with contextlib.suppress(Exception):
            _EVENT_SINK(event_obj)
    return event_obj


def observe_latency(name: str, seconds: float, **fields) -> None:
    if name == "checkpoint":
        checkpoint_duration_seconds.observe(seconds)
    LOGGER.info(name, extra={"component": "latency", "latency_ms": round(seconds * 1000, 3), **fields})


def set_breaker_state(name: str, state: typing.Any) -> None:
    state_val = state.value if isinstance(state, BreakerState) else str(state)
    numeric = {BreakerState.CLOSED.value: 0.0, BreakerState.HALF_OPEN.value: 0.5, BreakerState.OPEN.value: 1.0}
    circuit_breaker_state.labels(dep=name).set(numeric.get(state_val, 0.0))

