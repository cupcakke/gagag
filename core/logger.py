from core.common import *
from core.models import EventType
from core.models import BreakerState, Event
from core.metrics import *

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat()
        data = {
            "timestamp": timestamp,
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
LOGGER.setLevel(
    getattr(
        logging,
        os.environ.get("APP_OBSERVABILITY__LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )
)
LOGGER.propagate = False

_WS_CLIENTS: typing.Set[
    typing.Tuple[asyncio.AbstractEventLoop, asyncio.Queue]
] = set()
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
    sanitized_value = redact_instavm_value(payload or {})
    sanitized_payload = (
        sanitized_value if isinstance(sanitized_value, dict) else {}
    )
    type_str = (
        event_type.value
        if isinstance(event_type, EventType)
        else str(event_type)
    )
    task_id_str = str(task_id)
    normalized_tokens_used = int(tokens_used)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

    LOGGER.info(
        type_str,
        extra={
            "component": "event",
            "task_id": task_id_str,
            "step": step,
            "action": sanitized_payload.get("action"),
            "outcome": outcome,
            "tokens_used": normalized_tokens_used,
        },
    )

    event_data = {
        "id": 0,
        "type": type_str,
        "task_id": task_id_str,
        "payload": sanitized_payload,
        "tokens_used": normalized_tokens_used,
        "step": step,
        "outcome": outcome,
        "timestamp": timestamp,
    }

    with _WS_CLIENTS_LOCK:
        clients = tuple(_WS_CLIENTS)

    dead_clients = []
    for loop, queue_obj in clients:
        try:
            loop.call_soon_threadsafe(
                _queue_event,
                queue_obj,
                event_data,
            )
        except RuntimeError:
            dead_clients.append((loop, queue_obj))

    if dead_clients:
        with _WS_CLIENTS_LOCK:
            for client in dead_clients:
                _WS_CLIENTS.discard(client)

    try:
        event_obj = model_validate(
            Event,
            {
                "id": 0,
                "type": type_str,
                "task_id": task_id_str,
                "payload": sanitized_payload,
                "tokens_used": normalized_tokens_used,
                "step": step,
                "outcome": outcome,
                "timestamp": timestamp,
            },
        )
    except Exception:
        event_obj = event_data

    event_sink = _EVENT_SINK
    if event_sink is not None:
        with contextlib.suppress(Exception):
            event_sink(event_obj)

    return event_obj


def observe_latency(name: str, seconds: float, **fields: typing.Any) -> None:
    normalized_seconds = float(seconds)
    if name == "checkpoint":
        checkpoint_duration_seconds.observe(normalized_seconds)

    extra = dict(fields)
    extra["component"] = "latency"
    extra["latency_ms"] = round(normalized_seconds * 1000.0, 3)
    LOGGER.info(name, extra=extra)


def set_breaker_state(name: str, state: typing.Any) -> None:
    state_value = (
        state.value
        if isinstance(state, BreakerState)
        else str(state)
    )
    numeric_states = {
        BreakerState.CLOSED.value: 0.0,
        BreakerState.HALF_OPEN.value: 0.5,
        BreakerState.OPEN.value: 1.0,
    }
    circuit_breaker_state.labels(dep=name).set(
        numeric_states.get(state_value, 0.0)
    )
