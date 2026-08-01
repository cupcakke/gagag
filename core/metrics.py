from core.common import *

class _FallbackMetric:
    def __init__(self, name: str, description: str, labels: tuple = (), kind: str = "counter"):
        self.name = name
        self.description = description
        self.labels_names = tuple(labels)
        self.kind = kind
        self.values: typing.Dict[tuple, float] = {}
        self.counts: typing.Dict[tuple, int] = {}
        self.sums: typing.Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def labels(self, *values, **kwargs) -> "_FallbackMetricChild":
        if kwargs:
            key = tuple(str(kwargs.get(name, "")) for name in self.labels_names)
        elif values:
            key = tuple(str(value) for value in values)
        else:
            key = ()
        if len(key) != len(self.labels_names):
            raise ValueError("incorrect metric label count")
        return _FallbackMetricChild(self, key)

    def inc(self, amount: float = 1) -> None:
        self.labels().inc(amount)

    def set(self, value: float) -> None:
        self.labels().set(value)

    def observe(self, value: float) -> None:
        self.labels().observe(value)


class _FallbackMetricChild:
    def __init__(self, parent: _FallbackMetric, key: tuple):
        self.parent = parent
        self.key = key

    def inc(self, amount: float = 1) -> None:
        with self.parent._lock:
            self.parent.values[self.key] = self.parent.values.get(self.key, 0.0) + float(amount)

    def set(self, value: float) -> None:
        with self.parent._lock:
            self.parent.values[self.key] = float(value)

    def observe(self, value: float) -> None:
        with self.parent._lock:
            numeric = float(value)
            self.parent.counts[self.key] = self.parent.counts.get(self.key, 0) + 1
            self.parent.sums[self.key] = self.parent.sums.get(self.key, 0.0) + numeric
            self.parent.values[self.key] = numeric


_METRICS_REGISTRY = prometheus_client.CollectorRegistry() if _HAS_PROMETHEUS else None


def _make_metric(name: str, description: str, labels: tuple = (), kind: str = "counter") -> typing.Any:
    if _HAS_PROMETHEUS:
        if kind == "gauge":
            return prometheus_client.Gauge(name, description, list(labels), registry=_METRICS_REGISTRY)
        if kind == "histogram":
            return prometheus_client.Histogram(name, description, list(labels), registry=_METRICS_REGISTRY)
        if kind == "counter":
            return prometheus_client.Counter(name, description, list(labels), registry=_METRICS_REGISTRY)
        raise ValueError(f"unsupported metric kind: {kind}")
    return _FallbackMetric(name, description, labels, kind)


loop_iterations_total = _make_metric("loop_iterations_total", "Agent loop iterations")
tool_calls_total = _make_metric("tool_calls_total", "Tool calls executed")
model_calls_total = _make_metric("model_calls_total", "Model generate calls")
errors_total = _make_metric("errors_total", "Errors by type", ("type",))
checkpoints_total = _make_metric("checkpoints_total", "Checkpoint writes")
checkpoint_duration_seconds = _make_metric("checkpoint_duration_seconds", "Checkpoint write latency", (), "histogram")
memory_size_bytes = _make_metric("memory_size_bytes", "Memory tier sizes", ("tier",), "gauge")
circuit_breaker_state = _make_metric("circuit_breaker_state", "Circuit breaker state per dependency", ("dep",), "gauge")
heartbeats_total = _make_metric("heartbeats_total", "Agent heartbeats")
restarts_total = _make_metric("restarts_total", "Supervisor restarts")
livelocks_total = _make_metric("livelocks_total", "Livelock detections")
retries_total = _make_metric("retries_total", "Scheduler retries")
backoff_seconds_sum = _make_metric("backoff_seconds_sum", "Total backoff seconds")

