from core.common import *

import threading
import typing

try:
    import prometheus_client
except ImportError:
    prometheus_client = None

_HAS_PROMETHEUS = prometheus_client is not None


class _FallbackMetric:
    def __init__(
        self,
        name: str,
        description: str,
        labels: tuple = (),
        kind: str = "counter",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("metric name must be a non-empty string")
        if not isinstance(description, str) or not description:
            raise ValueError("metric description must be a non-empty string")
        if kind not in {"counter", "gauge", "histogram"}:
            raise ValueError(f"unsupported metric kind: {kind}")

        labels_names = tuple(labels)
        if any(not isinstance(label, str) or not label for label in labels_names):
            raise ValueError("metric label names must be non-empty strings")
        if len(labels_names) != len(set(labels_names)):
            raise ValueError("metric label names must be unique")

        self.name = name
        self.description = description
        self.labels_names = labels_names
        self.kind = kind
        self.values: typing.Dict[tuple, float] = {}
        self.counts: typing.Dict[tuple, int] = {}
        self.sums: typing.Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def labels(self, *values: typing.Any, **kwargs: typing.Any) -> "_FallbackMetricChild":
        if values and kwargs:
            raise ValueError("metric labels must be supplied either positionally or by keyword")

        if kwargs:
            unknown_labels = set(kwargs).difference(self.labels_names)
            missing_labels = set(self.labels_names).difference(kwargs)
            if unknown_labels:
                raise ValueError(
                    f"unknown metric label names: {', '.join(sorted(unknown_labels))}"
                )
            if missing_labels:
                raise ValueError(
                    f"missing metric label names: {', '.join(sorted(missing_labels))}"
                )
            key = tuple(str(kwargs[name]) for name in self.labels_names)
        elif values:
            key = tuple(str(value) for value in values)
        else:
            key = ()

        if len(key) != len(self.labels_names):
            raise ValueError(
                f"incorrect metric label count: expected {len(self.labels_names)}, received {len(key)}"
            )

        return _FallbackMetricChild(self, key)

    def inc(self, amount: float = 1) -> None:
        self.labels().inc(amount)

    def set(self, value: float) -> None:
        self.labels().set(value)

    def observe(self, value: float) -> None:
        self.labels().observe(value)


class _FallbackMetricChild:
    def __init__(self, parent: _FallbackMetric, key: tuple) -> None:
        self.parent = parent
        self.key = key

    def inc(self, amount: float = 1) -> None:
        if self.parent.kind not in {"counter", "gauge"}:
            raise TypeError("increment is only supported for counter and gauge metrics")

        numeric = float(amount)
        if self.parent.kind == "counter" and numeric < 0:
            raise ValueError("counter metrics cannot be decreased")

        with self.parent._lock:
            self.parent.values[self.key] = self.parent.values.get(self.key, 0.0) + numeric

    def set(self, value: float) -> None:
        if self.parent.kind != "gauge":
            raise TypeError("set is only supported for gauge metrics")

        numeric = float(value)
        with self.parent._lock:
            self.parent.values[self.key] = numeric

    def observe(self, value: float) -> None:
        if self.parent.kind != "histogram":
            raise TypeError("observe is only supported for histogram metrics")

        numeric = float(value)
        with self.parent._lock:
            self.parent.counts[self.key] = self.parent.counts.get(self.key, 0) + 1
            self.parent.sums[self.key] = self.parent.sums.get(self.key, 0.0) + numeric
            self.parent.values[self.key] = numeric


_METRICS_REGISTRY = (
    prometheus_client.CollectorRegistry()
    if _HAS_PROMETHEUS and prometheus_client is not None
    else None
)


def _make_metric(
    name: str,
    description: str,
    labels: tuple = (),
    kind: str = "counter",
) -> typing.Any:
    if kind not in {"counter", "gauge", "histogram"}:
        raise ValueError(f"unsupported metric kind: {kind}")

    if _HAS_PROMETHEUS and prometheus_client is not None:
        label_names = list(labels)
        if kind == "gauge":
            return prometheus_client.Gauge(
                name,
                description,
                label_names,
                registry=_METRICS_REGISTRY,
            )
        if kind == "histogram":
            return prometheus_client.Histogram(
                name,
                description,
                label_names,
                registry=_METRICS_REGISTRY,
            )
        return prometheus_client.Counter(
            name,
            description,
            label_names,
            registry=_METRICS_REGISTRY,
        )

    return _FallbackMetric(name, description, labels, kind)


loop_iterations_total = _make_metric(
    "loop_iterations_total",
    "Agent loop iterations",
)
tool_calls_total = _make_metric(
    "tool_calls_total",
    "Tool calls executed",
)
model_calls_total = _make_metric(
    "model_calls_total",
    "Model generate calls",
)
errors_total = _make_metric(
    "errors_total",
    "Errors by type",
    ("type",),
)
checkpoints_total = _make_metric(
    "checkpoints_total",
    "Checkpoint writes",
)
checkpoint_duration_seconds = _make_metric(
    "checkpoint_duration_seconds",
    "Checkpoint write latency",
    (),
    "histogram",
)
memory_size_bytes = _make_metric(
    "memory_size_bytes",
    "Memory tier sizes",
    ("tier",),
    "gauge",
)
circuit_breaker_state = _make_metric(
    "circuit_breaker_state",
    "Circuit breaker state per dependency",
    ("dep",),
    "gauge",
)
heartbeats_total = _make_metric(
    "heartbeats_total",
    "Agent heartbeats",
)
restarts_total = _make_metric(
    "restarts_total",
    "Supervisor restarts",
)
livelocks_total = _make_metric(
    "livelocks_total",
    "Livelock detections",
)
retries_total = _make_metric(
    "retries_total",
    "Scheduler retries",
)
backoff_seconds_sum = _make_metric(
    "backoff_seconds_sum",
    "Total backoff seconds",
)
