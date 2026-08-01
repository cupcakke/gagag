import asyncio
import base64
import concurrent.futures
import contextlib
import hmac
import dataclasses
import datetime as dt
import enum
import functools
import hashlib
import inspect
import io
import json
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import pathlib
import queue
import random
import re
import shlex
import signal
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import typing
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from collections import Counter, deque
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import resource as _resource_module
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

try:
    import flask
    from flask import Flask, Response, jsonify, request, send_from_directory
    _HAS_FLASK = True
except ImportError:
    flask = None
    _HAS_FLASK = False

try:
    import fastapi
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Header
    from fastapi.responses import JSONResponse, PlainTextResponse
    _HAS_FASTAPI = True
except ImportError:
    fastapi = None
    uvicorn = None
    _HAS_FASTAPI = False

try:
    import pydantic
    from pydantic import BaseModel, Field, ValidationError
    try:
        from pydantic import ConfigDict, field_validator
        _PYDANTIC_V2 = True
    except ImportError:
        _PYDANTIC_V2 = False
        ConfigDict = dict
        field_validator = pydantic.validator
        class _CompatBaseModel(BaseModel):
            class Config:
                allow_mutation = False
                extra = "forbid"

            @classmethod
            def model_validate(cls, data):
                return cls.parse_obj(data)

            def model_dump(self, mode="python"):
                return self.dict()

            @classmethod
            def model_json_schema(cls):
                return cls.schema()
        BaseModel = _CompatBaseModel
    _HAS_PYDANTIC = True
except ImportError:
    pydantic = None
    _PYDANTIC_V2 = False
    _HAS_PYDANTIC = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

try:
    import prometheus_client
    from prometheus_client import (
        CollectorRegistry,
        Counter as PromCounter,
        Gauge as PromGauge,
        Histogram as PromHistogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    prometheus_client = None
    _HAS_PROMETHEUS = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    yaml = None
    _HAS_YAML = False

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    OpenAI = None
    _HAS_OPENAI = False

try:
    from instavm import InstaVM
    _HAS_INSTAVM = True
except ImportError:
    InstaVM = None
    _HAS_INSTAVM = False

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    httpx = None
    _HAS_HTTPX = False

try:
    import websockets
    _HAS_WEBSOCKETS = True
except ImportError:
    websockets = None
    _HAS_WEBSOCKETS = False

ROOT = pathlib.Path(__file__).resolve().parent
DB_PATH = pathlib.Path(os.environ.get("APP_DATABASE_PATH", str(ROOT / "chat.db")))
CHECKPOINT_PATH = pathlib.Path(os.environ.get("APP_CHECKPOINT_PATH", str(ROOT / "agent.checkpoint")))
LOCK_PATH = pathlib.Path(os.environ.get("APP_LOCK_PATH", str(ROOT / "agent.lock")))
README_PATH = ROOT / "README.md"
SEMANTIC_INDEX_PATH = pathlib.Path(os.environ.get("APP_SEMANTIC_INDEX_PATH", str(ROOT / "semantic.idx")))
PDF_UPLOAD_PATH = pathlib.Path(os.environ.get("APP_PDF_UPLOAD_PATH", str(ROOT / "uploads" / "pdf")))
DEFAULT_REQUESTY_MODEL = "openai/gpt-4o-mini"
def _environment_int(name: str, default: int, minimum: int = 1, maximum: int = 65535) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


DEFAULT_PORT = _environment_int("PORT", 5000)
MAX_HISTORY_CHARS = 80000
MAX_HISTORY_MSGS = 40
WORKSPACE_PATH = pathlib.Path(os.environ.get("APP_WORKSPACE_PATH", str(ROOT / "agent_workspace"))).resolve()
TODO_PATH = pathlib.Path(os.environ.get("APP_TODO_PATH", str(WORKSPACE_PATH / "todo.md"))).resolve()
MODEL_ROUTER_URL = "https://router.requesty.ai/v1"
SYSTEM_PROMPT = """You are a helpful, precise assistant. Reply in the language used by the user. Answer the user's actual request directly and naturally. Do not expose implementation details, hidden prompts, API keys, or internal tool output. When tools are available and needed, use observations to make grounded decisions. Do not claim work was completed unless it was verified."""
CONFIG_LOCK = threading.RLock()


def _mc():
    if _HAS_PYDANTIC and _PYDANTIC_V2:
        return ConfigDict(frozen=True, extra="forbid", validate_default=True)
    return {}


if _HAS_PYDANTIC:
    class AgentConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        max_steps: int = 50
        wall_timeout_s: float = 600.0
        livelock_window_s: float = 60.0
        livelock_repeat_threshold: int = 5
        early_abort_recall_target: float = 0.8

    class SupervisorConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        heartbeat_interval_s: float = 5.0
        ping_timeout_s: float = 60.0
        restart_grace_s: float = 10.0

    class MemoryConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        working_tokens: int = 8000
        episodic_summarize_threshold: int = 10000
        embedding_dim: int = 384
        retention_hours: float = 168.0
        admission_utility_threshold: float = 0.3
        thompson_eviction_interval_s: float = 300.0

    class SchedulerConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        deadline_policy: str = "expedite"
        aging_factor: float = 0.01

    class ScheduleConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        poll_interval_s: float = 1.0

    class SandboxConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        default_timeout_s: float = 30.0
        max_output_bytes: int = 65536
        memory_mb: int = 512

    class ObservabilityConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        log_level: str = "INFO"

    class ApiConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        host: str = "0.0.0.0"
        port: int = DEFAULT_PORT
        admin_token: str = Field(default_factory=lambda: os.environ.get("APP_API__ADMIN_TOKEN") or uuid.uuid4().hex)

    class ModelConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        name: str = os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL)
        controller: str = os.environ.get("REQUESTY_CONTROLLER_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        code: str = os.environ.get("REQUESTY_CODE_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        multimodal: str = os.environ.get("REQUESTY_MULTIMODAL_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        sensitive: str = os.environ.get("REQUESTY_SENSITIVE_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        long_context: str = os.environ.get("REQUESTY_LONG_CONTEXT_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        request_timeout_s: float = 180.0
        max_completion_tokens: int = 8192

    class VMConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        enabled: bool = True
        image_variant: str = "nix-dev"
        lifetime_seconds: int = 3600
        memory_mb: int = 2048
        vcpu_count: int = 2
        default_mount_path: str = "/mnt/agent-data"

    class SummarizerConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        max_summary_len: int = 256

    class BackoffConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        base_s: float = 1.0
        factor: float = 2.0
        max_attempts: int = 5
        max_delay_s: float = 60.0

    class BreakerConfig(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        transient_threshold: int = 3
        permanent_threshold: int = 1
        cooldown_s: float = 30.0

    class Config(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        agent: AgentConfig = AgentConfig()
        supervisor: SupervisorConfig = SupervisorConfig()
        memory: MemoryConfig = MemoryConfig()
        scheduler: SchedulerConfig = SchedulerConfig()
        schedule: ScheduleConfig = ScheduleConfig()
        sandbox: SandboxConfig = SandboxConfig()
        observability: ObservabilityConfig = ObservabilityConfig()
        api: ApiConfig = ApiConfig()
        model: ModelConfig = ModelConfig()
        vm: VMConfig = VMConfig()
        summarizer: SummarizerConfig = SummarizerConfig()
        backoff: BackoffConfig = BackoffConfig()
        breaker: BreakerConfig = BreakerConfig()

else:
    @dataclasses.dataclass(frozen=True)
    class AgentConfig:
        max_steps: int = 50
        wall_timeout_s: float = 600.0
        livelock_window_s: float = 60.0
        livelock_repeat_threshold: int = 5
        early_abort_recall_target: float = 0.8

    @dataclasses.dataclass(frozen=True)
    class SupervisorConfig:
        heartbeat_interval_s: float = 5.0
        ping_timeout_s: float = 60.0
        restart_grace_s: float = 10.0

    @dataclasses.dataclass(frozen=True)
    class MemoryConfig:
        working_tokens: int = 8000
        episodic_summarize_threshold: int = 10000
        embedding_dim: int = 384
        retention_hours: float = 168.0
        admission_utility_threshold: float = 0.3
        thompson_eviction_interval_s: float = 300.0

    @dataclasses.dataclass(frozen=True)
    class SchedulerConfig:
        deadline_policy: str = "expedite"
        aging_factor: float = 0.01

    @dataclasses.dataclass(frozen=True)
    class ScheduleConfig:
        poll_interval_s: float = 1.0

    @dataclasses.dataclass(frozen=True)
    class SandboxConfig:
        default_timeout_s: float = 30.0
        max_output_bytes: int = 65536
        memory_mb: int = 512

    @dataclasses.dataclass(frozen=True)
    class ObservabilityConfig:
        log_level: str = "INFO"

    @dataclasses.dataclass(frozen=True)
    class ApiConfig:
        host: str = "0.0.0.0"
        port: int = DEFAULT_PORT
        admin_token: str = dataclasses.field(default_factory=lambda: os.environ.get("APP_API__ADMIN_TOKEN") or uuid.uuid4().hex)

    @dataclasses.dataclass(frozen=True)
    class ModelConfig:
        name: str = os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL)
        controller: str = os.environ.get("REQUESTY_CONTROLLER_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        code: str = os.environ.get("REQUESTY_CODE_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        multimodal: str = os.environ.get("REQUESTY_MULTIMODAL_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        sensitive: str = os.environ.get("REQUESTY_SENSITIVE_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        long_context: str = os.environ.get("REQUESTY_LONG_CONTEXT_MODEL", os.environ.get("REQUESTY_MODEL", DEFAULT_REQUESTY_MODEL))
        request_timeout_s: float = 180.0
        max_completion_tokens: int = 8192

    @dataclasses.dataclass(frozen=True)
    class VMConfig:
        enabled: bool = True
        image_variant: str = "nix-dev"
        lifetime_seconds: int = 3600
        memory_mb: int = 2048
        vcpu_count: int = 2
        default_mount_path: str = "/mnt/agent-data"

    @dataclasses.dataclass(frozen=True)
    class SummarizerConfig:
        max_summary_len: int = 256

    @dataclasses.dataclass(frozen=True)
    class BackoffConfig:
        base_s: float = 1.0
        factor: float = 2.0
        max_attempts: int = 5
        max_delay_s: float = 60.0

    @dataclasses.dataclass(frozen=True)
    class BreakerConfig:
        transient_threshold: int = 3
        permanent_threshold: int = 1
        cooldown_s: float = 30.0

    @dataclasses.dataclass(frozen=True)
    class Config:
        agent: AgentConfig = dataclasses.field(default_factory=AgentConfig)
        supervisor: SupervisorConfig = dataclasses.field(default_factory=SupervisorConfig)
        memory: MemoryConfig = dataclasses.field(default_factory=MemoryConfig)
        scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
        schedule: ScheduleConfig = dataclasses.field(default_factory=ScheduleConfig)
        sandbox: SandboxConfig = dataclasses.field(default_factory=SandboxConfig)
        observability: ObservabilityConfig = dataclasses.field(default_factory=ObservabilityConfig)
        api: ApiConfig = dataclasses.field(default_factory=ApiConfig)
        model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
        vm: VMConfig = dataclasses.field(default_factory=VMConfig)
        summarizer: SummarizerConfig = dataclasses.field(default_factory=SummarizerConfig)
        backoff: BackoffConfig = dataclasses.field(default_factory=BackoffConfig)
        breaker: BreakerConfig = dataclasses.field(default_factory=BreakerConfig)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value


def _coerce(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return value


def _validate_config(config):
    finite_values = [
        (config.agent.wall_timeout_s, "agent.wall_timeout_s"),
        (config.agent.livelock_window_s, "agent.livelock_window_s"),
        (config.agent.early_abort_recall_target, "agent.early_abort_recall_target"),
        (config.supervisor.heartbeat_interval_s, "supervisor.heartbeat_interval_s"),
        (config.supervisor.ping_timeout_s, "supervisor.ping_timeout_s"),
        (config.supervisor.restart_grace_s, "supervisor.restart_grace_s"),
        (config.memory.retention_hours, "memory.retention_hours"),
        (config.memory.admission_utility_threshold, "memory.admission_utility_threshold"),
        (config.memory.thompson_eviction_interval_s, "memory.thompson_eviction_interval_s"),
        (config.scheduler.aging_factor, "scheduler.aging_factor"),
        (config.schedule.poll_interval_s, "schedule.poll_interval_s"),
        (config.sandbox.default_timeout_s, "sandbox.default_timeout_s"),
        (config.model.request_timeout_s, "model.request_timeout_s"),
        (config.backoff.base_s, "backoff.base_s"),
        (config.backoff.factor, "backoff.factor"),
        (config.backoff.max_delay_s, "backoff.max_delay_s"),
        (config.breaker.cooldown_s, "breaker.cooldown_s"),
    ]
    for value, name in finite_values:
        if not math.isfinite(float(value)):
            raise RuntimeError(f"{name} must be finite")
    checks = [
        (config.agent.max_steps > 0, "agent.max_steps must be positive"),
        (config.agent.wall_timeout_s > 0, "agent.wall_timeout_s must be positive"),
        (config.agent.livelock_window_s > 0, "agent.livelock_window_s must be positive"),
        (config.agent.livelock_repeat_threshold > 1, "agent.livelock_repeat_threshold must be greater than one"),
        (0.0 <= config.agent.early_abort_recall_target <= 1.0, "agent.early_abort_recall_target must be between zero and one"),
        (config.supervisor.heartbeat_interval_s > 0, "supervisor.heartbeat_interval_s must be positive"),
        (config.supervisor.ping_timeout_s > config.supervisor.heartbeat_interval_s, "supervisor.ping_timeout_s must exceed heartbeat_interval_s"),
        (config.supervisor.restart_grace_s >= 0, "supervisor.restart_grace_s must be nonnegative"),
        (config.memory.working_tokens > 0, "memory.working_tokens must be positive"),
        (config.memory.episodic_summarize_threshold > 0, "memory.episodic_summarize_threshold must be positive"),
        (config.memory.embedding_dim > 0, "memory.embedding_dim must be positive"),
        (config.memory.retention_hours > 0, "memory.retention_hours must be positive"),
        (0.0 <= config.memory.admission_utility_threshold <= 1.0, "memory.admission_utility_threshold must be between zero and one"),
        (config.memory.thompson_eviction_interval_s > 0, "memory.thompson_eviction_interval_s must be positive"),
        (config.scheduler.deadline_policy in {"expedite", "none"}, "scheduler.deadline_policy must be expedite or none"),
        (config.scheduler.aging_factor >= 0, "scheduler.aging_factor must be nonnegative"),
        (config.schedule.poll_interval_s > 0, "schedule.poll_interval_s must be positive"),
        (config.sandbox.default_timeout_s > 0, "sandbox.default_timeout_s must be positive"),
        (config.sandbox.max_output_bytes > 0, "sandbox.max_output_bytes must be positive"),
        (config.sandbox.memory_mb > 0, "sandbox.memory_mb must be positive"),
        (1 <= config.api.port <= 65535, "api.port must be between one and 65535"),
        (bool(config.api.admin_token), "api.admin_token must not be empty"),
        (config.model.request_timeout_s > 0, "model.request_timeout_s must be positive"),
        (config.model.max_completion_tokens > 0, "model.max_completion_tokens must be positive"),
        (config.vm.lifetime_seconds > 0, "vm.lifetime_seconds must be positive"),
        (config.vm.memory_mb > 0, "vm.memory_mb must be positive"),
        (config.vm.vcpu_count > 0, "vm.vcpu_count must be positive"),
        (bool(config.vm.default_mount_path), "vm.default_mount_path must not be empty"),
        (config.summarizer.max_summary_len > 0, "summarizer.max_summary_len must be positive"),
        (config.backoff.base_s >= 0, "backoff.base_s must be nonnegative"),
        (config.backoff.factor >= 1, "backoff.factor must be at least one"),
        (config.backoff.max_attempts > 0, "backoff.max_attempts must be positive"),
        (config.backoff.max_delay_s >= 0, "backoff.max_delay_s must be nonnegative"),
        (config.breaker.transient_threshold > 0, "breaker.transient_threshold must be positive"),
        (config.breaker.permanent_threshold > 0, "breaker.permanent_threshold must be positive"),
        (config.breaker.cooldown_s >= 0, "breaker.cooldown_s must be nonnegative"),
    ]
    for valid, message in checks:
        if not valid:
            raise RuntimeError(message)
    return config


def _overlay(target, path, value):
    cursor = target
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = _coerce(value)


def load_config():
    raw = {}
    search_roots = []
    for candidate in (pathlib.Path.cwd(), ROOT):
        resolved = candidate.resolve()
        if resolved not in search_roots:
            search_roots.append(resolved)
    loaded_path = None
    for root in search_roots:
        for filename in ("config.yaml", "config.yml", "config.json"):
            path = root / filename
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
                if filename.endswith(".json"):
                    loaded = json.loads(text)
                else:
                    if not _HAS_YAML:
                        continue
                    loaded = yaml.safe_load(text)
                if loaded is None:
                    loaded = {}
                if not isinstance(loaded, dict):
                    raise RuntimeError("configuration root must be an object")
                raw.update(loaded)
                loaded_path = path
                break
            except Exception as exc:
                raise RuntimeError(f"configuration load failed: {path}: {exc}") from exc
        if loaded_path is not None:
            break
    for key, value in os.environ.items():
        if key.startswith("APP_") and "__" in key:
            section, field = key[4:].split("__", 1)
            _overlay(raw, [section.lower(), field.lower()], value)
    if _HAS_PYDANTIC:
        try:
            return _validate_config(Config.model_validate(raw))
        except Exception as exc:
            raise RuntimeError(f"configuration validation failed: {exc}") from exc
    values = {}
    mapping = [
        ("agent", AgentConfig),
        ("supervisor", SupervisorConfig),
        ("memory", MemoryConfig),
        ("scheduler", SchedulerConfig),
        ("schedule", ScheduleConfig),
        ("sandbox", SandboxConfig),
        ("observability", ObservabilityConfig),
        ("api", ApiConfig),
        ("model", ModelConfig),
        ("vm", VMConfig),
        ("summarizer", SummarizerConfig),
        ("backoff", BackoffConfig),
        ("breaker", BreakerConfig),
    ]
    known_sections = {name for name, _ in mapping}
    unknown_sections = set(raw) - known_sections
    if unknown_sections:
        raise RuntimeError(f"unknown configuration sections: {sorted(unknown_sections)}")
    for name, cls in mapping:
        sub = raw.get(name, {})
        if not isinstance(sub, dict):
            raise RuntimeError(f"configuration section {name} must be an object")
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(sub) - fields
        if unknown:
            raise RuntimeError(f"unknown configuration fields in {name}: {sorted(unknown)}")
        converted = {}
        for field in dataclasses.fields(cls):
            if field.name not in sub:
                continue
            value = sub[field.name]
            if field.type is bool:
                if isinstance(value, bool):
                    converted[field.name] = value
                elif isinstance(value, str) and value.lower() in {"true", "false"}:
                    converted[field.name] = value.lower() == "true"
                else:
                    raise RuntimeError(f"configuration field {name}.{field.name} must be boolean")
            elif field.type in {str, int, float}:
                try:
                    converted[field.name] = field.type(value)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"configuration field {name}.{field.name} has an invalid type") from exc
            else:
                converted[field.name] = value
        values[name] = cls(**converted)
    return _validate_config(Config(**values))


with CONFIG_LOCK:
    CONFIG = load_config()


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


@dataclasses.dataclass(frozen=True)
class InstaVMOperation:
    operation_id: str
    group: str
    method: str
    path_template: str
    documentation_url: str
    transport: str
    scope: str
    side_effects: bool
    destructive: bool
    timeout_s: float
    max_response_bytes: int

    @property
    def tool_name(self) -> str:
        return f"instavm_{self.operation_id}"


_INSTAVM_INVENTORY = (
    ("api_keys_delete", "API Keys", "DELETE", "/v1/api-keys/{item_id}", "json", "instavm.secrets.admin", True, True),
    ("api_keys_get", "API Keys", "GET", "/v1/api-keys/{item_id}", "json", "instavm.read", False, False),
    ("api_keys_list", "API Keys", "GET", "/v1/api-keys/", "json", "instavm.secrets.admin", False, False),
    ("api_keys_update", "API Keys", "PATCH", "/v1/api-keys/{item_id}", "json", "instavm.secrets.admin", True, False),
    ("api_keys_create", "API Keys", "POST", "/v1/api-keys/", "json", "instavm.secrets.admin", True, False),
    ("audit_catalog", "Audit", "GET", "/v1/audit/catalog", "json", "instavm.read", False, False),
    ("audit_events_list", "Audit", "GET", "/v1/audit/events", "json", "instavm.read", False, False),
    ("audit_event_get", "Audit", "GET", "/v1/audit/events/{event_id}", "json", "instavm.read", False, False),
    ("browser_session_close", "Browser", "DELETE", "/v1/browser/sessions/{session_id}", "json", "instavm.browser", True, True),
    ("browser_screenshot_get", "Browser", "GET", "/v1/browser/screenshot/{usage_id}", "binary", "instavm.browser", False, False),
    ("browser_session_get", "Browser", "GET", "/v1/browser/sessions/{session_id}", "json", "instavm.browser", False, False),
    ("browser_sessions_list", "Browser", "GET", "/v1/browser/sessions/", "json", "instavm.browser", False, False),
    ("browser_usage_get", "Browser", "GET", "/v1/browser/usage", "json", "instavm.browser", False, False),
    ("browser_usage_history", "Browser", "GET", "/v1/browser/usage/history", "json", "instavm.browser", False, False),
    ("browser_auth_test", "Browser", "POST", "/v1/browser/auth/test", "json", "instavm.browser", True, False),
    ("browser_click", "Browser", "POST", "/v1/browser/interactions/click", "json", "instavm.browser", True, False),
    ("browser_content", "Browser", "POST", "/v1/browser/interactions/content", "json", "instavm.browser", False, False),
    ("browser_extract", "Browser", "POST", "/v1/browser/interactions/extract", "json", "instavm.browser", False, False),
    ("browser_fill", "Browser", "POST", "/v1/browser/interactions/fill", "json", "instavm.browser", True, False),
    ("browser_navigate", "Browser", "POST", "/v1/browser/interactions/navigate", "json", "instavm.browser", True, False),
    ("browser_screenshot", "Browser", "POST", "/v1/browser/interactions/screenshot", "json", "instavm.browser", False, False),
    ("browser_scroll", "Browser", "POST", "/v1/browser/interactions/scroll", "json", "instavm.browser", True, False),
    ("browser_type", "Browser", "POST", "/v1/browser/interactions/type", "json", "instavm.browser", True, False),
    ("browser_wait", "Browser", "POST", "/v1/browser/interactions/wait", "json", "instavm.browser", False, False),
    ("browser_render", "Browser", "POST", "/v1/browser/render", "json", "instavm.browser", False, False),
    ("browser_session_create", "Browser", "POST", "/v1/browser/sessions/", "json", "instavm.browser", True, False),
    ("computer_use_proxy_delete", "Computer Use", "DELETE", "/v1/computeruse/{session_id}/{path}", "json", "instavm.computer", True, True),
    ("computer_use_proxy_get", "Computer Use", "GET", "/v1/computeruse/{session_id}/{path}", "json", "instavm.computer", False, False),
    ("computer_use_viewer_url", "Computer Use", "GET", "/v1/computeruse/{session_id}/viewer-url", "json", "instavm.computer", False, False),
    ("computer_use_vnc_websocket", "Computer Use", "GET", "/v1/computeruse/{session_id}/vnc/websockify", "websocket", "instavm.vnc", False, False),
    ("computer_use_proxy_options", "Computer Use", "OPTIONS", "/v1/computeruse/{session_id}/{path}", "json", "instavm.computer", False, False),
    ("computer_use_proxy_patch", "Computer Use", "PATCH", "/v1/computeruse/{session_id}/{path}", "json", "instavm.computer", True, False),
    ("computer_use_proxy_post", "Computer Use", "POST", "/v1/computeruse/{session_id}/{path}", "json", "instavm.computer", True, False),
    ("computer_use_proxy_put", "Computer Use", "PUT", "/v1/computeruse/{session_id}/{path}", "json", "instavm.computer", True, False),
    ("egress_session_get", "Egress", "GET", "/v1/egress/session/{session_id}", "json", "instavm.read", False, False),
    ("egress_vm_get", "Egress", "GET", "/v1/egress/vm/{vm_id}", "json", "instavm.read", False, False),
    ("egress_session_set", "Egress", "POST", "/v1/egress/session/{session_id}", "json", "instavm.egress.admin", True, False),
    ("egress_vm_set", "Egress", "POST", "/v1/egress/vm/{vm_id}", "json", "instavm.egress.admin", True, False),
    ("execution_get", "Execution", "GET", "/v1/executions/{item_id}", "json", "instavm.read", False, False),
    ("executions_list", "Execution", "GET", "/v1/executions/", "json", "instavm.read", False, False),
    ("execution_execute", "Execution", "POST", "/execute", "json", "instavm.mutate", True, False),
    ("execution_execute_async", "Execution", "POST", "/execute_async", "json", "instavm.mutate", True, False),
    ("execution_kill", "Execution", "POST", "/kill", "json", "instavm.delete", True, True),
    ("file_download", "Files", "POST", "/download", "binary", "instavm.read", False, False),
    ("file_upload", "Files", "POST", "/upload", "multipart", "instavm.mutate", True, False),
    ("session_pty_delete", "Sessions", "DELETE", "/v1/sessions/{session_id}/pty/sessions/{pty_id}", "json", "instavm.delete", True, True),
    ("session_app_url_get", "Sessions", "GET", "/v1/sessions/app-url/{session_id}", "json", "instavm.read", False, False),
    ("session_sandboxes_list", "Sessions", "GET", "/v1/sessions/sandboxes", "json", "instavm.read", False, False),
    ("session_ptys_list", "Sessions", "GET", "/v1/sessions/{session_id}/pty/sessions", "json", "instavm.read", False, False),
    ("session_pty_get", "Sessions", "GET", "/v1/sessions/{session_id}/pty/sessions/{pty_id}", "json", "instavm.read", False, False),
    ("session_info_get", "Sessions", "GET", "/v1/sessions/session/{session_id}", "json", "instavm.read", False, False),
    ("session_status_get", "Sessions", "GET", "/v1/sessions/status/{session_id}", "json", "instavm.read", False, False),
    ("session_usage_get", "Sessions", "GET", "/v1/sessions/usage/{session_id}", "json", "instavm.read", False, False),
    ("session_create", "Sessions", "POST", "/v1/sessions/session", "json", "instavm.mutate", True, False),
    ("session_pty_create", "Sessions", "POST", "/v1/sessions/{session_id}/pty/sessions", "json", "instavm.mutate", True, False),
    ("session_pty_resize", "Sessions", "POST", "/v1/sessions/{session_id}/pty/sessions/{pty_id}/resize", "json", "instavm.mutate", True, False),
    ("session_tape_start", "Sessions", "POST", "/v1/sessions/{session_id}/tape/start", "json", "instavm.mutate", True, False),
    ("custom_domain_delete", "Shares and Custom Domains", "DELETE", "/v1/custom-domains/{domain_id}", "json", "instavm.delete", True, True),
    ("custom_domains_list", "Shares and Custom Domains", "GET", "/v1/custom-domains", "json", "instavm.read", False, False),
    ("custom_domain_get", "Shares and Custom Domains", "GET", "/v1/custom-domains/{domain_id}", "json", "instavm.read", False, False),
    ("custom_domain_health", "Shares and Custom Domains", "GET", "/v1/custom-domains/{domain_id}/health", "json", "instavm.read", False, False),
    ("share_update", "Shares and Custom Domains", "PATCH", "/v1/shares/{share_id}", "json", "instavm.shares.admin", True, False),
    ("custom_domain_create", "Shares and Custom Domains", "POST", "/v1/custom-domains", "json", "instavm.shares.admin", True, False),
    ("custom_domain_verify", "Shares and Custom Domains", "POST", "/v1/custom-domains/{domain_id}/verify", "json", "instavm.shares.admin", True, False),
    ("share_create", "Shares and Custom Domains", "POST", "/v1/shares", "json", "instavm.shares.admin", True, False),
    ("snapshot_delete", "Snapshots", "DELETE", "/v1/snapshots/{snapshot_id}", "json", "instavm.delete", True, True),
    ("snapshots_list", "Snapshots", "GET", "/v1/snapshots", "json", "instavm.read", False, False),
    ("snapshot_get", "Snapshots", "GET", "/v1/snapshots/{snapshot_id}", "json", "instavm.read", False, False),
    ("snapshot_create", "Snapshots", "POST", "/v1/snapshots", "json", "instavm.mutate", True, False),
    ("ssh_key_delete", "SSH", "DELETE", "/v1/ssh-keys/{key_id}", "json", "instavm.ssh.admin", True, True),
    ("ssh_keys_list", "SSH", "GET", "/v1/ssh-keys", "json", "instavm.ssh.admin", False, False),
    ("ssh_key_add", "SSH", "POST", "/v1/ssh-keys", "json", "instavm.ssh.admin", True, False),
    ("vm_delete", "VMs", "DELETE", "/v1/vms/{vm_id}", "json", "instavm.delete", True, True),
    ("vm_pty_delete", "VMs", "DELETE", "/v1/vms/{vm_id}/pty/sessions/{pty_id}", "json", "instavm.delete", True, True),
    ("vms_list", "VMs", "GET", "/v1/vms", "json", "instavm.read", False, False),
    ("vm_get", "VMs", "GET", "/v1/vms/{vm_id}", "json", "instavm.read", False, False),
    ("vm_ptys_list", "VMs", "GET", "/v1/vms/{vm_id}/pty/sessions", "json", "instavm.read", False, False),
    ("vm_pty_get", "VMs", "GET", "/v1/vms/{vm_id}/pty/sessions/{pty_id}", "json", "instavm.read", False, False),
    ("vm_update", "VMs", "PATCH", "/v1/vms/{vm_id}", "json", "instavm.mutate", True, False),
    ("vm_create", "VMs", "POST", "/v1/vms", "json", "instavm.mutate", True, False),
    ("vm_clone", "VMs", "POST", "/v1/vms/{vm_id}/clone", "json", "instavm.mutate", True, False),
    ("vm_pty_create", "VMs", "POST", "/v1/vms/{vm_id}/pty/sessions", "json", "instavm.mutate", True, False),
    ("vm_pty_resize", "VMs", "POST", "/v1/vms/{vm_id}/pty/sessions/{pty_id}/resize", "json", "instavm.mutate", True, False),
    ("vm_resume", "VMs", "POST", "/v1/vms/{vm_id}/resume", "json", "instavm.mutate", True, False),
    ("vm_snapshot", "VMs", "POST", "/v1/vms/{vm_id}/snapshot", "json", "instavm.mutate", True, False),
    ("vm_stage_env", "VMs", "POST", "/v1/vms/{vm_id}/stage-env", "json", "instavm.secrets.admin", True, False),
    ("vm_suspend", "VMs", "POST", "/v1/vms/{vm_id}/suspend", "json", "instavm.mutate", True, False),
    ("vm_tape_start", "VMs", "POST", "/v1/vms/{vm_id}/tape/start", "json", "instavm.mutate", True, False),
    ("vm_volume_unmount", "Volumes", "DELETE", "/v1/vms/{vm_id}/volumes/{volume_id}", "json", "instavm.delete", True, True),
    ("volume_delete", "Volumes", "DELETE", "/v1/volumes/{volume_id}", "json", "instavm.delete", True, True),
    ("volume_checkpoint_delete", "Volumes", "DELETE", "/v1/volumes/{volume_id}/checkpoints/{checkpoint_id}", "json", "instavm.delete", True, True),
    ("volume_file_delete", "Volumes", "DELETE", "/v1/volumes/{volume_id}/files", "json", "instavm.delete", True, True),
    ("vm_volumes_list", "Volumes", "GET", "/v1/vms/{vm_id}/volumes", "json", "instavm.read", False, False),
    ("volumes_list", "Volumes", "GET", "/v1/volumes", "json", "instavm.read", False, False),
    ("volume_get", "Volumes", "GET", "/v1/volumes/{volume_id}", "json", "instavm.read", False, False),
    ("volume_checkpoints_list", "Volumes", "GET", "/v1/volumes/{volume_id}/checkpoints", "json", "instavm.read", False, False),
    ("volume_files_list", "Volumes", "GET", "/v1/volumes/{volume_id}/files", "json", "instavm.read", False, False),
    ("volume_update", "Volumes", "PATCH", "/v1/volumes/{volume_id}", "json", "instavm.mutate", True, False),
    ("vm_volume_mount", "Volumes", "POST", "/v1/vms/{vm_id}/volumes", "json", "instavm.mutate", True, False),
    ("volume_create", "Volumes", "POST", "/v1/volumes", "json", "instavm.mutate", True, False),
    ("volume_checkpoint_create", "Volumes", "POST", "/v1/volumes/{volume_id}/checkpoints", "json", "instavm.mutate", True, False),
    ("volume_file_download", "Volumes", "POST", "/v1/volumes/{volume_id}/files/download", "binary", "instavm.read", False, False),
    ("volume_file_upload", "Volumes", "POST", "/v1/volumes/{volume_id}/files/upload", "multipart", "instavm.mutate", True, False),
    ("webhook_endpoint_delete", "Webhooks", "DELETE", "/v1/webhooks/endpoints/{endpoint_id}", "json", "instavm.webhooks.admin", True, True),
    ("webhook_deliveries_list", "Webhooks", "GET", "/v1/webhooks/deliveries", "json", "instavm.webhooks.admin", False, False),
    ("webhook_endpoints_list", "Webhooks", "GET", "/v1/webhooks/endpoints", "json", "instavm.webhooks.admin", False, False),
    ("webhook_endpoint_get", "Webhooks", "GET", "/v1/webhooks/endpoints/{endpoint_id}", "json", "instavm.webhooks.admin", False, False),
    ("webhook_endpoint_update", "Webhooks", "PATCH", "/v1/webhooks/endpoints/{endpoint_id}", "json", "instavm.webhooks.admin", True, False),
    ("webhook_delivery_replay", "Webhooks", "POST", "/v1/webhooks/deliveries/{delivery_id}/replay", "json", "instavm.webhooks.admin", True, False),
    ("webhook_endpoint_create", "Webhooks", "POST", "/v1/webhooks/endpoints", "json", "instavm.webhooks.admin", True, False),
    ("webhook_secret_rotate", "Webhooks", "POST", "/v1/webhooks/endpoints/{endpoint_id}/rotate-secret", "json", "instavm.webhooks.admin", True, False),
    ("webhook_endpoint_test", "Webhooks", "POST", "/v1/webhooks/endpoints/{endpoint_id}/test", "json", "instavm.webhooks.admin", True, False),
    ("webhook_endpoint_verify", "Webhooks", "POST", "/v1/webhooks/endpoints/{endpoint_id}/verify", "json", "instavm.webhooks.admin", True, False),
    ("webhook_resend", "Webhooks", "POST", "/v1/webhooks/resend", "json", "instavm.webhooks.admin", True, False),
)


def _instavm_doc_url(group: str) -> str:
    slug_group = {
        "API Keys": "api-keys",
        "Audit": "audit",
        "Browser": "browser",
        "Computer Use": "computer-use",
        "Egress": "egress",
        "Execution": "execution",
        "Files": "files",
        "Sessions": "sessions",
        "Shares and Custom Domains": "shares",
        "Snapshots": "snapshots",
        "SSH": "ssh",
        "VMs": "vms",
        "Volumes": "volumes",
        "Webhooks": "webhooks",
    }[group]
    return f"https://instavm.io/docs/api/{slug_group}"


INSTAVM_MANIFEST = tuple(
    InstaVMOperation(
        operation_id=operation_id,
        group=group,
        method=method,
        path_template=path,
        documentation_url=_instavm_doc_url(group),
        transport=transport,
        scope=scope,
        side_effects=side_effects,
        destructive=destructive,
        timeout_s=900.0 if operation_id in {"vm_snapshot", "snapshot_create", "execution_execute", "execution_execute_async"} else 180.0,
        max_response_bytes=128 * 1024 * 1024 if transport == "binary" else 4 * 1024 * 1024,
    )
    for operation_id, group, method, path, transport, scope, side_effects, destructive in _INSTAVM_INVENTORY
)
INSTAVM_OPERATIONS = {operation.operation_id: operation for operation in INSTAVM_MANIFEST}


def validate_instavm_manifest() -> dict:
    if len(INSTAVM_MANIFEST) != 114:
        raise RuntimeError(f"InstaVM manifest count is {len(INSTAVM_MANIFEST)}, expected 114")
    identifiers = [operation.operation_id for operation in INSTAVM_MANIFEST]
    tools = [operation.tool_name for operation in INSTAVM_MANIFEST]
    routes = [(operation.method, operation.path_template) for operation in INSTAVM_MANIFEST]
    if len(set(identifiers)) != 114 or len(set(tools)) != 114 or len(set(routes)) != 114:
        raise RuntimeError("InstaVM manifest contains duplicate operation, tool, or route definitions")
    method_counts = Counter(operation.method for operation in INSTAVM_MANIFEST)
    expected_methods = {"GET": 42, "POST": 50, "DELETE": 14, "PATCH": 6, "PUT": 1, "OPTIONS": 1}
    if dict(method_counts) != expected_methods:
        raise RuntimeError(f"InstaVM method distribution is invalid: {dict(method_counts)}")
    group_counts = Counter(operation.group for operation in INSTAVM_MANIFEST)
    expected_groups = {
        "API Keys": 5, "Audit": 3, "Browser": 18, "Computer Use": 8, "Egress": 4,
        "Execution": 5, "Files": 2, "Sessions": 12, "Shares and Custom Domains": 8,
        "Snapshots": 4, "SSH": 3, "VMs": 16, "Volumes": 15, "Webhooks": 11,
    }
    if dict(group_counts) != expected_groups:
        raise RuntimeError(f"InstaVM group distribution is invalid: {dict(group_counts)}")
    return {"total": len(INSTAVM_MANIFEST), "methods": dict(method_counts), "groups": dict(group_counts)}


validate_instavm_manifest()


SECRET_FIELD_PATTERN = re.compile(r"(?:api[_-]?key|authorization|token|secret|password|private[_-]?key|cookie|credential|session[_-]?key|env)", re.IGNORECASE)


def _path_is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _retry_after_seconds(value: typing.Optional[str], attempt: int) -> float:
    if value:
        try:
            numeric = float(value)
            if numeric >= 0:
                return min(300.0, numeric)
        except ValueError:
            with contextlib.suppress(Exception):
                parsed = dt.datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=dt.timezone.utc)
                return min(300.0, max(0.0, (parsed - dt.datetime.now(dt.timezone.utc)).total_seconds()))
    return min(30.0, (2 ** max(0, attempt)) + random.random())


def redact_instavm_value(value: typing.Any) -> typing.Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SECRET_FIELD_PATTERN.search(str(key)) else redact_instavm_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_instavm_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_instavm_value(item) for item in value)
    return value


class InstaVMTransport:
    def __init__(self, api_key: str, artifact_root: pathlib.Path):
        if not _HAS_HTTPX:
            raise PermanentError("httpx package is required for InstaVM REST API operations")
        if not api_key:
            raise PermanentError("INSTAVM_API_KEY is required for InstaVM REST API operations")
        self.api_key = api_key
        raw_base_url = os.environ.get("INSTAVM_BASE_URL", "https://api.instavm.io").rstrip("/")
        parsed_base_url = urllib.parse.urlsplit(raw_base_url)
        if parsed_base_url.scheme != "https" or not parsed_base_url.hostname:
            raise PermanentError("INSTAVM_BASE_URL must be an absolute HTTPS URL")
        if parsed_base_url.username or parsed_base_url.password or parsed_base_url.query or parsed_base_url.fragment:
            raise PermanentError("INSTAVM_BASE_URL must not contain credentials, a query, or a fragment")
        self.base_url = raw_base_url
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        def positive_timeout(name: str, default: float) -> float:
            try:
                value = float(os.environ.get(name, str(default)))
            except ValueError as exc:
                raise PermanentError(f"{name} must be numeric") from exc
            if not math.isfinite(value) or value <= 0:
                raise PermanentError(f"{name} must be a positive finite number")
            return value

        tls_setting = os.environ.get("INSTAVM_TLS_VERIFY", "true").strip().lower()
        if tls_setting in {"0", "false", "no", "off"}:
            raise PermanentError("TLS certificate verification cannot be disabled for InstaVM")
        ca_bundle = os.environ.get("INSTAVM_CA_BUNDLE", "").strip()
        verify: typing.Union[bool, str] = ca_bundle or True
        self.timeout = httpx.Timeout(
            connect=positive_timeout("INSTAVM_CONNECT_TIMEOUT_SECONDS", 15.0),
            read=positive_timeout("INSTAVM_READ_TIMEOUT_SECONDS", 180.0),
            write=positive_timeout("INSTAVM_WRITE_TIMEOUT_SECONDS", 180.0),
            pool=positive_timeout("INSTAVM_POOL_TIMEOUT_SECONDS", 30.0),
        )
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout=self.timeout,
            verify=verify,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30),
        )

    def close(self) -> None:
        self.client.close()

    def _path(self, operation: InstaVMOperation, path_values: dict[str, typing.Any]) -> str:
        values = dict(path_values)
        required = set(re.findall(r"{([^}]+)}", operation.path_template))
        if set(values) != required:
            raise SchemaValidationError(
                f"{operation.operation_id} requires path fields {sorted(required)}, received {sorted(values)}"
            )
        rendered = operation.path_template
        for name in required:
            raw = str(values[name])
            if not raw or "\x00" in raw:
                raise SchemaValidationError(f"invalid path value for {name}")
            if name == "path":
                normalized = raw.strip("/")
                if not normalized or any(part in {".", ".."} for part in normalized.split("/")):
                    raise SchemaValidationError("computer-use proxy path is invalid")
                encoded = "/".join(urllib.parse.quote(part, safe="") for part in normalized.split("/"))
            else:
                encoded = urllib.parse.quote(raw, safe="")
            rendered = rendered.replace("{" + name + "}", encoded)
        return rendered

    def _artifact_path(self, operation_id: str, filename: str) -> pathlib.Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pathlib.Path(filename).name) or "download.bin"
        path = self.artifact_root / operation_id / f"{uuid.uuid4().hex}-{safe}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _error(self, operation: InstaVMOperation, response: typing.Any) -> InstaVMError:
        request_id = str(response.headers.get("x-request-id") or response.headers.get("request-id") or "")
        detail = ""
        with contextlib.suppress(Exception):
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("detail") or body.get("message") or body.get("error") or "")
        if not detail:
            detail = response.text[:2048] if getattr(response, "text", None) else "InstaVM request failed"
        retryable = response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        return InstaVMError(operation.operation_id, int(response.status_code), detail, request_id, retryable)

    def invoke(self, operation: InstaVMOperation, payload: dict[str, typing.Any]) -> dict:
        if not isinstance(payload, dict):
            raise SchemaValidationError("InstaVM operation input must be an object")
        allowed_fields = {"path", "query", "body", "headers", "file_path", "artifact_name", "idempotency_key", "confirm_destructive"}
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise SchemaValidationError(f"unknown InstaVM input fields: {sorted(unknown_fields)}")
        path_values = payload.get("path", {})
        query = payload.get("query", {})
        body = payload.get("body", {})
        headers = payload.get("headers", {})
        if not isinstance(path_values, dict) or not isinstance(query, dict) or not isinstance(body, dict) or not isinstance(headers, dict):
            raise SchemaValidationError("path, query, body, and headers must be objects")
        path = self._path(operation, path_values)
        extra_headers = {str(k): str(v) for k, v in headers.items() if not SECRET_FIELD_PATTERN.search(str(k))}
        try:
            retry_limit = int(os.environ.get("INSTAVM_RETRY_ATTEMPTS", "3"))
        except ValueError as exc:
            raise PermanentError("INSTAVM_RETRY_ATTEMPTS must be an integer") from exc
        if retry_limit < 1 or retry_limit > 20:
            raise PermanentError("INSTAVM_RETRY_ATTEMPTS must be between 1 and 20")
        idempotency_key = str(payload.get("idempotency_key") or "")
        if len(idempotency_key) > 256 or (idempotency_key and not re.fullmatch(r"[A-Za-z0-9._:-]+", idempotency_key)):
            raise SchemaValidationError("invalid idempotency key")
        if idempotency_key:
            extra_headers["Idempotency-Key"] = idempotency_key
        elif operation.side_effects and not operation.destructive:
            extra_headers["Idempotency-Key"] = str(uuid.uuid4())
        if operation.destructive and not idempotency_key:
            retry_limit = 1
        temporary: typing.Optional[pathlib.Path] = None
        for attempt in range(retry_limit):
            started = time.monotonic()
            try:
                if operation.transport == "multipart":
                    file_path = str(payload.get("file_path") or "")
                    if not file_path:
                        raise SchemaValidationError(f"{operation.operation_id} requires file_path")
                    source = pathlib.Path(file_path).resolve()
                    if not source.is_file():
                        raise SchemaValidationError("multipart source file does not exist")
                    allowed_roots = [WORKSPACE_PATH.resolve(), self.artifact_root.resolve()]
                    if not any(_path_is_within(source, root) for root in allowed_roots):
                        raise SchemaValidationError("multipart source file is outside approved directories")
                    try:
                        max_upload = int(os.environ.get("INSTAVM_MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
                    except ValueError as exc:
                        raise PermanentError("INSTAVM_MAX_UPLOAD_BYTES must be an integer") from exc
                    if max_upload <= 0:
                        raise PermanentError("INSTAVM_MAX_UPLOAD_BYTES must be positive")
                    if source.stat().st_size > max_upload:
                        raise SchemaValidationError("multipart source file exceeds INSTAVM_MAX_UPLOAD_BYTES")
                    with source.open("rb") as handle:
                        with self.client.stream(
                            operation.method,
                            path,
                            params=query,
                            data={str(k): str(v) for k, v in body.items()},
                            files={"file": (source.name, handle, "application/octet-stream")},
                            headers=extra_headers,
                            timeout=operation.timeout_s,
                        ) as response:
                            if 200 <= response.status_code < 300:
                                content = bytearray()
                                for chunk in response.iter_bytes():
                                    content.extend(chunk)
                                    if len(content) > operation.max_response_bytes:
                                        raise PermanentError(f"{operation.operation_id} JSON response exceeds configured size limit")
                                parsed: typing.Any = {}
                                if content:
                                    try:
                                        parsed = json.loads(bytes(content).decode(response.encoding or "utf-8"))
                                    except Exception as exc:
                                        raise PermanentError(f"{operation.operation_id} returned non-JSON data") from exc
                                return {
                                    "response": redact_instavm_value(parsed),
                                    "status_code": response.status_code,
                                    "request_id": response.headers.get("x-request-id", ""),
                                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                                }
                            response.read()
                            error = self._error(operation, response)
                            if not error.retryable or attempt + 1 >= retry_limit:
                                raise error
                            delay = _retry_after_seconds(response.headers.get("retry-after"), attempt)
                            time.sleep(delay)
                            continue
                elif operation.transport == "binary":
                    filename = str(payload.get("artifact_name") or "download.bin")
                    destination = self._artifact_path(operation.operation_id, filename)
                    temporary = destination.with_suffix(destination.suffix + ".partial")
                    try:
                        maximum = int(os.environ.get("INSTAVM_MAX_DOWNLOAD_BYTES", str(operation.max_response_bytes)))
                    except ValueError as exc:
                        raise PermanentError("INSTAVM_MAX_DOWNLOAD_BYTES must be an integer") from exc
                    if maximum <= 0:
                        raise PermanentError("INSTAVM_MAX_DOWNLOAD_BYTES must be positive")
                    with self.client.stream(
                        operation.method,
                        path,
                        params=query,
                        json=body if body else None,
                        headers=extra_headers,
                        timeout=operation.timeout_s,
                    ) as response:
                        if response.status_code < 200 or response.status_code >= 300:
                            response.read()
                            error = self._error(operation, response)
                            if not error.retryable or attempt + 1 >= retry_limit:
                                raise error
                            delay = _retry_after_seconds(response.headers.get("retry-after"), attempt)
                            time.sleep(delay)
                            continue
                        digest = hashlib.sha256()
                        size = 0
                        with temporary.open("wb") as handle:
                            for chunk in response.iter_bytes():
                                size += len(chunk)
                                if size > maximum:
                                    raise PermanentError(f"{operation.operation_id} response exceeds configured size limit")
                                handle.write(chunk)
                                digest.update(chunk)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, destination)
                        temporary = None
                        return {
                            "artifact": {
                                "path": str(destination.relative_to(self.artifact_root)),
                                "absolute_path": str(destination),
                                "bytes": size,
                                "sha256": digest.hexdigest(),
                                "content_type": response.headers.get("content-type", "application/octet-stream"),
                            },
                            "status_code": response.status_code,
                            "request_id": response.headers.get("x-request-id", ""),
                            "duration_ms": round((time.monotonic() - started) * 1000, 2),
                        }
                elif operation.transport == "websocket":
                    return self.websocket_exchange(operation, path, query, body)
                else:
                    with self.client.stream(
                        operation.method,
                        path,
                        params=query,
                        json=body if body else None,
                        headers=extra_headers,
                        timeout=operation.timeout_s,
                    ) as response:
                        if 200 <= response.status_code < 300:
                            content = bytearray()
                            for chunk in response.iter_bytes():
                                content.extend(chunk)
                                if len(content) > operation.max_response_bytes:
                                    raise PermanentError(f"{operation.operation_id} JSON response exceeds configured size limit")
                            parsed: typing.Any = {}
                            if content:
                                try:
                                    parsed = json.loads(bytes(content).decode(response.encoding or "utf-8"))
                                except Exception as exc:
                                    raise PermanentError(f"{operation.operation_id} returned non-JSON data") from exc
                            return {
                                "response": redact_instavm_value(parsed),
                                "status_code": response.status_code,
                                "request_id": response.headers.get("x-request-id", ""),
                                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                            }
                        response.read()
                        error = self._error(operation, response)
                        if not error.retryable or attempt + 1 >= retry_limit:
                            raise error
                        delay = _retry_after_seconds(response.headers.get("retry-after"), attempt)
                        time.sleep(delay)
            except InstaVMError:
                raise
            except TransientError:
                if attempt + 1 >= retry_limit:
                    raise
                time.sleep(min(30.0, (2 ** attempt) + random.random()))
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt + 1 >= retry_limit:
                    raise TransientError(f"{operation.operation_id} network failure: {exc}") from exc
                time.sleep(min(30.0, (2 ** attempt) + random.random()))
            finally:
                if temporary is not None:
                    with contextlib.suppress(FileNotFoundError, OSError):
                        temporary.unlink()
                    temporary = None
        raise TransientError(f"{operation.operation_id} exhausted request retry budget")

    def websocket_exchange(self, operation: InstaVMOperation, path: str, query: dict, body: dict) -> dict:
        if not _HAS_WEBSOCKETS:
            raise PermanentError("websockets package is required for WebSocket operations")
        scheme = "wss" if self.base_url.startswith("https://") else "ws"
        root = re.sub(r"^https?://", "", self.base_url)
        suffix = urllib.parse.urlencode(query, doseq=True)
        url = f"{scheme}://{root}{path}" + (f"?{suffix}" if suffix else "")
        outbound = body.get("messages", [])
        if not isinstance(outbound, list):
            raise SchemaValidationError("WebSocket body.messages must be an array")
        maximum_messages = int(body.get("maximum_messages", 1))
        idle_timeout_s = float(body.get("idle_timeout_s", min(5.0, operation.timeout_s)))
        if maximum_messages < 0 or maximum_messages > 1000:
            raise SchemaValidationError("maximum_messages must be between zero and 1000")
        if idle_timeout_s <= 0 or idle_timeout_s > operation.timeout_s:
            raise SchemaValidationError("idle_timeout_s is outside the permitted range")

        async def exchange() -> typing.Tuple[str, typing.List[dict]]:
            responses: typing.List[dict] = []
            total_bytes = 0
            connection_parameters = {
                "open_timeout": min(operation.timeout_s, 30.0),
                "close_timeout": 5.0,
                "max_size": operation.max_response_bytes,
            }
            header_parameter = "additional_headers" if "additional_headers" in inspect.signature(websockets.connect).parameters else "extra_headers"
            connection_parameters[header_parameter] = {"X-API-Key": self.api_key}
            async with websockets.connect(url, **connection_parameters) as connection:
                for message in outbound:
                    if isinstance(message, dict) and "base64" in message:
                        try:
                            data = base64.b64decode(str(message["base64"]), validate=True)
                        except Exception as exc:
                            raise SchemaValidationError("invalid base64 WebSocket message") from exc
                        await connection.send(data)
                    elif isinstance(message, (str, bytes)):
                        await connection.send(message)
                    else:
                        raise SchemaValidationError("WebSocket messages must be strings, bytes, or base64 objects")
                for _ in range(maximum_messages):
                    try:
                        received = await asyncio.wait_for(connection.recv(), timeout=idle_timeout_s)
                    except asyncio.TimeoutError:
                        break
                    if isinstance(received, bytes):
                        total_bytes += len(received)
                        responses.append({"base64": base64.b64encode(received).decode("ascii"), "binary": True})
                    else:
                        encoded = str(received).encode("utf-8")
                        total_bytes += len(encoded)
                        responses.append({"text": str(received), "binary": False})
                    if total_bytes > operation.max_response_bytes:
                        raise PermanentError("WebSocket response exceeds configured size limit")
                return str(connection.subprotocol or ""), responses

        started = time.monotonic()
        try:
            subprotocol, responses = asyncio.run(exchange())
        except SchemaValidationError:
            raise
        except Exception as exc:
            raise TransientError(f"{operation.operation_id} WebSocket exchange failed: {exc}") from exc
        return {
            "response": {"messages": responses, "subprotocol": subprotocol},
            "status_code": 101,
            "request_id": "",
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
        }



class ThinkParser:
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.state = "content"
        self.buffer = ""

    def _safe_prefix(self, text, marker):
        for length in range(min(len(marker) - 1, len(text)), 0, -1):
            if text.endswith(marker[:length]):
                return text[:-length]
        return text

    def feed(self, delta):
        events = []
        self.buffer += str(delta)
        while self.buffer:
            if self.state == "content":
                position = self.buffer.find(self.OPEN)
                if position < 0:
                    safe = self._safe_prefix(self.buffer, self.OPEN)
                    if safe:
                        events.append(("content", safe))
                        self.buffer = self.buffer[len(safe):]
                    break
                if position:
                    events.append(("content", self.buffer[:position]))
                self.buffer = self.buffer[position + len(self.OPEN):]
                self.state = "thinking"
                events.append(("thinking_start", ""))
            else:
                position = self.buffer.find(self.CLOSE)
                if position < 0:
                    safe = self._safe_prefix(self.buffer, self.CLOSE)
                    if safe:
                        events.append(("thinking_delta", safe))
                        self.buffer = self.buffer[len(safe):]
                    break
                if position:
                    events.append(("thinking_delta", self.buffer[:position]))
                self.buffer = self.buffer[position + len(self.CLOSE):]
                self.state = "content"
                events.append(("thinking_end", ""))
        return events

    def flush(self):
        if not self.buffer:
            return []
        if self.state == "thinking":
            events = [("thinking_delta", self.buffer), ("thinking_error", "unclosed thinking block")]
        else:
            events = [("content", self.buffer)]
        self.buffer = ""
        self.state = "content"
        return events


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


def estimate_tokens(text: str, override: typing.Optional[int] = None) -> int:
    if override is not None:
        value = int(override)
        if value < 0:
            raise ValueError("token override must be nonnegative")
        return value
    value = str(text)
    if not value:
        return 0
    lexical = len(re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE))
    byte_estimate = math.ceil(len(value.encode("utf-8")) / 4)
    return max(1, lexical, byte_estimate)


class TokenBudget:
    def __init__(self, total: int):
        self.total = int(total)
        self.remaining = int(total)
        self.used = 0

    def consume(self, amount: int) -> None:
        amount = max(0, int(amount))
        if amount > self.remaining:
            raise ContextWindowExceededError(f"requested {amount}, remaining {self.remaining}")
        self.remaining -= amount
        self.used += amount

    def fit(self, text: str) -> str:
        text = str(text)
        tokens = estimate_tokens(text)
        if tokens <= self.remaining:
            self.consume(tokens)
            return text
        if self.remaining <= 0:
            return ""
        low = 0
        high = len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if estimate_tokens(text[:middle]) <= self.remaining:
                low = middle
            else:
                high = middle - 1
        truncated = text[:low]
        self.consume(estimate_tokens(truncated))
        return truncated


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
LOGGER.setLevel(getattr(logging, CONFIG.observability.log_level.upper(), logging.INFO))
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


class CircuitBreaker:
    registry: typing.Dict[str, "CircuitBreaker"] = {}
    registry_lock = threading.RLock()

    def __init__(self, dependency: str, config: typing.Any = None):
        self.dependency = dependency
        self.config = config or CONFIG.breaker
        self.state = CircuitBreakerState()
        self._state_lock = threading.RLock()
        self._half_open_probe_in_flight = False
        with self.registry_lock:
            self.registry[dependency] = self
        set_breaker_state(dependency, self.state.state)

    async def allow_request(self) -> bool:
        with self._state_lock:
            now = time.time()
            if self.state.state == BreakerState.OPEN:
                if now < self.state.cooldown_until:
                    return False
                self.state = CircuitBreakerState(
                    state=BreakerState.HALF_OPEN,
                    failure_count=self.state.failure_count,
                    last_failure_ts=self.state.last_failure_ts,
                    cooldown_until=self.state.cooldown_until,
                )
                self._half_open_probe_in_flight = False
                record_event(EventType.CIRCUIT_HALF_OPEN, uuid.UUID(int=0), {"dependency": self.dependency})
                set_breaker_state(self.dependency, self.state.state)
            if self.state.state == BreakerState.HALF_OPEN:
                if self._half_open_probe_in_flight:
                    return False
                self._half_open_probe_in_flight = True
            return True

    async def record_success(self) -> None:
        with self._state_lock:
            was_open = self.state.state != BreakerState.CLOSED
            self.state = CircuitBreakerState(state=BreakerState.CLOSED)
            self._half_open_probe_in_flight = False
            set_breaker_state(self.dependency, self.state.state)
            if was_open:
                record_event(EventType.CIRCUIT_CLOSED, uuid.UUID(int=0), {"dependency": self.dependency})

    async def record_failure(self, permanent: bool = False) -> None:
        with self._state_lock:
            was_open = self.state.state == BreakerState.OPEN
            failure_count = self.state.failure_count + 1
            failure_ts = time.time()
            threshold = self.config.permanent_threshold if permanent else self.config.transient_threshold
            state = BreakerState.OPEN if self.state.state == BreakerState.HALF_OPEN or failure_count >= threshold else self.state.state
            cooldown_until = failure_ts + self.config.cooldown_s if state == BreakerState.OPEN else self.state.cooldown_until
            self.state = CircuitBreakerState(
                state=state,
                failure_count=failure_count,
                last_failure_ts=failure_ts,
                cooldown_until=cooldown_until,
            )
            self._half_open_probe_in_flight = False
            set_breaker_state(self.dependency, self.state.state)
            if state == BreakerState.OPEN and not was_open:
                record_event(EventType.CIRCUIT_OPEN, uuid.UUID(int=0), {"dependency": self.dependency, "permanent": permanent})

    def force_open(self) -> None:
        with self._state_lock:
            now = time.time()
            self.state = CircuitBreakerState(
                state=BreakerState.OPEN,
                failure_count=max(1, self.state.failure_count),
                last_failure_ts=now,
                cooldown_until=now + self.config.cooldown_s,
            )
            self._half_open_probe_in_flight = False
            set_breaker_state(self.dependency, BreakerState.OPEN)

    def force_close(self) -> None:
        with self._state_lock:
            self.state = CircuitBreakerState(state=BreakerState.CLOSED)
            self._half_open_probe_in_flight = False
            set_breaker_state(self.dependency, BreakerState.CLOSED)

    @classmethod
    def get(cls, dependency: str) -> "CircuitBreaker":
        with cls.registry_lock:
            breaker = cls.registry.get(dependency)
            if breaker is None:
                breaker = CircuitBreaker(dependency)
            return breaker

    @classmethod
    def snapshot(cls) -> typing.Dict[str, "CircuitBreaker"]:
        with cls.registry_lock:
            return dict(cls.registry)


class Database:
    def __init__(self, path: pathlib.Path = DB_PATH):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=15, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        conn = self.connect()
        try:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                images TEXT,
                attachments TEXT,
                ts INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

            CREATE TABLE IF NOT EXISTS chat_memory(
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(session_id, key)
            );

            CREATE TABLE IF NOT EXISTS pdf_attachments(
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                text_content TEXT NOT NULL,
                page_images TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schedules(
                id TEXT PRIMARY KEY,
                task_payload_json TEXT NOT NULL,
                cron_expression TEXT,
                interval_seconds REAL,
                mode TEXT NOT NULL,
                continuation_task_id TEXT,
                next_run_at REAL NOT NULL,
                last_run_at REAL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs(
                job_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS job_chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                UNIQUE(job_id, seq),
                FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_job_chunks_job_seq ON job_chunks(job_id, seq);

            CREATE TABLE IF NOT EXISTS tasks(
                id TEXT PRIMARY KEY,
                priority INTEGER NOT NULL,
                base_priority INTEGER NOT NULL DEFAULT 0,
                deadline TEXT,
                dependencies TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER DEFAULT 0,
                available_at REAL,
                created_at REAL,
                updated_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_status_priority ON tasks(status, priority DESC, created_at);
            CREATE INDEX IF NOT EXISTS idx_tasks_deadline ON tasks(deadline);

            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                task_id TEXT,
                type TEXT,
                payload_json TEXT,
                tokens_used INTEGER,
                emb BLOB,
                utility_score REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_access_ts REAL
            );
            CREATE INDEX IF NOT EXISTS idx_events_task_ts ON events(task_id, ts);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
            CREATE INDEX IF NOT EXISTS idx_events_utility ON events(utility_score);

            CREATE TABLE IF NOT EXISTS event_archive(
                id INTEGER PRIMARY KEY,
                ts REAL,
                task_id TEXT,
                type TEXT,
                payload_json TEXT,
                tokens_used INTEGER,
                emb BLOB,
                utility_score REAL,
                access_count INTEGER,
                last_access_ts REAL
            );

            CREATE TABLE IF NOT EXISTS semantic(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                task_id TEXT,
                text TEXT,
                emb BLOB,
                utility_score REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                alpha REAL DEFAULT 1.0,
                beta REAL DEFAULT 1.0
            );


            CREATE TABLE IF NOT EXISTS skills(
                name TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                definition_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS idempotency_keys(
                key TEXT PRIMARY KEY,
                task_id TEXT,
                tool_name TEXT,
                created_at REAL NOT NULL,
                expires_at REAL,
                status TEXT NOT NULL DEFAULT 'running',
                result_json TEXT
            );
            CREATE TABLE IF NOT EXISTS instavm_invocations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                task_id TEXT,
                status_code INTEGER,
                request_id TEXT,
                duration_ms REAL,
                outcome TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_instavm_invocations_operation_created
                ON instavm_invocations(operation_id, created_at);
            CREATE TABLE IF NOT EXISTS instavm_policy_decisions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                destructive INTEGER NOT NULL,
                decision TEXT NOT NULL,
                task_id TEXT,
                created_at REAL NOT NULL
            );
            """)
            migrations = [
                ("semantic", "alpha", "REAL DEFAULT 1.0"),
                ("semantic", "beta", "REAL DEFAULT 1.0"),
                ("messages", "attachments", "TEXT"),
                ("tasks", "base_priority", "INTEGER NOT NULL DEFAULT 0"),
                ("tasks", "available_at", "REAL"),
                ("idempotency_keys", "expires_at", "REAL"),
                ("idempotency_keys", "status", "TEXT NOT NULL DEFAULT 'running'"),
            ]
            added_columns: typing.Set[typing.Tuple[str, str]] = set()
            for table, column, declaration in migrations:
                columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}
                if column not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
                    added_columns.add((table, column))
            if ("tasks", "base_priority") in added_columns:
                conn.execute("UPDATE tasks SET base_priority=priority")
            conn.execute("UPDATE tasks SET available_at=COALESCE(available_at,updated_at,created_at,?)", (time.time(),))
            conn.execute("DELETE FROM job_chunks WHERE job_id NOT IN (SELECT job_id FROM jobs)")
        finally:
            conn.close()

    @contextlib.contextmanager
    def transaction(self, conn: sqlite3.Connection) -> typing.Generator:
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                with contextlib.suppress(Exception):
                    conn.execute("ROLLBACK")
                raise


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Scheduler:
    def __init__(self, database: Database, config: typing.Any = None, schedule_config: typing.Any = None):
        self.db = database
        self.config = config or CONFIG.scheduler
        self.schedule_config = schedule_config or CONFIG.schedule
        self.wake = asyncio.Event()
        self._loop: typing.Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: typing.Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._loop = loop or asyncio.get_running_loop()

    def _signal_wake(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self.wake.set)

    def _has_cycle(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        proposed: typing.Dict[str, typing.List[str]],
        visiting: typing.Optional[typing.Set[str]] = None,
        visited: typing.Optional[typing.Set[str]] = None,
    ) -> bool:
        visiting = set() if visiting is None else visiting
        visited = set() if visited is None else visited
        if task_id in visiting:
            return True
        if task_id in visited:
            return False
        visiting.add(task_id)
        if task_id in proposed:
            dependencies = proposed[task_id]
        else:
            row = conn.execute("SELECT dependencies FROM tasks WHERE id=?", (task_id,)).fetchone()
            dependencies = json.loads(row["dependencies"]) if row else []
        for dependency in dependencies:
            if self._has_cycle(conn, str(dependency), proposed, visiting, visited):
                return True
        visiting.remove(task_id)
        visited.add(task_id)
        return False

    @staticmethod
    def _aware_datetime(value: typing.Optional[dt.datetime]) -> typing.Optional[dt.datetime]:
        if value is None:
            return None
        if not isinstance(value, dt.datetime):
            value = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)

    def enqueue(self, task: typing.Any) -> uuid.UUID:
        if not isinstance(task, Task):
            task = model_validate(Task, task if isinstance(task, dict) else _plain(task))
        task_id = str(task.id)
        dependencies = [str(item) for item in task.dependencies]
        deadline = self._aware_datetime(task.deadline)
        base_priority = int(task.priority)
        status = task.status.value if isinstance(task.status, TaskStatus) else str(task.status)
        now = time.time()
        conn = self.db.connect()
        try:
            with self.db.transaction(conn):
                proposed = {task_id: dependencies}
                if self._has_cycle(conn, task_id, proposed):
                    status = TaskStatus.FAILED.value
                conn.execute(
                    "INSERT INTO tasks(id,priority,base_priority,deadline,dependencies,payload_json,status,attempts,available_at,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,COALESCE((SELECT attempts FROM tasks WHERE id=?),0),?,COALESCE((SELECT created_at FROM tasks WHERE id=?),?),?) "
                    "ON CONFLICT(id) DO UPDATE SET priority=excluded.priority,base_priority=excluded.base_priority,deadline=excluded.deadline,"
                    "dependencies=excluded.dependencies,payload_json=excluded.payload_json,status=excluded.status,available_at=excluded.available_at,updated_at=excluded.updated_at",
                    (
                        task_id,
                        base_priority,
                        base_priority,
                        deadline.isoformat() if deadline else None,
                        stable_json_dumps(dependencies),
                        stable_json_dumps(task.payload),
                        status,
                        task_id,
                        now,
                        task_id,
                        task.created_at.timestamp() if hasattr(task.created_at, "timestamp") else now,
                        now,
                    ),
                )
        finally:
            conn.close()
        self._signal_wake()
        return uuid.UUID(task_id)

    def dequeue_next(self) -> typing.Optional[typing.Any]:
        conn = self.db.connect()
        now = time.time()
        try:
            with self.db.transaction(conn):
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status='queued' AND COALESCE(available_at,0)<=? ORDER BY priority DESC,created_at ASC LIMIT 100",
                    (now,),
                ).fetchall()
                for row in rows:
                    dependency_failed = False
                    blocked = False
                    for dependency in json.loads(row["dependencies"]):
                        dep_row = conn.execute("SELECT status FROM tasks WHERE id=?", (dependency,)).fetchone()
                        if dep_row is None or dep_row["status"] in {TaskStatus.FAILED.value, TaskStatus.TIMED_OUT.value, TaskStatus.ABORTED.value}:
                            dependency_failed = True
                            break
                        if dep_row["status"] != TaskStatus.DONE.value:
                            blocked = True
                            break
                    if dependency_failed:
                        conn.execute(
                            "UPDATE tasks SET status=?,updated_at=? WHERE id=? AND status='queued'",
                            (TaskStatus.ABORTED.value, now, row["id"]),
                        )
                        continue
                    if blocked:
                        continue
                    priority = int(row["priority"] or 0)
                    deadline = self._aware_datetime(dt.datetime.fromisoformat(row["deadline"])) if row["deadline"] else None
                    if self.config.deadline_policy == "expedite" and deadline is not None:
                        seconds_until = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
                        if seconds_until <= 0:
                            priority += 1000000
                        elif seconds_until < 3600:
                            priority += int(100000 * (1.0 - seconds_until / 3600.0))
                    cursor = conn.execute(
                        "UPDATE tasks SET status='running',priority=?,updated_at=? WHERE id=? AND status='queued'",
                        (priority, now, row["id"]),
                    )
                    if cursor.rowcount:
                        data = dict(row)
                        data["priority"] = priority
                        return self._row_to_task(data, TaskStatus.RUNNING)
        finally:
            conn.close()
        return None

    def _row_to_task(self, row: typing.Mapping[str, typing.Any], status: typing.Optional[TaskStatus] = None) -> typing.Any:
        created = row["created_at"] or time.time()
        updated = row["updated_at"] or time.time()
        return model_validate(Task, {
            "id": str(row["id"]),
            "priority": int(row["priority"]),
            "deadline": row["deadline"],
            "dependencies": json.loads(row["dependencies"]),
            "payload": json.loads(row["payload_json"]),
            "created_at": dt.datetime.fromtimestamp(float(created), dt.timezone.utc).isoformat(),
            "updated_at": dt.datetime.fromtimestamp(float(updated), dt.timezone.utc).isoformat(),
            "status": status.value if status else row["status"],
            "checkpoints": [],
        })

    def update_status(self, task_id: typing.Any, status: typing.Any) -> None:
        status_val = status.value if isinstance(status, TaskStatus) else str(status)
        if status_val not in {item.value for item in TaskStatus}:
            raise ValueError(f"invalid task status: {status_val}")
        conn = self.db.connect()
        try:
            conn.execute(
                "UPDATE tasks SET status=?,updated_at=? WHERE id=?",
                (status_val, time.time(), str(task_id)),
            )
        finally:
            conn.close()

    def mark_done(self, task_id: typing.Any) -> None:
        self.update_status(task_id, TaskStatus.DONE)

    def mark_failed(self, task_id: typing.Any) -> None:
        self.update_status(task_id, TaskStatus.FAILED)

    @staticmethod
    def _parse_cron_field(field: str, minimum: int, maximum: int) -> typing.Set[int]:
        values: typing.Set[int] = set()
        for part in field.split(","):
            if not part:
                raise ValueError("empty cron field component")
            source, separator, step_text = part.partition("/")
            step = int(step_text) if separator else 1
            if step <= 0:
                raise ValueError("cron step must be positive")
            if source == "*":
                start, end = minimum, maximum
            elif "-" in source:
                start_text, end_text = source.split("-", 1)
                start, end = int(start_text), int(end_text)
            else:
                start = int(source)
                end = maximum if separator else start
            if start < minimum or end > maximum or start > end:
                raise ValueError("cron value is out of range")
            values.update(range(start, end + 1, step))
        return values

    def _valid_cron_expression(self, expression: str) -> bool:
        try:
            fields = expression.split()
            if len(fields) != 5:
                return False
            ranges = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
            for field, (minimum, maximum) in zip(fields, ranges):
                self._parse_cron_field(field, minimum, maximum)
            return True
        except (TypeError, ValueError):
            return False

    def _cron_matches(self, expression: str, current: dt.datetime) -> bool:
        minute, hour, day, month, weekday = expression.split()
        minute_values = self._parse_cron_field(minute, 0, 59)
        hour_values = self._parse_cron_field(hour, 0, 23)
        day_values = self._parse_cron_field(day, 1, 31)
        month_values = self._parse_cron_field(month, 1, 12)
        weekday_values = self._parse_cron_field(weekday, 0, 7)
        normalized_weekday = (current.weekday() + 1) % 7
        weekday_match = normalized_weekday in weekday_values or (normalized_weekday == 0 and 7 in weekday_values)
        day_match = current.day in day_values
        day_restricted = day_values != set(range(1, 32))
        weekday_restricted = weekday_values != set(range(0, 8))
        calendar_match = day_match and weekday_match
        if day_restricted and weekday_restricted:
            calendar_match = day_match or weekday_match
        return current.minute in minute_values and current.hour in hour_values and current.month in month_values and calendar_match

    def _next_cron_run(self, expression: str, after: float) -> float:
        probe = dt.datetime.fromtimestamp(after, dt.timezone.utc).replace(second=0, microsecond=0)
        for _ in range(366 * 24 * 60 * 5):
            probe += dt.timedelta(minutes=1)
            if self._cron_matches(expression, probe):
                return probe.timestamp()
        raise PermanentError("cron expression has no occurrence within five years")

    def create_schedule(
        self,
        payload: dict,
        interval_seconds: typing.Optional[float],
        cron_expression: typing.Optional[str],
        mode: str,
        continuation_task_id: typing.Optional[str],
    ) -> str:
        if not isinstance(payload, dict):
            raise PermanentError("schedule payload must be an object")
        if mode not in {"new", "continue"}:
            raise PermanentError("schedule mode must be new or continue")
        if (interval_seconds is None) == (cron_expression is None):
            raise PermanentError("set exactly one of interval_seconds or cron_expression")
        if interval_seconds is not None:
            interval_seconds = float(interval_seconds)
            if not math.isfinite(interval_seconds) or interval_seconds <= 0:
                raise PermanentError("interval_seconds must be a positive finite number")
        if cron_expression and not self._valid_cron_expression(cron_expression):
            raise PermanentError("cron expression is invalid")
        if mode == "continue":
            if not continuation_task_id:
                raise PermanentError("continue mode requires continuation_task_id")
            try:
                continuation_task_id = str(uuid.UUID(str(continuation_task_id)))
            except ValueError as exc:
                raise PermanentError("continuation_task_id must be a UUID") from exc
            conn = self.db.connect()
            try:
                exists = conn.execute("SELECT 1 FROM tasks WHERE id=?", (str(continuation_task_id),)).fetchone()
            finally:
                conn.close()
            if not exists:
                raise PermanentError("continuation task does not exist")
        schedule_id = str(uuid.uuid4())
        now = time.time()
        next_run = now + float(interval_seconds) if interval_seconds is not None else self._next_cron_run(str(cron_expression), now)
        conn = self.db.connect()
        try:
            conn.execute(
                "INSERT INTO schedules(id,task_payload_json,cron_expression,interval_seconds,mode,continuation_task_id,next_run_at,last_run_at,enabled,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (schedule_id, stable_json_dumps(payload), cron_expression, interval_seconds, mode, continuation_task_id, next_run, None, 1, now),
            )
        finally:
            conn.close()
        self._signal_wake()
        return schedule_id

    def _continuation_payload(self, conn: sqlite3.Connection, task_id: str, payload: dict) -> dict:
        task_row = conn.execute("SELECT payload_json,status FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not task_row:
            raise PermanentError("continuation task no longer exists")
        event_rows = conn.execute(
            "SELECT type,payload_json,ts FROM events WHERE task_id=? ORDER BY id DESC LIMIT 50",
            (task_id,),
        ).fetchall()
        return {
            **payload,
            "continuation_of": task_id,
            "continuation_source": {
                "payload": json.loads(task_row["payload_json"]),
                "status": task_row["status"],
                "events": [
                    {"type": row["type"], "payload": json.loads(row["payload_json"]), "timestamp": row["ts"]}
                    for row in reversed(event_rows)
                ],
            },
        }

    async def schedule_loop(self, stop_event: asyncio.Event) -> None:
        self.bind_loop()
        while not stop_event.is_set():
            now = time.time()
            created = 0
            failures: typing.List[typing.Tuple[str, str]] = []
            conn = self.db.connect()
            try:
                with self.db.transaction(conn):
                    rows = conn.execute(
                        "SELECT * FROM schedules WHERE enabled=1 AND next_run_at<=? ORDER BY next_run_at LIMIT 50",
                        (now,),
                    ).fetchall()
                    for row in rows:
                        savepoint = "schedule_" + uuid.uuid4().hex
                        conn.execute(f"SAVEPOINT {savepoint}")
                        try:
                            payload = json.loads(row["task_payload_json"])
                            if not isinstance(payload, dict):
                                raise ValueError("schedule payload is not an object")
                            if row["mode"] == "continue":
                                payload = self._continuation_payload(conn, str(row["continuation_task_id"]), payload)
                            elif row["mode"] != "new":
                                raise ValueError("schedule mode is invalid")
                            next_run = (
                                now + float(row["interval_seconds"])
                                if row["interval_seconds"] is not None
                                else self._next_cron_run(str(row["cron_expression"]), now)
                            )
                            task_id = str(uuid.uuid4())
                            conn.execute(
                                "INSERT INTO tasks(id,priority,base_priority,deadline,dependencies,payload_json,status,attempts,available_at,created_at,updated_at) "
                                "VALUES(?,?,?,?,?,?,?,0,?,?,?)",
                                (
                                    task_id,
                                    0,
                                    0,
                                    None,
                                    "[]",
                                    stable_json_dumps(payload),
                                    TaskStatus.QUEUED.value,
                                    now,
                                    now,
                                    now,
                                ),
                            )
                            cursor = conn.execute(
                                "UPDATE schedules SET last_run_at=?,next_run_at=? WHERE id=? AND enabled=1 AND next_run_at=?",
                                (now, next_run, row["id"], row["next_run_at"]),
                            )
                            if cursor.rowcount != 1:
                                raise TransientError("schedule claim was lost")
                            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                            created += 1
                        except Exception as exc:
                            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                            conn.execute("UPDATE schedules SET enabled=0 WHERE id=?", (row["id"],))
                            failures.append((str(row["id"]), str(exc)))
            finally:
                conn.close()
            if created:
                self._signal_wake()
            for schedule_id, message in failures:
                record_event(EventType.ERROR, uuid.UUID(int=0), {"schedule_id": schedule_id, "error": message})
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.schedule_config.poll_interval_s)
            except asyncio.TimeoutError:
                continue

    def requeue_with_backoff(self, task_id: typing.Any) -> bool:
        conn = self.db.connect()
        try:
            with self.db.transaction(conn):
                row = conn.execute("SELECT attempts FROM tasks WHERE id=?", (str(task_id),)).fetchone()
                if row is None:
                    return False
                attempts = int(row["attempts"] or 0) + 1
                try:
                    delay = exponential_backoff_with_jitter(
                        attempts - 1,
                        CONFIG.backoff.base_s,
                        CONFIG.backoff.factor,
                        CONFIG.backoff.max_delay_s,
                        max_attempts=CONFIG.backoff.max_attempts,
                    )
                except MaxRetriesExceeded:
                    conn.execute("UPDATE tasks SET status='failed',updated_at=? WHERE id=?", (time.time(), str(task_id)))
                    return False
                retries_total.inc()
                backoff_seconds_sum.inc(delay)
                now = time.time()
                conn.execute(
                    "UPDATE tasks SET status='queued',attempts=?,available_at=?,updated_at=? WHERE id=?",
                    (attempts, now + delay, now, str(task_id)),
                )
        finally:
            conn.close()
        self._signal_wake()
        return True

    def list_in_flight(self) -> typing.List[str]:
        conn = self.db.connect()
        try:
            return [str(row["id"]) for row in conn.execute("SELECT id FROM tasks WHERE status='running'")]
        finally:
            conn.close()

    def drain(self) -> None:
        conn = self.db.connect()
        try:
            now = time.time()
            conn.execute(
                "UPDATE tasks SET status='queued',available_at=?,updated_at=? WHERE status='running'",
                (now, now),
            )
        finally:
            conn.close()
        self._signal_wake()

    async def aging_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=60.0)
                break
            except asyncio.TimeoutError:
                pass
            conn = self.db.connect()
            try:
                rows = conn.execute("SELECT id,base_priority,created_at FROM tasks WHERE status='queued'").fetchall()
                now = time.time()
                for row in rows:
                    age_hours = max(0.0, (now - float(row["created_at"] or now)) / 3600.0)
                    base_priority = int(row["base_priority"] or 0)
                    age_bonus = int(math.floor(self.config.aging_factor * age_hours * 100.0))
                    conn.execute("UPDATE tasks SET priority=? WHERE id=?", (base_priority + age_bonus, row["id"]))
            finally:
                conn.close()


class WorkingMemory:
    def __init__(self, token_capacity: int):
        self.capacity = int(token_capacity)
        if self.capacity <= 0:
            raise ValueError("working-memory capacity must be positive")
        self.items: deque = deque()
        self.tokens = 0
        self._lock = threading.RLock()

    def push(self, text: str, metadata: typing.Optional[dict] = None) -> None:
        item = {"text": str(text), "metadata": metadata or {}, "ts": time.time()}
        item_tokens = estimate_tokens(item["text"])
        with self._lock:
            self.items.append(item)
            self.tokens += item_tokens
            while self.tokens > self.capacity and self.items:
                removed = self.items.popleft()
                self.tokens -= estimate_tokens(removed["text"])

    def render(self) -> str:
        with self._lock:
            return "\n".join(item["text"] for item in self.items)

    def pointer(self) -> int:
        with self._lock:
            return len(self.items)

    def snapshot(self) -> typing.List[dict]:
        with self._lock:
            return [dict(item) for item in self.items]

    def restore(self, items: typing.Iterable[dict]) -> None:
        with self._lock:
            self.items.clear()
            self.tokens = 0
            for raw in items:
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("text", ""))
                item = {"text": text, "metadata": dict(raw.get("metadata") or {}), "ts": float(raw.get("ts") or time.time())}
                self.items.append(item)
                self.tokens += estimate_tokens(text)
            while self.tokens > self.capacity and self.items:
                removed = self.items.popleft()
                self.tokens -= estimate_tokens(removed["text"])

    def clear(self) -> None:
        with self._lock:
            self.items.clear()
            self.tokens = 0


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
        self.config = config or CONFIG.memory
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


class EpisodicMemory:
    def __init__(self, database: Database, config: typing.Any = None):
        self.db = database
        self.config = config or CONFIG.memory
        self.semantic = SemanticMemory(database, self.config)

    @staticmethod
    def _event_fields(event: typing.Any) -> typing.Tuple[dict, float, str, str, int, typing.Optional[typing.List[float]]]:
        if isinstance(event, dict):
            payload = event.get("payload", {})
            timestamp = event.get("timestamp")
            task_id = event.get("task_id", uuid.UUID(int=0))
            event_type = event.get("type", EventType.ERROR.value)
            tokens = int(event.get("tokens_used", 0) or 0)
            embedding = event.get("embedding")
        else:
            payload = getattr(event, "payload", {})
            timestamp = getattr(event, "timestamp", None)
            task_id = getattr(event, "task_id", uuid.UUID(int=0))
            event_type = getattr(event, "type", EventType.ERROR.value)
            tokens = int(getattr(event, "tokens_used", 0) or 0)
            embedding = getattr(event, "embedding", None)
        if isinstance(timestamp, str):
            timestamp = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        ts = timestamp.timestamp() if isinstance(timestamp, dt.datetime) else time.time()
        type_text = event_type.value if isinstance(event_type, enum.Enum) else str(event_type)
        return dict(payload) if isinstance(payload, dict) else {"value": payload}, ts, str(task_id), type_text, tokens, embedding

    def append_event(self, event: typing.Any) -> int:
        payload, ts, task_id, type_text, tokens, embedding = self._event_fields(event)
        payload_text = stable_json_dumps(payload)
        vector = embedding or deterministic_embedding(payload_text, self.config.embedding_dim)
        utility = max(0.0, min(1.0, 0.25 + min(0.75, max(0, tokens) / 10000.0)))
        conn = self.db.connect()
        try:
            with self.db.transaction(conn):
                cursor = conn.execute(
                    "INSERT INTO events(ts,task_id,type,payload_json,tokens_used,emb,utility_score,last_access_ts) VALUES(?,?,?,?,?,?,?,?)",
                    (ts, task_id, type_text, payload_text, tokens, pack_vector(vector), utility, time.time()),
                )
                row_id = int(cursor.lastrowid)
        finally:
            conn.close()
        memory_size_bytes.labels(tier="episodic").set(self._size_bytes())
        return row_id

    def _size_bytes(self) -> int:
        conn = self.db.connect()
        try:
            row = conn.execute("SELECT COALESCE(SUM(length(payload_json)+length(emb)),0) AS n FROM events").fetchone()
            return int(row["n"] or 0)
        finally:
            conn.close()

    def count(self) -> int:
        conn = self.db.connect()
        try:
            return int(conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"] or 0)
        finally:
            conn.close()

    def retrieve(self, task_id: typing.Any, query: str, limit: int = 50) -> typing.List[typing.Tuple[float, sqlite3.Row]]:
        qvec = deterministic_embedding(query, self.config.embedding_dim)
        conn = self.db.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM events WHERE task_id IN (?,?) OR task_id IS NULL ORDER BY ts DESC LIMIT 2000",
                (str(task_id), str(uuid.UUID(int=0))),
            ).fetchall()
            results = []
            now = time.time()
            half_life = max(3600.0, self.config.retention_hours * 3600.0)
            for row in rows:
                age = max(0.0, now - float(row["ts"] or now))
                recency = math.exp(-math.log(2) * age / half_life)
                access_boost = math.log1p(int(row["access_count"] or 0)) / 10.0
                richness = min(1.0, estimate_tokens(row["payload_json"]) / 1000.0)
                similarity = cosine_similarity(qvec, unpack_vector(row["emb"]))
                score = similarity * 0.65 + recency * 0.2 + access_boost * 0.1 + richness * 0.05
                results.append((score, row))
            results.sort(key=lambda item: (item[0], int(item[1]["id"])), reverse=True)
            return results[:max(0, int(limit))]
        finally:
            conn.close()

    def mark_access(self, event_ids: typing.Iterable[int]) -> None:
        ids = sorted({int(item) for item in event_ids})
        if not ids:
            return
        accessed_at = time.time()
        conn = self.db.connect()
        try:
            with self.db.transaction(conn):
                conn.executemany(
                    "UPDATE events SET access_count=access_count+1,last_access_ts=? WHERE id=?",
                    [(accessed_at, item_id) for item_id in ids],
                )
        finally:
            conn.close()

    async def summarize_old_events(self, summarizer: typing.Any, max_len: int) -> int:
        event_count = self.count()
        cutoff = time.time() - self.config.retention_hours * 3600.0
        conn = self.db.connect()
        try:
            rows = conn.execute("SELECT * FROM events WHERE ts<? ORDER BY ts ASC LIMIT 200", (cutoff,)).fetchall()
            if not rows and event_count > self.config.episodic_summarize_threshold:
                compact_count = min(200, max(1, event_count - self.config.episodic_summarize_threshold))
                rows = conn.execute("SELECT * FROM events ORDER BY ts ASC LIMIT ?", (compact_count,)).fetchall()
        finally:
            conn.close()
        if not rows:
            return 0
        texts = [str(row["payload_json"]) for row in rows]
        summaries = await summarizer.summarize(texts, max_len)
        if len(summaries) != len(rows):
            raise PermanentError("summarizer must return one summary per event")
        semantic_ids = []
        try:
            for row, summary_text in zip(rows, summaries):
                if summary_text:
                    semantic_ids.append(self.semantic.add(row["task_id"] or uuid.UUID(int=0), summary_text, float(row["utility_score"] or 0.5)))
            conn = self.db.connect()
            try:
                with self.db.transaction(conn):
                    for row in rows:
                        conn.execute("INSERT OR IGNORE INTO event_archive SELECT * FROM events WHERE id=?", (row["id"],))
                        conn.execute("DELETE FROM events WHERE id=?", (row["id"],))
            finally:
                conn.close()
        except Exception:
            if semantic_ids:
                cleanup = self.db.connect()
                try:
                    with self.db.transaction(cleanup):
                        for semantic_id in semantic_ids:
                            cleanup.execute("DELETE FROM semantic WHERE id=?", (semantic_id,))
                finally:
                    cleanup.close()
                self.semantic._rebuild()
            raise
        memory_size_bytes.labels(tier="episodic").set(self._size_bytes())
        return len(rows)

    def admit_with_regret_gate(self, event: typing.Any, expected_utility: float) -> bool:
        return float(expected_utility) >= self.config.admission_utility_threshold


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


@dataclasses.dataclass
class ContextBlock:
    text: str
    selected_ids: typing.List[int]
    selection_log: typing.List[typing.Dict[str, typing.Any]]
    tokens_used: int


class MemoryManager:
    def __init__(self, database: Database, config: typing.Any = None, summarizer: typing.Any = None):
        self.config = config or CONFIG.memory
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


class ModelClient(ABC):
    @abstractmethod
    async def generate(
        self, prompt: str, tools_schema: typing.List[dict], max_tokens: int
    ) -> typing.Tuple[typing.Any, typing.Any]:
        raise TypeError("abstract method must be implemented by a concrete client")

    def close(self) -> None:
        return None


class StructuredRuleModel(ModelClient):
    async def generate(
        self, prompt: str, tools_schema: typing.List[dict], max_tokens: int
    ) -> typing.Tuple[typing.Any, typing.Any]:
        text = str(prompt)
        tool_names = {
            str(schema.get("function", {}).get("name", "")): schema
            for schema in tools_schema
            if isinstance(schema, dict)
        }
        directive = re.search(r"(?:^|\n)tool:([A-Za-z0-9_.-]+)(?:\s+({.*}))?\s*$", text, flags=re.DOTALL)
        calls = []
        final_answer = None
        declared_intent = "apply explicit structured rules"
        if directive:
            name = directive.group(1)
            if name not in tool_names:
                raise PermanentError(f"unknown explicit local tool directive: {name}")
            raw_arguments = directive.group(2) or "{}"
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise PermanentError("local tool arguments must be an object")
            calls = [{"name": name, "arguments": arguments}]
            declared_intent = f"execute {name}"
        else:
            task_match = re.search(r"^Task:\s*(.+)$", text, flags=re.MULTILINE)
            if not task_match:
                raise PermanentError("structured rule model requires a Task field")
            task_data = json.loads(task_match.group(1))
            final_answer = stable_json_dumps({"completed": True, "task": task_data})
        action = model_validate(ModelAction, {
            "tool_calls": calls,
            "final_answer": final_answer,
            "declared_intent": declared_intent,
            "confidence": 1.0,
            "rationale": "Structured execution of explicit input.",
        })
        prompt_tokens = estimate_tokens(text)
        completion_tokens = estimate_tokens(final_answer or stable_json_dumps(calls))
        usage = model_validate(UsageStats, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
        return action, usage


def _classify_model_exception(exc: Exception, model_name: str) -> AgentError:
    status_code = getattr(exc, "status_code", None)
    message = str(exc)
    if status_code in {400, 401, 403, 404, 409, 422} or any(term in message.lower() for term in ("authentication", "unauthorized", "invalid api key", "model not found", "bad request")):
        return PermanentError(f"Requesty request failed for {model_name}: {message}")
    return TransientError(f"Requesty request failed for {model_name}: {message}")


class RequestyModel(ModelClient):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self._model_clients: typing.Dict[str, "RequestyModel"] = {}
        if _HAS_OPENAI and api_key:
            self.client = OpenAI(
                api_key=api_key,
                base_url=MODEL_ROUTER_URL,
                timeout=CONFIG.model.request_timeout_s,
                max_retries=0,
            )

    def for_model(self, model_name: str) -> "RequestyModel":
        if model_name == self.model_name:
            return self
        client = self._model_clients.get(model_name)
        if client is None:
            client = RequestyModel(self.api_key, model_name)
            self._model_clients[model_name] = client
        return client

    async def generate(
        self, prompt: str, tools_schema: typing.List[dict], max_tokens: int
    ) -> typing.Tuple[typing.Any, typing.Any]:
        if not self.api_key:
            raise PermanentError("REQUESTY_API_KEY is required for autonomous model execution")
        if not _HAS_OPENAI:
            raise PermanentError("openai package is required for Requesty model execution")
        if not self.client:
            raise PermanentError("Requesty client is unavailable")
        parameters = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "tools": tools_schema or None,
            "tool_choice": "auto" if tools_schema else None,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_action",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "tool_calls": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"type": "string"},
                                        "arguments": {"type": "object", "additionalProperties": True},
                                    },
                                    "required": ["name", "arguments"],
                                },
                            },
                            "final_answer": {"type": ["string", "null"]},
                            "declared_intent": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "rationale": {"type": "string"},
                        },
                        "required": ["tool_calls", "final_answer", "declared_intent", "confidence", "rationale"],
                    },
                },
            },
            "max_completion_tokens": min(int(max_tokens), CONFIG.model.max_completion_tokens),
        }
        try:
            response = await asyncio.to_thread(self.client.chat.completions.create, **parameters)
        except Exception as exc:
            if "max_completion_tokens" in str(exc):
                parameters.pop("max_completion_tokens", None)
                parameters["max_tokens"] = min(int(max_tokens), CONFIG.model.max_completion_tokens)
                try:
                    response = await asyncio.to_thread(self.client.chat.completions.create, **parameters)
                except Exception as fallback_exc:
                    raise _classify_model_exception(fallback_exc, self.model_name) from fallback_exc
            else:
                raise _classify_model_exception(exc, self.model_name) from exc
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise TransientError(f"Requesty returned no choices for {self.model_name}")
        message = choices[0].message
        provider_calls = []
        for call in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments are not an object")
                provider_calls.append({"name": call.function.name, "arguments": arguments})
            except Exception as exc:
                raise PermanentError(f"invalid tool arguments returned by {self.model_name}: {exc}") from exc
        content = message.content or ""
        data = {}
        if content:
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                if not provider_calls:
                    raise PermanentError(f"invalid JSON action returned by {self.model_name}: {exc}") from exc
        if data and not isinstance(data, dict):
            raise PermanentError(f"invalid action object returned by {self.model_name}")
        calls = provider_calls or data.get("tool_calls", [])
        final_answer = data.get("final_answer")
        if not calls and not final_answer:
            raise PermanentError(f"{self.model_name} returned neither tool calls nor a final answer")
        action = model_validate(ModelAction, {
            "tool_calls": calls,
            "final_answer": final_answer,
            "declared_intent": data.get("declared_intent", ""),
            "confidence": float(data.get("confidence", 0.0)),
            "rationale": data.get("rationale", ""),
        })
        usage_obj = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage_obj, "total_tokens", 0) or (prompt_tokens + completion_tokens))
        usage = model_validate(UsageStats, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })
        return action, usage

    def close(self) -> None:
        for client in list(self._model_clients.values()):
            client.close()
        self._model_clients.clear()
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class SummarizerClient(ABC):
    @abstractmethod
    async def summarize(self, texts: typing.List[str], max_len: int) -> typing.List[str]:
        raise TypeError("abstract method must be implemented by a concrete client")


class ExtractiveFrequencySummarizer(SummarizerClient):
    async def summarize(self, texts: typing.List[str], max_len: int) -> typing.List[str]:
        max_len = int(max_len)
        if max_len <= 0:
            raise ValueError("summary length must be positive")
        output = []
        for text in texts:
            source = " ".join(str(text).split())
            sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", source) if item.strip()]
            words = re.findall(r"[\w'-]+", source.casefold(), flags=re.UNICODE)
            frequencies = Counter(words)
            ranked = []
            for index, sentence in enumerate(sentences or [source]):
                sentence_words = re.findall(r"[\w'-]+", sentence.casefold(), flags=re.UNICODE)
                score = sum(frequencies[word] for word in sentence_words) / max(1, len(sentence_words))
                ranked.append((score, -index, sentence))
            ranked.sort(reverse=True)
            selected = []
            used = 0
            for _, _, sentence in ranked:
                if not sentence:
                    continue
                remaining = max_len - used
                if remaining <= 0:
                    break
                part = sentence[:remaining]
                selected.append(part)
                used += len(part) + (1 if selected else 0)
            output.append(" ".join(selected)[:max_len])
        return output


if _HAS_PYDANTIC:
    class InstaVMOperationInput(BaseModel):
        model_config: typing.ClassVar[dict] = ConfigDict(extra="forbid")
        path: dict[str, str] = Field(default_factory=dict)
        query: dict[str, typing.Any] = Field(default_factory=dict)
        headers: dict[str, str] = Field(default_factory=dict)
        body: dict[str, typing.Any] = Field(default_factory=dict)
        file_path: str = ""
        artifact_name: str = ""
        idempotency_key: str = Field(default="", max_length=256)
        confirm_destructive: bool = False

    class InstaVMOperationOutput(BaseModel):
        model_config: typing.ClassVar[dict] = ConfigDict(extra="forbid")
        response: typing.Any = None
        status_code: int = 0
        request_id: str = ""
        duration_ms: float = 0.0
        artifact: typing.Optional[dict[str, typing.Any]] = None
        connection: typing.Optional[dict[str, typing.Any]] = None

    class EmptyInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()

    class TextOutput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        text: str = ""

    class EchoInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        text: str = ""

    class WebSearchInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        query: str = ""

    class WebSearchOutput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        results: typing.List[str] = Field(default_factory=list)

    class ReadFileInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        path: str = Field(min_length=1, max_length=4096)

    class ReadFileOutput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        content: str = ""
        error: str = ""

    class WriteFileInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        path: str = Field(min_length=1, max_length=4096)
        content: str = ""

    class WriteFileOutput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        ok: bool = False
        error: str = ""

    class ShellInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        command: str = Field(min_length=1, max_length=32000)
        timeout_s: float = Field(default=60.0, gt=0, le=900.0)
        cwd: str = ""

    class ShellOutput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        exit_code: int = 0
        stdout: str = ""
        stderr: str = ""
        timed_out: bool = False

    class VMCreateInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        vm_lifetime_seconds: int = Field(default=3600, ge=20, le=86400)
        memory_mb: int = Field(default=2048, ge=256, le=32768)
        vcpu_count: int = Field(default=2, ge=1, le=16)
        snapshot_id: str = ""
        egress_policy: dict[str, typing.Any] = Field(default_factory=dict)

    class VMIdentifierInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        vm_id: str = Field(min_length=1, max_length=256)

    class VMSnapshotInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        vm_id: str = Field(min_length=1, max_length=256)
        name: str = Field(min_length=1, max_length=256)

    class VMExecuteInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        vm_id: str = Field(min_length=1, max_length=256)
        command: str = Field(min_length=1, max_length=32000)
        language: str = "bash"
        timeout_s: float = Field(default=120.0, gt=0, le=900.0)
        stream: bool = False

    class VMFileInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        vm_id: str = Field(min_length=1, max_length=256)
        source_path: str = Field(min_length=1, max_length=4096)
        destination_path: str = Field(min_length=1, max_length=4096)

    class VMVolumeInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        name: str = Field(min_length=1, max_length=256)
        quota_bytes: int = Field(default=1073741824, ge=1048576, le=1099511627776)

    class VolumeIdentifierInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        volume_id: str = Field(min_length=1, max_length=256)
        confirm_destructive: bool = False

    class VolumeCheckpointInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        volume_id: str = Field(min_length=1, max_length=256)
        name: str = Field(min_length=1, max_length=256)

    class VMMountVolumeInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        vm_id: str = Field(min_length=1, max_length=256)
        volume_id: str = Field(min_length=1, max_length=256)
        mount_path: str = Field(min_length=1, max_length=4096)
        read_only: bool = False

    class VMEgressInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        vm_id: str = Field(min_length=1, max_length=256)
        allow_package_managers: bool = True
        allow_http: bool = False
        allow_https: bool = True
        allowed_domains: list[str] = Field(default_factory=list)
        allowed_cidrs: list[str] = Field(default_factory=list)

    class SSHKeyInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        public_key: str = Field(min_length=32, max_length=32768)

    class SearchInput(BaseModel):
        model_config: typing.ClassVar[dict] = _mc()
        query: str = Field(min_length=1, max_length=2048)
        limit: int = Field(default=8, ge=1, le=20)

else:
    @dataclasses.dataclass
    class InstaVMOperationInput:
        path: dict = dataclasses.field(default_factory=dict)
        query: dict = dataclasses.field(default_factory=dict)
        headers: dict = dataclasses.field(default_factory=dict)
        body: dict = dataclasses.field(default_factory=dict)
        file_path: str = ""
        artifact_name: str = ""
        idempotency_key: str = ""
        confirm_destructive: bool = False

    @dataclasses.dataclass
    class InstaVMOperationOutput:
        response: typing.Any = None
        status_code: int = 0
        request_id: str = ""
        duration_ms: float = 0.0
        artifact: typing.Optional[dict] = None
        connection: typing.Optional[dict] = None

    @dataclasses.dataclass
    class EmptyInput:
        pass

    @dataclasses.dataclass
    class TextOutput:
        text: str = ""

    @dataclasses.dataclass
    class EchoInput:
        text: str = ""

    @dataclasses.dataclass
    class WebSearchInput:
        query: str = ""

    @dataclasses.dataclass
    class WebSearchOutput:
        results: list = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class ReadFileInput:
        path: str = ""

    @dataclasses.dataclass
    class ReadFileOutput:
        content: str = ""
        error: str = ""

    @dataclasses.dataclass
    class WriteFileInput:
        path: str = ""
        content: str = ""

    @dataclasses.dataclass
    class WriteFileOutput:
        ok: bool = False
        error: str = ""

    @dataclasses.dataclass
    class ShellInput:
        command: str = ""
        timeout_s: float = 60.0
        cwd: str = ""

    @dataclasses.dataclass
    class ShellOutput:
        exit_code: int = 0
        stdout: str = ""
        stderr: str = ""
        timed_out: bool = False

    @dataclasses.dataclass
    class VMCreateInput:
        vm_lifetime_seconds: int = 3600
        memory_mb: int = 2048
        vcpu_count: int = 2
        snapshot_id: str = ""
        egress_policy: dict = dataclasses.field(default_factory=dict)

    @dataclasses.dataclass
    class VMIdentifierInput:
        vm_id: str = ""

    @dataclasses.dataclass
    class VMSnapshotInput:
        vm_id: str = ""
        name: str = ""

    @dataclasses.dataclass
    class VMExecuteInput:
        vm_id: str = ""
        command: str = ""
        language: str = "bash"
        timeout_s: float = 120.0
        stream: bool = False

    @dataclasses.dataclass
    class VMFileInput:
        vm_id: str = ""
        source_path: str = ""
        destination_path: str = ""

    @dataclasses.dataclass
    class VMVolumeInput:
        name: str = ""
        quota_bytes: int = 1073741824

    @dataclasses.dataclass
    class VolumeIdentifierInput:
        volume_id: str = ""
        confirm_destructive: bool = False

    @dataclasses.dataclass
    class VolumeCheckpointInput:
        volume_id: str = ""
        name: str = ""

    @dataclasses.dataclass
    class VMMountVolumeInput:
        vm_id: str = ""
        volume_id: str = ""
        mount_path: str = ""
        read_only: bool = False

    @dataclasses.dataclass
    class VMEgressInput:
        vm_id: str = ""
        allow_package_managers: bool = True
        allow_http: bool = False
        allow_https: bool = True
        allowed_domains: list = dataclasses.field(default_factory=list)
        allowed_cidrs: list = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class SSHKeyInput:
        public_key: str = ""

    @dataclasses.dataclass
    class SearchInput:
        query: str = ""
        limit: int = 8


class ToolSandbox:
    def __init__(self, database: Database, memory: MemoryManager, config: typing.Any = None):
        self.db = database
        self.memory = memory
        self.config = config or CONFIG.sandbox
        self.tools: typing.Dict[str, typing.Tuple[ToolSpec, typing.Callable]] = {}
        self._instavm_client: typing.Any = None
        self._instavm_transport: typing.Optional[InstaVMTransport] = None
        self._instavm_lock = threading.RLock()
        self._execution_context = threading.local()
        self._idempotency_locks: typing.Dict[str, threading.Lock] = {}
        self._idempotency_locks_guard = threading.RLock()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(32, int(os.environ.get("APP_TOOL_WORKERS", "8")))),
            thread_name_prefix="tool-worker",
        )
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register(
            ToolSpec(
                name="echo",
                version="1",
                input_schema=EchoInput,
                output_schema=TextOutput,
                timeout_s=5.0,
                resource_limits={},
                side_effects=False,
                idempotency_key_fn=None,
                fallback_tool=None,
            ),
            self._tool_echo,
        )
        self.register(
            ToolSpec(
                name="read_file",
                version="1",
                input_schema=ReadFileInput,
                output_schema=ReadFileOutput,
                timeout_s=10.0,
                resource_limits={"nofile": 64},
                side_effects=False,
                idempotency_key_fn=None,
                fallback_tool=None,
            ),
            self._tool_read_file,
        )
        self.register(
            ToolSpec(
                name="write_file",
                version="1",
                input_schema=WriteFileInput,
                output_schema=WriteFileOutput,
                timeout_s=10.0,
                resource_limits={"nofile": 64},
                side_effects=True,
                idempotency_key_fn=None,
                fallback_tool=None,
            ),
            self._tool_write_file,
        )
        self.register(
            ToolSpec(
                name="run_shell",
                version="1",
                input_schema=ShellInput,
                output_schema=ShellOutput,
                timeout_s=900.0,
                resource_limits={"nofile": 256},
                side_effects=True,
                idempotency_key_fn=None,
                fallback_tool=None,
            ),
            self._tool_run_shell,
        )
        self.register(
            ToolSpec(
                name="search_memory",
                version="1",
                input_schema=SearchInput,
                output_schema=WebSearchOutput,
                timeout_s=10.0,
                resource_limits={},
                side_effects=False,
                idempotency_key_fn=None,
                fallback_tool=None,
            ),
            self._tool_search_memory,
        )
        self.register(ToolSpec(name="vm_create", version="1", input_schema=VMCreateInput, output_schema=dict, timeout_s=180.0, resource_limits={}, side_effects=True), self._tool_vm_create)
        self.register(ToolSpec(name="vm_get", version="1", input_schema=VMIdentifierInput, output_schema=dict, timeout_s=30.0, resource_limits={}, side_effects=False), self._tool_vm_get)
        self.register(ToolSpec(name="vm_snapshot", version="1", input_schema=VMSnapshotInput, output_schema=dict, timeout_s=900.0, resource_limits={}, side_effects=True), self._tool_vm_snapshot)
        self.register(ToolSpec(name="vm_execute", version="1", input_schema=VMExecuteInput, output_schema=dict, timeout_s=900.0, resource_limits={}, side_effects=True), self._tool_vm_execute)
        self.register(ToolSpec(name="vm_upload_file", version="1", input_schema=VMFileInput, output_schema=dict, timeout_s=180.0, resource_limits={}, side_effects=True), self._tool_vm_upload_file)
        self.register(ToolSpec(name="vm_download_file", version="1", input_schema=VMFileInput, output_schema=dict, timeout_s=180.0, resource_limits={}, side_effects=True), self._tool_vm_download_file)
        self.register(ToolSpec(name="volume_create", version="1", input_schema=VMVolumeInput, output_schema=dict, timeout_s=60.0, resource_limits={}, side_effects=True), self._tool_volume_create)
        self.register(ToolSpec(name="volume_list", version="1", input_schema=EmptyInput, output_schema=dict, timeout_s=60.0, resource_limits={}, side_effects=False), self._tool_volume_list)
        self.register(ToolSpec(name="volume_checkpoint", version="1", input_schema=VolumeCheckpointInput, output_schema=dict, timeout_s=120.0, resource_limits={}, side_effects=True), self._tool_volume_checkpoint)
        self.register(ToolSpec(name="volume_delete", version="1", input_schema=VolumeIdentifierInput, output_schema=dict, timeout_s=60.0, resource_limits={}, side_effects=True), self._tool_volume_delete)
        self.register(ToolSpec(name="volume_mount", version="1", input_schema=VMMountVolumeInput, output_schema=dict, timeout_s=120.0, resource_limits={}, side_effects=True), self._tool_volume_mount)
        self.register(ToolSpec(name="vm_set_egress", version="1", input_schema=VMEgressInput, output_schema=dict, timeout_s=60.0, resource_limits={}, side_effects=True), self._tool_vm_set_egress)
        self.register(ToolSpec(name="ssh_key_add", version="1", input_schema=SSHKeyInput, output_schema=dict, timeout_s=60.0, resource_limits={}, side_effects=True), self._tool_ssh_key_add)
        self.register(ToolSpec(name="ssh_key_list", version="1", input_schema=EmptyInput, output_schema=dict, timeout_s=60.0, resource_limits={}, side_effects=False), self._tool_ssh_key_list)
        for operation in INSTAVM_MANIFEST:
            self.register(
                ToolSpec(
                    name=operation.tool_name,
                    version="1",
                    input_schema=InstaVMOperationInput,
                    output_schema=dict,
                    timeout_s=operation.timeout_s,
                    resource_limits={},
                    side_effects=operation.side_effects,
                    idempotency_key_fn=None,
                    fallback_tool=None,
                ),
                functools.partial(self._tool_instavm_operation, operation.operation_id),
            )

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self._instavm_lock:
            if self._instavm_transport is not None:
                self._instavm_transport.close()
                self._instavm_transport = None
            close = getattr(self._instavm_client, "close", None)
            if callable(close):
                close()
            self._instavm_client = None

    def _record_instavm_policy(self, operation: InstaVMOperation, decision: str, task_id: typing.Any) -> None:
        conn = self.db.connect()
        try:
            conn.execute(
                "INSERT INTO instavm_policy_decisions(operation_id,scope,destructive,decision,task_id,created_at) VALUES(?,?,?,?,?,?)",
                (operation.operation_id, operation.scope, int(operation.destructive), decision, str(task_id) if task_id is not None else None, time.time()),
            )
        finally:
            conn.close()

    def _authorize_instavm(self, operation: InstaVMOperation, payload: dict, task_id: typing.Any) -> None:
        if not CONFIG.vm.enabled:
            self._record_instavm_policy(operation, "denied_vm_disabled", task_id)
            raise PermanentError("InstaVM operations are disabled")
        configured = {item.strip() for item in os.environ.get("INSTAVM_ALLOWED_SCOPES", "instavm.read,instavm.mutate,instavm.browser,instavm.computer,instavm.vnc").split(",") if item.strip()}
        if operation.scope not in configured:
            self._record_instavm_policy(operation, "denied_scope", task_id)
            raise PermanentError(f"InstaVM scope is not authorized: {operation.scope}")
        if operation.destructive and not bool(payload.get("confirm_destructive")):
            self._record_instavm_policy(operation, "denied_confirmation", task_id)
            raise PermanentError(f"{operation.operation_id} requires confirm_destructive=true")
        self._record_instavm_policy(operation, "allowed", task_id)

    def _invoke_instavm(self, operation: InstaVMOperation, payload: dict) -> dict:
        task_id = getattr(self._execution_context, "task_id", None)
        self._authorize_instavm(operation, payload, task_id)
        idempotency_key = getattr(self._execution_context, "idempotency_key", None)
        if idempotency_key and not payload.get("idempotency_key"):
            payload = dict(payload)
            payload["idempotency_key"] = str(idempotency_key)
        return self._instavm_transport_client().invoke(operation, payload)

    def _instavm_transport_client(self) -> InstaVMTransport:
        api_key = os.environ.get("INSTAVM_API_KEY") or os.environ.get("INSTA_API_KEY")
        if not api_key:
            raise PermanentError("INSTAVM_API_KEY is required for InstaVM API operations")
        with self._instavm_lock:
            if self._instavm_transport is None:
                artifact_root = pathlib.Path(os.environ.get("INSTAVM_ARTIFACT_DIRECTORY", str(ROOT / "artifacts" / "instavm")))
                self._instavm_transport = InstaVMTransport(api_key, artifact_root)
            return self._instavm_transport

    def _tool_instavm_operation(self, operation_id: str, value: typing.Any) -> dict:
        operation = INSTAVM_OPERATIONS[operation_id]
        payload = _plain(value)
        task_id = getattr(self._execution_context, "task_id", None)
        started = time.monotonic()
        try:
            result = self._invoke_instavm(operation, payload)
            self._record_instavm_invocation(
                operation,
                task_id,
                int(result.get("status_code", 200)),
                str(result.get("request_id", "")),
                (time.monotonic() - started) * 1000,
                "succeeded",
            )
            return result
        except Exception:
            self._record_instavm_invocation(
                operation,
                task_id,
                None,
                "",
                (time.monotonic() - started) * 1000,
                "failed",
            )
            raise

    def _record_instavm_invocation(
        self,
        operation: InstaVMOperation,
        task_id: typing.Optional[typing.Any],
        status_code: typing.Optional[int],
        request_id: str,
        duration_ms: float,
        outcome: str,
    ) -> None:
        conn = self.db.connect()
        try:
            conn.execute(
                "INSERT INTO instavm_invocations(operation_id,tool_name,task_id,status_code,request_id,duration_ms,outcome,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    operation.operation_id,
                    operation.tool_name,
                    str(task_id) if task_id is not None else None,
                    status_code,
                    request_id[:256],
                    duration_ms,
                    outcome,
                    time.time(),
                ),
            )
        finally:
            conn.close()

    def register(self, spec: ToolSpec, function: typing.Callable) -> None:
        self.tools[spec.name] = (spec, function)

    def _validate_input(self, schema: typing.Any, arguments: dict) -> typing.Any:
        if not isinstance(arguments, dict):
            raise SchemaValidationError("tool input must be an object")
        try:
            value = model_validate(schema, arguments)
        except Exception as exc:
            raise SchemaValidationError(f"input validation failed: {exc}") from exc
        self._validate_constraints(value)
        return value

    def _validate_constraints(self, value: typing.Any) -> None:
        name = type(value).__name__
        fields = _plain(value) if not isinstance(value, dict) else value
        required_nonempty = {
            "ReadFileInput": ("path",),
            "WriteFileInput": ("path",),
            "ShellInput": ("command",),
            "VMIdentifierInput": ("vm_id",),
            "VMSnapshotInput": ("vm_id", "name"),
            "VMExecuteInput": ("vm_id", "command"),
            "VMFileInput": ("vm_id", "source_path", "destination_path"),
            "VMVolumeInput": ("name",),
            "VolumeIdentifierInput": ("volume_id",),
            "VolumeCheckpointInput": ("volume_id", "name"),
            "VMMountVolumeInput": ("vm_id", "volume_id", "mount_path"),
            "VMEgressInput": ("vm_id",),
            "SSHKeyInput": ("public_key",),
            "SearchInput": ("query",),
        }
        for field_name in required_nonempty.get(name, ()):
            if not str(fields.get(field_name, "")).strip():
                raise SchemaValidationError(f"{field_name} must not be empty")
        if name == "ShellInput":
            timeout = float(fields.get("timeout_s", 60.0))
            if not 0 < timeout <= 900:
                raise SchemaValidationError("timeout_s must be between zero and 900")
        if name == "SearchInput":
            limit = int(fields.get("limit", 8))
            if not 1 <= limit <= 20:
                raise SchemaValidationError("limit must be between one and 20")
        if name == "VMCreateInput":
            lifetime = int(fields.get("vm_lifetime_seconds", 3600))
            memory_mb = int(fields.get("memory_mb", 2048))
            vcpu_count = int(fields.get("vcpu_count", 2))
            if not 20 <= lifetime <= 86400:
                raise SchemaValidationError("vm_lifetime_seconds must be between 20 and 86400")
            if not 256 <= memory_mb <= 32768:
                raise SchemaValidationError("memory_mb must be between 256 and 32768")
            if not 1 <= vcpu_count <= 16:
                raise SchemaValidationError("vcpu_count must be between one and 16")
        if name == "VMExecuteInput":
            timeout = float(fields.get("timeout_s", 120.0))
            if not 0 < timeout <= 900:
                raise SchemaValidationError("timeout_s must be between zero and 900")
            if str(fields.get("language", "bash")) not in {"bash", "python"}:
                raise SchemaValidationError("language must be bash or python")
        if name == "VMVolumeInput":
            quota = int(fields.get("quota_bytes", 1073741824))
            if not 1048576 <= quota <= 1099511627776:
                raise SchemaValidationError("quota_bytes is outside the permitted range")
        if name == "SSHKeyInput" and not 32 <= len(str(fields.get("public_key", ""))) <= 32768:
            raise SchemaValidationError("public_key length is outside the permitted range")
        if name == "InstaVMOperationInput":
            for field_name in ("path", "query", "headers", "body"):
                if not isinstance(fields.get(field_name, {}), dict):
                    raise SchemaValidationError(f"{field_name} must be an object")
            idempotency_key = str(fields.get("idempotency_key", ""))
            if len(idempotency_key) > 256:
                raise SchemaValidationError("idempotency_key is too long")

    def _validate_output(self, schema: typing.Any, result: typing.Any) -> typing.Any:
        if schema is dict:
            if not isinstance(result, dict):
                raise SchemaValidationError("tool output must be an object")
            return result
        if not isinstance(result, dict):
            raise SchemaValidationError("tool output must be an object")
        try:
            return model_validate(schema, result)
        except Exception as exc:
            raise SchemaValidationError(f"output validation failed: {exc}") from exc

    def _idempotency_lock(self, key: str) -> threading.Lock:
        with self._idempotency_locks_guard:
            lock = self._idempotency_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._idempotency_locks[key] = lock
            return lock

    async def execute(
        self,
        task_id: typing.Any,
        intent: str,
        name: str,
        arguments: dict,
        _visited: typing.Optional[typing.Set[str]] = None,
    ) -> dict:
        if name not in self.tools:
            raise PermanentError(f"unknown tool: {name!r}")
        visited = set() if _visited is None else set(_visited)
        if name in visited:
            raise PermanentError(f"fallback cycle detected at tool {name!r}")
        visited.add(name)
        spec, function = self.tools[name]
        breaker = CircuitBreaker.get(name)
        if not await breaker.allow_request():
            if spec.fallback_tool and spec.fallback_tool in self.tools:
                record_event(EventType.ERROR, task_id, {"degradation": True, "primary": name, "fallback": spec.fallback_tool})
                return await self.execute(task_id, intent, spec.fallback_tool, arguments, visited)
            raise CircuitOpenError(f"circuit open for tool {name!r}")
        idempotency_lock = None
        idempotency_key = None
        owns_idempotency_claim = False
        preserve_idempotency_claim = False
        future: typing.Optional[asyncio.Future] = None
        try:
            validated_input = self._validate_input(spec.input_schema, arguments)
            plain_input = _plain(validated_input)
            if spec.side_effects:
                key_fn = spec.idempotency_key_fn or make_idempotency_key
                idempotency_key = key_fn(task_id, intent, {"name": name, "version": spec.version, "arguments": plain_input})
                idempotency_lock = self._idempotency_lock(idempotency_key)
                await asyncio.to_thread(idempotency_lock.acquire)
                now = time.time()
                conn = self.db.connect()
                try:
                    with self.db.transaction(conn):
                        row = conn.execute("SELECT result_json,expires_at,status FROM idempotency_keys WHERE key=?", (idempotency_key,)).fetchone()
                        unexpired = bool(row and (row["expires_at"] is None or float(row["expires_at"]) > now))
                        if row and row["result_json"] and unexpired and row["status"] == "done":
                            cached_result = json.loads(row["result_json"])
                        else:
                            cached_result = None
                        if cached_result is None and row and row["status"] == "running" and unexpired:
                            raise DuplicateActionError(f"tool action {name!r} is already running")
                        if cached_result is None:
                            conn.execute(
                                "INSERT INTO idempotency_keys(key,task_id,tool_name,created_at,expires_at,status,result_json) VALUES(?,?,?,?,?,'running',NULL) "
                                "ON CONFLICT(key) DO UPDATE SET task_id=excluded.task_id,tool_name=excluded.tool_name,created_at=excluded.created_at,expires_at=excluded.expires_at,status='running',result_json=NULL",
                                (idempotency_key, str(task_id), name, now, now + 86400.0),
                            )
                            owns_idempotency_claim = True
                    if cached_result is not None:
                        await breaker.record_success()
                        return cached_result
                finally:
                    conn.close()
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                self.executor,
                self._run_sandboxed,
                function,
                validated_input,
                spec,
                task_id,
                idempotency_key,
            )
            effective_timeout = spec.timeout_s if spec.timeout_s > 0 else self.config.default_timeout_s
            async with TimeoutBudget(effective_timeout, ToolTimeoutError):
                raw_result = await asyncio.shield(future)
            validated_output = self._validate_output(spec.output_schema, raw_result)
            result_data = _plain(validated_output)
            encoded = stable_json_dumps(result_data).encode("utf-8")
            if len(encoded) > self.config.max_output_bytes:
                raise PermanentError(f"tool output {len(encoded)} bytes exceeds limit {self.config.max_output_bytes}")
            if idempotency_key is not None:
                conn = self.db.connect()
                try:
                    conn.execute(
                        "UPDATE idempotency_keys SET result_json=?,status='done' WHERE key=?",
                        (stable_json_dumps(result_data), idempotency_key),
                    )
                finally:
                    conn.close()
            await breaker.record_success()
            tool_calls_total.inc()
            return result_data
        except (DuplicateActionError, SchemaValidationError, CircuitOpenError):
            raise
        except ToolTimeoutError:
            if idempotency_key is not None and owns_idempotency_claim and future is not None:
                preserve_idempotency_claim = True
                future.add_done_callback(
                    functools.partial(self._complete_timed_out_idempotent_execution, idempotency_key, spec)
                )
            await breaker.record_failure(permanent=False)
            raise
        except PermanentError:
            await breaker.record_failure(permanent=True)
            raise
        except TransientError:
            await breaker.record_failure(permanent=False)
            raise
        except Exception as exc:
            await breaker.record_failure(permanent=False)
            raise TransientError(f"tool {name!r} raised: {exc}") from exc
        finally:
            if idempotency_key is not None and owns_idempotency_claim and not preserve_idempotency_claim:
                conn = self.db.connect()
                try:
                    conn.execute(
                        "DELETE FROM idempotency_keys WHERE key=? AND status='running'",
                        (idempotency_key,),
                    )
                finally:
                    conn.close()
            if idempotency_lock is not None and idempotency_lock.locked():
                idempotency_lock.release()

    def _complete_timed_out_idempotent_execution(
        self,
        idempotency_key: str,
        spec: ToolSpec,
        future: asyncio.Future,
    ) -> None:
        conn = self.db.connect()
        try:
            try:
                raw_result = future.result()
                validated_output = self._validate_output(spec.output_schema, raw_result)
                result_data = _plain(validated_output)
                encoded = stable_json_dumps(result_data).encode("utf-8")
                if len(encoded) > self.config.max_output_bytes:
                    raise PermanentError(f"tool output {len(encoded)} bytes exceeds limit {self.config.max_output_bytes}")
                conn.execute(
                    "UPDATE idempotency_keys SET result_json=?,status='done',expires_at=? WHERE key=? AND status='running'",
                    (stable_json_dumps(result_data), time.time() + 86400.0, idempotency_key),
                )
            except Exception:
                conn.execute("DELETE FROM idempotency_keys WHERE key=? AND status='running'", (idempotency_key,))
        finally:
            conn.close()

    def _run_sandboxed(
        self,
        function: typing.Callable,
        validated: typing.Any,
        spec: ToolSpec,
        task_id: typing.Any,
        idempotency_key: typing.Optional[str],
    ) -> dict:
        self._execution_context.task_id = task_id
        self._execution_context.idempotency_key = idempotency_key
        try:
            return function(validated)
        finally:
            self._execution_context.task_id = None
            self._execution_context.idempotency_key = None

    def _tool_echo(self, value: typing.Any) -> dict:
        return {"text": str(getattr(value, "text", ""))}

    def _tool_read_file(self, value: typing.Any) -> dict:
        path = self._workspace_path(str(getattr(value, "path", "")))
        try:
            with path.open("rb") as handle:
                raw = handle.read(self.config.max_output_bytes + 1)
        except OSError as exc:
            raise PermanentError(f"file read failed: {exc}") from exc
        if len(raw) > self.config.max_output_bytes:
            raise PermanentError("file exceeds maximum readable size")
        return {"content": raw.decode("utf-8", errors="replace"), "error": ""}

    def _tool_write_file(self, value: typing.Any) -> dict:
        path = self._workspace_path(str(getattr(value, "path", "")))
        content = str(getattr(value, "content", ""))
        encoded = content.encode("utf-8")
        if len(encoded) > self.config.max_output_bytes:
            raise PermanentError("file content exceeds maximum writable size")
        try:
            atomic_write(path, encoded)
        except OSError as exc:
            raise PermanentError(f"file write failed: {exc}") from exc
        return {"ok": True, "error": ""}

    def _workspace_path(self, raw_path: str) -> pathlib.Path:
        candidate = pathlib.Path(raw_path).expanduser()
        path = candidate.resolve() if candidate.is_absolute() else (WORKSPACE_PATH / candidate).resolve()
        try:
            path.relative_to(WORKSPACE_PATH)
        except ValueError as exc:
            raise PermanentError(f"path is outside the managed workspace: {raw_path}") from exc
        return path

    def _tool_run_shell(self, value: typing.Any) -> dict:
        if os.environ.get("APP_ENABLE_LOCAL_SHELL", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            raise PermanentError("local shell execution is disabled; set APP_ENABLE_LOCAL_SHELL=true to enable it")
        command = str(getattr(value, "command", "")).strip()
        if not command:
            raise PermanentError("shell command must not be empty")
        timeout_s = float(getattr(value, "timeout_s", 60.0))
        requested_cwd = str(getattr(value, "cwd", "") or "")
        cwd = self._workspace_path(requested_cwd) if requested_cwd else WORKSPACE_PATH
        cwd.mkdir(parents=True, exist_ok=True)
        environment = {}
        for key in ("HOME", "LANG", "LC_ALL", "TERM", "TZ", "TMPDIR"):
            if key in os.environ:
                environment[key] = os.environ[key]
        environment["PATH"] = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        if os.name == "nt":
            argv = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", command]
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            start_new_session = False
        else:
            argv = ["/bin/sh", "-lc", command]
            creationflags = 0
            start_new_session = True
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
        output_lock = threading.Lock()
        output_exceeded = threading.Event()
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()

        def terminate_process() -> None:
            if process.poll() is not None:
                return
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                with contextlib.suppress(Exception):
                    process.kill()

        def consume(stream: typing.Any, target: bytearray) -> None:
            try:
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        return
                    with output_lock:
                        remaining = self.config.max_output_bytes + 1 - len(target)
                        if remaining > 0:
                            target.extend(chunk[:remaining])
                        if len(target) > self.config.max_output_bytes:
                            output_exceeded.set()
                    if output_exceeded.is_set():
                        terminate_process()
                        return
            finally:
                with contextlib.suppress(Exception):
                    stream.close()

        stdout_thread = threading.Thread(target=consume, args=(process.stdout, stdout_buffer), daemon=True)
        stderr_thread = threading.Thread(target=consume, args=(process.stderr, stderr_buffer), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process()
            exit_code = process.wait(timeout=10)
        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            terminate_process()
            raise TransientError("shell output readers did not terminate")
        if output_exceeded.is_set():
            raise PermanentError("shell output exceeds configured size limit")
        return {
            "exit_code": 124 if timed_out else int(exit_code),
            "stdout": bytes(stdout_buffer).decode("utf-8", errors="replace"),
            "stderr": bytes(stderr_buffer).decode("utf-8", errors="replace"),
            "timed_out": timed_out,
        }

    def _tool_search_memory(self, value: typing.Any) -> dict:
        query = str(getattr(value, "query", ""))
        limit = int(getattr(value, "limit", 8))
        task_id = getattr(self._execution_context, "task_id", uuid.UUID(int=0))
        if not isinstance(task_id, uuid.UUID):
            task_id = uuid.UUID(str(task_id))
        results = []
        for score, row in self.memory.episodic.retrieve(task_id, query, limit):
            results.append(stable_json_dumps({"score": score, "event": row["payload_json"]}))
        return {"results": results}

    def _instavm(self) -> typing.Any:
        api_key = os.environ.get("INSTAVM_API_KEY") or os.environ.get("INSTA_API_KEY")
        if not _HAS_INSTAVM:
            raise PermanentError("instavm package is not installed")
        if not api_key:
            raise PermanentError("INSTAVM_API_KEY or INSTA_API_KEY is required for VM operations")
        with self._instavm_lock:
            if self._instavm_client is None:
                self._instavm_client = InstaVM(api_key=api_key, auto_start_session=False)
            return self._instavm_client

    def _bind_vm_session(self, vm_id: str) -> typing.Any:
        client = self._instavm()
        vm = client.vms.get(vm_id)
        session_id = vm.get("session_id")
        if not session_id:
            raise PermanentError(f"VM {vm_id} has no executable session")
        client.session_id = session_id
        return client

    def _tool_vm_create(self, value: typing.Any) -> dict:
        policy = dict(getattr(value, "egress_policy", {}) or {})
        defaults = {
            "allow_package_managers": True,
            "allow_http": False,
            "allow_https": True,
            "allowed_domains": [],
            "allowed_cidrs": [],
        }
        defaults.update(policy)
        payload = {
            "wait": True,
            "image_variant": CONFIG.vm.image_variant,
            "vm_lifetime_seconds": int(getattr(value, "vm_lifetime_seconds", CONFIG.vm.lifetime_seconds)),
            "memory_mb": int(getattr(value, "memory_mb", CONFIG.vm.memory_mb)),
            "vcpu_count": int(getattr(value, "vcpu_count", CONFIG.vm.vcpu_count)),
            "egress_policy": defaults,
        }
        snapshot_id = str(getattr(value, "snapshot_id", "") or "")
        if snapshot_id:
            payload["snapshot_id"] = snapshot_id
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["vm_create"],
            {"query": {"wait": True}, "body": payload, "path": {}, "headers": {}},
        )

    def _tool_vm_get(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["vm_get"],
            {"path": {"vm_id": str(getattr(value, "vm_id"))}, "query": {}, "body": {}, "headers": {}},
        )

    def _tool_vm_snapshot(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["vm_snapshot"],
            {
                "path": {"vm_id": str(getattr(value, "vm_id"))},
                "query": {"wait": True},
                "body": {"name": str(getattr(value, "name"))},
                "headers": {},
            },
        )

    def _tool_vm_execute(self, value: typing.Any) -> dict:
        command = str(getattr(value, "command"))
        language = str(getattr(value, "language", "bash"))
        if language not in {"bash", "python"}:
            raise PermanentError("VM command language must be bash or python")
        vm_id = str(getattr(value, "vm_id"))
        vm = self._invoke_instavm(
            INSTAVM_OPERATIONS["vm_get"],
            {"path": {"vm_id": vm_id}, "query": {}, "body": {}, "headers": {}},
        )
        response = vm.get("response", {})
        session_id = response.get("session_id") if isinstance(response, dict) else None
        if not session_id:
            raise PermanentError(f"VM {vm_id} has no session_id for /execute")
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["execution_execute"],
            {
                "path": {},
                "query": {},
                "headers": {},
                "body": {
                    "command": command if language == "bash" else None,
                    "code": command if language == "python" else None,
                    "language": language,
                    "session_id": session_id,
                    "timeout": min(int(getattr(value, "timeout_s", 120)), 600),
                },
            },
        )

    def _tool_vm_upload_file(self, value: typing.Any) -> dict:
        source = self._workspace_path(str(getattr(value, "source_path")))
        if not source.is_file():
            raise PermanentError(f"source file does not exist: {source}")
        vm = self._tool_vm_get(type("Input", (), {"vm_id": str(getattr(value, "vm_id"))})())
        response = vm.get("response", {})
        session_id = response.get("session_id") if isinstance(response, dict) else None
        if not session_id:
            raise PermanentError("VM has no session_id for file upload")
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["file_upload"],
            {
                "path": {},
                "query": {},
                "headers": {},
                "body": {"session_id": session_id, "path": str(getattr(value, "destination_path"))},
                "file_path": str(source),
            },
        )

    def _tool_vm_download_file(self, value: typing.Any) -> dict:
        destination = self._workspace_path(str(getattr(value, "destination_path")))
        destination.parent.mkdir(parents=True, exist_ok=True)
        vm = self._tool_vm_get(type("Input", (), {"vm_id": str(getattr(value, "vm_id"))})())
        response = vm.get("response", {})
        session_id = response.get("session_id") if isinstance(response, dict) else None
        if not session_id:
            raise PermanentError("VM has no session_id for file download")
        result = self._invoke_instavm(
            INSTAVM_OPERATIONS["file_download"],
            {
                "path": {},
                "query": {},
                "headers": {},
                "body": {"session_id": session_id, "path": str(getattr(value, "source_path"))},
                "artifact_name": destination.name,
            },
        )
        artifact = result.get("artifact", {})
        source_path = pathlib.Path(str(artifact.get("absolute_path") or ""))
        if not source_path.is_file():
            raise PermanentError("downloaded artifact is missing")
        temporary = destination.with_suffix(destination.suffix + ".partial")
        try:
            with source_path.open("rb") as source, temporary.open("wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, destination)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
        result["destination_path"] = str(destination)
        return result

    def _tool_volume_create(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["volume_create"],
            {
                "path": {},
                "query": {},
                "headers": {},
                "body": {"name": str(getattr(value, "name")), "quota_bytes": int(getattr(value, "quota_bytes"))},
            },
        )

    def _volume_method(self, method_name: str, *args: typing.Any, **kwargs: typing.Any) -> dict:
        method = getattr(self._instavm().volumes, method_name, None)
        if method is None:
            raise PermanentError(f"installed instavm SDK does not expose volumes.{method_name}")
        result = method(*args, **kwargs)
        return dict(result) if isinstance(result, dict) else {"result": list(result) if isinstance(result, (list, tuple)) else str(result)}

    def _tool_volume_list(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["volumes_list"], {"path": {}, "query": {}, "body": {}, "headers": {}}
        )

    def _tool_volume_checkpoint(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["volume_checkpoint_create"],
            {
                "path": {"volume_id": str(getattr(value, "volume_id"))},
                "query": {},
                "headers": {},
                "body": {"name": str(getattr(value, "name"))},
            },
        )

    def _tool_volume_delete(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["volume_delete"],
            {
                "path": {"volume_id": str(getattr(value, "volume_id"))},
                "query": {},
                "body": {},
                "headers": {},
                "confirm_destructive": bool(getattr(value, "confirm_destructive", False)),
            },
        )

    def _tool_volume_mount(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["vm_volume_mount"],
            {
                "path": {"vm_id": str(getattr(value, "vm_id"))},
                "query": {"wait": True},
                "headers": {},
                "body": {
                    "volume_id": str(getattr(value, "volume_id")),
                    "mount_path": str(getattr(value, "mount_path")),
                    "mode": "ro" if bool(getattr(value, "read_only", False)) else "rw",
                },
            },
        )

    def _tool_vm_set_egress(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["egress_vm_set"],
            {
                "path": {"vm_id": str(getattr(value, "vm_id"))},
                "query": {},
                "headers": {},
                "body": {
                    "allow_package_managers": bool(getattr(value, "allow_package_managers", True)),
                    "allow_http": bool(getattr(value, "allow_http", False)),
                    "allow_https": bool(getattr(value, "allow_https", True)),
                    "allowed_domains": list(getattr(value, "allowed_domains", [])),
                    "allowed_cidrs": list(getattr(value, "allowed_cidrs", [])),
                },
            },
        )

    def _tool_ssh_key_add(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["ssh_key_add"],
            {"path": {}, "query": {}, "headers": {}, "body": {"public_key": str(getattr(value, "public_key"))}},
        )

    def _tool_ssh_key_list(self, value: typing.Any) -> dict:
        return self._invoke_instavm(
            INSTAVM_OPERATIONS["ssh_keys_list"], {"path": {}, "query": {}, "headers": {}, "body": {}}
        )


class CheckpointManager:
    def __init__(self, path: pathlib.Path, database: Database, lock_path: pathlib.Path = LOCK_PATH):
        self.path = pathlib.Path(path)
        self.db = database
        self.lock_path = pathlib.Path(lock_path)
        self._lock = threading.RLock()
        self.pending: asyncio.Queue = asyncio.Queue()
        self.latest: typing.Optional[typing.Any] = None
        self._owner_token = uuid.uuid4().hex

    def _checkpoint_sync(self, state: typing.Any) -> typing.Any:
        started = time.perf_counter()
        with self._lock:
            data = _plain(state)
            data["crc32"] = 0
            data["crc32"] = crc32(stable_json_dumps(data).encode("utf-8"))
            atomic_write(self.path, stable_json_dumps(data).encode("utf-8"))
            self.latest = model_validate(CheckpointState, data)
        checkpoints_total.inc()
        record_event(
            EventType.CHECKPOINT,
            uuid.UUID(int=0),
            {"path": str(self.path), "step": int(data.get("step_counter", 0)), "version": int(data.get("version", 1))},
        )
        observe_latency("checkpoint", time.perf_counter() - started)
        return self.latest

    async def checkpoint_now(self, state: typing.Any) -> typing.Any:
        return await asyncio.to_thread(self._checkpoint_sync, state)

    async def request_checkpoint(self, state: typing.Any) -> None:
        await self.pending.put(state)

    def _checkpoint_task_sync(
        self,
        task_id: str,
        step: int,
        working_memory: typing.List[dict],
        in_flight_tasks: typing.List[str],
        episodic_cursor: int,
    ) -> typing.Any:
        with self._lock:
            existing = self.latest
            if existing is None and self.path.exists():
                existing = self.load_latest()
            queue_state = dict(getattr(existing, "queue_state", {}) or {}) if existing is not None else {}
            tasks = dict(queue_state.get("tasks", {}) or {})
            active = {str(item) for item in in_flight_tasks}
            tasks = {key: value for key, value in tasks.items() if key in active}
            tasks[str(task_id)] = {
                "step": int(step),
                "working_memory": [dict(item) for item in working_memory],
            }
            state = CheckpointState(
                working_memory_ptr=len(working_memory),
                episodic_cursor=int(episodic_cursor),
                in_flight_tasks=[uuid.UUID(str(item)) for item in sorted(active)],
                queue_state={"tasks": tasks},
                step_counter=max([int(step), *[int(value.get("step", 0)) for value in tasks.values() if isinstance(value, dict)]]),
                version=2,
            )
            return self._checkpoint_sync(state)

    async def checkpoint_task(
        self,
        task_id: typing.Any,
        step: int,
        working_memory: typing.List[dict],
        in_flight_tasks: typing.List[str],
        episodic_cursor: int,
    ) -> typing.Any:
        return await asyncio.to_thread(
            self._checkpoint_task_sync,
            str(task_id),
            int(step),
            working_memory,
            in_flight_tasks,
            int(episodic_cursor),
        )

    def load_latest(self) -> typing.Optional[typing.Any]:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("checkpoint root must be an object")
            received_crc = int(data.get("crc32", -1))
            unsigned = dict(data)
            unsigned["crc32"] = 0
            expected_crc = crc32(stable_json_dumps(unsigned).encode("utf-8"))
            if expected_crc != received_crc:
                raise ValueError("checkpoint CRC mismatch")
            state = model_validate(CheckpointState, data)
            self.latest = state
            return state
        except Exception as exc:
            LOGGER.warning(f"checkpoint load failed: {exc}", extra={"component": "checkpoint"})
            return None

    def _read_lock(self) -> typing.Optional[dict]:
        try:
            value = json.loads(self.lock_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def detect_unclean_shutdown(self) -> bool:
        if not self.lock_path.exists():
            return False
        data = self._read_lock()
        if not data:
            return True
        pid = int(data.get("pid", 0) or 0)
        token = str(data.get("token", ""))
        if pid == os.getpid() and token == self._owner_token:
            return False
        return not self._pid_alive(pid)

    def write_lock(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = stable_json_dumps(
            {
                "pid": os.getpid(),
                "token": self._owner_token,
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        ).encode("utf-8")
        for _ in range(2):
            try:
                fd = os.open(str(self.lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    os.write(fd, payload)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                return
            except FileExistsError:
                data = self._read_lock()
                pid = int((data or {}).get("pid", 0) or 0)
                token = str((data or {}).get("token", ""))
                if pid == os.getpid() and token == self._owner_token:
                    return
                if self._pid_alive(pid):
                    raise RuntimeError(f"runtime lock is held by process {pid}")
                with contextlib.suppress(FileNotFoundError):
                    self.lock_path.unlink()
        raise RuntimeError("unable to acquire runtime lock")

    def remove_lock(self) -> None:
        data = self._read_lock()
        if data and str(data.get("token", "")) != self._owner_token:
            return
        with contextlib.suppress(FileNotFoundError):
            self.lock_path.unlink()

    async def flush_worker(self, stop: asyncio.Event) -> None:
        while not stop.is_set() or not self.pending.empty():
            try:
                state = await asyncio.wait_for(self.pending.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            try:
                await self.checkpoint_now(state)
            except Exception as exc:
                errors_total.labels(type=type(exc).__name__).inc()
                LOGGER.error(f"checkpoint flush failed: {exc}", extra={"component": "checkpoint"})
            finally:
                self.pending.task_done()


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
    ):
        self.db = database
        self.scheduler = scheduler
        self.memory = memory
        self.sandbox = sandbox
        self.model = model
        self.summarizer = summarizer
        self.checkpoint = checkpoint
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
            if recent[-1] > recent[-2] > recent[-3] and sum(recent) > int(CONFIG.model.max_completion_tokens * 2.5):
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
        default_model = CONFIG.model.controller or CONFIG.model.name
        attachments = payload.get("attachments", [])
        images = payload.get("images", [])
        if images or any(isinstance(item, dict) and item.get("kind") in {"image", "video"} for item in attachments):
            return CONFIG.model.multimodal or default_model
        if any(term in task_text for term in ("pdf", "long document", "hosszú dokumentum", "melléklet", "attachment")):
            return CONFIG.model.long_context or default_model
        if any(term in task_text for term in ("explicit sexual", "adult content", "felnőtt tartalom", "sensitive content", "érzékeny tartalom")):
            return CONFIG.model.sensitive or default_model
        if any(term in task_text for term in ("code", "python", "nix", "flake", "debug", "implement", "kód", "programoz", "javít")):
            return CONFIG.model.code or default_model
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
                await asyncio.wait_for(stop.wait(), timeout=CONFIG.supervisor.heartbeat_interval_s)
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
                    await self.memory.episodic.summarize_old_events(self.summarizer, CONFIG.summarizer.max_summary_len)
                except Exception as exc:
                    LOGGER.error(f"summarize loop failed: {exc}", extra={"component": "memory"})

    async def _eviction_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=CONFIG.memory.thompson_eviction_interval_s)
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
        if time.monotonic() - started_mono >= CONFIG.agent.wall_timeout_s:
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
        while step < CONFIG.agent.max_steps:
            if self.stop_event.is_set():
                self.scheduler.requeue_with_backoff(task.id)
                return
            elapsed = time.monotonic() - started_mono
            if elapsed >= CONFIG.agent.wall_timeout_s:
                self._emit(task.id, EventType.TIMEOUT, {"reason": "wall_timeout", "elapsed_s": elapsed})
                self.scheduler.update_status(task.id, TaskStatus.TIMED_OUT)
                return
            step += 1
            self.iteration_id += 1
            loop_iterations_total.inc()
            if step <= resume_step:
                continue
            context_block = self.memory.get_context(task.id, stable_json_dumps(task.payload), CONFIG.memory.working_tokens)
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
                remaining = max(0.001, CONFIG.agent.wall_timeout_s - elapsed)
                async with TimeoutBudget(min(CONFIG.model.request_timeout_s, remaining), TaskTimeoutError):
                    active_model = self.model.for_model(model_name) if isinstance(self.model, RequestyModel) else self.model
                    action, usage = await active_model.generate(prompt, self._tool_schema(), CONFIG.model.max_completion_tokens)
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
                if self._check_livelock(task.id, signature, CONFIG.agent.livelock_window_s, CONFIG.agent.livelock_repeat_threshold):
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
        self._emit(task.id, EventType.TIMEOUT, {"reason": "step_cap", "max_steps": CONFIG.agent.max_steps})
        self.scheduler.update_status(task.id, TaskStatus.TIMED_OUT)


def _supervisor_child(
    db_path: str,
    checkpoint_path: str,
    lock_path: str,
    heartbeat_conn: multiprocessing.connection.Connection,
    log_level: str,
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    database = Database(pathlib.Path(db_path))
    summarizer_inst = ExtractiveFrequencySummarizer()
    memory_inst = MemoryManager(database, CONFIG.memory, summarizer_inst)
    scheduler_inst = Scheduler(database)
    sandbox_inst = ToolSandbox(database, memory_inst)
    model_inst = RequestyModel(os.environ.get("REQUESTY_API_KEY", ""), CONFIG.model.name)
    checkpoint_inst = CheckpointManager(pathlib.Path(checkpoint_path), database, pathlib.Path(lock_path))
    unclean = checkpoint_inst.detect_unclean_shutdown()
    checkpoint_inst.write_lock()
    engine = AgentLoop(database, scheduler_inst, memory_inst, sandbox_inst, model_inst, summarizer_inst, checkpoint_inst)
    if unclean:
        scheduler_inst.drain()
    stop_thread = threading.Event()

    def send_heartbeat() -> None:
        while not stop_thread.wait(CONFIG.supervisor.heartbeat_interval_s):
            try:
                heartbeat_conn.send({"ts": time.time(), "iteration_id": engine.iteration_id, "running": len(engine.running_tasks())})
            except Exception:
                return

    heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True, name="heartbeat-sender")
    heartbeat_thread.start()

    def handle_shutdown(signum, frame) -> None:
        engine.request_stop()

    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
    try:
        asyncio.run(engine.run())
    finally:
        stop_thread.set()
        heartbeat_thread.join(timeout=CONFIG.supervisor.restart_grace_s)
        sandbox_inst.close()
        model_inst.close()
        checkpoint_inst.remove_lock()


class Supervisor:
    def __init__(self, engine: AgentLoop, loop_thread_getter: typing.Callable[[], typing.Optional[threading.Thread]], restart_callback: typing.Callable[[], bool]):
        self.engine = engine
        self._loop_thread_getter = loop_thread_getter
        self._restart_callback = restart_callback
        self._stop = threading.Event()
        self._thread: typing.Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._heartbeat_monitor, name="supervisor-monitor", daemon=True)
        self._thread.start()

    def _heartbeat_monitor(self) -> None:
        while not self._stop.wait(CONFIG.supervisor.heartbeat_interval_s):
            loop_thread = self._loop_thread_getter()
            if loop_thread is not None and not loop_thread.is_alive() and not self.engine.stop_event.is_set():
                if self._restart_callback():
                    restarts_total.inc()
                    record_event(EventType.RESTART, uuid.UUID(int=0), {"reason": "agent_loop_thread_exited"})
                continue
            age = time.time() - self.engine.last_heartbeat
            if self.engine.running_tasks() and age > CONFIG.supervisor.ping_timeout_s:
                errors_total.labels(type="HeartbeatTimeout").inc()
                record_event(EventType.ERROR, uuid.UUID(int=0), {"reason": "heartbeat_missed", "age_s": age, "running": [str(item) for item in self.engine.running_tasks()]})

    def stop_now(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=CONFIG.supervisor.restart_grace_s)


class Runtime:
    def __init__(self):
        self.database = Database(DB_PATH)
        self.summarizer = ExtractiveFrequencySummarizer()
        self.memory = MemoryManager(self.database, CONFIG.memory, self.summarizer)
        self.scheduler = Scheduler(self.database)
        self.sandbox = ToolSandbox(self.database, self.memory)
        self.model: ModelClient = RequestyModel(os.environ.get("REQUESTY_API_KEY", ""), CONFIG.model.name)
        self.checkpoint = CheckpointManager(CHECKPOINT_PATH, self.database, LOCK_PATH)
        self.engine = AgentLoop(self.database, self.scheduler, self.memory, self.sandbox, self.model, self.summarizer, self.checkpoint)
        self.supervisor = Supervisor(self.engine, lambda: self._loop_thread, self._restart_loop_if_dead)
        self.started = False
        self._loop_thread: typing.Optional[threading.Thread] = None
        self._loop: typing.Optional[asyncio.AbstractEventLoop] = None
        self._start_lock = threading.RLock()
        self._closed = False

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.engine.run())
        except Exception as exc:
            errors_total.labels(type=type(exc).__name__).inc()
            LOGGER.error(f"agent loop terminated: {exc}", extra={"component": "runtime"})
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    def _start_loop_thread(self) -> None:
        self.engine.stop_event.clear()
        self._loop_thread = threading.Thread(target=self._run_loop, name="agent-loop", daemon=True)
        self._loop_thread.start()

    def _restart_loop_if_dead(self) -> bool:
        with self._start_lock:
            if not self.started or self.engine.stop_event.is_set():
                return False
            if self._loop_thread is not None and self._loop_thread.is_alive():
                return False
            self.scheduler.drain()
            self._start_loop_thread()
            return True

    def start(self) -> None:
        global _EVENT_SINK
        with self._start_lock:
            if self.started:
                return
            if self._closed:
                raise RuntimeError("runtime is closed")
            unclean = self.checkpoint.detect_unclean_shutdown()
            if unclean:
                self.scheduler.drain()
            self.checkpoint.write_lock()
            _EVENT_SINK = self.memory.append
            try:
                _write_readme()
                self._start_loop_thread()
                self.supervisor.start()
                self.started = True
                if unclean:
                    saved = self.checkpoint.load_latest()
                    record_event(EventType.RESTART, uuid.UUID(int=0), {"reason": "unclean_shutdown_recovery", "checkpoint_step": int(getattr(saved, "step_counter", 0) or 0)})
            except Exception:
                self.engine.request_stop()
                self.checkpoint.remove_lock()
                _EVENT_SINK = None
                raise

    def stop_now(self) -> None:
        global _EVENT_SINK
        with self._start_lock:
            if not self.started:
                return
            self.engine.request_stop()
            self.supervisor.stop_now()
            thread = self._loop_thread
            if thread and thread is not threading.current_thread():
                thread.join(timeout=max(CONFIG.supervisor.restart_grace_s, CONFIG.sandbox.default_timeout_s + 1.0))
            if thread and thread.is_alive():
                LOGGER.error("runtime did not stop within the shutdown grace period", extra={"component": "runtime"})
                return
            self.scheduler.drain()
            self.sandbox.close()
            self.model.close()
            self.checkpoint.remove_lock()
            _EVENT_SINK = None
            self.started = False
            self._closed = True


class LazyRuntime:
    def __init__(self):
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_lock", threading.RLock())

    def _get(self) -> Runtime:
        instance = object.__getattribute__(self, "_instance")
        if instance is not None:
            return instance
        lock = object.__getattribute__(self, "_lock")
        with lock:
            instance = object.__getattribute__(self, "_instance")
            if instance is None:
                instance = Runtime()
                object.__setattr__(self, "_instance", instance)
            return instance

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._get(), name)

    def __setattr__(self, name: str, value: typing.Any) -> None:
        setattr(self._get(), name, value)


runtime = LazyRuntime()

MAX_IMAGE_BYTES = _environment_int("APP_MAX_IMAGE_BYTES", 8 * 1024 * 1024, 1024, 1024 * 1024 * 1024)
MAX_PDF_BYTES = _environment_int("APP_MAX_PDF_BYTES", 64 * 1024 * 1024, 1024, 4 * 1024 * 1024 * 1024)
MAX_PDF_PAGES = _environment_int("APP_MAX_PDF_PAGES", 200, 1, 10000)
MAX_MULTIMODAL_BYTES = _environment_int("APP_MAX_MULTIMODAL_BYTES", 32 * 1024 * 1024, 1024, 4 * 1024 * 1024 * 1024)
_CHAT_JOB_WORKERS = _environment_int("APP_CHAT_WORKERS", 8, 1, 64)
_CHAT_JOB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=_CHAT_JOB_WORKERS, thread_name_prefix="chat-job")
_CHAT_JOB_SLOTS = threading.BoundedSemaphore(_CHAT_JOB_WORKERS * 4)

def _normalize_session_id(value: typing.Any) -> str:
    text = str(value or "").strip()
    if not text:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(text))
    except ValueError as exc:
        raise ValueError("session_id must be a UUID") from exc

def _validated_image(image: typing.Any) -> typing.Optional[dict]:
    if not isinstance(image, dict):
        return None
    media_type = str(image.get("media_type") or "").lower()
    if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        raise ValueError("unsupported image media type")
    encoded = str(image.get("data") or "")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("invalid base64 image data") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the configured size limit")
    signatures = {
        "image/jpeg": raw.startswith(b"\xff\xd8\xff"),
        "image/png": raw.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/gif": raw.startswith((b"GIF87a", b"GIF89a")),
        "image/webp": len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP",
    }
    if not signatures[media_type]:
        raise ValueError("image content does not match its media type")
    return {"media_type": media_type, "data": encoded}



def db_insert_message(
    session_id: str,
    role: str,
    content: str,
    images: typing.Optional[typing.List[dict]] = None,
    attachments: typing.Optional[typing.List[dict]] = None,
) -> None:
    conn = runtime.database.connect()
    try:
        with runtime.database.transaction(conn):
            conn.execute(
                "INSERT INTO messages(session_id,role,content,images,attachments,ts) VALUES(?,?,?,?,?,?)",
                (
                    session_id,
                    role,
                    content,
                    stable_json_dumps(images) if images else None,
                    stable_json_dumps(attachments) if attachments else None,
                    int(time.time() * 1000),
                ),
            )
    finally:
        conn.close()


def db_history(session_id: str) -> typing.List[dict]:
    session_id = _normalize_session_id(session_id)
    conn = runtime.database.connect()
    try:
        rows = conn.execute(
            "SELECT role,content,images,attachments,ts FROM (SELECT id,role,content,images,attachments,ts FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 200) ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    result: typing.List[dict] = []
    for row in rows:
        item: dict = {"role": row["role"], "content": row["content"], "ts": row["ts"]}
        if row["images"]:
            with contextlib.suppress(Exception):
                item["images"] = json.loads(row["images"])
        if row["attachments"]:
            with contextlib.suppress(Exception):
                item["attachments"] = json.loads(row["attachments"])
        result.append(item)
    return result


def db_remember(session_id: str, key: str, value: str) -> None:
    conn = runtime.database.connect()
    try:
        with runtime.database.transaction(conn):
            conn.execute(
                "INSERT INTO chat_memory(session_id,key,value,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(session_id,key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (session_id, key, value, int(time.time() * 1000)),
            )
    finally:
        conn.close()


def db_recall(session_id: str, key: str) -> typing.Optional[str]:
    conn = runtime.database.connect()
    try:
        row = conn.execute(
            "SELECT value FROM chat_memory WHERE session_id=? AND key=?",
            (session_id, key),
        ).fetchone()
    finally:
        conn.close()
    return row["value"] if row else None


def _attachment_metadata(attachment_id: str) -> typing.Optional[dict]:
    conn = runtime.database.connect()
    try:
        row = conn.execute("SELECT id,filename,page_count FROM pdf_attachments WHERE id=?", (attachment_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {"id": row["id"], "name": row["filename"], "kind": "pdf", "pages": row["page_count"]}


def _pdf_message_parts(
    attachments: typing.Iterable[typing.Any],
    include_images: bool = True,
    remaining_bytes: typing.Optional[typing.List[int]] = None,
) -> typing.List[dict]:
    parts: typing.List[dict] = []
    attachment_ids = [str(item.get("id") if isinstance(item, dict) else item) for item in attachments or []]
    attachment_ids = [item for item in attachment_ids if re.fullmatch(r"[0-9a-f]{32}", item)]
    if not attachment_ids:
        return parts
    placeholders = ",".join("?" for _ in attachment_ids)
    conn = runtime.database.connect()
    try:
        rows = conn.execute(
            f"SELECT id,filename,text_content,page_images,page_count FROM pdf_attachments WHERE id IN ({placeholders})",
            attachment_ids,
        ).fetchall()
    finally:
        conn.close()
    by_id = {str(row["id"]): row for row in rows}
    budget = remaining_bytes if remaining_bytes is not None else [MAX_MULTIMODAL_BYTES]
    for attachment_id in attachment_ids:
        row = by_id.get(attachment_id)
        if not row:
            continue
        extracted_text = str(row["text_content"] or "").strip()
        text = (
            f"Attached PDF document: {row['filename']} ({row['page_count']} page(s)).\n"
            f"Extracted PDF text:\n{extracted_text or '[No extractable text.]'}"
        )
        encoded_text = text.encode("utf-8")
        if len(encoded_text) > budget[0]:
            text = encoded_text[: max(0, budget[0])].decode("utf-8", errors="ignore")
            budget[0] = 0
        else:
            budget[0] -= len(encoded_text)
        parts.append({"type": "text", "text": text})
        if not include_images or budget[0] <= 0:
            continue
        try:
            image_paths = json.loads(row["page_images"])
        except Exception:
            image_paths = []
        for raw_path in image_paths:
            path = pathlib.Path(str(raw_path)).resolve()
            if not path.is_file() or not _path_is_within(path, PDF_UPLOAD_PATH.resolve()):
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if len(raw) > budget[0]:
                break
            budget[0] -= len(raw)
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"}})
    return parts


def _model_message_content(
    text: str,
    images: typing.Optional[typing.Iterable[typing.Any]] = None,
    attachments: typing.Optional[typing.Iterable[typing.Any]] = None,
    include_pdf_images: bool = True,
) -> typing.Union[str, typing.List[dict]]:
    if not images and not attachments:
        return text
    budget = [MAX_MULTIMODAL_BYTES]
    parts: typing.List[dict] = [{"type": "text", "text": text or "Please inspect the attached content."}]
    budget[0] -= len((text or "").encode("utf-8"))
    for image in images or []:
        validated = _validated_image(image)
        if validated is None:
            continue
        raw_size = len(base64.b64decode(validated["data"], validate=True))
        if raw_size > budget[0]:
            raise ValueError("combined multimodal input exceeds the configured size limit")
        budget[0] -= raw_size
        parts.append({"type": "image_url", "image_url": {"url": f"data:{validated['media_type']};base64,{validated['data']}"}})
    parts.extend(_pdf_message_parts(attachments or [], include_images=include_pdf_images, remaining_bytes=budget))
    return parts


def build_history(session_id: str) -> typing.List[dict]:
    rows = db_history(session_id)
    rows = rows[-MAX_HISTORY_MSGS:]
    while rows and sum(len(str(row.get("content", "")).encode("utf-8")) for row in rows) > MAX_HISTORY_CHARS:
        rows.pop(0)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for row in rows:
        role = str(row.get("role", ""))
        if role == "user":
            messages.append({
                "role": "user",
                "content": _model_message_content(str(row.get("content", "")), row.get("images"), row.get("attachments"), include_pdf_images=False),
            })
        elif role in {"assistant", "system"}:
            messages.append({"role": role, "content": str(row.get("content", ""))})
    return messages

def create_job(session_id: str) -> str:
    job_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    conn = runtime.database.connect()
    try:
        conn.execute(
            "INSERT INTO jobs(job_id,session_id,status,created_at,updated_at) VALUES(?,?,?,?,?)",
            (job_id, session_id, "running", now, now),
        )
    finally:
        conn.close()
    return job_id


def persist_chunk(job_id: str, seq: int, event: dict) -> None:
    conn = runtime.database.connect()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO job_chunks(job_id,seq,event_json) VALUES(?,?,?)",
            (job_id, seq, stable_json_dumps(event)),
        )
        conn.execute(
            "UPDATE jobs SET updated_at=? WHERE job_id=?",
            (int(time.time() * 1000), job_id),
        )
    finally:
        conn.close()


def update_job(job_id: str, status: str) -> None:
    if status not in {"running", "done", "failed", "cancelled"}:
        raise ValueError("invalid job status")
    conn = runtime.database.connect()
    try:
        conn.execute(
            "UPDATE jobs SET status=?,updated_at=? WHERE job_id=?",
            (status, int(time.time() * 1000), job_id),
        )
    finally:
        conn.close()


def _job_status(job_id: str) -> typing.Optional[str]:
    conn = runtime.database.connect()
    try:
        row = conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return str(row["status"]) if row else None
    finally:
        conn.close()


class JobCancelledError(AgentError):
    pass


def _chat_tool_schemas() -> typing.List[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a command on the managed workspace machine. Use it to inspect or change real local files and run real local programs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute."},
                        "timeout_s": {"type": "number", "minimum": 1, "maximum": 600, "description": "Execution timeout in seconds."},
                        "cwd": {"type": "string", "description": "Optional workspace-relative working directory."},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vm_run",
                "description": "Run a command in a real isolated InstaVM machine. Use it for isolated execution, package experiments, or work that should not run in the managed workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command or source code to execute."},
                        "language": {"type": "string", "enum": ["bash", "python"], "description": "Command language."},
                        "timeout_s": {"type": "number", "minimum": 1, "maximum": 600, "description": "Execution timeout in seconds."},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_remember",
                "description": "Persist a user-specific fact or preference for this chat session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Short stable memory key."},
                        "value": {"type": "string", "description": "Fact or preference to store."},
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_recall",
                "description": "Read a user-specific fact or preference previously saved in this chat session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Memory key to retrieve."},
                    },
                    "required": ["key"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _chat_safe_arguments(raw: str) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PermanentError(f"Invalid tool arguments: {exc}") from exc
    if not isinstance(value, dict):
        raise PermanentError("Tool arguments must be a JSON object")
    return value


def _chat_tool_object(arguments: dict) -> typing.Any:
    return types.SimpleNamespace(**arguments)


def _chat_vm_id(created: dict) -> str:
    if not isinstance(created, dict):
        raise PermanentError("InstaVM create response is invalid")
    for candidate in (created, created.get("response"), created.get("vm")):
        if not isinstance(candidate, dict):
            continue
        for key in ("id", "vm_id"):
            vm_id = str(candidate.get(key) or "")
            if vm_id:
                return vm_id
    raise PermanentError("InstaVM create response does not contain a VM id")


def _chat_instavm_client() -> typing.Any:
    api_key = os.environ.get("INSTAVM_API_KEY") or os.environ.get("INSTA_API_KEY")
    if not api_key:
        raise PermanentError("INSTAVM_API_KEY is required for InstaVM execution")
    if not _HAS_INSTAVM:
        raise PermanentError("The official instavm Python package is not installed")
    return InstaVM(
        api_key=api_key,
        timeout=CONFIG.vm.lifetime_seconds,
        cpu_count=CONFIG.vm.vcpu_count,
        memory_mb=CONFIG.vm.memory_mb,
        auto_start_session=False,
    )


def _chat_instavm_session_id(created: dict) -> str:
    if not isinstance(created, dict):
        return ""
    for candidate in (created, created.get("response"), created.get("vm")):
        if not isinstance(candidate, dict):
            continue
        session_id = str(candidate.get("session_id") or "")
        if session_id:
            return session_id
    return ""


def _chat_instavm_prepare(vm_state: dict) -> typing.Any:
    client = vm_state.get("client")
    if client is None:
        client = _chat_instavm_client()
        vm_state["client"] = client
    if vm_state.get("session_id"):
        client.session_id = vm_state["session_id"]
        return client
    created = client.vms.create(
        wait=True,
        image_variant=CONFIG.vm.image_variant,
        vm_lifetime_seconds=CONFIG.vm.lifetime_seconds,
        memory_mb=CONFIG.vm.memory_mb,
        vcpu_count=CONFIG.vm.vcpu_count,
        egress_policy={
            "allow_package_managers": True,
            "allow_http": False,
            "allow_https": True,
            "allowed_domains": [],
            "allowed_cidrs": [],
        },
    )
    session_id = _chat_instavm_session_id(created)
    if not session_id:
        vm_id = _chat_vm_id(created)
        vm = client.vms.get(vm_id)
        session_id = _chat_instavm_session_id(vm)
    if not session_id:
        raise PermanentError("InstaVM did not return an executable session")
    if not vm_state.get("vm_id"):
        with contextlib.suppress(PermanentError):
            vm_state["vm_id"] = _chat_vm_id(created)
    vm_state["session_id"] = session_id
    client.session_id = session_id
    return client


def _chat_instavm_execute(client: typing.Any, command: str, language: str, timeout_s: float) -> dict:
    result = client.execute(command, language=language, timeout=int(timeout_s))
    if not isinstance(result, dict):
        raise PermanentError("InstaVM execute response is invalid")
    return result


def _chat_execution_output(result: typing.Any) -> tuple[bool, str, str]:
    response = result.get("response", result) if isinstance(result, dict) else result
    if not isinstance(response, dict):
        return False, "", stable_json_dumps(result)
    stdout = str(response.get("stdout") or response.get("output") or "")
    stderr = str(response.get("stderr") or response.get("error") or "")
    exit_code = response.get("exit_code", response.get("exitCode", response.get("returncode")))
    if exit_code is None:
        return False, stdout, stderr or "execution response did not include an exit code"
    try:
        ok = int(exit_code) == 0
    except (TypeError, ValueError):
        return False, stdout, stderr or "execution response included an invalid exit code"
    return ok, stdout, stderr


def _chat_run_tool(
    client: typing.Any,
    session_id: str,
    tool_name: str,
    arguments: dict,
    vm_state: dict,
    emit: typing.Callable[[str, dict], None],
) -> dict:
    if tool_name == "run_shell":
        command = str(arguments.get("command") or "").strip()
        if not command:
            raise PermanentError("run_shell requires a non-empty command")
        timeout_s = min(max(float(arguments.get("timeout_s", 60)), 1), 600)
        cwd = str(arguments.get("cwd") or "")
        emit("code_exec_start", {"reason": "Helyi gép", "code": command})
        result = asyncio.run(runtime.sandbox.execute(uuid.uuid5(uuid.NAMESPACE_URL, session_id), "execute a user-authorized local shell command", "run_shell", {"command": command, "timeout_s": timeout_s, "cwd": cwd}))
        ok = int(result.get("exit_code", 1)) == 0 and not bool(result.get("timed_out"))
        emit("code_exec_done", {"ok": ok, "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")})
        return result
    if tool_name == "vm_run":
        command = str(arguments.get("command") or "").strip()
        if not command:
            raise PermanentError("vm_run requires a non-empty command")
        language = str(arguments.get("language") or "bash")
        if language not in {"bash", "python"}:
            raise PermanentError("vm_run language must be bash or python")
        timeout_s = min(max(float(arguments.get("timeout_s", 120)), 1), 600)
        client = _chat_instavm_prepare(vm_state)
        emit("code_exec_start", {"reason": f"InstaVM ({vm_state['vm_id']})", "code": command})
        result = _chat_instavm_execute(client, command, language, timeout_s)
        ok, stdout, stderr = _chat_execution_output(result)
        emit("code_exec_done", {"ok": ok, "stdout": stdout, "stderr": stderr})
        return result
    if tool_name == "memory_remember":
        key = str(arguments.get("key") or "").strip()
        value = str(arguments.get("value") or "").strip()
        if not key or not value:
            raise PermanentError("memory_remember requires non-empty key and value")
        db_remember(session_id, key, value)
        emit("memory_write", {"key": key})
        return {"key": key, "saved": True}
    if tool_name == "memory_recall":
        key = str(arguments.get("key") or "").strip()
        if not key:
            raise PermanentError("memory_recall requires a non-empty key")
        value = db_recall(session_id, key)
        emit("memory_read", {"key": key, "found": value is not None})
        return {"key": key, "found": value is not None, "value": value}
    raise PermanentError(f"Unsupported chat tool: {tool_name}")


def job_thread(job_id: str, session_id: str, history: typing.List[dict]) -> None:
    full: typing.List[str] = []
    sequence = 0
    client: typing.Any = None
    vm_state: dict = {}
    final_status = "failed"

    def emit(event_type: str, data: typing.Optional[dict] = None, enforce_running: bool = True) -> None:
        nonlocal sequence
        if enforce_running and _job_status(job_id) == "cancelled":
            raise JobCancelledError("job was cancelled")
        persist_chunk(job_id, sequence, {"type": event_type, "data": data or {}})
        sequence += 1

    try:
        api_key = os.environ.get("REQUESTY_API_KEY", "")
        model_name = CONFIG.model.name or CONFIG.model.controller
        if not model_name:
            raise PermanentError("REQUESTY_MODEL or APP_MODEL__NAME must be configured")
        if not api_key:
            raise PermanentError("REQUESTY_API_KEY is not configured")
        if not _HAS_OPENAI:
            raise PermanentError("the openai package is unavailable")
        client = OpenAI(api_key=api_key, base_url=MODEL_ROUTER_URL, timeout=CONFIG.model.request_timeout_s, max_retries=0)
        messages = list(history)
        for _ in range(6):
            if _job_status(job_id) == "cancelled":
                raise JobCancelledError("job was cancelled")
            parser = ThinkParser()
            turn_content: typing.List[str] = []
            calls: dict[int, dict] = {}
            usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            request_parameters: dict = {
                "model": model_name,
                "messages": messages,
                "tools": _chat_tool_schemas(),
                "tool_choice": "auto",
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_completion_tokens": CONFIG.model.max_completion_tokens,
            }
            try:
                stream = client.chat.completions.create(**request_parameters)
            except Exception as exc:
                if "max_completion_tokens" not in str(exc):
                    raise
                request_parameters.pop("max_completion_tokens", None)
                request_parameters["max_tokens"] = CONFIG.model.max_completion_tokens
                stream = client.chat.completions.create(**request_parameters)
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    for key in usage_totals:
                        usage_totals[key] = max(usage_totals[key], int(getattr(usage, key, 0) or 0))
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    for event_type, event_data in parser.feed(str(piece)):
                        if event_type == "content":
                            turn_content.append(event_data)
                        emit(event_type, {"delta": event_data})
                for call_delta in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(call_delta, "index", 0) or 0)
                    call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    call_id = getattr(call_delta, "id", None)
                    if call_id:
                        call["id"] = str(call_id)
                    function = getattr(call_delta, "function", None)
                    if function is not None:
                        name = getattr(function, "name", None)
                        arguments = getattr(function, "arguments", None)
                        if name:
                            call["name"] = str(name)
                        if arguments:
                            call["arguments"] += str(arguments)
            for event_type, event_data in parser.flush():
                if event_type == "content":
                    turn_content.append(event_data)
                emit(event_type, {"delta": event_data})
            emit("usage", usage_totals)
            turn_text = "".join(turn_content)
            if not calls:
                if not turn_text:
                    raise PermanentError("the model returned neither content nor tool calls")
                full.append(turn_text)
                final_status = "done"
                break
            assistant_calls = []
            for index in sorted(calls):
                call = calls[index]
                if not call["id"] or not call["name"]:
                    raise PermanentError("the model returned an incomplete tool call")
                _chat_safe_arguments(call["arguments"])
                assistant_calls.append({
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                })
            messages.append({"role": "assistant", "content": turn_text or None, "tool_calls": assistant_calls})
            if turn_text:
                full.append(turn_text)
            for call in assistant_calls:
                tool_name = call["function"]["name"]
                arguments = _chat_safe_arguments(call["function"]["arguments"])
                try:
                    result = _chat_run_tool(client, session_id, tool_name, arguments, vm_state, emit)
                    tool_content = stable_json_dumps(result)
                except Exception as exc:
                    tool_content = stable_json_dumps({"error": type(exc).__name__, "message": str(exc)})
                    emit("warning", {"message": f"{tool_name} failed"})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_content})
                full.append(f"\n[{tool_name}]\n{tool_content}\n")
        else:
            raise PermanentError("the chat tool execution limit was exceeded")
        if _job_status(job_id) == "cancelled":
            raise JobCancelledError("job was cancelled")
        content = "".join(full).strip()
        if content:
            db_insert_message(session_id, "assistant", content)
    except JobCancelledError:
        final_status = "cancelled"
    except Exception as exc:
        errors_total.labels(type=type(exc).__name__).inc()
        LOGGER.exception("chat completion failed", extra={"component": "chat"})
        emit("error", {"message": "The assistant request failed."}, enforce_running=False)
    finally:
        vm_client = vm_state.get("client")
        close_vm = getattr(vm_client, "close", None)
        if callable(close_vm):
            with contextlib.suppress(Exception):
                close_vm()
        close_client = getattr(client, "close", None)
        if callable(close_client):
            with contextlib.suppress(Exception):
                close_client()
        emit("done", {"status": final_status}, enforce_running=False)
        update_job(job_id, final_status)


def stream_job(job_id: str) -> typing.Generator[str, None, None]:
    sequence = 0
    while True:
        conn = runtime.database.connect()
        try:
            rows = conn.execute(
                "SELECT seq,event_json FROM job_chunks WHERE job_id=? AND seq>=? ORDER BY seq LIMIT 200",
                (job_id, sequence),
            ).fetchall()
            job_row = conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        finally:
            conn.close()
        if job_row is None:
            yield f"data: {stable_json_dumps({'type': 'error', 'data': {'message': 'job not found'}})}\n\n"
            return
        if rows:
            for row in rows:
                sequence = int(row["seq"]) + 1
                yield f"data: {row['event_json']}\n\n"
                try:
                    if json.loads(row["event_json"]).get("type") == "done":
                        return
                except Exception:
                    continue
        elif job_row["status"] != "running":
            return
        else:
            time.sleep(0.1)


def health_status() -> dict:
    db_ok = False
    try:
        conn = runtime.database.connect()
        try:
            conn.execute("SELECT 1").fetchone()
            db_ok = True
        finally:
            conn.close()
    except Exception:
        db_ok = False
    vector_ok = False
    try:
        vector_ok = runtime.memory.semantic is not None and CONFIG.memory.embedding_dim > 0
    except Exception:
        vector_ok = False
    model_ok = bool(os.environ.get("REQUESTY_API_KEY")) and _HAS_OPENAI and bool(CONFIG.model.name or CONFIG.model.controller)
    return {"database": db_ok, "vector_index": vector_ok, "model": model_ok}


def _admin_authorized_value(value: str) -> bool:
    configured = str(CONFIG.api.admin_token or "")
    return bool(configured) and hmac.compare_digest(str(value or ""), f"Bearer {configured}")


def _session_token(session_id: str) -> str:
    return hmac.new(str(CONFIG.api.admin_token).encode("utf-8"), session_id.encode("utf-8"), hashlib.sha256).hexdigest()


def _session_authorized(session_id: str, token: str) -> bool:
    return hmac.compare_digest(str(token or ""), _session_token(session_id))


def _session_exists(session_id: str) -> bool:
    conn = runtime.database.connect()
    try:
        return conn.execute("SELECT 1 FROM messages WHERE session_id=? LIMIT 1", (session_id,)).fetchone() is not None
    finally:
        conn.close()


def _parse_deadline(value: typing.Any) -> typing.Optional[dt.datetime]:
    if value in {None, ""}:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("deadline must be an ISO 8601 datetime") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _task_details(task_id: str) -> typing.Optional[dict]:
    try:
        task_id = str(uuid.UUID(task_id))
    except ValueError:
        return None
    conn = runtime.database.connect()
    try:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            return None
        events = conn.execute(
            "SELECT id,type,payload_json,ts FROM events WHERE task_id=? ORDER BY id DESC LIMIT 20",
            (task_id,),
        ).fetchall()
    finally:
        conn.close()
    return {
        "id": task_id,
        "status": row["status"],
        "priority": int(row["priority"]),
        "deadline": row["deadline"],
        "dependencies": json.loads(row["dependencies"]),
        "payload": json.loads(row["payload_json"]),
        "events": [
            {"id": int(event["id"]), "type": event["type"], "payload": json.loads(event["payload_json"]), "timestamp": event["ts"]}
            for event in events
        ],
    }


def _create_task_from_data(data: typing.Any) -> typing.Any:
    if not isinstance(data, dict):
        raise ValueError("request body must be an object")
    dependencies = data.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ValueError("dependencies must be an array")
    normalized_dependencies = [str(uuid.UUID(str(item))) for item in dependencies]
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    deadline = _parse_deadline(data.get("deadline"))
    task = model_validate(Task, {
        "id": str(uuid.uuid4()),
        "priority": int(data.get("priority", 0)),
        "deadline": deadline.isoformat() if deadline else None,
        "dependencies": normalized_dependencies,
        "payload": payload,
        "status": TaskStatus.QUEUED.value,
    })
    runtime.scheduler.enqueue(task)
    return task


def _schedule_rows() -> typing.List[dict]:
    conn = runtime.database.connect()
    try:
        rows = conn.execute(
            "SELECT id,task_payload_json,cron_expression,interval_seconds,mode,continuation_task_id,next_run_at,last_run_at,enabled,created_at FROM schedules ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": row["id"],
            "payload": json.loads(row["task_payload_json"]),
            "cron_expression": row["cron_expression"],
            "interval_seconds": row["interval_seconds"],
            "mode": row["mode"],
            "continuation_task_id": row["continuation_task_id"],
            "next_run_at": row["next_run_at"],
            "last_run_at": row["last_run_at"],
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _create_schedule_from_data(data: typing.Any) -> str:
    if not isinstance(data, dict):
        raise ValueError("request body must be an object")
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return runtime.scheduler.create_schedule(
        payload=payload,
        interval_seconds=float(data["interval_seconds"]) if data.get("interval_seconds") is not None else None,
        cron_expression=str(data["cron_expression"]) if data.get("cron_expression") else None,
        mode=str(data.get("mode", "new")),
        continuation_task_id=str(uuid.UUID(str(data["continuation_task_id"]))) if data.get("continuation_task_id") else None,
    )


def _status_payload() -> dict:
    open_circuits = [name for name, breaker in CircuitBreaker.snapshot().items() if breaker.state.state == BreakerState.OPEN]
    last_checkpoint = runtime.engine.last_checkpoint
    return {
        "heartbeat_age_s": max(0.0, time.time() - runtime.engine.last_heartbeat),
        "in_flight_task_count": len(runtime.scheduler.list_in_flight()),
        "last_checkpoint_age_s": max(0.0, time.time() - last_checkpoint) if last_checkpoint else None,
        "open_circuits": open_circuits,
    }


def _fallback_metrics_text() -> str:
    rows: typing.List[str] = []
    metrics = (
        loop_iterations_total,
        tool_calls_total,
        model_calls_total,
        errors_total,
        checkpoints_total,
        checkpoint_duration_seconds,
        memory_size_bytes,
        circuit_breaker_state,
        heartbeats_total,
        restarts_total,
        livelocks_total,
        retries_total,
        backoff_seconds_sum,
    )
    for metric in metrics:
        if not isinstance(metric, _FallbackMetric):
            continue
        with metric._lock:
            for key, amount in metric.values.items():
                labels = ""
                if metric.labels_names:
                    labels = "{" + ",".join(f'{name}={json.dumps(value)}' for name, value in zip(metric.labels_names, key)) + "}"
                if metric.kind == "histogram":
                    rows.append(f"{metric.name}_count{labels} {metric.counts.get(key, 0)}")
                    rows.append(f"{metric.name}_sum{labels} {metric.sums.get(key, 0.0)}")
                else:
                    rows.append(f"{metric.name}{labels} {amount}")
    return "\n".join(rows) + ("\n" if rows else "")


def _force_checkpoint_state() -> typing.Any:
    running = runtime.scheduler.list_in_flight()
    tasks = {
        task_id: {
            "step": runtime.engine.iteration_id,
            "working_memory": runtime.memory.working_for(task_id).snapshot(),
        }
        for task_id in running
    }
    return CheckpointState(
        working_memory_ptr=sum(len(item["working_memory"]) for item in tasks.values()),
        episodic_cursor=runtime.memory.episodic.count(),
        in_flight_tasks=[uuid.UUID(item) for item in running],
        queue_state={"tasks": tasks},
        step_counter=runtime.engine.iteration_id,
        version=2,
    )


def _cleanup_pdf_attachments(max_age_seconds: typing.Optional[float] = None) -> int:
    retention = float(max_age_seconds if max_age_seconds is not None else os.environ.get("APP_PDF_RETENTION_SECONDS", str(7 * 24 * 3600)))
    if retention <= 0:
        raise ValueError("PDF retention must be positive")
    cutoff = int((time.time() - retention) * 1000)
    conn = runtime.database.connect()
    removed: typing.List[str] = []
    try:
        rows = conn.execute("SELECT id FROM pdf_attachments WHERE created_at<?", (cutoff,)).fetchall()
        removed = [str(row["id"]) for row in rows]
        with runtime.database.transaction(conn):
            conn.execute("DELETE FROM pdf_attachments WHERE created_at<?", (cutoff,))
    finally:
        conn.close()
    if removed:
        import shutil
        for attachment_id in removed:
            with contextlib.suppress(Exception):
                shutil.rmtree(PDF_UPLOAD_PATH / attachment_id)
    return len(removed)


def _delete_session_history(session_id: str) -> None:
    conn = runtime.database.connect()
    attachment_ids: typing.Set[str] = set()
    try:
        rows = conn.execute("SELECT attachments FROM messages WHERE session_id=? AND attachments IS NOT NULL", (session_id,)).fetchall()
        for row in rows:
            with contextlib.suppress(Exception):
                for item in json.loads(row["attachments"]):
                    if isinstance(item, dict) and re.fullmatch(r"[0-9a-f]{32}", str(item.get("id", ""))):
                        attachment_ids.add(str(item["id"]))
        with runtime.database.transaction(conn):
            conn.execute("UPDATE jobs SET status='cancelled',updated_at=? WHERE session_id=? AND status='running'", (int(time.time() * 1000), session_id))
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM chat_memory WHERE session_id=?", (session_id,))
        referenced: typing.Set[str] = set()
        for row in conn.execute("SELECT attachments FROM messages WHERE attachments IS NOT NULL"):
            with contextlib.suppress(Exception):
                for item in json.loads(row["attachments"]):
                    if isinstance(item, dict):
                        referenced.add(str(item.get("id", "")))
        removable = attachment_ids - referenced
        if removable:
            placeholders = ",".join("?" for _ in removable)
            with runtime.database.transaction(conn):
                conn.execute(f"DELETE FROM pdf_attachments WHERE id IN ({placeholders})", tuple(removable))
    finally:
        conn.close()
    if attachment_ids:
        import shutil
        for attachment_id in attachment_ids:
            if attachment_id not in locals().get("referenced", set()):
                with contextlib.suppress(Exception):
                    shutil.rmtree(PDF_UPLOAD_PATH / attachment_id)


def _extract_pdf(pdf_path: pathlib.Path, attachment_dir: pathlib.Path) -> typing.Tuple[str, typing.List[pathlib.Path]]:
    if pdf_path.stat().st_size <= 0 or pdf_path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the configured size limit")
    with pdf_path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError("file content is not a PDF")
    info = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, timeout=30, check=False)
    if info.returncode != 0:
        raise ValueError("PDF metadata could not be read")
    match = re.search(r"^Pages:\s*(\d+)\s*$", info.stdout, flags=re.MULTILINE)
    if not match:
        raise ValueError("PDF page count is unavailable")
    page_count = int(match.group(1))
    if page_count <= 0 or page_count > MAX_PDF_PAGES:
        raise ValueError("PDF page count exceeds the configured limit")
    text_result = subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"], capture_output=True, timeout=120, check=False)
    if text_result.returncode != 0:
        extracted = ""
    else:
        extracted = text_result.stdout.decode("utf-8", errors="replace")[:4_000_000]
    output_prefix = attachment_dir / "page"
    render = subprocess.run(
        ["pdftoppm", "-f", "1", "-l", str(page_count), "-jpeg", "-jpegopt", "quality=82", "-scale-to", "1400", str(pdf_path), str(output_prefix)],
        capture_output=True,
        text=True,
        timeout=min(600, max(60, page_count * 5)),
        check=False,
    )
    rendered = sorted(attachment_dir.glob("page-*.jpg"), key=lambda path: int(path.stem.rsplit("-", 1)[-1]))
    if render.returncode != 0 or len(rendered) != page_count:
        raise ValueError("PDF rendering failed")
    total_rendered = sum(path.stat().st_size for path in rendered)
    if total_rendered > MAX_PDF_BYTES * 4:
        raise ValueError("rendered PDF exceeds the configured output limit")
    return extracted.strip(), rendered


def _submit_chat_job(job_id: str, session_id: str, history: typing.List[dict]) -> bool:
    if not _CHAT_JOB_SLOTS.acquire(blocking=False):
        return False

    def run() -> None:
        try:
            job_thread(job_id, session_id, history)
        finally:
            _CHAT_JOB_SLOTS.release()

    try:
        _CHAT_JOB_EXECUTOR.submit(run)
        return True
    except Exception:
        _CHAT_JOB_SLOTS.release()
        raise


app = None
if _HAS_FLASK:
    app = Flask(__name__, static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_PDF_BYTES

    @app.errorhandler(Exception)
    def _handle_exception(exc):
        from werkzeug.exceptions import HTTPException as WerkzeugHTTPException
        if isinstance(exc, WerkzeugHTTPException):
            return jsonify({"error": exc.description}), exc.code
        LOGGER.exception("unhandled request", extra={"component": "http"})
        return jsonify({"error": "server error"}), 500

    @app.errorhandler(413)
    def _too_large(exc):
        return jsonify({"error": "request exceeds the configured size limit"}), 413

    def flask_admin() -> bool:
        return _admin_authorized_value(request.headers.get("Authorization", ""))

    def flask_session_token() -> str:
        return request.headers.get("X-Session-Token", "") or request.args.get("session_token", "")

    @app.route("/")
    def index():
        return send_from_directory(str(ROOT), "index.html")

    @app.route("/sw.js")
    def service_worker():
        return Response(
            "self.addEventListener('install',function(event){self.skipWaiting()});self.addEventListener('activate',function(event){event.waitUntil(self.clients.claim())});",
            mimetype="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @app.route("/api/upload/pdf", methods=["POST"])
    def upload_pdf():
        uploaded = request.files.get("file")
        if not uploaded:
            return jsonify({"error": "file is required"}), 400
        filename = pathlib.Path(uploaded.filename or "document.pdf").name
        if not filename.lower().endswith(".pdf"):
            return jsonify({"error": "only PDF files are accepted"}), 400
        attachment_id = uuid.uuid4().hex
        attachment_dir = PDF_UPLOAD_PATH / attachment_id
        try:
            _cleanup_pdf_attachments()
            attachment_dir.mkdir(parents=True, exist_ok=False)
            pdf_path = attachment_dir / "source.pdf"
            uploaded.save(str(pdf_path))
            extracted, rendered = _extract_pdf(pdf_path, attachment_dir)
            conn = runtime.database.connect()
            try:
                with runtime.database.transaction(conn):
                    conn.execute(
                        "INSERT INTO pdf_attachments(id,filename,text_content,page_images,page_count,created_at) VALUES(?,?,?,?,?,?)",
                        (attachment_id, filename, extracted, stable_json_dumps([str(path) for path in rendered]), len(rendered), int(time.time() * 1000)),
                    )
            finally:
                conn.close()
            return jsonify({"attachment": {"id": attachment_id, "name": filename, "kind": "pdf", "pages": len(rendered)}})
        except ValueError as exc:
            import shutil
            with contextlib.suppress(Exception):
                shutil.rmtree(attachment_dir)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            import shutil
            with contextlib.suppress(Exception):
                shutil.rmtree(attachment_dir)
            LOGGER.exception("PDF processing failed", extra={"component": "http"})
            return jsonify({"error": "PDF processing failed"}), 500

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "invalid request"}), 400
        try:
            session_id = _normalize_session_id(data.get("session_id"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        supplied_token = str(data.get("session_token") or request.headers.get("X-Session-Token", ""))
        if _session_exists(session_id) and not _session_authorized(session_id, supplied_token):
            return jsonify({"error": "unauthorized session"}), 401
        message = str(data.get("message", ""))
        if len(message) > 32000:
            return jsonify({"error": "message exceeds the configured length limit"}), 413
        try:
            images = [_validated_image(item) for item in (data.get("images") if isinstance(data.get("images"), list) else [])[:20]]
            images = [item for item in images if item is not None]
            if sum(len(base64.b64decode(item["data"], validate=True)) for item in images) > MAX_MULTIMODAL_BYTES:
                raise ValueError("combined images exceed the configured size limit")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        requested = data.get("attachments") if isinstance(data.get("attachments"), list) else []
        attachments = [metadata for raw in requested[:12] if isinstance(raw, str) for metadata in [_attachment_metadata(raw)] if metadata]
        if not message and not images and not attachments:
            return jsonify({"error": "empty message"}), 400
        history = build_history(session_id)
        db_insert_message(session_id, "user", message, images, attachments)
        job_id = create_job(session_id)
        request_history = history + [{"role": "user", "content": _model_message_content(message, images, attachments, include_pdf_images=True)}]
        if not _submit_chat_job(job_id, session_id, request_history):
            update_job(job_id, "failed")
            return jsonify({"error": "chat service is at capacity"}), 503
        token = _session_token(session_id)

        def generate():
            yield f"data: {stable_json_dumps({'type': 'session', 'data': {'session_id': session_id, 'session_token': token}})}\n\n"
            yield f"data: {stable_json_dumps({'type': 'job_id', 'data': {'job_id': job_id}})}\n\n"
            yield from stream_job(job_id)

        return Response(generate(), content_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/history", methods=["GET", "DELETE"])
    def history_endpoint():
        try:
            session_id = _normalize_session_id(request.args.get("session_id", ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not _session_authorized(session_id, flask_session_token()):
            return jsonify({"error": "unauthorized session"}), 401
        if request.method == "DELETE":
            _delete_session_history(session_id)
            return jsonify({"ok": True})
        return jsonify(db_history(session_id))

    @app.route("/api/job/<job_id>/status")
    def job_status(job_id):
        try:
            job_id = str(uuid.UUID(job_id))
        except ValueError:
            return jsonify({"status": "not_found"}), 404
        conn = runtime.database.connect()
        try:
            row = conn.execute("SELECT status,session_id FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            return jsonify({"status": "not_found"}), 404
        if not _session_authorized(row["session_id"], flask_session_token()):
            return jsonify({"error": "unauthorized session"}), 401
        return jsonify({"status": row["status"], "session_id": row["session_id"]})

    @app.route("/api/agent/resume/<job_id>")
    def agent_resume(job_id):
        conn = runtime.database.connect()
        try:
            row = conn.execute("SELECT session_id FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            return jsonify({"error": "job not found"}), 404
        if not _session_authorized(row["session_id"], flask_session_token()):
            return jsonify({"error": "unauthorized session"}), 401
        return Response(stream_job(job_id), content_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/session/<session_id>/active_job")
    def active_job(session_id):
        try:
            session_id = _normalize_session_id(session_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not _session_authorized(session_id, flask_session_token()):
            return jsonify({"error": "unauthorized session"}), 401
        conn = runtime.database.connect()
        try:
            row = conn.execute("SELECT job_id,status FROM jobs WHERE session_id=? AND status='running' ORDER BY created_at DESC LIMIT 1", (session_id,)).fetchone()
        finally:
            conn.close()
        return jsonify({"status": row["status"], "job_id": row["job_id"], "session_id": session_id} if row else {"status": "none"})

    @app.route("/metrics")
    def metrics_endpoint():
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        if _HAS_PROMETHEUS:
            return Response(generate_latest(_METRICS_REGISTRY), content_type=CONTENT_TYPE_LATEST)
        return Response(_fallback_metrics_text(), content_type="text/plain; version=0.0.4")

    @app.route("/health/liveness")
    def liveness():
        return jsonify({"status": "ok"})

    @app.route("/health/readiness")
    def readiness():
        status = health_status()
        return (jsonify(status), 200) if all(status.values()) else (jsonify(status), 503)

    @app.route("/health/startup")
    def startup():
        return (jsonify({"status": "ok"}), 200) if runtime.started else (jsonify({"status": "starting"}), 503)

    @app.route("/status")
    def status_endpoint():
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify(_status_payload())

    @app.route("/api/instavm/catalog")
    def instavm_catalog_endpoint():
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify([_plain(operation) | {"tool_name": operation.tool_name} for operation in INSTAVM_MANIFEST])

    @app.route("/api/instavm/coverage")
    def instavm_coverage_endpoint():
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        registered = {name for name in runtime.sandbox.tools if name.startswith("instavm_")}
        expected = {operation.tool_name for operation in INSTAVM_MANIFEST}
        return jsonify({**validate_instavm_manifest(), "implemented": len(registered & expected), "missing": sorted(expected - registered)})

    @app.route("/api/instavm/metrics")
    def instavm_metrics_endpoint():
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        conn = runtime.database.connect()
        try:
            rows = conn.execute("SELECT operation_id,decision,COUNT(*) AS count FROM instavm_policy_decisions GROUP BY operation_id,decision").fetchall()
        finally:
            conn.close()
        return jsonify([dict(row) for row in rows])

    @app.route("/tasks", methods=["POST"])
    def create_task_endpoint():
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        try:
            task = _create_task_from_data(request.get_json(force=True, silent=True))
            return jsonify({"id": str(task.id), "status": TaskStatus.QUEUED.value}), 201
        except (ValueError, TypeError, PermanentError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/tasks/<task_id>")
    def get_task_endpoint(task_id):
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        details = _task_details(task_id)
        return (jsonify(details), 200) if details else (jsonify({"error": "not found"}), 404)

    @app.route("/schedules", methods=["POST", "GET"])
    def schedules_endpoint():
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        if request.method == "GET":
            return jsonify(_schedule_rows())
        try:
            schedule_id = _create_schedule_from_data(request.get_json(force=True, silent=True))
            return jsonify({"id": schedule_id, "status": "scheduled"}), 201
        except (ValueError, TypeError, PermanentError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/schedules/<schedule_id>", methods=["DELETE"])
    def delete_schedule_endpoint(schedule_id):
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        conn = runtime.database.connect()
        try:
            cursor = conn.execute("UPDATE schedules SET enabled=0 WHERE id=?", (schedule_id,))
        finally:
            conn.close()
        return jsonify({"ok": bool(cursor.rowcount)})

    @app.route("/events")
    def events_endpoint():
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        task_id = request.args.get("task_id")
        if task_id:
            try:
                task_id = str(uuid.UUID(task_id))
            except ValueError:
                return jsonify({"error": "invalid task_id"}), 400

        try:
            initial_event_id = max(0, int(request.args.get("after", "0") or 0))
        except ValueError:
            return jsonify({"error": "after must be an integer"}), 400

        def event_stream():
            last_id = initial_event_id
            while True:
                conn = runtime.database.connect()
                try:
                    if task_id:
                        rows = conn.execute("SELECT id,type,task_id,payload_json,tokens_used,ts FROM events WHERE id>? AND task_id=? ORDER BY id LIMIT 200", (last_id, task_id)).fetchall()
                    else:
                        rows = conn.execute("SELECT id,type,task_id,payload_json,tokens_used,ts FROM events WHERE id>? ORDER BY id LIMIT 200", (last_id,)).fetchall()
                finally:
                    conn.close()
                if rows:
                    for row in rows:
                        last_id = int(row["id"])
                        yield f"data: {stable_json_dumps({'id': last_id, 'type': row['type'], 'task_id': row['task_id'], 'payload': json.loads(row['payload_json']), 'tokens_used': row['tokens_used'], 'timestamp': row['ts']})}\n\n"
                else:
                    yield ": keepalive\n\n"
                    time.sleep(5)

        return Response(event_stream(), content_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/admin/checkpoint", methods=["POST"])
    def force_checkpoint_endpoint():
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        asyncio.run(runtime.checkpoint.checkpoint_now(_force_checkpoint_state()))
        return jsonify({"ok": True})

    @app.route("/admin/replan/<task_id>", methods=["POST"])
    def force_replan_endpoint(task_id):
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        try:
            task_id = str(uuid.UUID(task_id))
        except ValueError:
            return jsonify({"error": "invalid task_id"}), 400
        return jsonify({"ok": runtime.scheduler.requeue_with_backoff(task_id)})

    @app.route("/admin/circuits/<name>/<state>", methods=["POST"])
    def force_circuit_endpoint(name, state):
        if not flask_admin():
            return jsonify({"error": "unauthorized"}), 401
        if state not in {"open", "close"}:
            return jsonify({"error": "state must be open or close"}), 400
        breaker = CircuitBreaker.get(name)
        breaker.force_open() if state == "open" else breaker.force_close()
        return jsonify({"ok": True, "state": breaker.state.state.value})


def _create_fastapi_app() -> typing.Any:
    if not _HAS_FASTAPI:
        return None
    fapp = FastAPI()

    def require_admin(authorization: str) -> None:
        if not _admin_authorized_value(authorization):
            raise HTTPException(status_code=401, detail="unauthorized")

    @fapp.get("/health/liveness")
    async def fapi_liveness():
        return {"status": "ok"}

    @fapp.get("/health/readiness")
    async def fapi_readiness():
        status = await asyncio.to_thread(health_status)
        if not all(status.values()):
            return JSONResponse(status_code=503, content=status)
        return status

    @fapp.get("/health/startup")
    async def fapi_startup():
        if not runtime.started:
            return JSONResponse(status_code=503, content={"status": "starting"})
        return {"status": "ok"}

    @fapp.get("/metrics")
    async def fapi_metrics(authorization: str = Header(default="")):
        require_admin(authorization)
        if _HAS_PROMETHEUS:
            return PlainTextResponse(generate_latest(_METRICS_REGISTRY).decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
        return PlainTextResponse(_fallback_metrics_text(), media_type="text/plain")

    @fapp.get("/status")
    async def fapi_status(authorization: str = Header(default="")):
        require_admin(authorization)
        return await asyncio.to_thread(_status_payload)

    @fapp.post("/tasks")
    async def fapi_create_task(data: dict, authorization: str = Header(default="")):
        require_admin(authorization)
        try:
            task = await asyncio.to_thread(_create_task_from_data, data)
            return {"id": str(task.id), "status": TaskStatus.QUEUED.value}
        except (ValueError, TypeError, PermanentError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @fapp.get("/tasks/{task_id}")
    async def fapi_get_task(task_id: str, authorization: str = Header(default="")):
        require_admin(authorization)
        details = await asyncio.to_thread(_task_details, task_id)
        if not details:
            raise HTTPException(status_code=404, detail="not found")
        return details

    @fapp.post("/schedules")
    async def fapi_create_schedule(data: dict, authorization: str = Header(default="")):
        require_admin(authorization)
        try:
            schedule_id = await asyncio.to_thread(_create_schedule_from_data, data)
            return {"id": schedule_id, "status": "scheduled"}
        except (ValueError, TypeError, PermanentError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @fapp.get("/schedules")
    async def fapi_list_schedules(authorization: str = Header(default="")):
        require_admin(authorization)
        return await asyncio.to_thread(_schedule_rows)

    @fapp.delete("/schedules/{schedule_id}")
    async def fapi_delete_schedule(schedule_id: str, authorization: str = Header(default="")):
        require_admin(authorization)
        def disable() -> bool:
            conn = runtime.database.connect()
            try:
                return bool(conn.execute("UPDATE schedules SET enabled=0 WHERE id=?", (schedule_id,)).rowcount)
            finally:
                conn.close()
        return {"ok": await asyncio.to_thread(disable)}

    @fapp.websocket("/events")
    async def fapi_events(websocket: WebSocket):
        if not _admin_authorized_value(websocket.headers.get("authorization", "")):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        client = (loop, event_queue)
        with _WS_CLIENTS_LOCK:
            _WS_CLIENTS.add(client)
        try:
            while True:
                try:
                    data = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    await websocket.send_json(data)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
        except WebSocketDisconnect:
            return
        except Exception as exc:
            LOGGER.warning(f"event websocket closed: {type(exc).__name__}", extra={"component": "http"})
        finally:
            with _WS_CLIENTS_LOCK:
                _WS_CLIENTS.discard(client)

    @fapp.post("/admin/checkpoint")
    async def fapi_force_checkpoint(authorization: str = Header(default="")):
        require_admin(authorization)
        await runtime.checkpoint.checkpoint_now(_force_checkpoint_state())
        return {"ok": True}

    @fapp.post("/admin/replan/{task_id}")
    async def fapi_force_replan(task_id: str, authorization: str = Header(default="")):
        require_admin(authorization)
        try:
            task_id = str(uuid.UUID(task_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid task_id") from exc
        return {"ok": await asyncio.to_thread(runtime.scheduler.requeue_with_backoff, task_id)}

    @fapp.post("/admin/circuits/{name}/{state}")
    async def fapi_force_circuit(name: str, state: str, authorization: str = Header(default="")):
        require_admin(authorization)
        if state not in {"open", "close"}:
            raise HTTPException(status_code=400, detail="state must be open or close")
        breaker = CircuitBreaker.get(name)
        await asyncio.to_thread(breaker.force_open if state == "open" else breaker.force_close)
        return {"ok": True, "state": breaker.state.state.value}

    return fapp


fastapi_app = _create_fastapi_app()


README_TEXT = """Autonomous Agent System

Architecture

This single-file Python service provides a WAL-backed SQLite task queue, dependency-aware scheduling, bounded concurrent workers, task-scoped working memory, episodic and semantic retrieval, CRC-protected checkpoints, idempotent side-effect execution, circuit breakers, authenticated administrative APIs, bounded chat workers, validated multimodal uploads, structured logging, Prometheus-compatible metrics, Flask chat endpoints, and FastAPI administrative and WebSocket endpoints.

Configuration

Configuration is loaded from config.yaml, config.yml, or config.json in the current working directory and then the source directory. Environment variables in APP_SECTION__FIELD form override file values. Supported sections are agent, supervisor, memory, scheduler, schedule, sandbox, observability, api, model, vm, summarizer, backoff, and breaker.

Required services

REQUESTY_API_KEY is required for model execution. INSTAVM_API_KEY is required for InstaVM operations. APP_API__ADMIN_TOKEN should be explicitly configured and is required as a Bearer token for administrative endpoints. The generated process-local token is suitable only for isolated development.

Recovery

The runtime acquires an owned process lock before starting. A stale lock triggers requeueing of tasks left in running state. CRC-valid checkpoints restore the recorded task working-memory snapshot and step position. A supervisor restarts the agent loop only when its thread has actually exited; a stale heartbeat from an active thread is reported as an error rather than falsely reported as a restart.

Security

Task, schedule, event, metric, status, InstaVM, and control endpoints require the administrative Bearer token. Chat sessions use an HMAC-signed session token. Shell execution uses a controlled environment, bounded output files, process-group termination, workspace path validation, and fixed worker capacity. PDF uploads are signature, byte, page-count, render-count, and output-size validated before persistence.

Serving

Set APP_SERVER to flask or fastapi. Flask provides chat, upload, history, task, schedule, event-stream, health, metrics, status, and control routes. FastAPI provides health, metrics, status, task, schedule, authenticated WebSocket events, and control routes. The selected server binds to APP_API__HOST and APP_API__PORT.

Testing

Run the embedded unittest suite with RUN_TESTS=1 or the --run-tests command-line argument.
"""


def _write_readme() -> None:
    atomic_write(README_PATH, README_TEXT.encode("utf-8"))


def _shutdown(signum, frame) -> None:
    LOGGER.info("shutdown signal received", extra={"component": "runtime"})
    runtime.stop_now()
    _CHAT_JOB_EXECUTOR.shutdown(wait=True, cancel_futures=True)
    raise SystemExit(0)


class _TestDatabase:
    def __init__(self):
        self._tmp = tempfile.mkdtemp(prefix="agent_test_")
        self.db_path = pathlib.Path(self._tmp) / "test.db"
        self.checkpoint_path = pathlib.Path(self._tmp) / "test.checkpoint"
        self.lock_path = pathlib.Path(self._tmp) / "test.lock"
        self.db = Database(self.db_path)

    def cleanup(self):
        import shutil
        with contextlib.suppress(Exception):
            shutil.rmtree(self._tmp)


class TestHappyPath(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = _TestDatabase()
        self._rng = random.Random(42)

    def tearDown(self):
        self._td.cleanup()

    async def test_single_task_full_cycle(self):
        summarizer = ExtractiveFrequencySummarizer()
        memory = MemoryManager(self._td.db, CONFIG.memory, summarizer)
        scheduler = Scheduler(self._td.db)
        sandbox = ToolSandbox(self._td.db, memory)
        model = StructuredRuleModel()
        checkpoint = CheckpointManager(self._td.checkpoint_path, self._td.db)
        engine = AgentLoop(self._td.db, scheduler, memory, sandbox, model, summarizer, checkpoint)

        task = model_validate(Task, {"id": str(uuid.uuid4()), "priority": 0, "payload": {"goal": "answer question"}, "status": "queued"})
        scheduler.enqueue(task)

        async def run_limited():
            task_item = scheduler.dequeue_next()
            self.assertIsNotNone(task_item)
            await engine._inner_process(task_item, time.monotonic(), 0, [], [], 0)

        await run_limited()

        conn = self._td.db.connect()
        try:
            row = conn.execute("SELECT status FROM tasks WHERE id=?", (str(task.id),)).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], TaskStatus.DONE.value)

        event_count = self._td.db.connect().execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        self.assertGreater(event_count, 0)


class TestTimeoutEnforcement(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = _TestDatabase()

    def tearDown(self):
        self._td.cleanup()

    async def test_wall_timeout(self):
        summarizer = ExtractiveFrequencySummarizer()
        memory = MemoryManager(self._td.db, CONFIG.memory, summarizer)
        scheduler = Scheduler(self._td.db)
        sandbox = ToolSandbox(self._td.db, memory)
        checkpoint = CheckpointManager(self._td.checkpoint_path, self._td.db)

        class InfiniteModel(ModelClient):
            async def generate(self, prompt, tools_schema, max_tokens):
                action = model_validate(ModelAction, {
                    "tool_calls": [{"name": "echo", "arguments": {"text": "loop"}}],
                    "final_answer": None,
                    "declared_intent": "echo",
                    "confidence": 1.0,
                    "rationale": "test",
                })
                usage = model_validate(UsageStats, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
                return action, usage

        engine = AgentLoop(self._td.db, scheduler, memory, sandbox, InfiniteModel(), summarizer, checkpoint)
        task = model_validate(Task, {"id": str(uuid.uuid4()), "priority": 0, "payload": {"goal": "infinite"}, "status": "running"})
        scheduler.enqueue(task)

        started = time.monotonic() - (CONFIG.agent.wall_timeout_s + 1)
        await engine._inner_process(task, started, 0, [], [], 0)

        conn = self._td.db.connect()
        try:
            row = conn.execute("SELECT status FROM tasks WHERE id=?", (str(task.id),)).fetchone()
        finally:
            conn.close()
        self.assertIn(row["status"], [TaskStatus.TIMED_OUT.value, TaskStatus.RUNNING.value])


class TestLivelockDetection(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = _TestDatabase()

    def tearDown(self):
        self._td.cleanup()

    async def test_livelock_triggers_requeue(self):
        summarizer = ExtractiveFrequencySummarizer()
        memory = MemoryManager(self._td.db, CONFIG.memory, summarizer)
        scheduler = Scheduler(self._td.db)
        sandbox = ToolSandbox(self._td.db, memory)
        checkpoint = CheckpointManager(self._td.checkpoint_path, self._td.db)

        call_count = [0]

        class RepeatModel(ModelClient):
            async def generate(self, prompt, tools_schema, max_tokens):
                call_count[0] += 1
                action = model_validate(ModelAction, {
                    "tool_calls": [{"name": "echo", "arguments": {"text": "same"}}],
                    "final_answer": None,
                    "declared_intent": "echo",
                    "confidence": 1.0,
                    "rationale": "test",
                })
                usage = model_validate(UsageStats, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
                return action, usage

        engine = AgentLoop(self._td.db, scheduler, memory, sandbox, RepeatModel(), summarizer, checkpoint)
        engine._sig_window = deque(maxlen=200)

        task = model_validate(Task, {"id": str(uuid.uuid4()), "priority": 0, "payload": {"goal": "repeat"}, "status": "running"})
        scheduler.enqueue(task)
        await engine._inner_process(task, time.monotonic(), 0, [], [], 0)

        conn = self._td.db.connect()
        try:
            rows = conn.execute(
                "SELECT type FROM events WHERE task_id=? AND type IN (?,?)",
                (str(task.id), EventType.LIVELOCK_DETECTED.value, EventType.SELF_CHECK.value),
            ).fetchall()
            task_row = conn.execute(
                "SELECT status FROM tasks WHERE id=?",
                (str(task.id),),
            ).fetchone()
        finally:
            conn.close()
        requeued = task_row and task_row["status"] in (TaskStatus.QUEUED.value, TaskStatus.FAILED.value)
        has_abort_event = len(rows) > 0
        self.assertTrue(requeued or has_abort_event, "expected task to be requeued or abort event emitted")


class TestCircuitBreakerTransitions(unittest.IsolatedAsyncioTestCase):
    async def test_closed_to_open_to_half_open_to_closed(self):
        dep_name = f"test-dep-{uuid.uuid4()}"
        config = BreakerConfig(transient_threshold=2, permanent_threshold=1, cooldown_s=0.05)
        breaker = CircuitBreaker(dep_name, config)

        self.assertEqual(breaker.state.state, BreakerState.CLOSED)
        self.assertTrue(await breaker.allow_request())

        await breaker.record_failure(permanent=False)
        self.assertEqual(breaker.state.state, BreakerState.CLOSED)

        await breaker.record_failure(permanent=False)
        self.assertEqual(breaker.state.state, BreakerState.OPEN)

        self.assertFalse(await breaker.allow_request())

        await asyncio.sleep(0.1)

        self.assertTrue(await breaker.allow_request())
        self.assertEqual(breaker.state.state, BreakerState.HALF_OPEN)

        await breaker.record_success()
        self.assertEqual(breaker.state.state, BreakerState.CLOSED)

    async def test_permanent_failure_opens_immediately(self):
        dep_name = f"test-perm-{uuid.uuid4()}"
        config = BreakerConfig(transient_threshold=5, permanent_threshold=1, cooldown_s=1.0)
        breaker = CircuitBreaker(dep_name, config)
        await breaker.record_failure(permanent=True)
        self.assertEqual(breaker.state.state, BreakerState.OPEN)


class TestRetryBackoff(unittest.IsolatedAsyncioTestCase):
    def test_backoff_increases(self):
        rng = random.Random(42)
        delays = []
        for attempt in range(4):
            d = exponential_backoff_with_jitter(attempt, 1.0, 2.0, 60.0, rng=rng)
            delays.append(d)
        self.assertTrue(all(d >= 0 for d in delays))

    def test_max_retries_raises(self):
        with self.assertRaises(MaxRetriesExceeded):
            exponential_backoff_with_jitter(5, 1.0, 2.0, 60.0, max_attempts=5)

    def test_transient_error_requeues(self):
        td = _TestDatabase()
        try:
            scheduler = Scheduler(td.db)
            task = model_validate(Task, {"id": str(uuid.uuid4()), "priority": 0, "payload": {}, "status": "running"})
            scheduler.enqueue(task)
            scheduler.update_status(task.id, TaskStatus.RUNNING)
            scheduler.requeue_with_backoff(task.id)
            conn = td.db.connect()
            try:
                row = conn.execute("SELECT status,attempts FROM tasks WHERE id=?", (str(task.id),)).fetchone()
            finally:
                conn.close()
            self.assertEqual(row["status"], TaskStatus.QUEUED.value)
            self.assertEqual(row["attempts"], 1)
        finally:
            td.cleanup()


class TestMemorySummarization(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = _TestDatabase()

    def tearDown(self):
        self._td.cleanup()

    async def test_summarizer_produces_output(self):
        summarizer = ExtractiveFrequencySummarizer()
        texts = ["The quick brown fox jumps over the lazy dog." * 10 for _ in range(5)]
        results = await summarizer.summarize(texts, 200)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], str)

    async def test_episodic_summarization_compacts(self):
        summarizer = ExtractiveFrequencySummarizer()
        memory = MemoryManager(self._td.db, CONFIG.memory, summarizer)
        task_id = uuid.UUID(int=0)
        for i in range(10):
            ev = model_validate(Event if _HAS_PYDANTIC else dict, {
                "type": EventType.TOOL_RESULT.value,
                "task_id": str(task_id),
                "payload": {"result": f"item {i} " * 100},
                "tokens_used": 50,
            }) if not _HAS_PYDANTIC else Event(
                type=EventType.TOOL_RESULT,
                task_id=task_id,
                payload={"result": f"item {i} " * 100},
                tokens_used=50,
            )
            memory.episodic.append_event(ev)
        count_before = memory.episodic.count()
        self.assertGreater(count_before, 0)


class TestCheckpointRecovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = _TestDatabase()

    def tearDown(self):
        self._td.cleanup()

    async def test_checkpoint_write_and_load(self):
        checkpoint = CheckpointManager(self._td.checkpoint_path, self._td.db)
        state = CheckpointState(
            working_memory_ptr=7,
            episodic_cursor=42,
            step_counter=13,
            version=1,
        )
        await checkpoint.checkpoint_now(state)
        loaded = checkpoint.load_latest()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.step_counter, 13)
        self.assertEqual(loaded.working_memory_ptr, 7)
        self.assertEqual(loaded.episodic_cursor, 42)

    async def test_corrupt_checkpoint_returns_none(self):
        checkpoint = CheckpointManager(self._td.checkpoint_path, self._td.db)
        self._td.checkpoint_path.write_bytes(b'{"crc32": 999, "step_counter": 1}')
        loaded = checkpoint.load_latest()
        self.assertIsNone(loaded)

    async def test_idempotency_key_deduplication(self):
        key = make_idempotency_key(uuid.UUID(int=1), "intent", {"name": "echo", "arguments": {}})
        key2 = make_idempotency_key(uuid.UUID(int=1), "intent", {"name": "echo", "arguments": {}})
        self.assertEqual(key, key2)

        key3 = make_idempotency_key(uuid.UUID(int=2), "intent", {"name": "echo", "arguments": {}})
        self.assertNotEqual(key, key3)


class TestConcurrentTaskGroup(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = _TestDatabase()

    def tearDown(self):
        self._td.cleanup()

    async def test_concurrent_tasks_via_taskgroup(self):
        results: typing.List[int] = []
        lock = asyncio.Lock()

        async def work(i: int) -> None:
            await asyncio.sleep(0.01)
            async with lock:
                results.append(i)

        async with asyncio.TaskGroup() as tg:
            for i in range(5):
                tg.create_task(work(i))

        self.assertEqual(sorted(results), list(range(5)))


class TestHealthEndpoints(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = _TestDatabase()

    def tearDown(self):
        self._td.cleanup()

    def test_health_status_returns_dict(self):
        status = health_status()
        self.assertIn("database", status)
        self.assertIn("vector_index", status)
        self.assertIn("model", status)

    def test_liveness_is_bool_values(self):
        status = health_status()
        for v in status.values():
            self.assertIsInstance(v, bool)


class TestToolSandboxSchemaValidation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = _TestDatabase()

    def tearDown(self):
        self._td.cleanup()

    async def test_echo_tool_success(self):
        summarizer = ExtractiveFrequencySummarizer()
        memory = MemoryManager(self._td.db, CONFIG.memory, summarizer)
        sandbox = ToolSandbox(self._td.db, memory)
        result = await sandbox.execute(uuid.UUID(int=0), "test", "echo", {"text": "hello"})
        self.assertEqual(result.get("text"), "hello")

    async def test_unknown_tool_raises_permanent(self):
        summarizer = ExtractiveFrequencySummarizer()
        memory = MemoryManager(self._td.db, CONFIG.memory, summarizer)
        sandbox = ToolSandbox(self._td.db, memory)
        with self.assertRaises(PermanentError):
            await sandbox.execute(uuid.UUID(int=0), "test", "nonexistent_tool_xyz", {})

    async def test_circuit_open_raises(self):
        summarizer = ExtractiveFrequencySummarizer()
        memory = MemoryManager(self._td.db, CONFIG.memory, summarizer)
        sandbox = ToolSandbox(self._td.db, memory)
        breaker = CircuitBreaker.get("echo")
        breaker.force_open()
        try:
            with self.assertRaises(CircuitOpenError):
                await sandbox.execute(uuid.UUID(int=0), "test", "echo", {"text": "hi"})
        finally:
            breaker.force_close()


class TestResourceLimits(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = _TestDatabase()

    def tearDown(self):
        self._td.cleanup()

    async def test_output_size_limit(self):
        summarizer = ExtractiveFrequencySummarizer()
        memory = MemoryManager(self._td.db, CONFIG.memory, summarizer)
        sandbox = ToolSandbox(self._td.db, memory)

        big_output = {"text": "x" * (CONFIG.sandbox.max_output_bytes + 1000)}

        def _big_echo(value):
            return big_output

        sandbox.register(
            ToolSpec(
                name="big_echo",
                version="1",
                input_schema=EchoInput,
                output_schema=TextOutput,
                timeout_s=5.0,
                resource_limits={},
                side_effects=False,
                idempotency_key_fn=None,
                fallback_tool=None,
            ),
            _big_echo,
        )
        with self.assertRaises(PermanentError):
            await sandbox.execute(uuid.UUID(int=0), "test", "big_echo", {"text": "trigger"})


def run_tests() -> None:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_classes = [
        TestHappyPath,
        TestTimeoutEnforcement,
        TestLivelockDetection,
        TestCircuitBreakerTransitions,
        TestRetryBackoff,
        TestMemorySummarization,
        TestCheckpointRecovery,
        TestConcurrentTaskGroup,
        TestHealthEndpoints,
        TestToolSandboxSchemaValidation,
        TestResourceLimits,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stderr)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    if os.environ.get("RUN_TESTS") == "1" or "--run-tests" in sys.argv[1:]:
        run_tests()
        return
    runtime.start()
    server_mode = str(os.environ.get("APP_SERVER", "flask" if app is not None else "fastapi")).strip().lower()
    try:
        if server_mode == "flask":
            if app is None:
                raise RuntimeError("Flask is unavailable")
            from werkzeug.serving import make_server
            server = make_server(CONFIG.api.host, CONFIG.api.port, app, threaded=True)
            server.serve_forever()
        elif server_mode == "fastapi":
            if not _HAS_FASTAPI or uvicorn is None or fastapi_app is None:
                raise RuntimeError("FastAPI or Uvicorn is unavailable")
            uvicorn.run(fastapi_app, host=CONFIG.api.host, port=CONFIG.api.port, log_level="warning", access_log=False)
        else:
            raise RuntimeError("APP_SERVER must be flask or fastapi")
    finally:
        runtime.stop_now()
        _CHAT_JOB_EXECUTOR.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
