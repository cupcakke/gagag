from core.common import *

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

