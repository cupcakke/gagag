from core.common import *
from config.settings import _mc, WORKSPACE_PATH
from core.models import *
from core.database import Database
from core.logger import record_event, set_breaker_state, observe_latency
from memory.manager import MemoryManager
from execution.instavm import *

class CircuitBreaker:
    registry: typing.Dict[str, "CircuitBreaker"] = {}
    registry_lock = threading.RLock()

    def __init__(self, dependency: str, config: typing.Any = None):
        self.dependency = dependency
        self.config = config or types.SimpleNamespace(transient_threshold=3, permanent_threshold=1, cooldown_s=30.0)
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
    def __init__(self, database: Database, memory: MemoryManager, config: typing.Any = None, vm_config: typing.Any = None):
        self.db = database
        self.memory = memory
        self.config = config
        self.vm_config = vm_config
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
        if self.vm_config is not None and not getattr(self.vm_config, "enabled", True):
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
            "image_variant": getattr(self.vm_config, "image_variant", "nix-dev"),
            "vm_lifetime_seconds": int(getattr(value, "vm_lifetime_seconds", getattr(self.vm_config, "lifetime_seconds", 3600))),
            "memory_mb": int(getattr(value, "memory_mb", getattr(self.vm_config, "memory_mb", 2048))),
            "vcpu_count": int(getattr(value, "vcpu_count", getattr(self.vm_config, "vcpu_count", 2))),
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

