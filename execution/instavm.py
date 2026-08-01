from core.common import *
from core.models import AgentError, PermanentError, TransientError, InstaVMError

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

