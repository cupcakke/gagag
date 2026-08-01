from core.common import *
from config.settings import *
from core.models import *
from core.database import Database
from core.logger import LOGGER, record_event, _WS_CLIENTS, _WS_CLIENTS_LOCK
from core.metrics import *
from memory.manager import MemoryManager
from llm.clients import RequestyModel
from execution.instavm import *
from execution.sandbox import CircuitBreaker

APP_CONFIG = None
runtime = None

def configure_routes(runtime_instance: typing.Any, config: typing.Any) -> None:
    global runtime, APP_CONFIG
    runtime = runtime_instance
    APP_CONFIG = config

def _cfg() -> typing.Any:
    if APP_CONFIG is None:
        raise RuntimeError("Az alkalmazás konfigurációja nincs inicializálva")
    return APP_CONFIG

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
        raise ValueError("A session_id értékének UUID formátumúnak kell lennie") from exc

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
        raise ValueError("Érvénytelen base64 képadat") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("A kép túllépi a beállított méretkorlátot")
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
            raise ValueError("A multimodális bemenet túllépi a beállított méretkorlátot")
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
        raise ValueError("Érvénytelen feladatállapot")
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
        raise PermanentError("Az eszközargumentumoknak JSON objektumnak kell lenniük")
    return value


def _chat_tool_object(arguments: dict) -> typing.Any:
    return types.SimpleNamespace(**arguments)


def _chat_vm_id(created: dict) -> str:
    if not isinstance(created, dict):
        raise PermanentError("Az InstaVM létrehozási válasza érvénytelen")
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
        raise PermanentError("Az InstaVM végrehajtáshoz az INSTAVM_API_KEY megadása kötelező")
    if not _HAS_INSTAVM:
        raise PermanentError("The official instavm Python package is not installed")
    return InstaVM(
        api_key=api_key,
        timeout=_cfg().vm.lifetime_seconds,
        cpu_count=_cfg().vm.vcpu_count,
        memory_mb=_cfg().vm.memory_mb,
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
        image_variant=_cfg().vm.image_variant,
        vm_lifetime_seconds=_cfg().vm.lifetime_seconds,
        memory_mb=_cfg().vm.memory_mb,
        vcpu_count=_cfg().vm.vcpu_count,
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
        raise PermanentError("Az InstaVM végrehajtási válasza érvénytelen")
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
        return False, stdout, stderr or "A végrehajtási válasz érvénytelen kilépési kódot tartalmazott"
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
            raise PermanentError("A run_shell nem üres parancsot igényel")
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
            raise PermanentError("A vm_run nem üres parancsot igényel")
        language = str(arguments.get("language") or "bash")
        if language not in {"bash", "python"}:
            raise PermanentError("A vm_run nyelvének bash vagy python értékűnek kell lennie")
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
            raise PermanentError("A memory_remember nem üres kulcsot és értéket igényel")
        db_remember(session_id, key, value)
        emit("memory_write", {"key": key})
        return {"key": key, "saved": True}
    if tool_name == "memory_recall":
        key = str(arguments.get("key") or "").strip()
        if not key:
            raise PermanentError("A memory_recall nem üres kulcsot igényel")
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
        model_name = _cfg().model.name or _cfg().model.controller
        if not model_name:
            raise PermanentError("A REQUESTY_MODEL vagy az APP_MODEL__NAME beállítása kötelező")
        if not api_key:
            raise PermanentError("REQUESTY_API_KEY is not configured")
        if not _HAS_OPENAI:
            raise PermanentError("the openai package is unavailable")
        client = OpenAI(api_key=api_key, base_url=MODEL_ROUTER_URL, timeout=_cfg().model.request_timeout_s, max_retries=0)
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
                "max_completion_tokens": _cfg().model.max_completion_tokens,
            }
            try:
                stream = client.chat.completions.create(**request_parameters)
            except Exception as exc:
                if "max_completion_tokens" not in str(exc):
                    raise
                request_parameters.pop("max_completion_tokens", None)
                request_parameters["max_tokens"] = _cfg().model.max_completion_tokens
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
                    emit("warning", {"message": f"{tool_name} sikertelen"})
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
        emit("error", {"message": "Az asszisztenskérés sikertelen."}, enforce_running=False)
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
            yield f"data: {stable_json_dumps({'type': 'error', 'data': {'message': 'A feladat nem található'}})}\n\n"
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
        vector_ok = runtime.memory.semantic is not None and _cfg().memory.embedding_dim > 0
    except Exception:
        vector_ok = False
    model_ok = bool(os.environ.get("REQUESTY_API_KEY")) and _HAS_OPENAI and bool(_cfg().model.name or _cfg().model.controller)
    return {"database": db_ok, "vector_index": vector_ok, "model": model_ok}


def _admin_authorized_value(value: str) -> bool:
    configured = str(_cfg().api.admin_token or "")
    return bool(configured) and hmac.compare_digest(str(value or ""), f"Bearer {configured}")


def _session_token(session_id: str) -> str:
    return hmac.new(str(_cfg().api.admin_token).encode("utf-8"), session_id.encode("utf-8"), hashlib.sha256).hexdigest()


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
        raise ValueError("A határidőnek ISO 8601 dátum-idő formátumúnak kell lennie") from exc
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
        raise ValueError("A kérés törzsének objektumnak kell lennie")
    dependencies = data.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ValueError("A függőségeknek tömbnek kell lenniük")
    normalized_dependencies = [str(uuid.UUID(str(item))) for item in dependencies]
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("A payload mezőnek objektumnak kell lennie")
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
        raise ValueError("A kérés törzsének objektumnak kell lennie")
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("A payload mezőnek objektumnak kell lennie")
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
        raise ValueError("A PDF megőrzési időnek pozitívnak kell lennie")
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
        raise ValueError("A PDF túllépi a beállított méretkorlátot")
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
        raise ValueError("A PDF oldalszáma túllépi a beállított korlátot")
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
        raise ValueError("A PDF renderelése sikertelen")
    total_rendered = sum(path.stat().st_size for path in rendered)
    if total_rendered > MAX_PDF_BYTES * 4:
        raise ValueError("A renderelt PDF túllépi a beállított kimeneti korlátot")
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
        return jsonify({"error": "Szerverhiba"}), 500

    @app.errorhandler(413)
    def _too_large(exc):
        return jsonify({"error": "A kérés túllépi a beállított méretkorlátot"}), 413

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
            return jsonify({"error": "Fájl megadása kötelező"}), 400
        filename = pathlib.Path(uploaded.filename or "document.pdf").name
        if not filename.lower().endswith(".pdf"):
            return jsonify({"error": "Csak PDF-fájlok fogadhatók el"}), 400
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
            LOGGER.exception("A PDF feldolgozása sikertelen", extra={"component": "http"})
            return jsonify({"error": "A PDF feldolgozása sikertelen"}), 500

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Érvénytelen kérés"}), 400
        try:
            session_id = _normalize_session_id(data.get("session_id"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        supplied_token = str(data.get("session_token") or request.headers.get("X-Session-Token", ""))
        if _session_exists(session_id) and not _session_authorized(session_id, supplied_token):
            return jsonify({"error": "Jogosulatlan munkamenet"}), 401
        message = str(data.get("message", ""))
        if len(message) > 32000:
            return jsonify({"error": "Az üzenet túllépi a beállított hosszkorlátot"}), 413
        try:
            images = [_validated_image(item) for item in (data.get("images") if isinstance(data.get("images"), list) else [])[:20]]
            images = [item for item in images if item is not None]
            if sum(len(base64.b64decode(item["data"], validate=True)) for item in images) > MAX_MULTIMODAL_BYTES:
                raise ValueError("A képek együttes mérete túllépi a beállított korlátot")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        requested = data.get("attachments") if isinstance(data.get("attachments"), list) else []
        attachments = [metadata for raw in requested[:12] if isinstance(raw, str) for metadata in [_attachment_metadata(raw)] if metadata]
        if not message and not images and not attachments:
            return jsonify({"error": "Üres üzenet"}), 400
        history = build_history(session_id)
        db_insert_message(session_id, "user", message, images, attachments)
        job_id = create_job(session_id)
        request_history = history + [{"role": "user", "content": _model_message_content(message, images, attachments, include_pdf_images=True)}]
        if not _submit_chat_job(job_id, session_id, request_history):
            update_job(job_id, "failed")
            return jsonify({"error": "A csevegőszolgáltatás jelenleg túlterhelt"}), 503
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
            return jsonify({"error": "Jogosulatlan munkamenet"}), 401
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
            return jsonify({"error": "Jogosulatlan munkamenet"}), 401
        return jsonify({"status": row["status"], "session_id": row["session_id"]})

    @app.route("/api/agent/resume/<job_id>")
    def agent_resume(job_id):
        conn = runtime.database.connect()
        try:
            row = conn.execute("SELECT session_id FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            return jsonify({"error": "A feladat nem található"}), 404
        if not _session_authorized(row["session_id"], flask_session_token()):
            return jsonify({"error": "Jogosulatlan munkamenet"}), 401
        return Response(stream_job(job_id), content_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/session/<session_id>/active_job")
    def active_job(session_id):
        try:
            session_id = _normalize_session_id(session_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not _session_authorized(session_id, flask_session_token()):
            return jsonify({"error": "Jogosulatlan munkamenet"}), 401
        conn = runtime.database.connect()
        try:
            row = conn.execute("SELECT job_id,status FROM jobs WHERE session_id=? AND status='running' ORDER BY created_at DESC LIMIT 1", (session_id,)).fetchone()
        finally:
            conn.close()
        return jsonify({"status": row["status"], "job_id": row["job_id"], "session_id": session_id} if row else {"status": "none"})

    @app.route("/metrics")
    def metrics_endpoint():
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
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
            return jsonify({"error": "Nincs jogosultság"}), 401
        return jsonify(_status_payload())

    @app.route("/api/instavm/catalog")
    def instavm_catalog_endpoint():
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
        return jsonify([_plain(operation) | {"tool_name": operation.tool_name} for operation in INSTAVM_MANIFEST])

    @app.route("/api/instavm/coverage")
    def instavm_coverage_endpoint():
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
        registered = {name for name in runtime.sandbox.tools if name.startswith("instavm_")}
        expected = {operation.tool_name for operation in INSTAVM_MANIFEST}
        return jsonify({**validate_instavm_manifest(), "implemented": len(registered & expected), "missing": sorted(expected - registered)})

    @app.route("/api/instavm/metrics")
    def instavm_metrics_endpoint():
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
        conn = runtime.database.connect()
        try:
            rows = conn.execute("SELECT operation_id,decision,COUNT(*) AS count FROM instavm_policy_decisions GROUP BY operation_id,decision").fetchall()
        finally:
            conn.close()
        return jsonify([dict(row) for row in rows])

    @app.route("/tasks", methods=["POST"])
    def create_task_endpoint():
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
        try:
            task = _create_task_from_data(request.get_json(force=True, silent=True))
            return jsonify({"id": str(task.id), "status": TaskStatus.QUEUED.value}), 201
        except (ValueError, TypeError, PermanentError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/tasks/<task_id>")
    def get_task_endpoint(task_id):
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
        details = _task_details(task_id)
        return (jsonify(details), 200) if details else (jsonify({"error": "Nem található"}), 404)

    @app.route("/schedules", methods=["POST", "GET"])
    def schedules_endpoint():
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
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
            return jsonify({"error": "Nincs jogosultság"}), 401
        conn = runtime.database.connect()
        try:
            cursor = conn.execute("UPDATE schedules SET enabled=0 WHERE id=?", (schedule_id,))
        finally:
            conn.close()
        return jsonify({"ok": bool(cursor.rowcount)})

    @app.route("/events")
    def events_endpoint():
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
        task_id = request.args.get("task_id")
        if task_id:
            try:
                task_id = str(uuid.UUID(task_id))
            except ValueError:
                return jsonify({"error": "Érvénytelen task_id"}), 400

        try:
            initial_event_id = max(0, int(request.args.get("after", "0") or 0))
        except ValueError:
            return jsonify({"error": "Az after értékének egész számnak kell lennie"}), 400

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
            return jsonify({"error": "Nincs jogosultság"}), 401
        asyncio.run(runtime.checkpoint.checkpoint_now(_force_checkpoint_state()))
        return jsonify({"ok": True})

    @app.route("/admin/replan/<task_id>", methods=["POST"])
    def force_replan_endpoint(task_id):
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
        try:
            task_id = str(uuid.UUID(task_id))
        except ValueError:
            return jsonify({"error": "Érvénytelen task_id"}), 400
        return jsonify({"ok": runtime.scheduler.requeue_with_backoff(task_id)})

    @app.route("/admin/circuits/<name>/<state>", methods=["POST"])
    def force_circuit_endpoint(name, state):
        if not flask_admin():
            return jsonify({"error": "Nincs jogosultság"}), 401
        if state not in {"open", "close"}:
            return jsonify({"error": "Az állapot csak open vagy close lehet"}), 400
        breaker = CircuitBreaker.get(name)
        breaker.force_open() if state == "open" else breaker.force_close()
        return jsonify({"ok": True, "state": breaker.state.state.value})


def _create_fastapi_app() -> typing.Any:
    if not _HAS_FASTAPI:
        return None
    fapp = FastAPI()

    def require_admin(authorization: str) -> None:
        if not _admin_authorized_value(authorization):
            raise HTTPException(status_code=401, detail="Nincs jogosultság")

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
            raise HTTPException(status_code=404, detail="Nem található")
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
            raise HTTPException(status_code=400, detail="Érvénytelen task_id") from exc
        return {"ok": await asyncio.to_thread(runtime.scheduler.requeue_with_backoff, task_id)}

    @fapp.post("/admin/circuits/{name}/{state}")
    async def fapi_force_circuit(name: str, state: str, authorization: str = Header(default="")):
        require_admin(authorization)
        if state not in {"open", "close"}:
            raise HTTPException(status_code=400, detail="Az állapot csak open vagy close lehet")
        breaker = CircuitBreaker.get(name)
        await asyncio.to_thread(breaker.force_open if state == "open" else breaker.force_close)
        return {"ok": True, "state": breaker.state.state.value}

    return fapp


fastapi_app = _create_fastapi_app()

