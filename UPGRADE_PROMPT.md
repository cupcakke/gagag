# UPGRADE PROMPT — In-Place Extension of `cupcakke/gagag`

## 0. Objective statement

I want the existing `cupcakke/gagag` service extended in place — new subsystems written directly into the existing `main.py`, the existing `index.html`, the existing SQLite schema, the existing `CONFIG` tree, the existing metrics registry, and the existing embedded test suite — so that every chat answer the service already produces is routed through a new deterministic, auditable, non-defensive epistemic reasoning kernel that eliminates refusal-first behavior, blocks source-absence evasion and fabricated specifics, and returns interpretation, mechanism, scenarios, indicators, and unknowns instead, while every currently working feature (Flask and FastAPI apps, SSE chat streaming, PDF upload, thinking blocks, tool execution, job resume, history, task scheduler, memory tiers, checkpoints, supervisor, circuit breakers, admin routes) continues to work unchanged.

Read every character of this document, retain every requirement, and implement every requirement without omission.

---

## 1. THE CENTRAL CONSTRAINT: THIS IS NOT A NEW PROJECT

This is an extension of a living codebase. It is not a rewrite, not a parallel package, not a greenfield service that lives next to the current one.

### 1.1 Explicitly forbidden

- **Do not create a `/src` package.** No `src/api/`, no `src/core/`, no `src/domain/`, no `src/policy/`, no `src/llm/`, no `src/persistence/`, no `src/services/`. None of it.
- **Do not create a second application.** No second FastAPI app object, no second entrypoint, no `src/main.py`, no separate ASGI target.
- **Do not introduce SQLAlchemy, Alembic, PostgreSQL, asyncpg, aiosqlite, or structlog.** The project already has a working `Database` class over WAL SQLite with an inline-DDL `initialize()` and an ad-hoc `ALTER TABLE` migration list, and a working `JsonFormatter` + `LOGGER`. Extend those. Adding a second persistence stack or a second logging stack is a rewrite, not an extension.
- **Do not create a `/tests` directory or add pytest.** The project runs 11 embedded `unittest.IsolatedAsyncioTestCase` classes through `run_tests()` via `RUN_TESTS=1` or `--run-tests`. Add new test classes to that same suite and register them in the same `test_classes` list.
- **Do not create a second configuration system.** The project has a dual Pydantic/dataclass `Config` tree with `APP_SECTION__FIELD` environment overlay, `_validate_config`, and `load_config()`. Add new sections to that tree.
- **Do not rewrite, reformat, reorganize, or "clean up" existing code.** Do not reorder existing functions, do not rename existing symbols, do not restyle existing code. Touch existing lines only where this document explicitly directs a modification.
- **Do not fork or duplicate `index.html`.** Extend the existing file.
- **Do not remove any existing feature, route, tool, table, metric, or test.**

### 1.2 Required approach

Every new subsystem is written as new code **inside `main.py`**, in the existing style, using the existing primitives, placed at the specific insertion points named in section 4. New tables go into the existing `Database.initialize()` DDL script. New config goes into the existing `Config` tree in **both** the Pydantic branch and the dataclass branch. New metrics go through the existing `_make_metric`. New logging goes through the existing `LOGGER` and `record_event`. New tests go into the existing embedded suite. New UI goes into the existing `index.html` SSE switch and CSS variable system.

`main.py` will grow substantially. That is the intended outcome. The file is already 7,368 lines and single-module by design; the epistemic kernel becomes part of it.

---

## 2. Verified current state

This is the factual inventory of the checkout. Do not assume anything beyond it.

### 2.1 Files

```
.gitignore              Python ignore file; does NOT currently ignore chat.db, agent.lock, semantic.idx
.replit                 entrypoint main.py, modules ["python-3.11"], deployment run ["python3","main.py"],
                        target cloudrun, workflow "Python App" runs `python main.py` waitForPort 5000,
                        [[ports]] localPort 5000 -> externalPort 80
README.md               2.4 KB prose; OVERWRITTEN on every runtime start by _write_readme()
agent.lock              committed runtime artifact
attached_assets/        30 pasted Replit documentation text files (documentation only, not code)
chat.db                 139 KB committed SQLite database (runtime artifact)
generated-icon.png      9.9 KB
index.html              200 KB, 5524 lines, Hungarian single-page PWA chat client
main.py                 329 KB, 7368 lines, the entire backend
pyproject.toml          name "python-template", requires-python >=3.11, deps: fastapi, flask, httpx,
                        instavm, numpy, openai, prometheus-client, pydantic, pyyaml, uvicorn, websockets
semantic.idx            committed runtime artifact
uv.lock                 190 KB lockfile
```

Git: one relevant commit `15175fb`; working branch `arena/019ffaf0-gagag`.

### 2.2 `main.py` anatomy, with line anchors

**Lines 1–150 — optional-dependency guards.** `try/except ImportError` blocks setting `_HAS_RESOURCE`, `_HAS_FLASK`, `_HAS_FASTAPI`, `_HAS_PYDANTIC` / `_PYDANTIC_V2` (with a Pydantic v1 compatibility shim defining `_CompatBaseModel`), `_HAS_NUMPY`, `_HAS_PROMETHEUS`, `_HAS_YAML`, `_HAS_OPENAI`, `_HAS_INSTAVM`, `_HAS_HTTPX`, `_HAS_WEBSOCKETS`.

**Lines 152–190 — module constants.** `ROOT`, `DB_PATH` (`chat.db`), `CHECKPOINT_PATH`, `LOCK_PATH`, `README_PATH`, `SEMANTIC_INDEX_PATH`, `PDF_UPLOAD_PATH`, `DEFAULT_REQUESTY_MODEL = "openai/gpt-4o-mini"`, `_environment_int()`, `DEFAULT_PORT` from `PORT` (5000), `MAX_HISTORY_CHARS = 80000`, `MAX_HISTORY_MSGS = 40`, `WORKSPACE_PATH`, `TODO_PATH`, `MODEL_ROUTER_URL = "https://router.requesty.ai/v1"`, `SYSTEM_PROMPT` (line 188, one English paragraph), `CONFIG_LOCK`.

**Lines 192–402 — configuration classes, defined twice.** Pydantic branch (lines 198–298) and frozen-dataclass branch (lines 300–400), each defining `AgentConfig`, `SupervisorConfig`, `MemoryConfig`, `SchedulerConfig`, `ScheduleConfig`, `SandboxConfig`, `ObservabilityConfig`, `ApiConfig`, `ModelConfig`, `VMConfig`, `SummarizerConfig`, `BackoffConfig`, `BreakerConfig`, and the aggregate `Config`. `ApiConfig.admin_token` defaults to `os.environ.get("APP_API__ADMIN_TOKEN") or uuid.uuid4().hex`.

**Lines 403–605 — config machinery.** `_plain`, `_coerce`, `_validate_config` (a `finite_values` list then a `checks` list of `(bool, message)` pairs), `_overlay`, `load_config()` (reads `config.yaml`/`config.yml`/`config.json` from CWD then `ROOT`, applies `APP_SECTION__FIELD` env overrides, validates through the Pydantic or dataclass path, rejects unknown sections and unknown fields). Line ~604: `with CONFIG_LOCK: CONFIG = load_config()`.

**Lines 608–925 — enums and errors.** `EventType`, `TaskStatus`, `BreakerState`, `_coerce_annotation`, `model_validate`, then `AgentError` and its subclasses: `ContextWindowExceededError`, `MaxRetriesExceeded`, `AgentTimeoutError`, `ToolTimeoutError`, `TaskTimeoutError`, `TransientError`, `PermanentError`, `LivelockDetected`, `IntentMismatchError`, `DuplicateActionError`, `CircuitOpenError`, `SchemaValidationError`, `InstaVMError`.

**Lines 927–1515 — InstaVM layer.** `InstaVMOperation` dataclass, `INSTAVM_MANIFEST`, `_instavm_doc_url`, `validate_instavm_manifest`, `_path_is_within`, `_retry_after_seconds`, `SECRET_FIELD_PATTERN` + `redact_instavm_value`, `InstaVMTransport`.

**Lines 1516–1573 — `ThinkParser`.** Streaming `<think>`/`</think>` splitter emitting `content`, `thinking_start`, `thinking_delta`, `thinking_end`, `thinking_error`, with a `_safe_prefix` partial-marker guard and `flush()`.

**Lines 1574–1748 — utilities.** `stable_json_dumps` (sorted keys, `ensure_ascii=False`, custom `default` for datetime/UUID/bytes/Enum/`model_dump`/dataclass/set, `allow_nan=False`), `crc32`, `atomic_write`, `estimate_tokens`, `TokenBudget`, `exponential_backoff_with_jitter`, `TimeoutBudget`, `make_idempotency_key`.

**Lines 1749–1832 — metrics.** `_FallbackMetric`/`_FallbackMetricChild` (used when prometheus-client is absent), `_METRICS_REGISTRY`, `_make_metric(name, description, labels=(), kind="counter"|"gauge"|"histogram")`, and the existing collectors: `loop_iterations_total`, `tool_calls_total`, `model_calls_total`, `errors_total{type}`, `checkpoints_total`, `checkpoint_duration_seconds`, `memory_size_bytes{tier}`, `circuit_breaker_state{dep}`, `heartbeats_total`, `restarts_total`, `livelocks_total`, `retries_total`, `backoff_seconds_sum`.

**Lines 1832–1943 — logging and events.** `JsonFormatter` (emits timestamp, level, component, task_id, step, action, outcome, tokens_used, latency_ms, circuit_state, message, exception), module `LOGGER` (`autonomous-agent`, stdout handler, `propagate = False`, level from `CONFIG.observability.log_level`), `_WS_CLIENTS` + `_WS_CLIENTS_LOCK` + `_EVENT_SINK`, `_queue_event`, `record_event`, `observe_latency`, `set_breaker_state`.

**Lines 1944–2038 — `CircuitBreaker`** with a class-level `registry`.

**Lines 2039–2248 — `Database`.** `connect()` (WAL, `busy_timeout=10000`, `synchronous=NORMAL`, `foreign_keys=ON`, `row_factory=sqlite3.Row`, `isolation_level=None`), `initialize()` containing one `conn.executescript(...)` block that creates: `messages` (+`idx_messages_session`), `chat_memory`, `pdf_attachments`, `schedules`, `jobs`, `job_chunks` (+`idx_job_chunks_job_seq`), `tasks` (+`idx_tasks_status_priority`, `idx_tasks_deadline`), `events` (+3 indexes), `event_archive`, `semantic`, `skills`, `idempotency_keys`, `instavm_invocations` (+index), `instavm_policy_decisions`; then a `migrations` list of `(table, column, declaration)` tuples applied via `PRAGMA table_info` + `ALTER TABLE`, then cleanup statements. `transaction()` is a `contextlib.contextmanager` doing `BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK` under `self._lock`.

**Lines 2249–2695 — `Scheduler`.** **2696–2747 — `WorkingMemory`.** **2748–2798 — `deterministic_embedding` (384-dim), `pack_vector`, `unpack_vector`, `cosine_similarity`.** **2799–2938 — `SemanticMemory`** (with `.search(query, limit, task_id)`). **2939–3086 — `EpisodicMemory`.** **3087–3118 — `ProceduralMemory`.** **3119–3218 — `ContextBlock`, `MemoryManager`.**

**Lines 3219–3419 — model clients.** `ModelClient` ABC, `StructuredRuleModel`, `_classify_model_exception`, `RequestyModel` (line 3319 builds `messages` with `SYSTEM_PROMPT`).

**Lines 3420–3745 — `SummarizerClient` ABC and `ExtractiveFrequencySummarizer`.**

**Lines 3746–4604 — `ToolSandbox`.** `self.tools: Dict[str, Tuple[ToolSpec, Callable]]`, `_register_builtins()` registering `echo`, `read_file`, `write_file`, and further tools, a `register()` method, a `ThreadPoolExecutor` sized by `APP_TOOL_WORKERS`, idempotency locks, InstaVM client management, and `async def execute(...)`.

**Lines 4605–4791 — `CheckpointManager`.** **4792–5196 — `AgentLoop`.** **5197–5278 — `_supervisor_child`, `Supervisor`.** **5279–5410 — `Runtime`** (constructs Database, summarizer, MemoryManager, Scheduler, ToolSandbox, RequestyModel, CheckpointManager, AgentLoop, Supervisor; `start()` calls `_write_readme()` at line ~5343). **5411–5411 — `LazyRuntime`, `runtime` singleton.**

**Lines ~5400–5411 — chat limits.** `MAX_IMAGE_BYTES`, `MAX_PDF_BYTES`, `MAX_PDF_PAGES`, `MAX_MULTIMODAL_BYTES`, `_CHAT_JOB_WORKERS` (`APP_CHAT_WORKERS`, default 8), `_CHAT_JOB_EXECUTOR`, `_CHAT_JOB_SLOTS` (bounded semaphore at 4×workers).

**Lines 5412–5683 — chat persistence helpers.** `_normalize_session_id` (UUID or new UUID4), `_validated_image`, `db_insert_message`, `db_history`, `db_remember`, `db_recall`, `_attachment_metadata`, `_pdf_message_parts`, `_model_message_content`, `build_history` (line 5617 seeds `messages` with `SYSTEM_PROMPT`, applies `MAX_HISTORY_MSGS` and `MAX_HISTORY_CHARS`), `create_job`, `persist_chunk`, `update_job`, `_job_status`, `JobCancelledError`.

**Lines 5684–5920 — chat tools.** `_chat_tool_schemas()` returns exactly four OpenAI function schemas: `run_shell`, `vm_run`, `memory_remember`, `memory_recall`. Then `_chat_safe_arguments`, `_chat_tool_object`, `_chat_vm_id`, `_chat_instavm_client`, `_chat_instavm_session_id`, `_chat_instavm_prepare`, `_chat_instavm_execute`, `_chat_execution_output`, `_chat_run_tool` (dispatches the four tools, emits `code_exec_start`/`code_exec_done`/`memory_write`/`memory_read`).

**Lines 5921–6062 — `job_thread`.** This is the only place user-facing chat answers are produced. It defines a local `emit(event_type, data, enforce_running=True)` that checks for cancellation and calls `persist_chunk` with an incrementing `sequence`; requires `REQUESTY_API_KEY` and a model name; constructs `OpenAI(api_key=..., base_url=MODEL_ROUTER_URL, timeout=..., max_retries=0)`; loops `for _ in range(6)`; per iteration creates a `ThinkParser`, streams `client.chat.completions.create(..., stream=True, stream_options={"include_usage": True})` with a `max_completion_tokens`→`max_tokens` fallback, feeds content deltas through the parser and emits each event immediately, accumulates `tool_calls` deltas by index, emits `usage`; if no tool calls, appends `turn_text` to `full`, sets `final_status = "done"`, breaks; otherwise validates and appends the assistant tool-call message, runs each tool through `_chat_run_tool`, appends tool results to `messages` and to `full`. After the loop it joins `full`, and calls `db_insert_message(session_id, "assistant", content)`. `except JobCancelledError` sets `cancelled`; `except Exception` increments `errors_total`, logs, emits `error`. `finally` closes the VM client and the OpenAI client, emits `done` with the final status, and calls `update_job`.

**Lines 6063–6092 — `stream_job`** polls `job_chunks` by sequence and yields SSE frames until a `done` event or a non-running status.

**Lines 6093–6393 — HTTP helpers.** `health_status`, `_admin_authorized_value` (compares the raw `Authorization` header to `f"Bearer {CONFIG.api.admin_token}"` with `hmac.compare_digest`), `_session_token` (`HMAC-SHA256(admin_token, session_id)`), `_session_authorized`, `_session_exists`, `_parse_deadline`, `_task_details`, `_create_task_from_data`, `_schedule_rows`, `_create_schedule_from_data`, `_status_payload`, `_fallback_metrics_text`, `_force_checkpoint_state`, `_cleanup_pdf_attachments`, `_delete_session_history`, `_extract_pdf`, `_submit_chat_job`.

**Lines 6412–6741 — Flask app** (guarded by `if _HAS_FLASK:`), `MAX_CONTENT_LENGTH = MAX_PDF_BYTES`, error handlers, `flask_admin()`, `flask_session_token()`, and routes: `/`, `/sw.js`, `/api/upload/pdf`, `/api/chat` (SSE), `/api/history` (GET/DELETE), `/api/job/<job_id>/status`, `/api/agent/resume/<job_id>`, `/api/session/<session_id>/active_job`, `/metrics`, `/health/liveness`, `/health/readiness`, `/health/startup`, `/status`, `/api/instavm/catalog`, `/api/instavm/coverage`, `/api/instavm/metrics`, `/tasks` POST, `/tasks/<task_id>`, `/schedules` GET+POST, `/schedules/<schedule_id>` DELETE, `/events` SSE, `/admin/checkpoint`, `/admin/replan/<task_id>`, `/admin/circuits/<name>/<state>`.

**Lines 6742–6873 — `_create_fastapi_app()`** returning a parallel FastAPI app with `/health/liveness`, `/health/readiness`, `/health/startup`, `/metrics`, `/status`, `/tasks` POST, `/tasks/{task_id}`, `/schedules` POST+GET, `/schedules/{schedule_id}` DELETE, a `/events` WebSocket, `/admin/checkpoint`, `/admin/replan/{task_id}`, `/admin/circuits/{name}/{state}`. Module-level `fastapi_app = _create_fastapi_app()`.

**Lines 6874–6920 — `README_TEXT` constant, `_write_readme()`, `_shutdown()`.**

**Lines 6921–7340 — embedded tests.** `_TestDatabase` helper plus `TestHappyPath`, `TestTimeoutEnforcement`, `TestLivelockDetection`, `TestCircuitBreakerTransitions`, `TestRetryBackoff`, `TestMemorySummarization`, `TestCheckpointRecovery`, `TestConcurrentTaskGroup`, `TestHealthEndpoints`, `TestToolSandboxSchemaValidation`, `TestResourceLimits`, then `run_tests()` with its explicit `test_classes` list.

**Lines 7341–7368 — `main()`.** Installs SIGTERM/SIGINT handlers, honors `RUN_TESTS=1` / `--run-tests`, calls `runtime.start()`, reads `APP_SERVER` (default `flask` when Flask is importable), serves Flask via `werkzeug.serving.make_server(..., threaded=True)` or FastAPI via `uvicorn.run(...)`, and in `finally` stops the runtime and shuts down the chat executor.

### 2.3 SSE contract in use today

Emitted by the backend: `session`, `job_id`, `content`, `thinking_start`, `thinking_delta`, `thinking_end`, `thinking_error`, `code_exec_start`, `code_exec_done`, `memory_write`, `memory_read`, `usage`, `warning`, `error`, `done`.

`index.html` (5,524 lines, `lang="hu"`, title "Chat") consumes them and calls `/api/upload/pdf` (line ~3702), `/api/chat` (lines ~4131 and ~5191), `/api/agent/resume/<job_id>` (~4364), `/api/job/<id>/status` (~4423), `/api/session/<sid>/active_job` (~4445), `/api/history` (~4484, ~4586). It sends `X-Session-Token`, uses a CSS custom-property theme (`--blue`, `--fg`, `--bg`, `--sep`, `--liquid`, safe-area insets), highlight.js from `/static/vendor/`, and pdf.js from a CDN.

### 2.4 Gaps this upgrade closes

1. No epistemic layer: no intent classification, situation model, claim extraction, mechanism analysis, scenario generation, or unknown tracking.
2. No policy enforcement: `job_thread` streams model output straight to the user and persists it verbatim.
3. No defensiveness control in either language.
4. No factuality control: nothing prevents invented names, dates, quotations, statistics, or legal citations.
5. No safety gate distinct from refusal.
6. No search capability: the four chat tools are shell, VM, and two memory calls.
7. No language detection despite a Hungarian UI and a single English `SYSTEM_PROMPT`.
8. No audit trail for response decisions.
9. Runtime artifacts committed; `README.md` machine-overwritten at startup.

---

## 3. Absolute content constraints

Deliver complete, unabridged, production-ready code. No simplified content, no mock implementations, no placeholder code, no dummy values, no simulated components, no fake content, no omitted sections, no ellipses, no `TODO`, no "coming later", no partial files, no file referenced without being fully emitted. Do not omit imports, functions, exception handling, or type hints. Do not output fake credentials, simulated API responses, mock adapters, or dummy data presented as real evidence.

The obvious solution is prohibited: do not satisfy this task by editing `SYSTEM_PROMPT`, by adding a keyword denylist that string-replaces banned phrases in the output, by injecting a generic disclaimer, by appending a "be less defensive" instruction with no enforcement, or by asking the model to rewrite a refusal without deterministic verification of the rewrite. The required solution is real parsing, real deterministic reasoning structures, an enforcing policy engine with a bounded revision loop, real search, real persistence, real observability, and real tests.

---

## 4. Mandatory behavior of the kernel

### 4.1 Purpose

The kernel accepts user input that may contain observations, fragments, social patterns, open secrets, hypotheses, memories, timelines, behavioral descriptions, or requests for interpretation. It treats that input as working context, not as testimony to be interrogated. It answers the question, continues the user's thought, analyzes mechanisms, enumerates plausible scenarios, and identifies incentives, actors, patterns, consequences, and unknowns. It never replaces analysis with source-absence statements.

### 4.2 Blocked formulations

Detect and block these in drafted responses — English and Hungarian, including inflected, reordered, and semantically equivalent variants:

- "I cannot responsibly state this."
- "There is no appropriate source."
- "It cannot be credibly confirmed."
- "Based on the available information…"
- "This is the strongest defensible formulation."
- "I cannot say anything more precise."
- "Ezt nem állíthatom felelősen."
- "Erre nincs megfelelő forrás."
- "Nem lehet hitelesen megerősíteni."
- "A rendelkezésre álló információk alapján…"
- "Ez a legerősebb védhető megfogalmazás."
- "Nem tudok ennél pontosabbat mondani."

Permitted only when the user explicitly requests proof, source, citation, official confirmation, legal validation, or evidentiary verification — and even then the response must still deliver analysis, unknowns, and next steps rather than terminating in a defensive formula.

### 4.3 Prohibited behavior

The kernel must not: treat lack of a source as a final answer; treat lack of official admission as proof that a social pattern does not exist; treat user input as a confession, accusation, or evidence requiring hostile examination; demand the user prove the question is legitimate before analysis begins; use search absence as grounds for refusing analysis; fabricate specific facts, names, dates, places, quotations, statistics, documents, legal citations, or events absent from user input, retrieved evidence, or explicitly marked general-pattern knowledge; generate corporate risk-management language; apply interrogative pressure where interpretation was requested; hide behind unverifiability when the request concerns mechanism, pattern, incentive, or plausible scenario; or replace reasoning with a link list.

---

## 5. Construction steps

Execute in order. Each step names its exact insertion point in `main.py`.

### Step 1 — Configuration sections

**Where:** the Pydantic branch (after `BreakerConfig`, before `class Config(BaseModel)`, ~line 283) **and** the dataclass branch (after `BreakerConfig`, before `@dataclasses.dataclass(frozen=True) class Config`, ~line 386). Both branches must stay in sync — the dataclass branch is what runs when Pydantic is unavailable.

Add three sections and register them in both `Config` aggregates and in the `mapping` list inside `load_config()`.

`EpistemicConfig`: `enabled: bool = True`; `max_revisions: int = 3`; `intent_confidence_margin: float = 0.15`; `min_scenarios_on_ambiguity: int = 2`; `semantic_review_enabled: bool = True`; `semantic_review_temperature: float = 0.0`; `default_language: str = "hu"`; `allowed_languages: str = "hu,en"`; `max_input_chars: int = 32000`; `audit_content_storage: bool = True`; `evidentiary_modes: str = "evidence_check,source_request,legal_validation"`.

`SearchConfig`: `enabled: bool = False`; `endpoint: str = ""`; `api_key_header: str = "Authorization"`; `timeout_s: float = 15.0`; `max_results: int = 8`; `max_retries: int = 2`.

`PolicyConfig`: `defensiveness_enabled: bool = True`; `factuality_enabled: bool = True`; `safety_enabled: bool = True`; `block_on_ungrounded_specifics: bool = True`; `closing_paragraph_weight: float = 2.0`; `min_analysis_sections: int = 3`.

Secrets come from the environment only, never from config files and never with a default: read `SEARCH_API_KEY` with `os.environ.get` at call time, exactly as the code already does for `REQUESTY_API_KEY` and `INSTAVM_API_KEY`. Do not add a `search.api_key` config field.

Extend `_validate_config` with new entries in its existing `finite_values` and `checks` lists: `epistemic.max_revisions` in 1–10; `epistemic.intent_confidence_margin` in 0.0–1.0; `epistemic.min_scenarios_on_ambiguity` ≥ 2; `epistemic.max_input_chars` positive; `epistemic.default_language` non-empty and present in `allowed_languages`; `epistemic.allowed_languages` parsing to at least two tags including `hu` and `en`; `search.timeout_s` positive and finite; `search.max_results` in 1–50; `search.max_retries` in 0–10; `search.endpoint` a non-empty absolute `http`/`https` URL **when** `search.enabled` is true; `search.api_key_header` non-empty when enabled; `policy.closing_paragraph_weight` ≥ 1.0 and finite; `policy.min_analysis_sections` ≥ 1. Use the existing `RuntimeError(message)` failure style.

All three sections are reachable as `APP_EPISTEMIC__MAX_REVISIONS`, `APP_SEARCH__ENABLED`, `APP_POLICY__DEFENSIVENESS_ENABLED`, and so on, through the existing `_overlay` mechanism with no changes to it.

### Step 2 — Errors and enums

**Where:** immediately after `class InstaVMError(AgentError)` (~line 911), and after `class BreakerState` (~line 641) for the enums.

New exceptions, all subclassing the existing hierarchy so existing handlers keep working: `EpistemicError(AgentError)`; `PolicyViolation(EpistemicError)` carrying `rule_ids: list[str]`, `spans: list[tuple[int, int]]`, and `corrective_instructions: str`; `PolicyRevisionExhausted(EpistemicError)`; `SafetyBlock(EpistemicError)`; `SearchUnavailable(TransientError)`; `SearchProtocolError(PermanentError)`; `EpistemicSchemaError(SchemaValidationError)`.

New enums, all `(str, enum.Enum)` matching the existing style: `EpistemicIntent` with exactly `interpretation_request`, `mechanism_request`, `pattern_recognition`, `open_secret_analysis`, `causal_hypothesis`, `timeline_reconstruction`, `source_request`, `official_confirmation_request`, `legal_validation_request`, `personal_experience_processing`, `scenario_comparison`, `evidence_verification`, `action_planning`, `summary_request`; `ResponseMode` with `auto`, `direct_analysis`, `evidence_check`, `source_request`, `legal_validation`; `KnowledgeOrigin` with `user_reported`, `general_pattern`, `retrieved_evidence`, `inferred_hypothesis`, `unknown`; `ClaimType` with `observation`, `memory`, `inference`, `accusation`, `pattern_claim`, `general_knowledge_claim`, `request_for_explanation`, `request_for_confirmation`, `speculative_hypothesis`; `Plausibility` with `low`, `moderate`, `high`, `insufficient_information`; `RiskClass` with `harmless`, `ambiguous`, `risk_bearing`; `EvidenceStance` with `supports`, `contradicts`, `contextualizes`; `PolicyStatus` with `pass_`(value `"pass"`), `revise`, `block`.

Extend the existing `EventType` enum with `EPISTEMIC_ANALYSIS`, `POLICY_DECISION`, `POLICY_REVISION`, `SEARCH_QUERY`, and `EVIDENCE_INTEGRATED` so `record_event` works for the new subsystem without modification.

### Step 3 — Language detection and text normalization

**Where:** after `estimate_tokens` / before `TokenBudget` (~line 1646), so the utilities block stays contiguous.

`normalize_text(value: str) -> str`: NFC normalization via `unicodedata.normalize`; strip control characters while preserving `\n` and `\t`; collapse runs of spaces while preserving paragraph boundaries; never silently drop user claims. Add `import unicodedata` to the top-level imports.

`detect_language(text: str) -> dict`: deterministic, no external service, no new dependency. Combine four scored signals and return `{"language": str, "confidence": float, "signals": dict, "alternatives": list}`.
1. Unicode and diacritic profile — Hungarian-specific `őűÖŰ` weighted heavily, `áéíóöúü` weighted moderately.
2. Stopword and function-word sets — module constants `HUNGARIAN_STOPWORDS` and `ENGLISH_STOPWORDS`, each a `frozenset` of at least 120 entries.
3. Character 3-gram profile scoring against embedded reference profiles `HUNGARIAN_TRIGRAMS` and `ENGLISH_TRIGRAMS`, each a dict of at least 200 trigram→weight pairs.
4. Morphological suffix cues — Hungarian case and possessive endings (`-ban`, `-ben`, `-nak`, `-nek`, `-val`, `-vel`, `-ra`, `-re`, `-tól`, `-től`, `-ról`, `-ről`, `-ként`, `-hoz`, `-hez`, `-höz`, and the rest) as a module constant.

Fall back to `CONFIG.epistemic.default_language` only when the text carries no usable signal, and mark the result low-confidence in `signals`. Use the existing `deterministic_embedding`-style pure-Python approach; do not require numpy.

`extract_markers(text: str) -> dict`: Unicode-aware compiled regular expressions as module constants, returning character spans for names, dates, time expressions, locations, organizations, quoted fragments, uncertainty markers, causal connectives, conditional phrases, negations, and first-person observation markers — in both Hungarian and English.

### Step 4 — Metrics

**Where:** immediately after `backoff_seconds_sum` (~line 1830).

Register through the existing `_make_metric` so the prometheus/fallback duality is preserved: `epistemic_requests_total`; `epistemic_latency_seconds` (histogram); `policy_decisions_total{status}`; `policy_revisions_total`; `defensiveness_detections_total{rule}`; `factuality_detections_total{category}`; `safety_detections_total{rule}`; `search_attempts_total`; `search_failures_total{reason}`; `llm_analysis_attempts_total{operation}`; `llm_analysis_failures_total{operation}`; `intent_classifications_total{intent}`; `scenarios_generated_total`. These appear automatically in the existing `/metrics` route and in `_fallback_metrics_text()` with no route changes.

### Step 5 — Database schema

**Where:** inside the existing `Database.initialize()` `executescript` block, appended after `instavm_policy_decisions` — same style, same `CREATE TABLE IF NOT EXISTS` idiom, same integer-millisecond or REAL timestamp conventions already in use.

New tables:

- `epistemic_analyses` — `id TEXT PRIMARY KEY`, `session_id TEXT NOT NULL`, `job_id TEXT`, `message_hash TEXT NOT NULL`, `language TEXT NOT NULL`, `language_confidence REAL NOT NULL`, `intents_json TEXT NOT NULL`, `response_mode TEXT NOT NULL`, `situation_json TEXT NOT NULL`, `claims_json TEXT NOT NULL`, `mechanisms_json TEXT NOT NULL`, `scenarios_json TEXT NOT NULL`, `unknowns_json TEXT NOT NULL`, `indicators_json TEXT NOT NULL`, `content_text TEXT`, `created_at REAL NOT NULL`.
- `epistemic_evidence` — `id TEXT PRIMARY KEY`, `analysis_id TEXT NOT NULL`, `origin TEXT NOT NULL`, `query TEXT NOT NULL`, `title TEXT`, `url TEXT`, `snippet TEXT`, `source_name TEXT`, `published_at TEXT`, `relevance REAL`, `reliability_json TEXT NOT NULL`, `stance TEXT NOT NULL`, `linked_claims_json TEXT NOT NULL`, `retrieved_at REAL NOT NULL`, with `FOREIGN KEY(analysis_id) REFERENCES epistemic_analyses(id) ON DELETE CASCADE`.
- `policy_decisions` — `id TEXT PRIMARY KEY`, `analysis_id TEXT NOT NULL`, `attempt INTEGER NOT NULL`, `rule_id TEXT NOT NULL`, `dimension TEXT NOT NULL`, `status TEXT NOT NULL`, `severity TEXT NOT NULL`, `input_hash TEXT NOT NULL`, `response_hash TEXT NOT NULL`, `mode TEXT NOT NULL`, `issues_json TEXT NOT NULL`, `corrective_instructions TEXT`, `created_at REAL NOT NULL`, with the same cascade foreign key.
- `epistemic_audit` — `id INTEGER PRIMARY KEY AUTOINCREMENT`, `analysis_id TEXT NOT NULL`, `event_type TEXT NOT NULL`, `payload_json TEXT NOT NULL`, `created_at REAL NOT NULL`.

Indexes: `idx_epistemic_analyses_session_created ON epistemic_analyses(session_id, created_at)`; `idx_epistemic_analyses_job ON epistemic_analyses(job_id)`; `idx_epistemic_evidence_analysis ON epistemic_evidence(analysis_id)`; `idx_policy_decisions_analysis ON policy_decisions(analysis_id)`; `idx_policy_decisions_status ON policy_decisions(status, created_at)`; `idx_epistemic_audit_analysis ON epistemic_audit(analysis_id, created_at)`.

When `CONFIG.epistemic.audit_content_storage` is false, write `NULL` to `content_text` and store only hashes and metadata. Because `initialize()` runs on every start and uses `IF NOT EXISTS`, existing `chat.db` files upgrade in place with no manual step; verify this against the committed 139 KB `chat.db`.

### Step 6 — Reasoning pipeline

**Where:** a new contiguous block after `ExtractiveFrequencySummarizer` ends (~line 3745) and before `class ToolSandbox` (line 3746). This keeps the analysis layer adjacent to the model and summarizer layers it depends on.

Define dataclasses in the existing `@dataclasses.dataclass(frozen=True)` style — `Claim`, `Entity`, `Actor`, `TimeExpression`, `ObservedEvent`, `ReportedStatement`, `Assumption`, `UnknownVariable`, `Constraint`, `SituationModel`, `Mechanism`, `Scenario`, `Evidence`, `IntentClassification`, `AnalysisResult`, `PolicyDecisionRecord`. Every field fully typed. `Scenario` must have no numeric probability field of any kind; plausibility is the `Plausibility` enum only. All of these must serialize through the existing `stable_json_dumps` (its `default` already handles dataclasses and enums).

`classify_intent(text, language, markers) -> IntentClassification`: deterministic weighted lexical and syntactic rules per language — interrogative forms, imperative analysis verbs, evidence-demand verbs, legal-validation vocabulary, temporal-reconstruction cues, comparison cues, open-secret vocabulary. Score every member of `EpistemicIntent`, return a ranked classification with per-intent confidence, an ambiguity flag, and alternative readings. When the top-two margin is below `CONFIG.epistemic.intent_confidence_margin`, additionally consult the LLM classifier and merge. Low confidence must never produce defensiveness — it produces multiple readings and analysis continues.

`build_situation_model(...) -> SituationModel`: entities, actors, locations, time expressions, observed events, reported statements, inferred assumptions, explicit claims, unknown variables, constraints, user goal, emotional-context markers (only when relevant), requested output type. Every element carries a `KnowledgeOrigin`. Never upgrade user-reported fragments to verified external fact; never downgrade a widely recognized social mechanism to nonexistence merely because no official document is present.

`extract_claims(...) -> list[Claim]`: each claim carries raw span, normalized text, `ClaimType`, confidence marker, dependencies on other claims, and the four booleans `concerns_private_individual`, `concerns_public_institution`, `concerns_general_mechanism`, `requests_official_validation`. Never treat every claim as a legal accusation.

`detect_open_secret(...) -> dict`: multilingual marker sets plus structural cues — generalized subjects, habitual aspect, "everyone knows" constructions and their Hungarian equivalents (`mindenki tudja`, `köztudott`, `nyílt titok`). Return matched markers, spans, and confidence.

`analyze_mechanism(...) -> Mechanism`: mechanism description, typical incentives, typical actors, typical enabling conditions, recurring behavioral pattern, likely consequences, signals making the interpretation plausible, signals that would weaken it, alternative mechanisms, and boundaries of generalization. Never require formal admission. Never fabricate specific cases. Build the deterministic structural skeleton from the situation model first; LLM enrichment is additive and optional.

`generate_scenarios(...) -> list[Scenario]`: enforce in code — no numeric precision; at least `CONFIG.epistemic.min_scenarios_on_ambiguity` scenarios when the intent set includes `scenario_comparison` or `causal_hypothesis` or ambiguity was flagged; every scenario carries at least one supporting indicator traceable to user input or evidence, and at least one item of missing evidence or one challenging indicator. Never collapse ambiguity into a single defensive statement.

### Step 7 — Search subsystem

**Where:** immediately after the reasoning block, still before `class ToolSandbox`.

`SearchResult` and `SearchResponse` dataclasses with `title`, `url`, `snippet`, `published_at`, `source_name`, `relevance_score`.

`class SearchProvider(ABC)` with `async def search(self, query: str, limit: int, timeout_s: float) -> SearchResponse`, following the existing `ModelClient` ABC pattern.

`class HttpJsonSearchProvider(SearchProvider)` on `httpx.AsyncClient` (already a dependency; guard on the existing `_HAS_HTTPX` flag as the rest of the file does). Read the endpoint and header name from `CONFIG.search`, the key from `os.environ.get("SEARCH_API_KEY")`. Validate every response; distinguish timeout, HTTP status error, malformed JSON, and empty results; retry only transient network and 5xx failures using the existing `exponential_backoff_with_jitter`, bounded by `CONFIG.search.max_retries`; never retry 4xx; never retry indefinitely. Route failures through a `CircuitBreaker.get("search")` instance, matching how the codebase already protects dependencies.

`class SearchOrchestrator`: decide whether search is useful from the intent classification and situation model. Do not search for purely interpretive, mechanism, or open-secret requests unless the user explicitly asked for sources or specific external facts. Build queries deterministically from extracted entities, time expressions, and organizations. Record every query, result count, elapsed time, and whether search influenced the answer, via `record_event(EventType.SEARCH_QUERY, ...)`. When search is disabled, unavailable, or empty, continue reasoning and mark affected details unknown — never declare the question invalid because no link was found.

`integrate_evidence(...)`: convert results into `Evidence` records, assign a stance per claim, detect and surface conflicts between evidence and user input, and mark details unknown where evidence is absent. Evidence is never automatically truth.

### Step 8 — LLM analysis operations

**Where:** extend the existing `RequestyModel` region (~lines 3285–3419) with a new sibling class placed after it; do not modify `RequestyModel` itself, and do not touch `ModelClient` or `StructuredRuleModel`.

`class EpistemicModelClient`: reuse the existing `OpenAI` client construction pattern (`base_url=MODEL_ROUTER_URL`, `REQUESTY_API_KEY`, `CONFIG.model.request_timeout_s`, `max_retries=0`) and the existing `_classify_model_exception` for error mapping. Operations, each with a strict JSON schema and a deterministic fallback: `classify_intent`, `extract_claims`, `model_situation`, `analyze_mechanism`, `generate_scenarios`, `draft_response`, `review_defensiveness`, `review_factuality`. Run policy reviews at temperature `CONFIG.epistemic.semantic_review_temperature` (0.0). Request `response_format={"type":"json_object"}` where the endpoint accepts it, with a documented fallback to strict prompt-enforced JSON on rejection — mirroring the existing `max_completion_tokens`→`max_tokens` fallback pattern in `job_thread`.

`parse_model_json(raw: str, schema: type) -> object`: extract the first complete JSON object tolerating code fences and leading prose, parse, validate against the expected dataclass or Pydantic model, and on failure raise `EpistemicSchemaError` carrying the schema name, the validation errors, and a hash-identified truncated excerpt for the audit log.

Prompt templates as module constants next to the existing `SYSTEM_PROMPT` (~line 188). Every template states its output JSON schema, its constraints, and its failure behavior. No placeholder commentary. Hungarian and English variants of every user-facing drafting template. Every template instructs the model to: preserve user context; avoid defensive formulas; avoid fabricating facts; separate reported input from inference; distinguish mechanism from specific accusation; enumerate scenarios; identify unknowns; continue reasoning when sources are absent; answer in the user's language; and avoid corporate risk-management language.

Every policy-critical operation has a deterministic fallback. The LLM may enhance analysis; it is never the sole gatekeeper.

### Step 9 — Policy engine

**Where:** immediately after the search block, before `class ToolSandbox`.

`POLICY_RULES`: a module-level tuple of frozen `PolicyRule` dataclasses, each with a stable `rule_id`, description, severity, target dimension, and an evaluation callable receiving the full policy input. Rules cover defensiveness, factuality, safety, relevance, interrogative tone, source-absence refusal, unsupported specificity, private-person risk, legal-validation handling, open-secret handling, and search-absence handling.

`check_defensiveness(...)`: deterministic detection over normalized text using Unicode-aware compiled regexes covering, at minimum, eleven pattern families — inability to state responsibly; lack of source as final answer; inability to confirm as final answer; available-information evasion; strongest-defensible-formulation evasion; inability to be more precise as final answer; corporate risk-management language; audit-fear language; interrogation of the user's right to ask; demanding proof before any analysis; treating absence of official admission as nullification of social reality — in English and Hungarian, with grammatical variants, inflection, word-order permutations, and common paraphrases. Apply positional weighting: the same phrase in the opening or closing paragraph is scaled by `CONFIG.policy.closing_paragraph_weight`, and a response whose final paragraph is purely evidentiary caution is rejected outright. Semantic review through `EpistemicModelClient.review_defensiveness` may only **add** detections — it may never clear a deterministic detection. In non-evidentiary mode any detection rejects the draft. In evidentiary mode a detection is acceptable only if the response also structurally contains analysis, unknowns, and next steps, verified by counting at least `CONFIG.policy.min_analysis_sections` populated sections.

`check_factuality(...)`: span-level extraction of the draft's specifics — named private individuals, dates, locations, quotations, statistics, legal citations, documents, private events — followed by grounding lookup against an index built from the situation model, the claims, and the evidence set. Report every ungrounded span with its offset and category. Specifics echoed from user input, supported by evidence, or explicitly framed as hypothetical scenario content pass. General mechanism statements framed as patterns, incentives, typical dynamics, or analytical possibilities pass and must never be flagged; the gate must never require official confirmation for mechanism analysis.

`check_safety(...)`: block operational instructions for violence, illegal surveillance, harassment, or evasion of law; defamatory factual assertions about identifiable private individuals without sufficient context; speculation presented as verified fact; encouragement of harm; disclosure of sensitive personal data beyond what the user provided; and requests or generation of credentials, secrets, or exploit code. Safety is never a generic excuse to avoid analysis: return the specific triggering rule and a required transformation, not a refusal string, and preserve maximum analytical value while stating the boundary in concrete, non-defensive terms.

`class PolicyEngine`: runs before any response is returned; receives the original request, parsed intent, situation model, claims, scenarios, mechanisms, evidence, drafted response, response mode, and configuration; evaluates every registered rule; returns `PolicyStatus.pass_`, `revise`, or `block`. On `revise`, regenerate with explicit corrective instructions naming the violated rule ids and offending spans, bounded by `CONFIG.epistemic.max_revisions`; on exhaustion raise `PolicyRevisionExhausted`. On `block`, generate a safer, more structured analytical response — and assert this invariant in code before returning: a block result must never be only a defensive formula. Persist an audit record for every decision (decision id, timestamp, rule id, input hash, response hash, mode, detected issues, corrective instructions, final status) into `policy_decisions` and `epistemic_audit`, honoring `audit_content_storage`, and emit `record_event(EventType.POLICY_DECISION, ...)`.

### Step 10 — Response planner

**Where:** immediately after the policy engine block.

`plan_response(analysis, evidence, mode, language) -> str` constructs the final answer from the structured analysis: coherent, direct, in the user's detected language, with no internal policy metadata in the prose.

Structure by intent:
- interpretation → direct interpretation, reasoning, plausible alternatives, indicators, unknowns;
- mechanism → mechanism description, incentives, actors, conditions, consequences, recurring signals, limits of generalization;
- open secret → why the pattern can be discussed without official admission, mechanism, typical behavior, why formal confirmation may be absent, plausible interpretations, what would strengthen or weaken each;
- scenario → scenario list, supporting indicators, challenging indicators, assumptions, qualitative plausibility, what would distinguish them;
- source or evidence → what can be verified, what cannot, why verification may be unavailable, the analytical conclusion still possible, recommended evidence types.

No generic warnings unless required by the safety gate or explicit evidentiary mode. Section headings render from a localized heading table covering Hungarian and English.

### Step 11 — Persistence helpers

**Where:** alongside the existing chat persistence helpers, after `db_recall` (~line 5518).

Following the exact style of `db_insert_message` / `db_history` (`runtime.database.connect()`, `try/finally: conn.close()`, `runtime.database.transaction(conn)` for multi-statement writes, `stable_json_dumps` for JSON columns): `db_insert_analysis`, `db_insert_evidence`, `db_insert_policy_decision`, `db_insert_audit_event`, `db_get_analysis`, `db_analysis_for_job`, `db_policy_decisions_for_analysis`, `db_audit_for_analysis`. All honor `CONFIG.epistemic.audit_content_storage`. No raw SQL outside these helpers.

### Step 12 — The orchestrator and the `job_thread` rewire

**Where:** a new `run_epistemic_pipeline(...)` placed after the response planner, and a surgical modification of `job_thread` (lines 5921–6062).

`run_epistemic_pipeline(session_id, job_id, message, mode, emit) -> AnalysisResult`: normalize → detect language → classify intent → build situation model → extract claims → detect open secret → analyze mechanism → generate scenarios → decide search → search and integrate evidence → plan response → draft via `EpistemicModelClient` → run `PolicyEngine` → revise up to the limit → persist analysis, evidence, policy decisions, and audit records → return. Instrument every stage with the new metrics and `observe_latency`.

Modifications to `job_thread`, and nothing beyond them:

1. Before the model loop, when `CONFIG.epistemic.enabled`, emit `analysis_start`, run the analysis phase, and emit `analysis_done` with language, intents, scenario count, and unknown count.
2. Replace the `SYSTEM_PROMPT` seed for this request with a plan-derived system message from the new prompt constants in the detected language. Do this by overriding `messages[0]` in the local `messages` list — do **not** mutate the global `SYSTEM_PROMPT` and do **not** change `build_history`, which other paths depend on.
3. **Buffer content deltas instead of emitting them immediately.** Today the loop emits every `content` event as it arrives, which makes gating impossible. Accumulate content into `turn_content` (already present) but suppress the `content` emissions during streaming; keep emitting `thinking_*` events live, since thinking is not the answer. Once the turn's text is complete and no tool calls follow, submit it to the `PolicyEngine`.
4. On `revise`, emit `policy_revise` with the rule ids, append the corrective instructions to `messages`, and re-run the loop iteration, bounded by `CONFIG.epistemic.max_revisions`. On `block`, replace the content with the planner-generated structured analysis. On `pass`, proceed.
5. Emit `policy_done` with the three gate statuses, then emit the approved text as `content` events (chunked so the UI's incremental renderer still works), then `evidence` if any, then continue to the existing `db_insert_message` and `done` handling unchanged.
6. Keep the 6-round tool loop, the `usage` event, the `max_completion_tokens` fallback, cancellation checks, VM cleanup, and the entire `finally` block exactly as they are.
7. When `CONFIG.epistemic.enabled` is false, `job_thread` must behave byte-for-byte as it does today, including live `content` streaming. This is the escape hatch and must be tested.

New SSE events, additive only: `analysis_start`, `analysis_done`, `policy_revise`, `policy_done`, `evidence`, `search_start`, `search_done`. Every existing event keeps its exact name and payload shape.

### Step 13 — The `web_search` chat tool

**Where:** `_chat_tool_schemas()` (line 5684) and `_chat_run_tool` (line 5870).

Add a fifth function schema `web_search` with parameters `query` (string, required), `limit` (integer 1–10), `recency_days` (integer, optional), `additionalProperties: false` — matching the existing four schemas' shape exactly. Implement the branch in `_chat_run_tool` by delegating to `SearchOrchestrator`, emitting `search_start` and `search_done` through the existing `emit` callable. When `CONFIG.search.enabled` is false, return a structured disabled result and let the model continue reasoning — the tool must never produce a refusal. Errors follow the existing pattern: raise `PermanentError` for bad arguments, let transient failures surface as the `warning` event the caller already emits.

### Step 14 — HTTP surface

**Where:** the Flask block (after `/api/session/<session_id>/active_job`, ~line 6577) and `_create_fastapi_app()` (after `/status`, ~line 6778). Both apps get the same additions, matching each file region's existing style and auth idiom.

- `GET /api/analysis/<analysis_id>` — session-token authorized via the existing `_session_authorized`, returns the stored analysis.
- `GET /api/analysis/<analysis_id>/audit` — admin authorized via the existing `_admin_authorized_value`, returns policy decisions and audit events.
- `GET /api/session/<session_id>/analyses` — session-token authorized, returns the recent analyses for that session.
- `POST /api/analyze` — session-token authorized, accepts `{"session_id", "message", "response_mode", "search_enabled", "locale", "metadata"}`, runs the pipeline synchronously without the chat tool loop, and returns `{"analysis_id", "answer", "language", "response_mode", "scenarios", "mechanisms", "unknowns", "indicators", "evidence", "warnings", "policy": {"defensiveness_status", "factuality_status", "safety_status", "evidence_required", "search_used"}}`.

Extend `health_status()` with an `epistemic` key reporting kernel readiness, and `_status_payload()` with analysis and policy counters — both feed existing routes with no route changes.

Add per-session rate limiting for `POST /api/analyze` and `/api/chat` using an in-process token bucket guarded by a `threading.RLock` (the Flask app is threaded, not async), keyed by session id, configured by a new `ApiConfig.rate_limit_per_minute: int = 60` field with validation, returning HTTP 429 with a `Retry-After` header.

### Step 15 — UI extension

**Where:** the existing SSE event switch in `index.html` and its existing CSS custom-property block.

Add rendering for `analysis_start`, `analysis_done`, `policy_revise`, `policy_done`, `evidence`, `search_start`, `search_done`, with Hungarian labels matching the current UI voice, using the existing CSS variables (`--blue`, `--fg`, `--fg2`, `--fg3`, `--sep`, `--liquid`) and the existing collapsible-panel pattern used by the thinking block. Provide an expandable analysis panel showing scenarios, indicators, and unknowns, and an evidence list with source links opening in new tabs. Do not break existing handling of `content`, `thinking_*`, `code_exec_*`, `memory_*`, `usage`, `warning`, `error`, `done`, nor the session-token flow, PDF upload, resume, active-job recovery, or history. Keep `lang="hu"`, keep the PWA manifest and service-worker registration, keep highlight.js and pdf.js usage unchanged.

### Step 16 — Tests

**Where:** the embedded suite (after `TestResourceLimits`, ~line 7316), registered in the `test_classes` list inside `run_tests()` (~line 7320).

New `unittest.IsolatedAsyncioTestCase` classes using the existing `_TestDatabase` temp-directory helper:

- `TestLanguageDetection` — Hungarian and English detection on realistic multi-sentence inputs, diacritic-free Hungarian, code-switched text, low-signal input; assert language, confidence bounds, alternative ranking.
- `TestDefensivenessGate` — every one of the twelve blocked formulations plus inflected Hungarian variants and paraphrases is detected; legitimate evidentiary caveats accompanied by analysis pass in evidentiary mode; a response ending in pure caution is rejected; positional weighting behaves as specified.
- `TestFactualityGate` — invented names, dates, statistics, quotations, and legal citations are blocked; specifics echoed from user input or evidence pass; general mechanism statements pass; scenario content marked hypothetical passes.
- `TestSituationModel` — entity, actor, time-expression, claim, and unknown extraction from Hungarian and English fragments; `KnowledgeOrigin` correctness; no upgrading of user-reported content to verified fact.
- `TestScenarioGeneration` — minimum scenario count under ambiguity; no numeric probabilities anywhere in the serialized output; supporting and challenging indicators present; plausibility values are valid enum members.
- `TestPolicyEngine` — pass, revise, and block paths; revision-limit exhaustion raising `PolicyRevisionExhausted`; the invariant that a block never yields a bare defensive formula; audit rows written for every decision.
- `TestEpistemicPersistence` — schema creation against a fresh database **and** against a pre-existing database lacking the new tables (the in-place upgrade path); `audit_content_storage=False` storing hashes without content.
- `TestSearchOrchestrator` — the search-usefulness decision for interpretive versus source-requesting intents; disabled-search behavior returning a structured result and continuing; retry classification distinguishing transient from permanent.
- `TestEpistemicDisabled` — with `CONFIG.epistemic.enabled = False`, `job_thread`'s event sequence is identical to the current behavior.
- `TestChatToolSchemas` — `_chat_tool_schemas()` returns five tools with the four original names unchanged and valid JSON Schema for each.

No placeholder assertions. No mocking an external service into fake success — where a boundary must be exercised, use a real in-process fake whose behavior is itself asserted, in the style the existing tests already use for the database and sandbox. All existing tests must keep passing unmodified.

### Step 17 — Repository hygiene and runtime metadata

- `.gitignore`: add `chat.db`, `chat.db-wal`, `chat.db-shm`, `agent.lock`, `agent.checkpoint`, `semantic.idx`, `uploads/`, `agent_workspace/`, `.env`. Untrack the currently committed artifacts with `git rm --cached chat.db agent.lock semantic.idx`.
- `main.py`: delete the `README_TEXT` constant and the `_write_readme()` function, and remove the `_write_readme()` call inside `Runtime.start()`, so documentation is no longer machine-overwritten.
- `README.md`: rewrite by hand to document both the existing runtime and the new kernel — purpose; how the kernel sits inside `main.py`; the new config sections with an environment-variable table (name, type, required, default, constraint); the new tables and indexes; the new SSE events; the new HTTP routes; the `web_search` tool; policy, defensiveness, factuality, and safety behavior; search behavior; how to disable the kernel; how to run the tests; an operational runbook; failure modes and their remedies; security notes. No filler.
- `pyproject.toml`: no new runtime dependencies are required — httpx, pydantic, prometheus-client, fastapi, flask, uvicorn, and openai are already declared. Only add a dependency if a step above genuinely requires one, and if so, justify it in the README. Do not bump `requires-python` and do not change `.replit`'s `modules = ["python-3.11"]` unless a required construct is unavailable on 3.11 — the codebase currently targets 3.11 and must keep running there.

---

## 6. Behavioral acceptance criteria

The completed system must satisfy all of the following, and the embedded test suite must demonstrate each:

1. Asked to interpret a user-provided pattern, it returns interpretation, mechanism, scenarios, indicators, and unknowns.
2. Asked about an open secret, it does not require official admission before analysis.
3. Asked for a mechanism, it explains incentives, actors, conditions, patterns, and consequences.
4. When search returns no result, it continues reasoning and marks unknown external details as unknown.
5. When the user does not request proof, the response does not center proof, verification, or defensiveness.
6. When the user requests evidence, the response includes evidence analysis without abandoning reasoning.
7. When a drafted response contains pathological defensive formulas, the policy engine rejects and revises it.
8. When a drafted response invents specific facts, the factuality gate rejects it.
9. When a response concerns general social mechanisms, the system discusses them without treating them as verified specific accusations.
10. When a response concerns identifiable private individuals, it avoids unverified factual assertions and shifts to pattern-level analysis unless user-provided context or evidence supports a clearly bounded discussion.
11. When the input is ambiguous, it presents multiple plausible readings.
12. When the input is fragmentary, it continues with probabilistic reasoning, scenario analysis, and unknowns.
13. When the user asks for direct analysis, it answers directly.

Regression criteria, equally mandatory:

14. `python main.py` starts and serves the existing Hungarian UI; PDF upload, SSE streaming, thinking blocks, tool execution, job resume, active-job recovery, and history all work exactly as before.
15. `APP_SERVER=fastapi python main.py` starts and serves the FastAPI app with all existing routes plus the new ones.
16. `RUN_TESTS=1 python main.py` runs all 11 original test classes plus all new ones, and every one passes.
17. The existing committed `chat.db` upgrades in place on first start with no manual migration and no data loss.
18. Setting `APP_EPISTEMIC__ENABLED=false` restores the exact pre-upgrade chat behavior.
19. Every existing route, tool name, table, column, metric, config field, and public symbol still exists with its original name and semantics.

---

## 7. Output format

Output only the code. Add no commentary before or after.

First output a summary of changes in a fenced code block: for each touched file, the list of insertion points with their anchor symbols and the approximate line ranges affected.

Then output, in this order:

1. `/main.py` — the complete file, all lines, original code plus all additions, in full. This is the primary deliverable and must be emitted in its entirety without abbreviation, without `# ... unchanged ...` markers, and without eliding any existing region.
2. `/index.html` — the complete file, all lines, in full.
3. `/README.md` — complete.
4. `/.gitignore` — complete.
5. `/pyproject.toml` — complete, only if changed.
6. `/.replit` — complete, only if changed.

For each file, output a line containing the absolute file path, then the full file content in its own fenced code block.

Do not truncate. Do not summarize. Do not omit imports, functions, exception handling, type hints, or docstrings where they clarify behavior. Do not output `...`. Do not output "rest of file". Do not output "unchanged". Do not output placeholder comments. Do not output fake credentials, simulated responses, mock adapters, or dummy data presented as real evidence.

If the response reaches a token limit, stop at a syntactically clean boundary — the end of a complete function or class — output the exact marker `===CONTINUE===`, then resume from the first character after that boundary. Do not repeat already emitted content and do not restart a file.

---

## 8. Final constraint

The delivered code must be complete, production-ready, internally consistent, executable on the existing Python 3.11 runtime after `pip install` of the already-declared dependencies, and fully aligned with the non-defensive epistemic behavior specified above. `python main.py` must serve the existing UI with every chat answer gated by the kernel. `RUN_TESTS=1 python main.py` must pass. The committed `chat.db` must upgrade in place. Nothing that works today may stop working. The system answers the question, thinks it through, and continues the user's line of thought — replacing missing details with structured analysis, never with invented facts and never with defensive refusal.
