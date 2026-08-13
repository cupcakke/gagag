# Autonomous Agent with the Epistemic Reasoning Kernel

A single-file Python service (`main.py`) that runs an autonomous task agent, a Hungarian chat front end (`index.html`), and an epistemic reasoning kernel that turns user input into an explicit situation model, mechanisms, scenarios, indicators and open questions, and then enforces non-defensive, grounded, safe answers through a three-dimensional policy engine.

The kernel is not a system prompt and not a keyword filter. It is a deterministic analysis pipeline with an auditable decision trail: every analysis, every retrieved source and every policy decision is written to SQLite and is retrievable over HTTP.

## Contents

- [Runtime layout](#runtime-layout)
- [Requirements and installation](#requirements-and-installation)
- [Running](#running)
- [Configuration](#configuration)
- [The epistemic kernel](#the-epistemic-kernel)
- [Policy engine](#policy-engine)
- [Search and evidence](#search-and-evidence)
- [Database schema](#database-schema)
- [HTTP API](#http-api)
- [SSE events](#sse-events)
- [Chat tools](#chat-tools)
- [Observability](#observability)
- [Disabling the kernel](#disabling-the-kernel)
- [Tests](#tests)
- [Operational runbook](#operational-runbook)
- [Failure modes and remedies](#failure-modes-and-remedies)
- [Security notes](#security-notes)

## Runtime layout

| Path | Role |
| --- | --- |
| `main.py` | The whole backend: configuration, database, agent loop, tool sandbox, epistemic kernel, policy engine, Flask and FastAPI routes, embedded test suite. |
| `index.html` | Hungarian single-page chat UI (PWA, service worker, highlight.js, pdf.js), served at `/`. |
| `pyproject.toml` | Dependency declaration. Targets Python 3.11 or later. |
| `.replit` | Replit entrypoint (`python3 main.py`, port 5000 mapped to 80). |
| `chat.db` | SQLite database (WAL). Created on first start; git-ignored. |
| `agent.lock`, `agent.checkpoint`, `semantic.idx` | Process lock, CRC-protected checkpoint, vector index. Runtime artifacts; git-ignored. |
| `agent_workspace/` | Working directory for the `run_shell` tool; git-ignored. |

Internally `main.py` is organised as: configuration loader and validator, enums and error hierarchy, epistemic prompt constants, logging and metrics, `Database`, `Scheduler`, memory tiers, model clients, the epistemic block (types, `EpistemicModelClient`, analysis functions, search and evidence, policy engine, response planner, pipeline, persistence helpers), the tool sandbox, `AgentLoop` and `Supervisor`, `Runtime`, the Flask and FastAPI applications, and the embedded `unittest` suite.

## Requirements and installation

Python 3.11 or later. Declared dependencies: `fastapi`, `flask`, `httpx`, `instavm`, `numpy`, `openai`, `prometheus-client`, `pydantic`, `pyyaml`, `uvicorn`, `websockets`. The epistemic kernel introduces no new dependency: it uses `httpx` for search (already declared), `prometheus-client` for metrics (already declared), and the standard library for everything else.

```
uv sync
```

Every third-party import in `main.py` is optional at import time and guarded by a `_HAS_*` flag. With no dependencies installed the module still imports, the deterministic kernel still runs, and the embedded test suite still passes; only the HTTP servers, the model client and live search require their respective packages.

## Running

```
REQUESTY_API_KEY=... APP_API__ADMIN_TOKEN=... python3 main.py
```

`APP_SERVER` selects the server: `flask` (default when Flask is importable) serves the chat UI, uploads, history, SSE, analysis, admin and health routes; `fastapi` serves health, metrics, status, analysis, task, schedule, WebSocket event and admin routes. Both bind to `APP_API__HOST` and `APP_API__PORT`.

## Configuration

Configuration is read from `config.yaml`, `config.yml` or `config.json` in the working directory and then next to `main.py`, and is then overridden by environment variables in `APP_<SECTION>__<FIELD>` form (for example `APP_EPISTEMIC__ENABLED=false`). Unknown sections and unknown fields are rejected at startup. All values are validated; a failed check raises at import time rather than at first use.

Sections: `agent`, `supervisor`, `memory`, `scheduler`, `schedule`, `sandbox`, `observability`, `api`, `model`, `vm`, `summarizer`, `backoff`, `breaker`, `epistemic`, `search`, `policy`.

### `epistemic`

| Variable | Type | Default | Constraint |
| --- | --- | --- | --- |
| `APP_EPISTEMIC__ENABLED` | bool | `true` | — |
| `APP_EPISTEMIC__MAX_REVISIONS` | int | `3` | ≥ 0 |
| `APP_EPISTEMIC__INTENT_CONFIDENCE_MARGIN` | float | `0.15` | 0.0 – 1.0 |
| `APP_EPISTEMIC__MIN_SCENARIOS_ON_AMBIGUITY` | int | `2` | ≥ 1 |
| `APP_EPISTEMIC__SEMANTIC_REVIEW_ENABLED` | bool | `true` | — |
| `APP_EPISTEMIC__SEMANTIC_REVIEW_TEMPERATURE` | float | `0.0` | 0.0 – 2.0 |
| `APP_EPISTEMIC__DEFAULT_LANGUAGE` | str | `hu` | non-empty, must be listed in `allowed_languages` |
| `APP_EPISTEMIC__ALLOWED_LANGUAGES` | str | `hu,en` | comma-separated, at least one entry |
| `APP_EPISTEMIC__MAX_INPUT_CHARS` | int | `32000` | > 0 |
| `APP_EPISTEMIC__AUDIT_CONTENT_STORAGE` | bool | `true` | `false` stores hashes instead of text |
| `APP_EPISTEMIC__EVIDENTIARY_MODES` | str | `evidence_check,source_request,legal_validation` | comma-separated intents that force retrieval |

### `search`

| Variable | Type | Default | Constraint |
| --- | --- | --- | --- |
| `APP_SEARCH__ENABLED` | bool | `false` | requires `endpoint` when true |
| `APP_SEARCH__ENDPOINT` | str | `""` | `http://` or `https://` when search is enabled |
| `APP_SEARCH__API_KEY_HEADER` | str | `Authorization` | non-empty |
| `APP_SEARCH__TIMEOUT_S` | float | `15.0` | > 0 |
| `APP_SEARCH__MAX_RESULTS` | int | `8` | > 0 |
| `APP_SEARCH__MAX_RETRIES` | int | `2` | ≥ 0 |
| `SEARCH_API_KEY` | str | `""` | secret, no default, environment only |

### `policy`

| Variable | Type | Default | Constraint |
| --- | --- | --- | --- |
| `APP_POLICY__DEFENSIVENESS_ENABLED` | bool | `true` | — |
| `APP_POLICY__FACTUALITY_ENABLED` | bool | `true` | — |
| `APP_POLICY__SAFETY_ENABLED` | bool | `true` | — |
| `APP_POLICY__BLOCK_ON_UNGROUNDED_SPECIFICS` | bool | `true` | — |
| `APP_POLICY__CLOSING_PARAGRAPH_WEIGHT` | float | `2.0` | ≥ 1.0 |
| `APP_POLICY__MIN_ANALYSIS_SECTIONS` | int | `3` | ≥ 1 |

### `api` and secrets

| Variable | Type | Default | Constraint |
| --- | --- | --- | --- |
| `APP_API__HOST` | str | `0.0.0.0` | — |
| `APP_API__PORT` | int | `5000` | — |
| `APP_API__ADMIN_TOKEN` | str | random per process | secret; set it explicitly in production |
| `APP_API__RATE_LIMIT_PER_MINUTE` | int | `60` | > 0 |
| `REQUESTY_API_KEY` | str | none | secret, required for model execution |
| `REQUESTY_MODEL`, `REQUESTY_CONTROLLER_MODEL`, `REQUESTY_CODE_MODEL`, `REQUESTY_MULTIMODAL_MODEL`, `REQUESTY_SENSITIVE_MODEL`, `REQUESTY_LONG_CONTEXT_MODEL` | str | provider default | model routing |
| `INSTAVM_API_KEY` | str | none | secret, required for `vm_run` |

Secrets have no defaults in configuration files and are never logged: log payloads pass through redaction before they are emitted.

## The epistemic kernel

`run_epistemic_pipeline(...)` (and its synchronous twin `run_epistemic_pipeline_sync`) executes the following ordered stages, each of which is timed into `epistemic_stage_seconds`:

1. **Language detection** — deterministic scoring over stopword ratio, suffix ratio, character trigram profile and diacritic ratio for Hungarian and English. Returns the language, a confidence in `[0, 1]`, the per-language signals and the ranked alternatives. Low-signal input falls back to `epistemic.default_language` with confidence `0.0`, and the function is pure: the same input always yields the same output.
2. **Intent classification** — 14 intents: `interpretation_request`, `mechanism_explanation`, `scenario_analysis`, `open_secret_analysis`, `evidence_check`, `source_request`, `legal_validation`, `risk_assessment`, `actor_mapping`, `timeline_reconstruction`, `terminology_clarification`, `decision_support`, `emotional_context`, `general_question`. A lexical scorer always produces a result; when a model client is available its classification is merged on top and the deterministic result stays as fallback. The primary intent selects the response mode: `analytical`, `evidentiary`, `explanatory`, `scenario` or `direct`.
3. **Claim extraction** — every sentence becomes a `Claim` with a type (`observed_event`, `reported_statement`, `inference`, `assumption`, `evaluation`, `question`, `emotion`, `norm_reference`, `quantity`), a `KnowledgeOrigin` (`user_statement`, `retrieved_source`, `model_general_knowledge`, `derived_inference`, `unknown`), a confidence, the source span, and flags for private individuals, public institutions, general mechanisms and requests for official validation. User-reported content is never promoted to verified fact.
4. **Situation model** — summary, entities, actors with interests and capabilities, time expressions, observed events, reported statements (with an explicit `verified` flag), assumptions, unknowns (question, why it matters, how to resolve) and constraints.
5. **Open-secret detection** — marker set, knowledge distribution (`asymmetric` or `diffuse`), the reasons silence is individually rational, and the threshold at which the silence breaks.
6. **Mechanism analysis** — mechanisms drawn from a structural library (alignment of incentives, information asymmetry, procedural routine, diffusion of responsibility, resource constraint, network dependence) with preconditions, incentives, typical indicators, counter-indicators and generality.
7. **Scenario generation** — at least `min_scenarios_on_ambiguity` scenarios (raised to three when several unknowns exist or the input is an open secret), each with a qualitative plausibility of `low`, `moderate`, `high` or `insufficient_information`, a reason, supporting and contradicting indicators, a distinguishing test and the information required to settle it. No numeric probabilities are produced anywhere.
8. **Search and evidence integration** — only when retrieval is useful; see below.
9. **Response planning** — `plan_response` builds a structured answer whose sections depend on the primary intent, using the Hungarian or English heading table.
10. **Policy loop** — the draft is evaluated, corrected and re-drafted up to `max_revisions` times; see below.
11. **Persistence** — the analysis, the evidence, every policy decision and an audit trail are written to SQLite.

Without a model client every stage still yields a complete deterministic result, so the kernel degrades in quality but never in availability.

## Policy engine

`PolicyEngine.evaluate` returns one `PolicyOutcome` per enabled dimension, each with a status of `pass`, `revise` or `block`, the individual issues, a corrective instruction and a numeric score.

**Defensiveness.** Regex rules over the lowercased answer covering refusal, disclaimer, referral to a professional, generic caution, moralising filler, apology, model identity hedges, refusal to reason about possibilities, unverifiability used as a stopping reason, and empty closing formulas — in Hungarian and English. Issues in the closing paragraph are weighted by `policy.closing_paragraph_weight`. A closing paragraph consisting only of caution is flagged separately, and an empty answer never passes. Any issue results in `revise`, never in silent acceptance. When a model client is available and `semantic_review_enabled` is set, a semantic review at `semantic_review_temperature` (0.0 by default) adds paraphrases the regexes miss.

**Factuality.** Every specific span in the answer — dates, years, amounts, statute references, section references, case numbers and long quotations — must appear in the grounding corpus assembled from the user message, the situation model, the claims and the retrieved evidence, or in the mechanism descriptions. Ungrounded spans and URLs that are not part of the retrieved evidence are flagged; statutes, sections, case numbers and quotations are high severity. General mechanism statements and hypothetical scenario wording pass, because they are grounded in the mechanism library rather than asserted as fact.

**Safety.** Operational identifying detail about a private individual, categorical guilt attribution, instructions for physical harm and covert surveillance instructions are high severity and cause a `block`. Certainty asserted about a private individual without adjudicated proof is medium severity and causes a `revise`.

A `block` never produces a bare refusal. `PolicyEngine.blocked_response` emits a structured answer containing the situation, the mechanisms, a named list of what was left out and why, and next steps — the answer is narrowed, not withheld. Exhausting the revision limit with a model client in the loop raises `PolicyRevisionExhausted`, which the HTTP layer maps to 422; without a model client the deterministic plan is served instead.

## Search and evidence

Retrieval is off by default. `SearchOrchestrator.should_search` returns true only when search is enabled and configured, and either the response mode is `evidentiary` or one of the detected intents is listed in `epistemic.evidentiary_modes`. Interpretive questions therefore never trigger a network call.

`HttpJsonSearchProvider` is a generic JSON HTTP adapter: it posts `{"query": ..., "limit": ...}` to `search.endpoint`, sends the API key in `search.api_key_header` (as a bearer token when that header is `Authorization`), and accepts responses shaped as a bare array or as an object with a `results`, `data`, `items`, `organic`, `webPages` or `hits` field. Result fields are picked by alias, so most commercial search APIs work without a custom adapter. Requests go through the `search` circuit breaker and retry `search.max_retries` times.

Failures are classified, never fatal: `SearchUnavailable` (transient, breaker open, provider down) and `SearchProtocolError` (permanent, malformed payload) both end retrieval, are recorded as warnings on the analysis, and the pipeline continues from the material already available.

`integrate_evidence` deduplicates by lowercased URL, links each item to the claims it overlaps with, assigns a stance of `supports`, `contradicts` or `neutral`, scores source reliability by host category (official, international body, academic, press, general web), sorts by reliability plus relevance, and caps the set at `search.max_results`.

## Database schema

Existing tables (`messages`, `chat_memory`, `pdf_attachments`, `jobs`, `job_chunks`, `tasks`, `events`, `event_archive`, `schedules`, `semantic`, `skills`, `idempotency_keys`, `instavm_invocations`, `instavm_policy_decisions`) are unchanged. The kernel adds four tables and their indexes, created with `CREATE TABLE IF NOT EXISTS` so an existing `chat.db` is upgraded in place on the next start without data loss:

| Table | Purpose | Indexes |
| --- | --- | --- |
| `epistemic_analyses` | One row per analysis: message hash, language and confidence, intents, response mode, situation, claims, mechanisms, scenarios, unknowns, indicators, optional answer text. | `(session_id, created_at)`, `(job_id)`, `(message_hash)` |
| `epistemic_evidence` | Retrieved sources with reliability, stance and linked claims. Cascades from the analysis. | `(analysis_id)` |
| `policy_decisions` | One row per dimension per attempt: rule ids, status, severity, input and response hashes, issues, corrective instructions. Cascades from the analysis. | `(analysis_id, attempt)`, `(status)` |
| `epistemic_audit` | Append-only event trail for the analysis. | `(analysis_id, created_at)` |

Re-persisting an analysis is an upsert on the primary key, so the related evidence and policy rows survive the answer being rewritten.

## HTTP API

Chat, upload, history, resume and active-job routes are unchanged. Session-scoped routes require the HMAC session token in `X-Session-Token`; admin routes require `Authorization: Bearer <APP_API__ADMIN_TOKEN>`, compared with a constant-time comparison.

| Method and path | Auth | Description |
| --- | --- | --- |
| `POST /api/analyze` | session token | Runs the pipeline synchronously and returns the analysis. |
| `GET /api/analysis/<id>` | session token | The stored analysis with its evidence. |
| `GET /api/analysis/<id>/audit` | admin | Policy decisions and the audit trail. |
| `GET /api/session/<id>/analyses?limit=` | session token | Analyses for the session, newest first. |
| `GET /status` | admin | Includes an `epistemic` block with analysis, evidence, audit and policy counters. |
| `GET /metrics` | admin | Prometheus exposition, including the kernel metrics. |

All of these are mirrored on the FastAPI application.

`POST /api/analyze` request:

```json
{
  "session_id": "9f2406bf-7f8c-49f2-ab1f-cb0cb6154104",
  "message": "Mindenki tudja a faluban, hogy a hivatal és a cég összejátszik, de senki nem mondja ki. Mit jelent ez?",
  "response_mode": "analytical",
  "search_enabled": false,
  "locale": "hu",
  "metadata": {}
}
```

Only `message` is required. Responses: `200` with the analysis, `attempts`, `search_used` and a refreshed `session_token`; `400` for an empty message, an unknown `response_mode`, a non-boolean `search_enabled`, a non-object `metadata`, a malformed `session_id` or an unsupported `locale`; `401` for a session token mismatch; `413` above `epistemic.max_input_chars`; `422` when the policy engine blocks or exhausts its revisions; `429` with a `Retry-After` header above `api.rate_limit_per_minute`; `503` when the kernel is disabled; `502` for any other kernel failure.

## SSE events

`GET /api/agent/resume/<job_id>` and the chat stream carry the existing events (`content`, `thinking_*`, `code_exec_*`, `memory_*`, `usage`, `warning`, `error`, `done`, `session`, `job_id`) plus:

| Event | Payload |
| --- | --- |
| `analysis_start` | `analysis_id`, `language`, `confidence` |
| `search_start` | `analysis_id`, `queries` |
| `search_done` | `analysis_id`, `queries`, `results`, `sources[]` |
| `analysis_done` | `analysis_id`, `language`, `intents`, `response_mode`, `scenarios`, `unknowns`, `mechanisms`, `evidence` |
| `analysis_payload` | The full renderable analysis: intents, mechanisms, scenarios with plausibility, indicators, unknowns, open-secret flag, warnings |
| `policy_revise` | `analysis_id`, `attempt`, `rules` |
| `policy_done` | `analysis_id`, per-dimension statuses, `evidence_required`, `search_used` |
| `evidence` | `analysis_id`, `count`, `items[]` with title, url, source, publication date, stance and reliability |

The UI renders these with Hungarian labels: a status bar for the analysis and policy phases reusing the existing `srch-bar` pattern, a collapsible analysis panel listing mechanisms, scenarios with a qualitative plausibility badge, indicators and open questions, and an evidence list whose links open in a new tab. Answers are streamed in chunks of about 400 characters after the policy engine has approved them, so the user never sees a paragraph that is later retracted.

## Chat tools

The model is offered five tools: `run_shell`, `vm_run`, `memory_remember`, `memory_recall` (all unchanged) and `web_search`. `web_search` takes a `query`, an optional `limit` between 1 and 10 and an optional `recency_days` between 1 and 3650. When search is not configured it returns `{"enabled": false, "results": [], "reason": "external search is not configured; continue the analysis from the material already available and name the missing datum"}` and instructs the model to continue from the available material and name the missing datum, rather than refusing.

## Observability

Structured JSON logs with a component, task id, step, action, outcome, token count and latency field; payloads are redacted before logging. Prometheus metrics added by the kernel:

`epistemic_analyses_total{mode}`, `epistemic_intent_total{intent}`, `epistemic_language_total{language}`, `policy_decisions_total{dimension,status}`, `policy_revisions_total`, `policy_blocks_total{dimension}`, `defensiveness_hits_total{rule}`, `search_queries_total{outcome}`, `evidence_items_total{origin}`, `epistemic_pipeline_seconds`, `epistemic_stage_seconds{stage}`.

`/status` reports open circuit breakers, heartbeat age, in-flight jobs and the epistemic counters. `/health/liveness`, `/health/readiness` and `/health/startup` report database, vector index, model and epistemic-schema availability.

## Disabling the kernel

```
APP_EPISTEMIC__ENABLED=false
```

With this set, `job_thread` skips analysis entirely and the SSE event sequence is byte-for-byte the pre-upgrade sequence (`content`, `usage`, `done`); `/api/analyze` answers `503`; no rows are written to the kernel tables. Individual gates can also be turned off with `APP_POLICY__DEFENSIVENESS_ENABLED`, `APP_POLICY__FACTUALITY_ENABLED` and `APP_POLICY__SAFETY_ENABLED`, and semantic review with `APP_EPISTEMIC__SEMANTIC_REVIEW_ENABLED=false`.

## Tests

```
RUN_TESTS=1 python3 main.py
```

or `python3 main.py --run-tests`. The embedded suite covers the original agent behaviour (happy path, timeouts, livelock detection, circuit breaker transitions, retry backoff, memory summarisation, checkpoint recovery, concurrency, health, sandbox schema validation, resource limits) and the kernel: language detection, the defensiveness gate against every blocked formulation and its inflected variants, the factuality gate against invented dates, amounts, statutes, sections, quotations and links, the situation model and claim taxonomy, scenario generation, the policy engine including revision exhaustion and audit writing, persistence including the in-place upgrade path and hash-only mode, search orchestration and failure classification, the disabled-kernel event sequence, and the chat tool schemas.

No test contacts the network. Where a boundary must be exercised, an in-process fake is used whose own behaviour is asserted. The suite is deterministic under configuration overrides: each kernel test pins the `epistemic`, `search` and `policy` sections to their defaults, so `RUN_TESTS=1` passes regardless of the ambient environment.

## Operational runbook

- **First start.** Set `REQUESTY_API_KEY` and `APP_API__ADMIN_TOKEN`, then run `python3 main.py`. The database, the checkpoint and the lock file are created automatically.
- **Upgrading an existing deployment.** Stop the process, replace `main.py` and `index.html`, start it again. The new tables and columns are created on start; existing rows are untouched. No migration command is required.
- **Enabling search.** Set `APP_SEARCH__ENABLED=true`, `APP_SEARCH__ENDPOINT` and `SEARCH_API_KEY`. Verify with an `evidence_check` request and watch `search_queries_total`.
- **Inspecting a bad answer.** Take the `analysis_id` from the `analysis_done` event or from `/api/session/<id>/analyses`, then read `/api/analysis/<id>/audit` for the rule ids, severities and corrective instructions of every attempt.
- **Tightening or loosening the gates.** Adjust `policy.closing_paragraph_weight` and `epistemic.max_revisions`. Raising the revision limit increases latency and model cost linearly.
- **Rotating the admin token.** Change `APP_API__ADMIN_TOKEN` and restart. Session tokens are derived from it, so all clients re-authenticate.
- **Clean shutdown.** SIGTERM or SIGINT stops the runtime, drains the chat executor and releases the lock. An unclean shutdown is detected on the next start via the stale lock, tasks left in `running` are requeued, and a CRC-valid checkpoint restores the working memory and step position.

## Failure modes and remedies

| Symptom | Cause | Remedy |
| --- | --- | --- |
| Startup raises a configuration message | A validation check failed | Fix the named field; the message states the constraint. |
| `RuntimeError: FastAPI or Uvicorn is unavailable` | `APP_SERVER=fastapi` without the packages | Install the dependencies or set `APP_SERVER=flask`. |
| Answers arrive without the analysis panel | Kernel disabled, or the pipeline raised | Check `APP_EPISTEMIC__ENABLED` and look for the `warning` event and the logged exception type. |
| `422` from `/api/analyze` | Policy block or revision exhaustion | Read the audit trail; the rule ids identify the dimension. Recurrent exhaustion usually means the drafting model ignores the corrective instructions. |
| `429` with `Retry-After` | Rate limit per session | Back off, or raise `APP_API__RATE_LIMIT_PER_MINUTE`. |
| `search_unavailable` warnings | Provider down or the circuit breaker is open | The answer is still produced; check the endpoint and the key, then reset the breaker with the admin circuit route. |
| Factuality flags legitimate specifics | The value is genuinely absent from the input and the evidence | Supply it in the message, or enable search so it can be grounded. |
| Database is locked | Concurrent writers on a slow filesystem | WAL and a ten-second busy timeout are enabled; move `chat.db` to a local disk. |

## Security notes

- Admin routes require a bearer token compared with `hmac.compare_digest`; the per-process fallback token is for isolated development only.
- Chat sessions are authenticated with an HMAC session token derived from the admin token; session ids must be UUIDs.
- `POST /api/analyze` and `POST /api/chat` are rate limited per session with a sixty-second window and a `Retry-After` header.
- With `APP_EPISTEMIC__AUDIT_CONTENT_STORAGE=false` the answer text, evidence snippets and long audit strings are replaced by hashes, so the audit trail stays verifiable without retaining content.
- Log payloads are redacted; secrets are read from the environment and never written to configuration files or logs.
- `run_shell` executes in a controlled environment with a validated workspace path, bounded output, process-group termination and fixed worker capacity; `vm_run` executes in an isolated InstaVM machine.
- PDF uploads are validated for signature, byte size, page count, render count and output size before persistence; image uploads are validated for media type, base64 correctness and magic bytes.
- Evidence links are rendered with `rel="noopener noreferrer"` and only after protocol validation; anything that is not `http` or `https` is rendered as plain text rather than as a link.
