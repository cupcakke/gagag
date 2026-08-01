Autonomous Agent System

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
