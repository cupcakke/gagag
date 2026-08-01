from core.common import *
from config.settings import _plain
from core.models import *

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

