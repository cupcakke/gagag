from core.common import *
from core.models import *
from core.database import Database
from core.logger import record_event

class Scheduler:
    def __init__(self, database: Database, config: typing.Any = None, schedule_config: typing.Any = None, backoff_config: typing.Any = None):
        self.db = database
        self.config = config
        self.schedule_config = schedule_config
        self.backoff_config = backoff_config
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
                        getattr(self.backoff_config, "base_s", 1.0),
                        getattr(self.backoff_config, "factor", 2.0),
                        getattr(self.backoff_config, "max_delay_s", 60.0),
                        max_attempts=getattr(self.backoff_config, "max_attempts", 5),
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

