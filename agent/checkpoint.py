from core.common import *
from core.models import *
from core.database import Database
from core.logger import record_event
from core.metrics import *
from config.settings import LOCK_PATH, _plain

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
            if not isinstance(data, dict):
                raise TypeError("checkpoint state must serialize to an object")
            data = dict(data)
            data["crc32"] = 0
            data["crc32"] = crc32(stable_json_dumps(data).encode("utf-8"))
            validated_state = model_validate(CheckpointState, data)
            atomic_write(self.path, stable_json_dumps(data).encode("utf-8"))
            self.latest = validated_state
        checkpoints_total.inc()
        record_event(
            EventType.CHECKPOINT,
            uuid.UUID(int=0),
            {
                "path": str(self.path),
                "step": int(data.get("step_counter", 0)),
                "version": int(data.get("version", 1)),
            },
        )
        observe_latency("checkpoint", time.perf_counter() - started)
        return validated_state

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
            queue_state_value = getattr(existing, "queue_state", {}) if existing is not None else {}
            queue_state = dict(queue_state_value) if isinstance(queue_state_value, dict) else {}
            tasks_value = queue_state.get("tasks", {})
            tasks = dict(tasks_value) if isinstance(tasks_value, dict) else {}
            active = {str(uuid.UUID(str(item))) for item in in_flight_tasks}
            normalized_task_id = str(uuid.UUID(str(task_id)))
            active.add(normalized_task_id)
            tasks = {
                str(key): dict(value)
                for key, value in tasks.items()
                if str(key) in active and isinstance(value, dict)
            }
            tasks[normalized_task_id] = {
                "step": int(step),
                "working_memory": [dict(item) for item in working_memory],
            }
            step_values = [
                int(value.get("step", 0))
                for value in tasks.values()
                if isinstance(value, dict)
            ]
            state = CheckpointState(
                working_memory_ptr=len(working_memory),
                episodic_cursor=int(episodic_cursor),
                in_flight_tasks=[uuid.UUID(item) for item in sorted(active)],
                queue_state={"tasks": tasks},
                step_counter=max([int(step), *step_values]),
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
        with self._lock:
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
                LOGGER.warning(
                    f"checkpoint load failed: {exc}",
                    extra={"component": "checkpoint"},
                )
                return None

    def _read_lock(self) -> typing.Optional[dict]:
        try:
            value = json.loads(self.lock_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    @staticmethod
    def _lock_pid(data: typing.Optional[dict]) -> int:
        if not data:
            return 0
        try:
            return int(data.get("pid", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return 0

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
        pid = self._lock_pid(data)
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
                fd = os.open(
                    str(self.lock_path),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    view = memoryview(payload)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise OSError("failed to write runtime lock")
                        view = view[written:]
                    os.fsync(fd)
                except Exception:
                    with contextlib.suppress(OSError):
                        os.close(fd)
                    with contextlib.suppress(FileNotFoundError):
                        self.lock_path.unlink()
                    raise
                else:
                    os.close(fd)
                return
            except FileExistsError:
                data = self._read_lock()
                pid = self._lock_pid(data)
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
        if not data:
            return
        if str(data.get("token", "")) != self._owner_token:
            return
        if self._lock_pid(data) != os.getpid():
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
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors_total.labels(type=type(exc).__name__).inc()
                LOGGER.error(
                    f"checkpoint flush failed: {exc}",
                    extra={"component": "checkpoint"},
                )
            finally:
                self.pending.task_done()
