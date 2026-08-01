from core.common import *
from config.settings import *
from core.models import *
from core.database import Database
from core.logger import LOGGER, record_event
import core.logger as logger_module
from core.metrics import *
from memory.manager import MemoryManager
from llm.clients import ModelClient, RequestyModel, StructuredRuleModel, ExtractiveFrequencySummarizer
from execution.scheduler import Scheduler
from execution.sandbox import ToolSandbox
from agent.checkpoint import CheckpointManager
from agent.loop import AgentLoop
from api.server import create_flask_app, create_fastapi_app

CONFIG = load_config()


class Supervisor:
    def __init__(self, engine: AgentLoop, loop_thread_getter: typing.Callable[[], typing.Optional[threading.Thread]], restart_callback: typing.Callable[[], bool], config: typing.Any):
        self.engine = engine
        self._loop_thread_getter = loop_thread_getter
        self._restart_callback = restart_callback
        self.config = config
        self._stop = threading.Event()
        self._thread: typing.Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._heartbeat_monitor, name="supervisor-monitor", daemon=True)
        self._thread.start()

    def _heartbeat_monitor(self) -> None:
        while not self._stop.wait(self.config.supervisor.heartbeat_interval_s):
            loop_thread = self._loop_thread_getter()
            if loop_thread is not None and not loop_thread.is_alive() and not self.engine.stop_event.is_set():
                if self._restart_callback():
                    restarts_total.inc()
                    record_event(EventType.RESTART, uuid.UUID(int=0), {"reason": "agent_loop_thread_exited"})
                continue
            age = time.time() - self.engine.last_heartbeat
            if self.engine.running_tasks() and age > self.config.supervisor.ping_timeout_s:
                errors_total.labels(type="HeartbeatTimeout").inc()
                record_event(EventType.ERROR, uuid.UUID(int=0), {"reason": "heartbeat_missed", "age_s": age, "running": [str(item) for item in self.engine.running_tasks()]})

    def stop_now(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=self.config.supervisor.restart_grace_s)


README_TEXT = """Autonóm ügynökrendszer

Architektúra

Ez a Python szolgáltatás SQLite-alapú feladatkezelést, függőségérzékeny ütemezést, korlátozott párhuzamos végrehajtást, munkamemóriát, epizodikus és szemantikus visszakeresést, CRC-védett ellenőrzőpontokat, idempotens eszközvégrehajtást, megszakítókat, hitelesített adminisztratív API-kat, korlátozott csevegési dolgozókat, validált multimodális feltöltéseket, strukturált naplózást, Prometheus-kompatibilis metrikákat, Flask csevegési végpontokat és FastAPI adminisztratív valamint WebSocket végpontokat biztosít.

Konfiguráció

A konfiguráció a config.yaml, config.yml vagy config.json fájlokból töltődik be az aktuális munkakönyvtárból vagy a forráskönyvtárból. Az APP_SECTION__FIELD formátumú környezeti változók felülírják a fájlban megadott értékeket.

Kiszolgálás

Állítsd az APP_SERVER változót flask vagy fastapi értékre. A kiválasztott szerver az APP_API__HOST és APP_API__PORT beállítások alapján indul.
"""


def _write_readme() -> None:
    atomic_write(README_PATH, README_TEXT.encode("utf-8"))


class Runtime:
    def __init__(self, config: typing.Any = None):
        self.config = config or load_config()
        self.database = Database(DB_PATH)
        self.summarizer = ExtractiveFrequencySummarizer()
        self.memory = MemoryManager(self.database, self.config.memory, self.summarizer)
        self.scheduler = Scheduler(self.database, self.config.scheduler, self.config.schedule, self.config.backoff)
        self.sandbox = ToolSandbox(self.database, self.memory, self.config.sandbox, self.config.vm)
        self.model: ModelClient = RequestyModel(os.environ.get("REQUESTY_API_KEY", ""), self.config.model.name, self.config.model)
        self.checkpoint = CheckpointManager(CHECKPOINT_PATH, self.database, LOCK_PATH)
        self.engine = AgentLoop(self.database, self.scheduler, self.memory, self.sandbox, self.model, self.summarizer, self.checkpoint, self.config)
        self.supervisor = Supervisor(self.engine, lambda: self._loop_thread, self._restart_loop_if_dead, self.config)
        self.started = False
        self._loop_thread: typing.Optional[threading.Thread] = None
        self._loop: typing.Optional[asyncio.AbstractEventLoop] = None
        self._start_lock = threading.RLock()
        self._closed = False

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.engine.run())
        except Exception as exc:
            errors_total.labels(type=type(exc).__name__).inc()
            LOGGER.error(f"az ügynökciklus megszakadt: {exc}", extra={"component": "runtime"})
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    def _start_loop_thread(self) -> None:
        self.engine.stop_event.clear()
        self._loop_thread = threading.Thread(target=self._run_loop, name="agent-loop", daemon=True)
        self._loop_thread.start()

    def _restart_loop_if_dead(self) -> bool:
        with self._start_lock:
            if not self.started or self.engine.stop_event.is_set():
                return False
            if self._loop_thread is not None and self._loop_thread.is_alive():
                return False
            self.scheduler.drain()
            self._start_loop_thread()
            return True

    def start(self) -> None:
        with self._start_lock:
            if self.started:
                return
            if self._closed:
                raise RuntimeError("A futtatókörnyezet már le van zárva")
            unclean = self.checkpoint.detect_unclean_shutdown()
            if unclean:
                self.scheduler.drain()
            self.checkpoint.write_lock()
            logger_module._EVENT_SINK = self.memory.append
            try:
                _write_readme()
                self._start_loop_thread()
                self.supervisor.start()
                self.started = True
                if unclean:
                    saved = self.checkpoint.load_latest()
                    record_event(EventType.RESTART, uuid.UUID(int=0), {"reason": "unclean_shutdown_recovery", "checkpoint_step": int(getattr(saved, "step_counter", 0) or 0)})
            except Exception:
                self.engine.request_stop()
                self.checkpoint.remove_lock()
                logger_module._EVENT_SINK = None
                raise

    def stop_now(self) -> None:
        with self._start_lock:
            if not self.started:
                return
            self.engine.request_stop()
            self.supervisor.stop_now()
            thread = self._loop_thread
            if thread and thread is not threading.current_thread():
                thread.join(timeout=max(self.config.supervisor.restart_grace_s, self.config.sandbox.default_timeout_s + 1.0))
            if thread and thread.is_alive():
                LOGGER.error("a futtatókörnyezet nem állt le a türelmi időn belül", extra={"component": "runtime"})
                return
            self.scheduler.drain()
            self.sandbox.close()
            self.model.close()
            self.checkpoint.remove_lock()
            logger_module._EVENT_SINK = None
            self.started = False
            self._closed = True


class LazyRuntime:
    def __init__(self, config: typing.Any = None):
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_config", config)

    def _get(self) -> Runtime:
        instance = object.__getattribute__(self, "_instance")
        if instance is not None:
            return instance
        lock = object.__getattribute__(self, "_lock")
        with lock:
            instance = object.__getattribute__(self, "_instance")
            if instance is None:
                instance = Runtime(object.__getattribute__(self, "_config"))
                object.__setattr__(self, "_instance", instance)
            return instance

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._get(), name)

    def __setattr__(self, name: str, value: typing.Any) -> None:
        setattr(self._get(), name, value)


runtime = LazyRuntime(CONFIG)
app = create_flask_app(runtime, CONFIG)
fastapi_app = create_fastapi_app(runtime, CONFIG)


def _shutdown(signum, frame) -> None:
    LOGGER.info("leállítási jel érkezett", extra={"component": "runtime"})
    runtime.stop_now()
    raise SystemExit(0)


def run_tests() -> bool:
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main() -> None:
    if os.environ.get("RUN_TESTS") == "1" or "--run-tests" in sys.argv:
        raise SystemExit(0 if run_tests() else 1)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    runtime.start()
    selected = os.environ.get("APP_SERVER", "flask").strip().lower()
    if selected == "fastapi":
        if not _HAS_FASTAPI or fastapi_app is None:
            raise RuntimeError("A FastAPI kiszolgálóhoz telepíteni kell a fastapi és uvicorn csomagokat")
        uvicorn.run(fastapi_app, host=CONFIG.api.host, port=CONFIG.api.port, log_level="warning", access_log=False)
    else:
        if not _HAS_FLASK or app is None:
            raise RuntimeError("A Flask kiszolgálóhoz telepíteni kell a flask csomagot")
        from werkzeug.serving import make_server
        server = make_server(CONFIG.api.host, CONFIG.api.port, app, threaded=True)
        try:
            server.serve_forever()
        finally:
            runtime.stop_now()


if __name__ == "__main__":
    main()
