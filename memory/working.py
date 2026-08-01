from core.common import *
from llm.parsers import estimate_tokens

class WorkingMemory:
    def __init__(self, token_capacity: int):
        self.capacity = int(token_capacity)
        if self.capacity <= 0:
            raise ValueError("working-memory capacity must be positive")
        self.items: deque = deque()
        self.tokens = 0
        self._lock = threading.RLock()

    def push(self, text: str, metadata: typing.Optional[dict] = None) -> None:
        item = {"text": str(text), "metadata": metadata or {}, "ts": time.time()}
        item_tokens = estimate_tokens(item["text"])
        with self._lock:
            self.items.append(item)
            self.tokens += item_tokens
            while self.tokens > self.capacity and self.items:
                removed = self.items.popleft()
                self.tokens -= estimate_tokens(removed["text"])

    def render(self) -> str:
        with self._lock:
            return "\n".join(item["text"] for item in self.items)

    def pointer(self) -> int:
        with self._lock:
            return len(self.items)

    def snapshot(self) -> typing.List[dict]:
        with self._lock:
            return [dict(item) for item in self.items]

    def restore(self, items: typing.Iterable[dict]) -> None:
        with self._lock:
            self.items.clear()
            self.tokens = 0
            for raw in items:
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("text", ""))
                item = {"text": text, "metadata": dict(raw.get("metadata") or {}), "ts": float(raw.get("ts") or time.time())}
                self.items.append(item)
                self.tokens += estimate_tokens(text)
            while self.tokens > self.capacity and self.items:
                removed = self.items.popleft()
                self.tokens -= estimate_tokens(removed["text"])

    def clear(self) -> None:
        with self._lock:
            self.items.clear()
            self.tokens = 0

