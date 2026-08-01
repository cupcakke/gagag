from core.common import *
from llm.parsers import estimate_tokens

import threading
import time
import typing
from collections import deque


class WorkingMemory:
    def __init__(self, token_capacity: int):
        self.capacity = int(token_capacity)
        if self.capacity <= 0:
            raise ValueError("working-memory capacity must be positive")
        self.items: deque = deque()
        self.tokens = 0
        self._lock = threading.RLock()

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        token_count = int(estimate_tokens(text))
        return max(0, token_count)

    @staticmethod
    def _normalize_metadata(metadata: typing.Optional[dict]) -> dict:
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None")
        return dict(metadata)

    @staticmethod
    def _normalize_timestamp(value: typing.Any) -> float:
        if value is None:
            return time.time()
        try:
            timestamp = float(value)
        except (TypeError, ValueError, OverflowError):
            return time.time()
        if timestamp != timestamp or timestamp in (float("inf"), float("-inf")):
            return time.time()
        return timestamp

    def _trim_to_capacity(self) -> None:
        while self.tokens > self.capacity and self.items:
            removed = self.items.popleft()
            self.tokens -= removed["_token_count"]
        if self.tokens < 0:
            self.tokens = 0

    def push(self, text: str, metadata: typing.Optional[dict] = None) -> None:
        normalized_text = str(text)
        item = {
            "text": normalized_text,
            "metadata": self._normalize_metadata(metadata),
            "ts": time.time(),
            "_token_count": self._estimate_token_count(normalized_text),
        }
        with self._lock:
            self.items.append(item)
            self.tokens += item["_token_count"]
            self._trim_to_capacity()

    def render(self) -> str:
        with self._lock:
            return "\n".join(item["text"] for item in self.items)

    def pointer(self) -> int:
        with self._lock:
            return len(self.items)

    def snapshot(self) -> typing.List[dict]:
        with self._lock:
            return [
                {
                    "text": item["text"],
                    "metadata": dict(item["metadata"]),
                    "ts": item["ts"],
                }
                for item in self.items
            ]

    def restore(self, items: typing.Iterable[dict]) -> None:
        if items is None:
            raise TypeError("items must be an iterable of dictionaries")

        restored_items = deque()
        restored_tokens = 0

        for raw in items:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text", ""))
            metadata_value = raw.get("metadata")
            metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
            token_count = self._estimate_token_count(text)
            restored_items.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "ts": self._normalize_timestamp(raw.get("ts")),
                    "_token_count": token_count,
                }
            )
            restored_tokens += token_count

        while restored_tokens > self.capacity and restored_items:
            removed = restored_items.popleft()
            restored_tokens -= removed["_token_count"]

        with self._lock:
            self.items = restored_items
            self.tokens = max(0, restored_tokens)

    def clear(self) -> None:
        with self._lock:
            self.items.clear()
            self.tokens = 0
