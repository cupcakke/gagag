from core.common import *
from core.models import ContextWindowExceededError

class ThinkParser:
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.state = "content"
        self.buffer = ""

    def _safe_prefix(self, text, marker):
        for length in range(min(len(marker) - 1, len(text)), 0, -1):
            if text.endswith(marker[:length]):
                return text[:-length]
        return text

    def feed(self, delta):
        events = []
        self.buffer += str(delta)
        while self.buffer:
            if self.state == "content":
                position = self.buffer.find(self.OPEN)
                if position < 0:
                    safe = self._safe_prefix(self.buffer, self.OPEN)
                    if safe:
                        events.append(("content", safe))
                        self.buffer = self.buffer[len(safe):]
                    break
                if position:
                    events.append(("content", self.buffer[:position]))
                self.buffer = self.buffer[position + len(self.OPEN):]
                self.state = "thinking"
                events.append(("thinking_start", ""))
            else:
                position = self.buffer.find(self.CLOSE)
                if position < 0:
                    safe = self._safe_prefix(self.buffer, self.CLOSE)
                    if safe:
                        events.append(("thinking_delta", safe))
                        self.buffer = self.buffer[len(safe):]
                    break
                if position:
                    events.append(("thinking_delta", self.buffer[:position]))
                self.buffer = self.buffer[position + len(self.CLOSE):]
                self.state = "content"
                events.append(("thinking_end", ""))
        return events

    def flush(self):
        if not self.buffer:
            return []
        if self.state == "thinking":
            events = [("thinking_delta", self.buffer), ("thinking_error", "unclosed thinking block")]
        else:
            events = [("content", self.buffer)]
        self.buffer = ""
        self.state = "content"
        return events

def estimate_tokens(text: str, override: typing.Optional[int] = None) -> int:
    if override is not None:
        value = int(override)
        if value < 0:
            raise ValueError("token override must be nonnegative")
        return value
    value = str(text)
    if not value:
        return 0
    lexical = len(re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE))
    byte_estimate = math.ceil(len(value.encode("utf-8")) / 4)
    return max(1, lexical, byte_estimate)

class TokenBudget:
    def __init__(self, total: int):
        self.total = int(total)
        self.remaining = int(total)
        self.used = 0

    def consume(self, amount: int) -> None:
        amount = max(0, int(amount))
        if amount > self.remaining:
            raise ContextWindowExceededError(f"requested {amount}, remaining {self.remaining}")
        self.remaining -= amount
        self.used += amount

    def fit(self, text: str) -> str:
        text = str(text)
        tokens = estimate_tokens(text)
        if tokens <= self.remaining:
            self.consume(tokens)
            return text
        if self.remaining <= 0:
            return ""
        low = 0
        high = len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if estimate_tokens(text[:middle]) <= self.remaining:
                low = middle
            else:
                high = middle - 1
        truncated = text[:low]
        self.consume(estimate_tokens(truncated))
        return truncated

