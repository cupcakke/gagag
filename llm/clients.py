from core.common import *
from core.models import *
from llm.parsers import estimate_tokens
from config.settings import MODEL_ROUTER_URL, SYSTEM_PROMPT

class ModelClient(ABC):
    @abstractmethod
    async def generate(
        self, prompt: str, tools_schema: typing.List[dict], max_tokens: int
    ) -> typing.Tuple[typing.Any, typing.Any]:
        raise TypeError("abstract method must be implemented by a concrete client")

    def close(self) -> None:
        return None

class StructuredRuleModel(ModelClient):
    async def generate(
        self, prompt: str, tools_schema: typing.List[dict], max_tokens: int
    ) -> typing.Tuple[typing.Any, typing.Any]:
        text = str(prompt)
        tool_names = {
            str(schema.get("function", {}).get("name", "")): schema
            for schema in tools_schema
            if isinstance(schema, dict)
        }
        directive = re.search(r"(?:^|\n)tool:([A-Za-z0-9_.-]+)(?:\s+({.*}))?\s*$", text, flags=re.DOTALL)
        calls = []
        final_answer = None
        declared_intent = "apply explicit structured rules"
        if directive:
            name = directive.group(1)
            if name not in tool_names:
                raise PermanentError(f"unknown explicit local tool directive: {name}")
            raw_arguments = directive.group(2) or "{}"
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise PermanentError("local tool arguments must be an object")
            calls = [{"name": name, "arguments": arguments}]
            declared_intent = f"execute {name}"
        else:
            task_match = re.search(r"^Task:\s*(.+)$", text, flags=re.MULTILINE)
            if not task_match:
                raise PermanentError("structured rule model requires a Task field")
            task_data = json.loads(task_match.group(1))
            final_answer = stable_json_dumps({"completed": True, "task": task_data})
        action = model_validate(ModelAction, {
            "tool_calls": calls,
            "final_answer": final_answer,
            "declared_intent": declared_intent,
            "confidence": 1.0,
            "rationale": "Structured execution of explicit input.",
        })
        prompt_tokens = estimate_tokens(text)
        completion_tokens = estimate_tokens(final_answer or stable_json_dumps(calls))
        usage = model_validate(UsageStats, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
        return action, usage

def _classify_model_exception(exc: Exception, model_name: str) -> AgentError:
    status_code = getattr(exc, "status_code", None)
    message = str(exc)
    if status_code in {400, 401, 403, 404, 409, 422} or any(term in message.lower() for term in ("authentication", "unauthorized", "invalid api key", "model not found", "bad request")):
        return PermanentError(f"Requesty request failed for {model_name}: {message}")
    return TransientError(f"Requesty request failed for {model_name}: {message}")

class RequestyModel(ModelClient):
    def __init__(self, api_key: str, model_name: str, config: typing.Any = None):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self.config = config
        self._model_clients: typing.Dict[str, "RequestyModel"] = {}
        if _HAS_OPENAI and api_key:
            self.client = OpenAI(
                api_key=api_key,
                base_url=MODEL_ROUTER_URL,
                timeout=getattr(self.config, "request_timeout_s", 180.0),
                max_retries=0,
            )

    def for_model(self, model_name: str) -> "RequestyModel":
        if model_name == self.model_name:
            return self
        client = self._model_clients.get(model_name)
        if client is None:
            client = RequestyModel(self.api_key, model_name, self.config)
            self._model_clients[model_name] = client
        return client

    async def generate(
        self, prompt: str, tools_schema: typing.List[dict], max_tokens: int
    ) -> typing.Tuple[typing.Any, typing.Any]:
        if not self.api_key:
            raise PermanentError("REQUESTY_API_KEY is required for autonomous model execution")
        if not _HAS_OPENAI:
            raise PermanentError("openai package is required for Requesty model execution")
        if not self.client:
            raise PermanentError("Requesty client is unavailable")
        parameters = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "tools": tools_schema or None,
            "tool_choice": "auto" if tools_schema else None,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_action",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "tool_calls": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"type": "string"},
                                        "arguments": {"type": "object", "additionalProperties": True},
                                    },
                                    "required": ["name", "arguments"],
                                },
                            },
                            "final_answer": {"type": ["string", "null"]},
                            "declared_intent": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "rationale": {"type": "string"},
                        },
                        "required": ["tool_calls", "final_answer", "declared_intent", "confidence", "rationale"],
                    },
                },
            },
            "max_completion_tokens": min(int(max_tokens), getattr(self.config, "max_completion_tokens", 8192)),
        }
        try:
            response = await asyncio.to_thread(self.client.chat.completions.create, **parameters)
        except Exception as exc:
            if "max_completion_tokens" in str(exc):
                parameters.pop("max_completion_tokens", None)
                parameters["max_tokens"] = min(int(max_tokens), getattr(self.config, "max_completion_tokens", 8192))
                try:
                    response = await asyncio.to_thread(self.client.chat.completions.create, **parameters)
                except Exception as fallback_exc:
                    raise _classify_model_exception(fallback_exc, self.model_name) from fallback_exc
            else:
                raise _classify_model_exception(exc, self.model_name) from exc
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise TransientError(f"Requesty returned no choices for {self.model_name}")
        message = choices[0].message
        provider_calls = []
        for call in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments are not an object")
                provider_calls.append({"name": call.function.name, "arguments": arguments})
            except Exception as exc:
                raise PermanentError(f"invalid tool arguments returned by {self.model_name}: {exc}") from exc
        content = message.content or ""
        data = {}
        if content:
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                if not provider_calls:
                    raise PermanentError(f"invalid JSON action returned by {self.model_name}: {exc}") from exc
        if data and not isinstance(data, dict):
            raise PermanentError(f"invalid action object returned by {self.model_name}")
        calls = provider_calls or data.get("tool_calls", [])
        final_answer = data.get("final_answer")
        if not calls and not final_answer:
            raise PermanentError(f"{self.model_name} returned neither tool calls nor a final answer")
        action = model_validate(ModelAction, {
            "tool_calls": calls,
            "final_answer": final_answer,
            "declared_intent": data.get("declared_intent", ""),
            "confidence": float(data.get("confidence", 0.0)),
            "rationale": data.get("rationale", ""),
        })
        usage_obj = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage_obj, "total_tokens", 0) or (prompt_tokens + completion_tokens))
        usage = model_validate(UsageStats, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })
        return action, usage

    def close(self) -> None:
        for client in list(self._model_clients.values()):
            client.close()
        self._model_clients.clear()
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

class SummarizerClient(ABC):
    @abstractmethod
    async def summarize(self, texts: typing.List[str], max_len: int) -> typing.List[str]:
        raise TypeError("abstract method must be implemented by a concrete client")

class ExtractiveFrequencySummarizer(SummarizerClient):
    async def summarize(self, texts: typing.List[str], max_len: int) -> typing.List[str]:
        max_len = int(max_len)
        if max_len <= 0:
            raise ValueError("summary length must be positive")
        output = []
        for text in texts:
            source = " ".join(str(text).split())
            sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", source) if item.strip()]
            words = re.findall(r"[\w'-]+", source.casefold(), flags=re.UNICODE)
            frequencies = Counter(words)
            ranked = []
            for index, sentence in enumerate(sentences or [source]):
                sentence_words = re.findall(r"[\w'-]+", sentence.casefold(), flags=re.UNICODE)
                score = sum(frequencies[word] for word in sentence_words) / max(1, len(sentence_words))
                ranked.append((score, -index, sentence))
            ranked.sort(reverse=True)
            selected = []
            used = 0
            for _, _, sentence in ranked:
                if not sentence:
                    continue
                remaining = max_len - used
                if remaining <= 0:
                    break
                part = sentence[:remaining]
                selected.append(part)
                used += len(part) + (1 if selected else 0)
            output.append(" ".join(selected)[:max_len])
        return output

