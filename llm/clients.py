from core.common import *
from core.models import *
from llm.parsers import estimate_tokens
from config.settings import MODEL_ROUTER_URL, SYSTEM_PROMPT

import asyncio
import json
import re
import typing
from abc import ABC, abstractmethod
from collections import Counter

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class ModelClient(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        tools_schema: typing.List[dict],
        max_tokens: int,
    ) -> typing.Tuple[typing.Any, typing.Any]:
        raise TypeError("abstract method must be implemented by a concrete client")

    def close(self) -> None:
        return None


class StructuredRuleModel(ModelClient):
    async def generate(
        self,
        prompt: str,
        tools_schema: typing.List[dict],
        max_tokens: int,
    ) -> typing.Tuple[typing.Any, typing.Any]:
        text = str(prompt)
        tool_names = {
            str(schema.get("function", {}).get("name", "")): schema
            for schema in tools_schema
            if isinstance(schema, dict)
            and isinstance(schema.get("function"), dict)
            and schema.get("function", {}).get("name")
        }
        directive = re.search(
            r"(?:^|\n)tool:([A-Za-z0-9_.-]+)(?:[ \t]+(\{[^\n]*\}))?[ \t]*$",
            text,
            flags=re.MULTILINE,
        )
        calls: typing.List[dict] = []
        final_answer: typing.Optional[str] = None
        declared_intent = "apply explicit structured rules"

        if directive:
            name = directive.group(1)
            if name not in tool_names:
                raise PermanentError(f"unknown explicit local tool directive: {name}")
            raw_arguments = directive.group(2) or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PermanentError(
                    f"invalid JSON arguments for local tool {name}: {exc}"
                ) from exc
            if not isinstance(arguments, dict):
                raise PermanentError("local tool arguments must be an object")
            calls = [{"name": name, "arguments": arguments}]
            declared_intent = f"execute {name}"
        else:
            task_match = re.search(r"^Task:[ \t]*(.+?)[ \t]*$", text, flags=re.MULTILINE)
            if not task_match:
                raise PermanentError("structured rule model requires a Task field")
            try:
                task_data = json.loads(task_match.group(1))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PermanentError(f"invalid JSON in Task field: {exc}") from exc
            final_answer = stable_json_dumps(
                {
                    "completed": True,
                    "task": task_data,
                }
            )

        action = model_validate(
            ModelAction,
            {
                "tool_calls": calls,
                "final_answer": final_answer,
                "declared_intent": declared_intent,
                "confidence": 1.0,
                "rationale": "Structured execution of explicit input.",
            },
        )
        prompt_tokens = estimate_tokens(text)
        completion_text = (
            final_answer if final_answer is not None else stable_json_dumps(calls)
        )
        completion_tokens = estimate_tokens(completion_text)
        usage = model_validate(
            UsageStats,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )
        return action, usage


def _classify_model_exception(exc: Exception, model_name: str) -> AgentError:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    message = str(exc)
    normalized_message = message.casefold()
    permanent_terms = (
        "authentication",
        "unauthorized",
        "invalid api key",
        "incorrect api key",
        "model not found",
        "bad request",
        "permission denied",
        "forbidden",
    )

    if status_code in {400, 401, 403, 404, 409, 422} or any(
        term in normalized_message for term in permanent_terms
    ):
        return PermanentError(
            f"Requesty request failed for {model_name}: {message}"
        )

    return TransientError(f"Requesty request failed for {model_name}: {message}")


class RequestyModel(ModelClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        config: typing.Any = None,
    ):
        self.api_key = str(api_key or "")
        self.model_name = str(model_name or "")
        self.client: typing.Any = None
        self.config = config
        self._model_clients: typing.Dict[str, "RequestyModel"] = {}

        if not self.model_name:
            raise ValueError("model_name must not be empty")

        if OpenAI is not None and self.api_key:
            request_timeout = getattr(self.config, "request_timeout_s", 180.0)
            if request_timeout is None:
                request_timeout = 180.0
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=MODEL_ROUTER_URL,
                timeout=float(request_timeout),
                max_retries=0,
            )

    def for_model(self, model_name: str) -> "RequestyModel":
        normalized_model_name = str(model_name or "")
        if not normalized_model_name:
            raise ValueError("model_name must not be empty")
        if normalized_model_name == self.model_name:
            return self

        client = self._model_clients.get(normalized_model_name)
        if client is None:
            client = RequestyModel(
                self.api_key,
                normalized_model_name,
                self.config,
            )
            self._model_clients[normalized_model_name] = client
        return client

    def _completion_limit(self, max_tokens: int) -> int:
        requested_limit = int(max_tokens)
        if requested_limit <= 0:
            raise PermanentError("max_tokens must be positive")

        configured_limit = getattr(
            self.config,
            "max_completion_tokens",
            8192,
        )
        if configured_limit is None:
            configured_limit = 8192

        configured_limit = int(configured_limit)
        if configured_limit <= 0:
            raise PermanentError("configured max_completion_tokens must be positive")

        return min(requested_limit, configured_limit)

    @staticmethod
    def _extract_message_content(message: typing.Any) -> str:
        content = getattr(message, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if not isinstance(content, (list, tuple)):
            raise PermanentError("Requesty returned unsupported message content")

        parts: typing.List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                item_text = item.get("text")
                if isinstance(item_text, str):
                    parts.append(item_text)
                    continue
                continue
            item_text = getattr(item, "text", None)
            if isinstance(item_text, str):
                parts.append(item_text)

        return "".join(parts)

    @staticmethod
    def _normalize_content_calls(value: typing.Any) -> typing.List[dict]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise PermanentError("tool_calls in the action must be an array")

        normalized_calls: typing.List[dict] = []
        for index, call in enumerate(value):
            if not isinstance(call, dict):
                raise PermanentError(
                    f"tool call at index {index} must be an object"
                )

            name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(name, str) or not name:
                raise PermanentError(
                    f"tool call at index {index} has an invalid name"
                )

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise PermanentError(
                        f"tool call at index {index} has invalid JSON arguments: {exc}"
                    ) from exc

            if not isinstance(arguments, dict):
                raise PermanentError(
                    f"tool call at index {index} arguments must be an object"
                )

            normalized_calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                }
            )

        return normalized_calls

    async def generate(
        self,
        prompt: str,
        tools_schema: typing.List[dict],
        max_tokens: int,
    ) -> typing.Tuple[typing.Any, typing.Any]:
        if not self.api_key:
            raise PermanentError(
                "REQUESTY_API_KEY is required for autonomous model execution"
            )
        if OpenAI is None:
            raise PermanentError(
                "openai package is required for Requesty model execution"
            )
        if self.client is None:
            raise PermanentError("Requesty client is unavailable")

        if tools_schema is None:
            tools_schema = []
        if not isinstance(tools_schema, list):
            raise PermanentError("tools_schema must be a list")

        completion_limit = self._completion_limit(max_tokens)
        parameters: typing.Dict[str, typing.Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": str(prompt),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_action",
                    "strict": False,
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
                                        "name": {
                                            "type": "string",
                                        },
                                        "arguments": {
                                            "type": "object",
                                            "additionalProperties": True,
                                        },
                                    },
                                    "required": [
                                        "name",
                                        "arguments",
                                    ],
                                },
                            },
                            "final_answer": {
                                "type": [
                                    "string",
                                    "null",
                                ],
                            },
                            "declared_intent": {
                                "type": "string",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "rationale": {
                                "type": "string",
                            },
                        },
                        "required": [
                            "tool_calls",
                            "final_answer",
                            "declared_intent",
                            "confidence",
                            "rationale",
                        ],
                    },
                },
            },
            "max_completion_tokens": completion_limit,
        }

        if tools_schema:
            parameters["tools"] = tools_schema
            parameters["tool_choice"] = "auto"

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **parameters,
            )
        except Exception as exc:
            normalized_error = str(exc).casefold()
            if "max_completion_tokens" in normalized_error:
                parameters.pop("max_completion_tokens", None)
                parameters["max_tokens"] = completion_limit
                try:
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        **parameters,
                    )
                except Exception as fallback_exc:
                    raise _classify_model_exception(
                        fallback_exc,
                        self.model_name,
                    ) from fallback_exc
            else:
                raise _classify_model_exception(
                    exc,
                    self.model_name,
                ) from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise TransientError(
                f"Requesty returned no choices for {self.model_name}"
            )

        message = getattr(choices[0], "message", None)
        if message is None:
            raise TransientError(
                f"Requesty returned a choice without a message for {self.model_name}"
            )

        refusal = getattr(message, "refusal", None)
        if refusal:
            raise PermanentError(
                f"{self.model_name} refused the request: {refusal}"
            )

        provider_calls: typing.List[dict] = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            try:
                function = getattr(call, "function", None)
                if function is None:
                    raise TypeError("tool call does not contain a function")

                name = getattr(function, "name", None)
                if not isinstance(name, str) or not name:
                    raise TypeError("tool name is missing")

                raw_arguments = getattr(function, "arguments", None) or "{}"
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments are not an object")

                provider_calls.append(
                    {
                        "name": name,
                        "arguments": arguments,
                    }
                )
            except Exception as exc:
                raise PermanentError(
                    f"invalid tool call at index {index} returned by "
                    f"{self.model_name}: {exc}"
                ) from exc

        content = self._extract_message_content(message)
        data: typing.Dict[str, typing.Any] = {}

        if content.strip():
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError as exc:
                if not provider_calls:
                    raise PermanentError(
                        f"invalid JSON action returned by {self.model_name}: {exc}"
                    ) from exc
            else:
                if not isinstance(parsed_content, dict):
                    raise PermanentError(
                        f"invalid action object returned by {self.model_name}"
                    )
                data = parsed_content

        content_calls = self._normalize_content_calls(data.get("tool_calls", []))
        calls = provider_calls if provider_calls else content_calls
        final_answer = data.get("final_answer")

        if final_answer is not None and not isinstance(final_answer, str):
            raise PermanentError(
                f"invalid final answer returned by {self.model_name}"
            )

        if not calls and final_answer is None:
            raise PermanentError(
                f"{self.model_name} returned neither tool calls nor a final answer"
            )

        declared_intent = data.get("declared_intent", "")
        rationale = data.get("rationale", "")
        confidence = data.get("confidence", 0.0)

        if not isinstance(declared_intent, str):
            raise PermanentError(
                f"invalid declared intent returned by {self.model_name}"
            )
        if not isinstance(rationale, str):
            raise PermanentError(
                f"invalid rationale returned by {self.model_name}"
            )

        try:
            normalized_confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise PermanentError(
                f"invalid confidence returned by {self.model_name}"
            ) from exc

        if not 0.0 <= normalized_confidence <= 1.0:
            raise PermanentError(
                f"confidence returned by {self.model_name} is outside the range 0 to 1"
            )

        action = model_validate(
            ModelAction,
            {
                "tool_calls": calls,
                "final_answer": final_answer,
                "declared_intent": declared_intent,
                "confidence": normalized_confidence,
                "rationale": rationale,
            },
        )

        usage_obj = getattr(response, "usage", None)
        prompt_tokens = int(
            getattr(usage_obj, "prompt_tokens", 0) or 0
        )
        completion_tokens = int(
            getattr(usage_obj, "completion_tokens", 0) or 0
        )
        total_tokens = int(
            getattr(usage_obj, "total_tokens", 0)
            or prompt_tokens + completion_tokens
        )

        usage = model_validate(
            UsageStats,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )
        return action, usage

    def close(self) -> None:
        clients = list(self._model_clients.values())
        self._model_clients.clear()

        first_error: typing.Optional[BaseException] = None
        for client in clients:
            try:
                client.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        close_method = getattr(self.client, "close", None)
        if callable(close_method):
            try:
                close_method()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        self.client = None

        if first_error is not None:
            raise first_error


class SummarizerClient(ABC):
    @abstractmethod
    async def summarize(
        self,
        texts: typing.List[str],
        max_len: int,
    ) -> typing.List[str]:
        raise TypeError("abstract method must be implemented by a concrete client")


class ExtractiveFrequencySummarizer(SummarizerClient):
    async def summarize(
        self,
        texts: typing.List[str],
        max_len: int,
    ) -> typing.List[str]:
        max_len = int(max_len)
        if max_len <= 0:
            raise ValueError("summary length must be positive")
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")

        output: typing.List[str] = []

        for text in texts:
            source = " ".join(str(text).split())
            if not source:
                output.append("")
                continue

            sentences = [
                item.strip()
                for item in re.split(r"(?<=[.!?])\s+", source)
                if item.strip()
            ]
            if not sentences:
                sentences = [source]

            words = re.findall(
                r"[\w'-]+",
                source.casefold(),
                flags=re.UNICODE,
            )
            frequencies = Counter(words)
            ranked: typing.List[typing.Tuple[float, int, str]] = []

            for index, sentence in enumerate(sentences):
                sentence_words = re.findall(
                    r"[\w'-]+",
                    sentence.casefold(),
                    flags=re.UNICODE,
                )
                score = sum(
                    frequencies[word] for word in sentence_words
                ) / max(1, len(sentence_words))
                ranked.append((score, -index, sentence))

            ranked.sort(reverse=True)
            selected: typing.List[str] = []
            used = 0

            for _, _, sentence in ranked:
                if not sentence:
                    continue

                separator_length = 1 if selected else 0
                remaining = max_len - used - separator_length
                if remaining <= 0:
                    break

                part = sentence[:remaining]
                if not part:
                    break

                selected.append(part)
                used += separator_length + len(part)

                if used >= max_len:
                    break

            output.append(" ".join(selected)[:max_len])

        return output
