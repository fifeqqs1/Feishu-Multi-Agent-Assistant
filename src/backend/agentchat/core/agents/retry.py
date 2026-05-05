import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    TIMEOUT = "timeout"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    PARAMETER = "parameter"
    EMPTY_RESULT = "empty_result"
    UNKNOWN = "unknown"


class SubAgentStatus(str, Enum):
    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    NON_RETRYABLE_FAILURE = "non_retryable_failure"


class AgentExecutionError(Exception):
    def __init__(self, message: str, *, error_type: str):
        super().__init__(message)
        self.error_type = error_type


class RetryableError(AgentExecutionError):
    pass


class NonRetryableError(AgentExecutionError):
    pass


class CircuitBreakerError(AgentExecutionError):
    pass


class AgentRetryConfig(BaseModel):
    enable_multi_agent_orchestration: bool = True
    max_mcp_tool_attempts: int = Field(default=3, ge=1, le=5)
    mcp_tool_backoff_base_seconds: float = Field(default=0.75, ge=0.0, le=10.0)
    max_sub_agent_same_query_retries: int = Field(default=1, ge=0, le=3)
    max_query_rewrite_retries: int = Field(default=1, ge=0, le=3)
    max_parallel_sub_agents: int = Field(default=3, ge=1, le=8)
    max_multi_agent_tasks: int = Field(default=4, ge=1, le=12)
    max_tool_call_count: int = Field(default=12, ge=1)
    max_consecutive_tool_failures: int = Field(default=2, ge=1)
    max_same_mcp_agent_calls: int = Field(default=2, ge=1)
    max_total_tokens: int = Field(default=12000, ge=1)
    max_round_timeout_seconds: float = Field(default=45.0, ge=1.0)


class SubAgentExecutionResult(BaseModel):
    status: SubAgentStatus
    query: str
    final_query: str
    attempts: int = 1
    answer: str = ""
    summary: str = ""
    raw_result: str = ""
    source_agent: Optional[str] = None
    error: Optional[str] = None
    confidence: Optional[float] = None
    evidence: list[str] = Field(default_factory=list)
    tool_name: Optional[str] = None
    error_type: Optional[str] = None
    raw_error: Optional[str] = None
    query_changed: bool = False


@dataclass
class MainAgentRuntimeState:
    started_at: float = field(default_factory=time.monotonic)
    total_tokens: int = 0
    total_tool_calls: int = 0
    tool_failure_streak: dict[str, int] = field(default_factory=dict)
    mcp_agent_call_counts: dict[str, int] = field(default_factory=dict)
    last_failed_tool_name: str | None = None

    def reset(self) -> None:
        self.started_at = time.monotonic()
        self.total_tokens = 0
        self.total_tool_calls = 0
        self.tool_failure_streak.clear()
        self.mcp_agent_call_counts.clear()
        self.last_failed_tool_name = None

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def record_tool_call(self, tool_name: str, *, is_mcp_tool: bool) -> None:
        self.total_tool_calls += 1
        if is_mcp_tool:
            self.mcp_agent_call_counts[tool_name] = (
                self.mcp_agent_call_counts.get(tool_name, 0) + 1
            )

    def record_tool_success(self, tool_name: str) -> None:
        self.tool_failure_streak[tool_name] = 0
        self.last_failed_tool_name = None

    def record_tool_failure(self, tool_name: str) -> int:
        if self.last_failed_tool_name != tool_name:
            failure_count = 1
        else:
            failure_count = self.tool_failure_streak.get(tool_name, 0) + 1
        self.tool_failure_streak[tool_name] = failure_count
        self.last_failed_tool_name = tool_name
        return failure_count


RETRYABLE_ERROR_TYPES = {
    ErrorType.TIMEOUT.value,
    ErrorType.NETWORK.value,
    ErrorType.RATE_LIMIT.value,
    ErrorType.SERVER_ERROR.value,
}

RETRYABLE_MESSAGE_HINTS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "rate limit",
    "too many requests",
    "connection reset",
    "connection aborted",
    "connection refused",
    "remoteprotocolerror",
    "transporterror",
)

PARAMETER_MESSAGE_HINTS = (
    "400",
    "401",
    "403",
    "404",
    "parameter",
    "argument",
    "schema",
    "validation",
    "invalid",
    "missing",
    "required",
    "not found",
    "permission",
    "unauthorized",
    "forbidden",
    "bad request",
)


def detect_error_type(error: Exception | str) -> str:
    if isinstance(error, AgentExecutionError):
        return error.error_type

    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return ErrorType.TIMEOUT.value

    error_type_name = getattr(type(error), "__name__", "").lower()
    error_message = str(error).lower()
    combined = f"{error_type_name} {error_message}"

    if "timeout" in combined or "timed out" in combined:
        return ErrorType.TIMEOUT.value

    if any(hint in combined for hint in ("rate limit", "too many requests", "429")):
        return ErrorType.RATE_LIMIT.value

    if any(hint in combined for hint in ("500", "502", "503", "504", "service unavailable", "bad gateway", "gateway timeout")):
        return ErrorType.SERVER_ERROR.value

    if any(hint in combined for hint in PARAMETER_MESSAGE_HINTS):
        return ErrorType.PARAMETER.value

    if any(hint in combined for hint in ("connect", "connection", "network", "dns", "socket", "transport")):
        return ErrorType.NETWORK.value

    return ErrorType.UNKNOWN.value


def is_retryable_error_type(error_type: Optional[str]) -> bool:
    return error_type in RETRYABLE_ERROR_TYPES


def classify_exception(error: Exception) -> AgentExecutionError:
    error_type = detect_error_type(error)
    error_message = str(error)
    if is_retryable_error_type(error_type):
        return RetryableError(error_message, error_type=error_type)
    return NonRetryableError(error_message, error_type=error_type)


def get_backoff_delay(attempt: int, base_delay: float) -> float:
    return min(base_delay * (2 ** (attempt - 1)), 8.0)


async def execute_with_retry(
    operation: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int,
    base_delay: float,
    operation_name: str,
) -> Any:
    last_error: Optional[AgentExecutionError] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except CircuitBreakerError:
            raise
        except Exception as error:
            classified_error = (
                error
                if isinstance(error, AgentExecutionError)
                else classify_exception(error)
            )
            last_error = classified_error

            if not isinstance(classified_error, RetryableError) or attempt >= max_attempts:
                raise classified_error from error

            delay = get_backoff_delay(attempt, base_delay)
            logger.warning(
                f"{operation_name} failed on attempt {attempt}/{max_attempts}, "
                f"retry after {delay:.2f}s: {classified_error}"
            )
            await asyncio.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RetryableError(
        f"{operation_name} failed without a captured exception",
        error_type=ErrorType.UNKNOWN.value,
    )


def extract_total_tokens_from_model_response(response: Any) -> int:
    visited: set[int] = set()
    queue = [response]

    while queue:
        current = queue.pop(0)
        if current is None:
            continue

        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        if isinstance(current, dict):
            usage = current.get("usage_metadata") or current.get("token_usage") or current.get("usage")
            if isinstance(usage, dict):
                total_tokens = usage.get("total_tokens")
                if total_tokens is not None:
                    return int(total_tokens)

                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                return int(input_tokens) + int(output_tokens)

            for key in ("result", "response", "message", "messages", "llm_output"):
                value = current.get(key)
                if isinstance(value, list):
                    queue.extend(value)
                elif value is not None:
                    queue.append(value)
            continue

        usage = getattr(current, "usage_metadata", None) or getattr(current, "usage", None)
        if usage:
            total_tokens = getattr(usage, "total_tokens", None)
            if total_tokens is not None:
                return int(total_tokens)

            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                return int(input_tokens) + int(output_tokens)

        for attr_name in ("result", "response", "message", "llm_output"):
            value = getattr(current, attr_name, None)
            if value is not None:
                queue.append(value)

        messages = getattr(current, "messages", None)
        if isinstance(messages, list):
            queue.extend(messages)

    return 0


def should_retry_with_rewrite(result: SubAgentExecutionResult) -> bool:
    return result.status == SubAgentStatus.RETRYABLE_FAILURE and result.error_type == ErrorType.EMPTY_RESULT.value
