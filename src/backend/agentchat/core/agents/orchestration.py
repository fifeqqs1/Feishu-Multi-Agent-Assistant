import asyncio
import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from agentchat.core.agents.retry import (
    AgentRetryConfig,
    CircuitBreakerError,
    MainAgentRuntimeState,
    SubAgentExecutionResult,
    SubAgentStatus,
    extract_total_tokens_from_model_response,
)
from agentchat.core.callbacks import usage_metadata_callback


class OrchestrationTask(BaseModel):
    worker_id: str
    query: str
    reason: str


class MultiAgentPlan(BaseModel):
    should_orchestrate: bool = False
    reasoning: str = ""
    tasks: List[OrchestrationTask] = Field(default_factory=list)
    aggregation_goal: str = ""


class MultiAgentAggregateResult(BaseModel):
    status: Literal["success", "partial", "failure"] = "failure"
    summary: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)
    successful_agents: List[str] = Field(default_factory=list)
    failed_agents: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    degraded: bool = False


class WorkerDescriptor(BaseModel):
    worker_id: str
    source_agent: str
    worker_type: Literal["knowledge", "skill", "mcp"]
    description: str


@dataclass
class RegisteredWorker:
    descriptor: WorkerDescriptor
    runner: Callable[[str], Awaitable[SubAgentExecutionResult]]


def summarize_text(text: str, max_length: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 3].rstrip() + "..."


def extract_evidence(text: str, max_items: int = 3, max_length: int = 180) -> list[str]:
    lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    evidence = []
    for line in lines:
        clipped = summarize_text(line, max_length=max_length)
        if clipped and clipped not in evidence:
            evidence.append(clipped)
        if len(evidence) >= max_items:
            break

    if evidence:
        return evidence

    summarized = summarize_text(text, max_length=max_length)
    return [summarized] if summarized else []


class MultiAgentOrchestrator:
    def __init__(self, model, retry_config: AgentRetryConfig, runtime_state: MainAgentRuntimeState):
        self.model = model
        self.retry_config = retry_config
        self.runtime_state = runtime_state

    def remaining_timeout(self) -> float:
        return max(
            self.retry_config.max_round_timeout_seconds - self.runtime_state.elapsed_seconds(),
            0.0,
        )

    def ensure_runtime_budget(self) -> None:
        if self.remaining_timeout() <= 0:
            raise CircuitBreakerError(
                "The current round exceeded the total timeout budget.",
                error_type="timeout",
            )
        if self.runtime_state.total_tokens >= self.retry_config.max_total_tokens:
            raise CircuitBreakerError(
                "The current round exceeded the token budget.",
                error_type="token_budget",
            )

    async def _structured_invoke(self, response_model: type[BaseModel], prompt: str) -> BaseModel:
        self.ensure_runtime_budget()
        response = await asyncio.wait_for(
            self.model.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "You are a coordinator that must return only one valid JSON object. "
                            "Do not output markdown. Do not add any extra text."
                        )
                    ),
                    HumanMessage(content=prompt),
                ],
                config={"callbacks": [usage_metadata_callback]},
            ),
            timeout=self.remaining_timeout(),
        )

        consumed_tokens = extract_total_tokens_from_model_response(response)
        if consumed_tokens:
            self.runtime_state.total_tokens += consumed_tokens
            if self.runtime_state.total_tokens > self.retry_config.max_total_tokens:
                raise CircuitBreakerError(
                    "The current round exceeded the token budget.",
                    error_type="token_budget",
                )

        content = str(getattr(response, "content", "")).strip()
        try:
            return response_model.model_validate_json(content)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                return response_model.model_validate_json(match.group(0))
            raise ValueError(f"Invalid structured JSON output: {content}")

    async def plan(self, query: str, workers: list[WorkerDescriptor]) -> MultiAgentPlan:
        worker_payload = json.dumps(
            [worker.model_dump() for worker in workers],
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""
User query:
{query}

Available workers:
{worker_payload}

Task:
Decide whether explicit multi-agent orchestration is useful for this query.
Only choose workers when their work is meaningfully complementary and can be executed independently.

Rules:
1. Use at most {self.retry_config.max_multi_agent_tasks} tasks.
2. Only set should_orchestrate=true when at least 2 workers add value together.
3. Prefer knowledge + skill + mcp combinations when they are complementary.
4. Each task query should be optimized for that worker.
5. Do not repeat the same worker_id more than once.
6. If orchestration is not needed, return should_orchestrate=false and an empty task list.

Output JSON fields:
- should_orchestrate: boolean
- reasoning: string
- tasks: array of {{worker_id, query, reason}}
- aggregation_goal: string
"""
        result = await self._structured_invoke(MultiAgentPlan, prompt)
        if len(result.tasks) > self.retry_config.max_multi_agent_tasks:
            result.tasks = result.tasks[: self.retry_config.max_multi_agent_tasks]
        return result

    async def aggregate(
        self,
        query: str,
        plan: MultiAgentPlan,
        results: list[SubAgentExecutionResult],
    ) -> MultiAgentAggregateResult:
        success_results = [
            {
                "source_agent": result.source_agent or result.tool_name,
                "summary": result.summary or result.answer,
                "raw_result": result.raw_result or result.answer,
                "confidence": result.confidence,
                "evidence": result.evidence,
            }
            for result in results
            if result.status == SubAgentStatus.SUCCESS
        ]
        failed_results = [
            {
                "source_agent": result.source_agent or result.tool_name,
                "error_type": result.error_type,
                "error": result.error or result.raw_error,
            }
            for result in results
            if result.status != SubAgentStatus.SUCCESS
        ]

        if not success_results:
            return MultiAgentAggregateResult(
                status="failure",
                summary="All sub-agents failed.",
                final_answer="I could not complete the multi-agent workflow because all selected sub-agents failed.",
                confidence=0.0,
                evidence=[],
                successful_agents=[],
                failed_agents=[item["source_agent"] or "unknown" for item in failed_results],
                degraded=True,
            )

        prompt = f"""
User query:
{query}

Aggregation goal:
{plan.aggregation_goal or "Merge the worker outputs into one stable final answer."}

Successful worker results:
{json.dumps(success_results, ensure_ascii=False, indent=2)}

Failed worker results:
{json.dumps(failed_results, ensure_ascii=False, indent=2)}

Task:
Aggregate the worker outputs.

Rules:
1. Judge which results succeeded and which failed.
2. Identify conflicts if any results disagree.
3. Prefer stronger evidence and more direct answers.
4. If some workers failed but the successful ones are enough, still produce a usable final answer and mark degraded=true.
5. The final answer must be directly user-facing and concise.

Output JSON fields:
- status: "success" | "partial" | "failure"
- summary: string
- final_answer: string
- confidence: number between 0 and 1
- evidence: string[]
- successful_agents: string[]
- failed_agents: string[]
- conflicts: string[]
- degraded: boolean
"""

        return await self._structured_invoke(MultiAgentAggregateResult, prompt)
