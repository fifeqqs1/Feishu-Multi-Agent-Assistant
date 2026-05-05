import asyncio
import copy
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, NotRequired

from loguru import logger
from pydantic import BaseModel, Field

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Command

from agentchat.api.services.agent_skill import AgentSkillService
from agentchat.api.services.llm import LLMService
from agentchat.api.services.mcp_server import MCPService
from agentchat.api.services.tool import ToolService
from agentchat.core.agents.mcp_agent import MCPAgent, MCPConfig
from agentchat.core.agents.orchestration import (
    MultiAgentOrchestrator,
    RegisteredWorker,
    WorkerDescriptor,
    extract_evidence,
    summarize_text,
)
from agentchat.core.agents.retry import (
    AgentRetryConfig,
    CircuitBreakerError,
    ErrorType,
    MainAgentRuntimeState,
    SubAgentExecutionResult,
    SubAgentStatus,
    extract_total_tokens_from_model_response,
    should_retry_with_rewrite,
)
from agentchat.core.agents.skill_agent import SkillAgent
from agentchat.core.callbacks import usage_metadata_callback
from agentchat.core.models.manager import ModelManager
from agentchat.database import AgentSkill
from agentchat.services.rag.handler import RagHandler
from agentchat.settings import app_settings
from agentchat.tools import AgentToolsWithName
from agentchat.tools.openapi_tool.adapter import OpenAPIToolAdapter


class StreamAgentState(AgentState):
    tool_call_count: NotRequired[int]
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]
    available_tools: NotRequired[List[BaseTool]]


class AgentConfig(BaseModel):
    user_id: str
    llm_id: str
    mcp_ids: List[str]
    knowledge_ids: List[str]
    tool_ids: List[str]
    agent_skill_ids: List[str]
    system_prompt: str
    enable_memory: bool = False
    name: str | None = None
    retry_config: AgentRetryConfig = Field(default_factory=AgentRetryConfig)


class EmitEventAgentMiddleware(AgentMiddleware):
    def __init__(
        self,
        name_resolver_func: Callable[[str], tuple[str, str]],
        tool_metadata_map: Dict[str, Dict[str, str]],
        retry_config: AgentRetryConfig,
        runtime_state: MainAgentRuntimeState,
    ):
        super().__init__()
        self.name_resolver_func = name_resolver_func
        self.tool_metadata_map = tool_metadata_map
        self.retry_config = retry_config
        self.runtime_state = runtime_state

    def _is_mcp_tool(self, tool_name: str) -> bool:
        return self.tool_metadata_map.get(tool_name, {}).get("type") == "MCP"

    def _enforce_pre_tool_limits(self, tool_name: str) -> None:
        if self.runtime_state.elapsed_seconds() > self.retry_config.max_round_timeout_seconds:
            raise CircuitBreakerError(
                "The current round exceeded the total timeout budget.",
                error_type="timeout",
            )

        if self.runtime_state.total_tokens >= self.retry_config.max_total_tokens:
            raise CircuitBreakerError(
                "The current round exceeded the token budget.",
                error_type="token_budget",
            )

        if self.runtime_state.total_tool_calls >= self.retry_config.max_tool_call_count:
            raise CircuitBreakerError(
                "The current round exceeded the tool call limit.",
                error_type="tool_call_limit",
            )

        if self._is_mcp_tool(tool_name):
            repeated_calls = self.runtime_state.mcp_agent_call_counts.get(tool_name, 0)
            if repeated_calls >= self.retry_config.max_same_mcp_agent_calls:
                raise CircuitBreakerError(
                    f"MCP sub-agent `{tool_name}` reached the repeat call limit.",
                    error_type="mcp_repeat_limit",
                )

    def _remaining_round_timeout(self) -> float:
        return max(
            self.retry_config.max_round_timeout_seconds - self.runtime_state.elapsed_seconds(),
            0.0,
        )

    async def aafter_model(
        self,
        state: StreamAgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return {"model_call_count": state["model_call_count"] + 1}
        return {"jump_to": "end"}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        try:
            if available_tools := request.state.get("available_tools", []):
                request.tools = available_tools

            remaining_timeout = self._remaining_round_timeout()
            if remaining_timeout <= 0:
                raise CircuitBreakerError(
                    "The current round exceeded the total timeout budget.",
                    error_type="timeout",
                )

            response = await asyncio.wait_for(handler(request), timeout=remaining_timeout)
            consumed_tokens = extract_total_tokens_from_model_response(response)
            if consumed_tokens:
                self.runtime_state.total_tokens += consumed_tokens
                if self.runtime_state.total_tokens > self.retry_config.max_total_tokens:
                    raise CircuitBreakerError(
                        "The current round exceeded the token budget.",
                        error_type="token_budget",
                    )

            return response
        except CircuitBreakerError:
            raise
        except Exception as error:
            logger.error(f"Model call error: {error}")
            raise ValueError(error) from error

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        writer = get_stream_writer()
        tool_name = request.tool_call["name"]
        tool_type, display_tool_name = self.name_resolver_func(tool_name)

        self._enforce_pre_tool_limits(tool_name)
        self.runtime_state.record_tool_call(
            tool_name,
            is_mcp_tool=self._is_mcp_tool(tool_name),
        )
        request.state["tool_call_count"] = request.state.get("tool_call_count", 0) + 1

        writer(
            {
                "status": "START",
                "title": f"Execute {tool_type}: {display_tool_name}",
                "message": f"Calling tool {display_tool_name}...",
            }
        )

        try:
            remaining_timeout = self._remaining_round_timeout()
            if remaining_timeout <= 0:
                raise CircuitBreakerError(
                    "The current round exceeded the total timeout budget.",
                    error_type="timeout",
                )

            tool_result = await asyncio.wait_for(handler(request), timeout=remaining_timeout)
            self.runtime_state.record_tool_success(tool_name)

            writer(
                {
                    "status": "END",
                    "title": f"Execute {tool_type}: {display_tool_name}",
                    "message": getattr(tool_result, "content", str(tool_result)),
                }
            )
            return tool_result
        except CircuitBreakerError as error:
            writer(
                {
                    "status": "ERROR",
                    "title": f"Execute {tool_type}: {display_tool_name}",
                    "message": str(error),
                }
            )
            raise
        except Exception as error:
            failure_count = self.runtime_state.record_tool_failure(tool_name)

            writer(
                {
                    "status": "ERROR",
                    "title": f"Execute {tool_type}: {display_tool_name}",
                    "message": str(error),
                }
            )

            if failure_count >= self.retry_config.max_consecutive_tool_failures:
                raise CircuitBreakerError(
                    f"Tool `{tool_name}` failed {failure_count} times in a row.",
                    error_type="tool_failure_limit",
                ) from error

            return ToolMessage(
                content=str(error),
                name=tool_name,
                tool_call_id=request.tool_call["id"],
            )


class GeneralAgent:
    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.retry_config = agent_config.retry_config
        self.runtime_state = MainAgentRuntimeState()

        self.conversation_model = None
        self.tool_invocation_model = None
        self.react_agent = None
        self.orchestrator: MultiAgentOrchestrator | None = None

        self.tools: List[BaseTool] = []
        self.mcp_agent_as_tools: List[BaseTool] = []
        self.skill_agent_as_tools: List[BaseTool] = []
        self.middlewares = []
        self.tool_metadata_map: Dict[str, Dict[str, str]] = {}
        self.sub_agent_workers: Dict[str, RegisteredWorker] = {}

        self.event_queue = asyncio.Queue()
        self.stop_streaming = False

    def wrap_event(self, data: Dict[Any, Any]):
        return {
            "type": "event",
            "timestamp": time.time(),
            "data": data,
        }

    def get_tool_display_name(self, tool_name: str):
        metadata = self.tool_metadata_map.get(tool_name)
        if not metadata:
            return "Tool", tool_name
        return metadata.get("type", "Tool"), metadata.get("name", tool_name)

    def get_last_user_query(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    def remaining_round_timeout(self) -> float:
        return max(
            self.retry_config.max_round_timeout_seconds - self.runtime_state.elapsed_seconds(),
            0.0,
        )

    def enforce_registered_worker_limits(self, worker: RegisteredWorker):
        if self.runtime_state.elapsed_seconds() > self.retry_config.max_round_timeout_seconds:
            raise CircuitBreakerError(
                "The current round exceeded the total timeout budget.",
                error_type="timeout",
            )

        if self.runtime_state.total_tokens >= self.retry_config.max_total_tokens:
            raise CircuitBreakerError(
                "The current round exceeded the token budget.",
                error_type="token_budget",
            )

        if self.runtime_state.total_tool_calls >= self.retry_config.max_tool_call_count:
            raise CircuitBreakerError(
                "The current round exceeded the tool call limit.",
                error_type="tool_call_limit",
            )

        if worker.descriptor.worker_type == "mcp":
            repeated_calls = self.runtime_state.mcp_agent_call_counts.get(worker.descriptor.worker_id, 0)
            if repeated_calls >= self.retry_config.max_same_mcp_agent_calls:
                raise CircuitBreakerError(
                    f"MCP sub-agent `{worker.descriptor.source_agent}` reached the repeat call limit.",
                    error_type="mcp_repeat_limit",
                )

    def register_worker(self, worker: RegisteredWorker):
        self.sub_agent_workers[worker.descriptor.worker_id] = worker

    async def init_agent(self):
        self.runtime_state.reset()
        self.mcp_agent_as_tools = await self.setup_mcp_agent_as_tools()
        self.tools = await self.setup_tools()
        self.skill_agent_as_tools = await self.setup_agent_skill_as_tools()

        await self.setup_knowledge_tool()
        await self.setup_language_model()

        self.orchestrator = MultiAgentOrchestrator(
            self.conversation_model,
            self.retry_config,
            self.runtime_state,
        )

        self.search_tool = self.setup_search_tool()
        self.middlewares = await self.setup_agent_middleware()
        self.react_agent = self.setup_react_agent()

    async def setup_agent_middleware(self):
        emit_event_middleware = EmitEventAgentMiddleware(
            self.get_tool_display_name,
            self.tool_metadata_map,
            self.retry_config,
            self.runtime_state,
        )
        return [emit_event_middleware]

    async def setup_language_model(self):
        if self.agent_config.llm_id:
            model_config = await LLMService.get_llm_by_id(self.agent_config.llm_id)
            self.conversation_model = ModelManager.get_user_model(**model_config)
        else:
            self.conversation_model = ModelManager.get_conversation_model()
        self.tool_invocation_model = ModelManager.get_tool_invocation_model()

    def setup_react_agent(self):
        return create_agent(
            model=self.conversation_model,
            tools=self.tools + self.mcp_agent_as_tools + self.skill_agent_as_tools,
            middleware=self.middlewares,
            state_schema=StreamAgentState,
        )

    def setup_search_tool(self):
        @tool(parse_docstring=True)
        def search_available_tools(query: str, tool_call_id):
            """
            Search available tools and activate matched tools for the current round.

            Args:
                query (str): Keyword used to search tool names or descriptions.

            Returns:
                str: A list of matched tools.
            """

            found_tools = []
            available_tools = self.tools + self.mcp_agent_as_tools
            for current_tool in available_tools:
                if current_tool.name == "search_available_tools":
                    continue
                tool_description = current_tool.description or ""
                if query.lower() in current_tool.name.lower() or query.lower() in tool_description.lower():
                    found_tools.append(current_tool)

            if not found_tools:
                content = "No related tools were found. Try another keyword."
            else:
                content = "Matched tools:\n" + "\n".join(tool_item.name for tool_item in found_tools)

            tool_message = ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name="search_available_tools",
            )
            return Command(update={"available_tools": found_tools, "messages": [tool_message]})

        return search_available_tools

    async def setup_tools(self) -> List[BaseTool]:
        def create_openapi_tool_executor(tool_adapter, tool_name):
            async def execute_wrapper(**kwargs):
                return await tool_adapter.execute(_tool_name=tool_name, **kwargs)

            return execute_wrapper

        tools: List[BaseTool] = []
        db_tools = await ToolService.get_tools_from_id(self.agent_config.tool_ids)
        for db_tool in db_tools:
            if db_tool.is_user_defined:
                tool_adapter = OpenAPIToolAdapter(
                    auth_config=db_tool.auth_config,
                    openapi_schema=db_tool.openapi_schema,
                )
                for openapi_tool in tool_adapter.tools:
                    tool_name = openapi_tool["function"].get("name", "")
                    tools.append(
                        StructuredTool(
                            name=tool_name,
                            description=openapi_tool["function"].get("description", ""),
                            coroutine=create_openapi_tool_executor(tool_adapter, tool_name),
                            args_schema=openapi_tool,
                        )
                    )
                    self.tool_metadata_map[tool_name] = {
                        "name": db_tool.display_name,
                        "type": "Tool",
                    }
            else:
                agent_tool = AgentToolsWithName.get(db_tool.name)
                if agent_tool:
                    tools.append(agent_tool)
                self.tool_metadata_map[db_tool.name] = {
                    "name": db_tool.display_name,
                    "type": "Tool",
                }
        return tools

    async def execute_registered_worker(self, worker_id: str, query: str) -> SubAgentExecutionResult:
        worker = self.sub_agent_workers[worker_id]
        self.enforce_registered_worker_limits(worker)
        self.runtime_state.record_tool_call(
            worker.descriptor.worker_id,
            is_mcp_tool=worker.descriptor.worker_type == "mcp",
        )
        current_result = await worker.runner(query)
        if current_result.status == SubAgentStatus.SUCCESS:
            self.runtime_state.record_tool_success(worker.descriptor.worker_id)
        else:
            self.runtime_state.record_tool_failure(worker.descriptor.worker_id)
        rewrite_attempts = 0

        while (
            rewrite_attempts < self.retry_config.max_query_rewrite_retries
            and should_retry_with_rewrite(current_result)
        ):
            rewritten_query = await self.rewrite_sub_agent_query(
                current_result.final_query,
                current_result,
            )
            if not rewritten_query or rewritten_query.strip() == current_result.final_query.strip():
                break

            rewrite_attempts += 1
            self.enforce_registered_worker_limits(worker)
            self.runtime_state.record_tool_call(
                worker.descriptor.worker_id,
                is_mcp_tool=worker.descriptor.worker_type == "mcp",
            )
            current_result = await worker.runner(rewritten_query)
            if current_result.status == SubAgentStatus.SUCCESS:
                self.runtime_state.record_tool_success(worker.descriptor.worker_id)
            else:
                self.runtime_state.record_tool_failure(worker.descriptor.worker_id)
            current_result = current_result.model_copy(
                update={
                    "final_query": rewritten_query,
                    "query_changed": rewritten_query != query,
                }
            )

            if current_result.status == SubAgentStatus.SUCCESS:
                break

        return current_result

    def format_sub_agent_failure(self, result: SubAgentExecutionResult) -> str:
        details = [
            f"source_agent={result.source_agent or result.tool_name or 'unknown'}",
            f"status={result.status.value if isinstance(result.status, SubAgentStatus) else result.status}",
            f"error_type={result.error_type or 'unknown'}",
            f"attempts={result.attempts}",
            f"query_changed={result.query_changed}",
        ]
        if result.error:
            details.append(f"error={result.error}")
        elif result.raw_error:
            details.append(f"error={result.raw_error}")
        if result.final_query and result.final_query != result.query:
            details.append(f"final_query={result.final_query}")
        return "Sub-agent failed: " + ", ".join(details)

    async def setup_agent_skill_as_tools(self) -> List[BaseTool]:
        agent_skill_as_tools = []
        agent_skills = await AgentSkillService.get_agent_skills_by_ids(self.agent_config.agent_skill_ids)

        def create_skill_agent_as_tool(worker_id: str, agent_skill: AgentSkill):
            @tool(agent_skill.as_tool_name, description=agent_skill.description)
            async def call_skill_agent(query: str):
                result = await self.execute_registered_worker(worker_id, query)
                if result.status == SubAgentStatus.SUCCESS:
                    return result.raw_result or result.answer
                raise ValueError(self.format_sub_agent_failure(result))

            return call_skill_agent

        for agent_skill in agent_skills:
            skill_agent = SkillAgent(
                agent_skill,
                self.agent_config.user_id,
                retry_config=self.retry_config,
            )
            worker_id = f"skill::{agent_skill.as_tool_name}"
            self.register_worker(
                RegisteredWorker(
                    descriptor=WorkerDescriptor(
                        worker_id=worker_id,
                        source_agent=agent_skill.name,
                        worker_type="skill",
                        description=agent_skill.description or "",
                    ),
                    runner=lambda query, agent=skill_agent: agent.ainvoke([HumanMessage(content=query)]),
                )
            )
            self.tool_metadata_map[agent_skill.as_tool_name] = {
                "name": agent_skill.name,
                "type": "Skill",
            }
            agent_skill_as_tools.append(create_skill_agent_as_tool(worker_id, agent_skill))

        return agent_skill_as_tools

    async def rewrite_sub_agent_query(
        self,
        query: str,
        result: SubAgentExecutionResult,
    ) -> str:
        remaining_timeout = self.remaining_round_timeout()
        if remaining_timeout <= 0:
            raise CircuitBreakerError(
                "The current round exceeded the total timeout budget.",
                error_type="timeout",
            )

        prompt = (
            "Please rewrite the query for a sub-agent call. "
            "Keep the original intent, make the query more concrete, and return only one rewritten query.\n"
            f"Sub-agent: {result.source_agent or result.tool_name or 'unknown'}\n"
            f"Original query: {query}\n"
            f"Failure type: {result.error_type}\n"
            f"Failure detail: {result.error or result.raw_error or 'empty result'}"
        )

        try:
            response = await asyncio.wait_for(
                self.conversation_model.ainvoke(
                    [HumanMessage(content=prompt)],
                    config={"callbacks": [usage_metadata_callback]},
                ),
                timeout=remaining_timeout,
            )
            consumed_tokens = extract_total_tokens_from_model_response(response)
            if consumed_tokens:
                self.runtime_state.total_tokens += consumed_tokens
                if self.runtime_state.total_tokens > self.retry_config.max_total_tokens:
                    raise CircuitBreakerError(
                        "The current round exceeded the token budget.",
                        error_type="token_budget",
                    )

            rewritten_query = str(getattr(response, "content", "")).strip()
            if not rewritten_query:
                return query
            return rewritten_query
        except Exception as error:
            logger.warning(f"Rewrite sub-agent query failed, fallback to original query: {error}")
            return query

    async def setup_mcp_agent_as_tools(self):
        mcp_agent_as_tools = []

        def create_mcp_agent_as_tool(worker_id: str, mcp_as_tool_name: str, description: str):
            @tool(mcp_as_tool_name, description=description)
            async def call_mcp_agent(query: str):
                result = await self.execute_registered_worker(worker_id, query)
                if result.status == SubAgentStatus.SUCCESS:
                    return result.raw_result or result.answer
                raise ValueError(self.format_sub_agent_failure(result))

            return call_mcp_agent

        for mcp_id in self.agent_config.mcp_ids:
            mcp_server = await MCPService.get_mcp_server_from_id(mcp_id)
            mcp_config = MCPConfig(**mcp_server)
            mcp_agent = MCPAgent(
                mcp_config,
                self.agent_config.user_id,
                retry_config=self.retry_config,
            )
            await mcp_agent.init_mcp_agent()

            tool_name = mcp_server.get("mcp_as_tool_name")
            description = mcp_server.get("description")
            worker_id = f"mcp::{tool_name}"

            self.register_worker(
                RegisteredWorker(
                    descriptor=WorkerDescriptor(
                        worker_id=worker_id,
                        source_agent=mcp_config.server_name,
                        worker_type="mcp",
                        description=description or "",
                    ),
                    runner=lambda query, agent=mcp_agent: agent.ainvoke([HumanMessage(content=query)]),
                )
            )

            self.tool_metadata_map[tool_name] = {
                "name": mcp_config.server_name,
                "type": "MCP",
            }
            mcp_agent_as_tools.append(create_mcp_agent_as_tool(worker_id, tool_name, description))

        return mcp_agent_as_tools

    async def build_knowledge_worker_result(self, query: str) -> SubAgentExecutionResult:
        source_agent = "Knowledge Retrieval"
        answer = ""
        if app_settings.rag and app_settings.rag.enable_summary:
            answer = await RagHandler.rag_query_summary(query, self.agent_config.knowledge_ids)
        if not answer or answer == "No relevant documents found.":
            answer = await RagHandler.retrieve_ranked_documents(query, self.agent_config.knowledge_ids)
        if answer and answer != "No relevant documents found.":
            return SubAgentExecutionResult(
                status=SubAgentStatus.SUCCESS,
                query=query,
                final_query=query,
                attempts=1,
                answer=answer,
                summary=summarize_text(answer),
                raw_result=answer,
                source_agent=source_agent,
                error=None,
                confidence=0.65,
                evidence=extract_evidence(answer),
                tool_name="retrival_knowledge",
            )

        return SubAgentExecutionResult(
            status=SubAgentStatus.RETRYABLE_FAILURE,
            query=query,
            final_query=query,
            attempts=1,
            answer="",
            summary="Knowledge worker completed but returned no relevant documents.",
            raw_result=answer or "",
            source_agent=source_agent,
            error="No relevant documents found.",
            confidence=0.0,
            evidence=[],
            tool_name="retrival_knowledge",
            error_type=ErrorType.EMPTY_RESULT.value,
            raw_error="No relevant documents found.",
        )

    async def setup_knowledge_tool(self):
        worker_id = "knowledge::default"

        async def run_knowledge_worker(query: str) -> SubAgentExecutionResult:
            return await self.build_knowledge_worker_result(query)

        @tool(parse_docstring=True)
        async def retrival_knowledge(query: str) -> str:
            """
            Retrieve knowledge base content relevant to the user query.

            Args:
                query (str): User question.

            Returns:
                str: Retrieved knowledge snippets.
            """

            result = await self.execute_registered_worker(worker_id, query)
            if result.status == SubAgentStatus.SUCCESS:
                return result.raw_result or result.answer
            raise ValueError(self.format_sub_agent_failure(result))

        if self.agent_config.knowledge_ids:
            self.register_worker(
                RegisteredWorker(
                    descriptor=WorkerDescriptor(
                        worker_id=worker_id,
                        source_agent="Knowledge Retrieval",
                        worker_type="knowledge",
                        description="Searches the bound knowledge bases and returns evidence-backed snippets.",
                    ),
                    runner=run_knowledge_worker,
                )
            )
            self.tools.append(retrival_knowledge)
            self.tool_metadata_map[retrival_knowledge.name] = {
                "name": "Knowledge Retrieval",
                "type": "Knowledge",
            }

    def should_attempt_multi_agent_orchestration(self, messages: List[BaseMessage]) -> bool:
        return (
            self.retry_config.enable_multi_agent_orchestration
            and self.orchestrator is not None
            and len(self.sub_agent_workers) >= 2
            and bool(self.get_last_user_query(messages).strip())
        )

    async def stream_text_response(self, text: str) -> AsyncGenerator[Dict[str, Any], None]:
        accumulated = ""
        if not text:
            return

        chunk_size = 120
        for index in range(0, len(text), chunk_size):
            chunk = text[index:index + chunk_size]
            accumulated += chunk
            yield {
                "type": "response_chunk",
                "timestamp": time.time(),
                "data": {
                    "chunk": chunk,
                    "accumulated": accumulated,
                },
            }

    async def try_multi_agent_orchestration(
        self,
        messages: List[BaseMessage],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        query = self.get_last_user_query(messages)
        assert self.orchestrator is not None

        worker_descriptors = [worker.descriptor for worker in self.sub_agent_workers.values()]
        yield self.wrap_event(
            {
                "status": "START",
                "title": "Multi-Agent Orchestration",
                "message": "Planning fan-out tasks across sub-agents...",
            }
        )

        plan = await self.orchestrator.plan(query, worker_descriptors)
        if not plan.should_orchestrate or len(plan.tasks) < 2:
            yield self.wrap_event(
                {
                    "status": "END",
                    "title": "Multi-Agent Orchestration",
                    "message": "Planner chose to fall back to the normal agent flow.",
                }
            )
            return

        yield self.wrap_event(
            {
                "status": "END",
                "title": "Multi-Agent Orchestration",
                "message": {
                    "reasoning": plan.reasoning,
                    "tasks": [task.model_dump() for task in plan.tasks],
                    "aggregation_goal": plan.aggregation_goal,
                },
            }
        )

        semaphore = asyncio.Semaphore(self.retry_config.max_parallel_sub_agents)

        async def run_task(task):
            async with semaphore:
                return task, await self.execute_registered_worker(task.worker_id, task.query)

        executable_tasks = []
        seen_worker_ids = set()
        for task in plan.tasks:
            if task.worker_id not in self.sub_agent_workers:
                continue
            if task.worker_id in seen_worker_ids:
                continue
            seen_worker_ids.add(task.worker_id)
            executable_tasks.append(task)
        for task in executable_tasks:
            worker = self.sub_agent_workers.get(task.worker_id)
            yield self.wrap_event(
                {
                    "status": "START",
                    "title": f"Fan-Out - {worker.descriptor.source_agent}",
                    "message": {
                        "worker_id": task.worker_id,
                        "query": task.query,
                        "reason": task.reason,
                        "worker_type": worker.descriptor.worker_type,
                    },
                }
            )

        if len(executable_tasks) < 2:
            yield self.wrap_event(
                {
                    "status": "END",
                    "title": "Multi-Agent Orchestration",
                    "message": "Planner output was incomplete, falling back to the normal agent flow.",
                }
            )
            return

        execution_pairs = await asyncio.gather(*(run_task(task) for task in executable_tasks))
        results: list[SubAgentExecutionResult] = []
        for task, result in execution_pairs:
            results.append(result)
            yield self.wrap_event(
                {
                    "status": "END" if result.status == SubAgentStatus.SUCCESS else "ERROR",
                    "title": f"Fan-In - {result.source_agent or task.worker_id}",
                    "message": {
                        "status": result.status,
                        "summary": result.summary,
                        "source_agent": result.source_agent,
                        "confidence": result.confidence,
                        "error": result.error or result.raw_error,
                        "evidence": result.evidence,
                    },
                }
            )

        aggregate = await self.orchestrator.aggregate(query, plan, results)
        yield self.wrap_event(
            {
                "status": "END",
                "title": "Aggregation",
                "message": aggregate.model_dump(),
            }
        )

        if aggregate.status == "failure":
            return

        async for chunk in self.stream_text_response(aggregate.final_answer):
            yield chunk

    async def astream(self, messages: List[BaseMessage]) -> AsyncGenerator[Dict[str, Any], None]:
        response_content = ""

        try:
            if self.should_attempt_multi_agent_orchestration(messages):
                try:
                    used_orchestration = False
                    async for event in self.try_multi_agent_orchestration(messages):
                        if event.get("type") == "response_chunk":
                            used_orchestration = True
                            response_content = event["data"]["accumulated"]
                        yield event

                    if used_orchestration:
                        return
                except Exception as orchestration_error:
                    logger.warning(
                        f"Multi-agent orchestration failed, fallback to normal agent flow: {orchestration_error}"
                    )
                    yield self.wrap_event(
                        {
                            "status": "ERROR",
                            "title": "Multi-Agent Orchestration",
                            "message": "Orchestration failed, falling back to the normal agent flow.",
                        }
                    )

            async for token, metadata in self.react_agent.astream(
                input={
                    "messages": copy.deepcopy(messages),
                    "model_call_count": 0,
                    "tool_call_count": 0,
                    "user_id": self.agent_config.user_id,
                },
                config={"callbacks": [usage_metadata_callback]},
                stream_mode=["messages", "custom"],
            ):
                if token == "custom":
                    yield self.wrap_event(metadata)
                elif isinstance(metadata[0], AIMessageChunk) and metadata[0].content:
                    response_content += metadata[0].content
                    yield {
                        "type": "response_chunk",
                        "timestamp": time.time(),
                        "data": {
                            "chunk": metadata[0].content,
                            "accumulated": response_content,
                        },
                    }
        except CircuitBreakerError as error:
            logger.warning(f"GeneralAgent stopped by circuit breaker: {error}")
            stop_message = (
                "I stopped retrying this round because the execution guardrail was reached. "
                f"Current blocker: {error}"
            )
            yield {
                "type": "response_chunk",
                "timestamp": time.time(),
                "data": {
                    "chunk": stop_message,
                    "accumulated": response_content + stop_message,
                },
            }
        except Exception as error:
            logger.error(f"LLM Model Error: {error}")
            fallback_message = "I ran into an unexpected error while processing this request."
            yield {
                "type": "response_chunk",
                "timestamp": time.time(),
                "data": {
                    "chunk": fallback_message,
                    "accumulated": response_content + fallback_message,
                },
            }

    def stop_streaming_callback(self):
        self.stop_streaming = True
