import asyncio
from typing import List

from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.prebuilt.tool_node import ToolCallRequest

from agentchat.api.services.mcp_user_config import MCPUserConfigService
from agentchat.core.agents.retry import (
    AgentRetryConfig,
    ErrorType,
    SubAgentExecutionResult,
    SubAgentStatus,
    detect_error_type,
    is_retryable_error_type,
)
from agentchat.core.agents.orchestration import extract_evidence, summarize_text
from agentchat.core.models.manager import ModelManager
from agentchat.prompts.completion import CALL_END_PROMPT
from agentchat.services.mcp.manager import MCPManager
from agentchat.utils.convert import convert_mcp_config


class MCPConfig(BaseModel):
    url: str
    type: str = "sse"
    tools: List[str] = []
    server_name: str
    mcp_server_id: str


class MCPAgent:
    def __init__(
        self,
        mcp_config: MCPConfig,
        user_id: str,
        retry_config: AgentRetryConfig | None = None,
    ):
        self.mcp_config = mcp_config
        self.retry_config = retry_config or AgentRetryConfig()
        self.mcp_manager = MCPManager(
            [convert_mcp_config(mcp_config.model_dump())],
            retry_config=self.retry_config,
        )

        self.user_id = user_id
        self.mcp_tools: List[BaseTool] = []

        self.conversation_model = None
        self.tool_invocation_model = None

        self.react_agent = None
        self.middlewares = None

    async def init_mcp_agent(self):
        if self.mcp_config:
            self.mcp_tools = await self.setup_mcp_tools()

        await self.setup_language_model()
        self.middlewares = await self.setup_agent_middlewares()
        self.react_agent = self.setup_react_agent()

    async def emit_event(self, event):
        writer = get_stream_writer()
        writer(event)

    async def setup_language_model(self):
        self.conversation_model = ModelManager.get_conversation_model()
        self.tool_invocation_model = ModelManager.get_tool_invocation_model()

    async def setup_mcp_tools(self):
        return await self.mcp_manager.get_mcp_tools()

    async def setup_agent_middlewares(self):
        @wrap_tool_call
        async def add_tool_call_args(request: ToolCallRequest, handler):
            tool_name = request.tool_call["name"]
            await self.emit_event(
                {
                    "status": "START",
                    "title": f"Sub-Agent - {self.mcp_config.server_name} execute tool: {tool_name}",
                    "message": f"Calling MCP tool {tool_name}...",
                }
            )

            try:
                user_config = await MCPUserConfigService.get_mcp_user_config(
                    self.user_id,
                    self.mcp_config.mcp_server_id,
                )
                if user_config:
                    request.tool_call["args"].update(user_config)

                tool_result = await handler(request)

                await self.emit_event(
                    {
                        "status": "END",
                        "title": f"Sub-Agent - {self.mcp_config.server_name} execute tool: {tool_name}",
                        "message": str(tool_result),
                    }
                )
                return tool_result
            except Exception as error:
                await self.emit_event(
                    {
                        "status": "ERROR",
                        "title": f"Sub-Agent - {self.mcp_config.server_name} execute tool: {tool_name}",
                        "message": str(error),
                    }
                )
                raise

        return [add_tool_call_args]

    def setup_react_agent(self):
        return create_agent(
            model=self.conversation_model,
            tools=self.mcp_tools,
            middleware=self.middlewares,
            system_prompt=CALL_END_PROMPT,
        )

    def _extract_query(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    def _collect_answer(self, result: dict) -> str:
        filtered_messages = []
        for message in result["messages"][:-1]:
            if not isinstance(message, (HumanMessage, SystemMessage)):
                filtered_messages.append(message)

        contents = []
        for message in filtered_messages:
            content = getattr(message, "content", "")
            if isinstance(content, list):
                contents.extend(
                    str(item.get("text", item))
                    if isinstance(item, dict)
                    else str(item)
                    for item in content
                )
            else:
                contents.append(str(content))

        return "\n".join(text for text in contents if text).strip()

    async def ainvoke(self, messages: List[BaseMessage]) -> SubAgentExecutionResult:
        query = self._extract_query(messages)
        max_attempts = 1 + self.retry_config.max_sub_agent_same_query_retries

        for attempt in range(1, max_attempts + 1):
            try:
                result = await asyncio.wait_for(
                    self.react_agent.ainvoke({"messages": messages}),
                    timeout=self.retry_config.max_round_timeout_seconds,
                )
                answer = self._collect_answer(result)

                if answer:
                    return SubAgentExecutionResult(
                        status=SubAgentStatus.SUCCESS,
                        query=query,
                        final_query=query,
                        attempts=attempt,
                        answer=answer,
                        summary=summarize_text(answer),
                        raw_result=answer,
                        source_agent=self.mcp_config.server_name,
                        confidence=0.75,
                        evidence=extract_evidence(answer),
                        tool_name=self.mcp_config.server_name,
                    )

                return SubAgentExecutionResult(
                    status=SubAgentStatus.RETRYABLE_FAILURE,
                    query=query,
                    final_query=query,
                    attempts=attempt,
                    summary="MCP sub-agent completed but returned no useful content.",
                    raw_result="",
                    source_agent=self.mcp_config.server_name,
                    error="MCP sub-agent returned an empty result.",
                    confidence=0.0,
                    evidence=[],
                    tool_name=self.mcp_config.server_name,
                    error_type=ErrorType.EMPTY_RESULT.value,
                    raw_error="MCP sub-agent returned an empty result.",
                )
            except Exception as error:
                error_type = detect_error_type(error)
                can_retry = is_retryable_error_type(error_type) and attempt < max_attempts
                if can_retry:
                    continue

                return SubAgentExecutionResult(
                    status=(
                        SubAgentStatus.RETRYABLE_FAILURE
                        if is_retryable_error_type(error_type)
                        else SubAgentStatus.NON_RETRYABLE_FAILURE
                    ),
                    query=query,
                    final_query=query,
                    attempts=attempt,
                    summary="MCP sub-agent execution failed.",
                    raw_result="",
                    source_agent=self.mcp_config.server_name,
                    error=str(error),
                    confidence=0.0,
                    evidence=[],
                    tool_name=self.mcp_config.server_name,
                    error_type=error_type,
                    raw_error=str(error),
                )

        return SubAgentExecutionResult(
            status=SubAgentStatus.RETRYABLE_FAILURE,
            query=query,
            final_query=query,
            attempts=max_attempts,
            summary="MCP sub-agent exhausted retries without a concrete error.",
            raw_result="",
            source_agent=self.mcp_config.server_name,
            error="MCP sub-agent exhausted retries without a concrete error.",
            confidence=0.0,
            evidence=[],
            tool_name=self.mcp_config.server_name,
            error_type=ErrorType.UNKNOWN.value,
            raw_error="MCP sub-agent exhausted retries without a concrete error.",
        )
