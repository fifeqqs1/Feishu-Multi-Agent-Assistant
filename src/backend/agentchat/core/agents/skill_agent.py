import asyncio
from typing import Dict, List, Optional, Union

from loguru import logger
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.prebuilt.tool_node import ToolCallRequest

from agentchat.core.agents.orchestration import extract_evidence, summarize_text
from agentchat.core.agents.retry import (
    AgentRetryConfig,
    ErrorType,
    SubAgentExecutionResult,
    SubAgentStatus,
    detect_error_type,
    is_retryable_error_type,
)
from agentchat.core.callbacks import usage_metadata_callback
from agentchat.core.models.manager import ModelManager
from agentchat.database import AgentSkill
from agentchat.schema.agent_skill import AgentSkillFile, AgentSkillFolder


class SkillAgent:
    def __init__(
        self,
        skill: AgentSkill,
        user_id: str,
        retry_config: AgentRetryConfig | None = None,
    ):
        self.skill = skill
        self.user_id = user_id
        self.retry_config = retry_config or AgentRetryConfig()
        self.skill_folder: Optional[AgentSkillFolder] = None
        self.file_cache: Dict[str, AgentSkillFile] = {}

        self.conversation_model = None
        self.tool_invocation_model = None
        self.tools = None
        self.middlewares = None
        self.react_agent = None

        self._initialized = False

    async def init_skill_agent(self):
        try:
            self.load_skill_folder(self.skill.folder)
            self.setup_language_model()
            self.tools = self.setup_skill_agent_tools()
            self.middlewares = await self.setup_agent_middlewares()
            self.react_agent = self.setup_react_agent()
            self._initialized = True
            logger.info(f"SkillAgent `{self.skill.name}` initialized successfully")
        except Exception as error:
            logger.error(f"SkillAgent initialization failed: {error}")
            raise

    def setup_react_agent(self):
        if not self.conversation_model:
            raise ValueError("Conversation model is not initialized")

        skill_md = self.get_skill_md()
        if not skill_md:
            logger.warning(f"Skill `{self.skill.name}` does not contain SKILL.md")
            skill_md = f"Skill Name: {self.skill.name}\nDescription: {self.skill.description or 'N/A'}"

        return create_agent(
            model=self.conversation_model,
            tools=self.tools,
            middleware=self.middlewares,
            system_prompt=self._build_system_prompt(skill_md),
        )

    def setup_language_model(self):
        self.conversation_model = ModelManager.get_conversation_model()
        self.tool_invocation_model = ModelManager.get_tool_invocation_model()

    async def emit_event(self, event):
        writer = get_stream_writer()
        writer(event)

    def _build_system_prompt(self, skill_md: str) -> str:
        return f"""You are a specialized skill agent responsible for executing the skill `{self.skill.name}`.

# Skill Document
{skill_md}

# Available Tools
You can use the `get_file_content` tool to read any file inside this skill package.

# Execution Rules
1. Follow the skill document strictly.
2. Read additional files proactively when needed.
3. Return a clear and structured answer.
4. If something fails, explain the failure precisely.
"""

    def load_skill_folder(self, json_data: dict) -> AgentSkillFolder:
        def parse_item(item_data: dict) -> Union[AgentSkillFile, AgentSkillFolder]:
            item_type = item_data.get("type", "file")

            if item_type == "file":
                file_obj = AgentSkillFile(
                    name=item_data["name"],
                    path=item_data["path"],
                    type=item_data["type"],
                    content=item_data.get("content", ""),
                )
                self.file_cache[file_obj.path] = file_obj
                return file_obj

            if item_type == "folder":
                folder_items = [parse_item(sub_item) for sub_item in item_data.get("folder", [])]
                return AgentSkillFolder(
                    name=item_data["name"],
                    path=item_data["path"],
                    type=item_data["type"],
                    folder=folder_items,
                )

            raise ValueError(f"Unknown type: {item_type}")

        self.skill_folder = parse_item(json_data)
        return self.skill_folder

    def get_file_content(self, path: str) -> Optional[str]:
        file_obj = self.file_cache.get(path)
        return file_obj.content if file_obj else None

    def list_files(self, pattern: str = None) -> List[str]:
        if pattern:
            return [path for path in self.file_cache.keys() if pattern in path]
        return list(self.file_cache.keys())

    def get_skill_md(self) -> Optional[str]:
        if not self.skill_folder or not getattr(self.skill_folder, "folder", None):
            return None

        for item in self.skill_folder.folder:
            if isinstance(item, AgentSkillFile) and item.name == "SKILL.md":
                return item.content
        return None

    async def setup_agent_middlewares(self):
        @wrap_tool_call
        async def add_tool_call_args(request: ToolCallRequest, handler):
            tool_name = request.tool_call["name"]
            await self.emit_event(
                {
                    "status": "START",
                    "title": f"Skill-Agent - {self.skill.name} execute tool: {tool_name}",
                    "message": f"Calling skill tool {tool_name}...",
                }
            )

            tool_result = await handler(request)

            await self.emit_event(
                {
                    "status": "END",
                    "title": f"Skill-Agent - {self.skill.name} execute tool: {tool_name}",
                    "message": str(tool_result),
                }
            )
            return tool_result

        return [add_tool_call_args]

    def setup_skill_agent_tools(self):
        @tool(parse_docstring=True)
        def get_file_content(file_path: str) -> str:
            """
            Read the content of a file inside the skill package.

            Args:
                file_path: File path inside the skill package.

            Returns:
                File content, or a helpful error message if the file does not exist.
            """

            content = self.get_file_content(file_path)
            if content is None:
                available_files = self.list_files()
                return (
                    f"Error: file `{file_path}` does not exist.\n"
                    f"Available files:\n" + "\n".join(f"  - {file_name}" for file_name in available_files)
                )
            return content

        @tool(parse_docstring=True)
        def list_skill_files(pattern: str = None) -> str:
            """
            List files inside the skill package.

            Args:
                pattern: Optional substring filter for the file path.

            Returns:
                File paths separated by new lines.
            """

            files = self.list_files(pattern)
            if not files:
                return "No files found."
            return "\n".join(files)

        return [get_file_content, list_skill_files]

    def _extract_query(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    def _collect_answer(self, result: dict) -> str:
        filtered_messages = [
            message for message in result["messages"]
            if not isinstance(message, (HumanMessage, SystemMessage))
        ]

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
        if not self._initialized:
            await self.init_skill_agent()

        query = self._extract_query(messages)
        max_attempts = 1 + self.retry_config.max_sub_agent_same_query_retries

        for attempt in range(1, max_attempts + 1):
            try:
                result = await asyncio.wait_for(
                    self.react_agent.ainvoke(
                        input={"messages": messages},
                        config={"callbacks": [usage_metadata_callback]},
                    ),
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
                        source_agent=self.skill.name,
                        error=None,
                        confidence=0.7,
                        evidence=extract_evidence(answer),
                        tool_name=self.skill.as_tool_name,
                    )

                return SubAgentExecutionResult(
                    status=SubAgentStatus.RETRYABLE_FAILURE,
                    query=query,
                    final_query=query,
                    attempts=attempt,
                    answer="",
                    summary="Skill sub-agent completed but returned no useful content.",
                    raw_result="",
                    source_agent=self.skill.name,
                    error="Skill sub-agent returned an empty result.",
                    confidence=0.0,
                    evidence=[],
                    tool_name=self.skill.as_tool_name,
                    error_type=ErrorType.EMPTY_RESULT.value,
                    raw_error="Skill sub-agent returned an empty result.",
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
                    answer="",
                    summary="Skill sub-agent execution failed.",
                    raw_result="",
                    source_agent=self.skill.name,
                    error=str(error),
                    confidence=0.0,
                    evidence=[],
                    tool_name=self.skill.as_tool_name,
                    error_type=error_type,
                    raw_error=str(error),
                )

        return SubAgentExecutionResult(
            status=SubAgentStatus.RETRYABLE_FAILURE,
            query=query,
            final_query=query,
            attempts=max_attempts,
            answer="",
            summary="Skill sub-agent exhausted retries without a concrete error.",
            raw_result="",
            source_agent=self.skill.name,
            error="Skill sub-agent exhausted retries without a concrete error.",
            confidence=0.0,
            evidence=[],
            tool_name=self.skill.as_tool_name,
            error_type=ErrorType.UNKNOWN.value,
            raw_error="Skill sub-agent exhausted retries without a concrete error.",
        )
