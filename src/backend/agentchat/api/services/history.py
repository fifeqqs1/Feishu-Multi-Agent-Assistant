import asyncio
from typing import List
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agentchat.api.services.dialog import DialogService
from agentchat.database.dao.history import HistoryDao
from agentchat.schema.chunk import ChunkModel
from agentchat.services.memory.client import memory_client
from agentchat.services.memory.session_context import session_context_manager
from agentchat.services.rag.es_client import client as es_client
from agentchat.services.rag.vector_db import milvus_client
from agentchat.settings import app_settings
from agentchat.utils.helpers import get_now_beijing_time

Assistant_Role = "assistant"
User_Role = "user"


class HistoryService:
    @classmethod
    async def create_history(cls, role: str, content: str, events: List[dict], dialog_id: str):
        try:
            await HistoryDao.create_history(role, content, events, dialog_id)
        except Exception as err:
            raise ValueError(f"Add history data appear error: {err}")

    @classmethod
    async def select_history(cls, dialog_id: str, top_k: int = 4) -> List[BaseMessage] | None:
        try:
            result = await HistoryDao.select_history_from_time(dialog_id, top_k)
            messages: List[BaseMessage] = []
            for data in result:
                if data.role == Assistant_Role:
                    messages.append(AIMessage(content=data.content))
                elif data.role == User_Role:
                    messages.append(HumanMessage(content=data.content))
            return messages
        except Exception as err:
            raise ValueError(f"Select history is appear error: {err}")

    @classmethod
    def _format_memory_results(cls, title: str, memory_results: list[dict], max_items: int) -> str:
        if not memory_results:
            return ""

        lines = [f"<{title}>"]
        for index, memory in enumerate(memory_results[:max_items], start=1):
            memory_text = str(memory.get("memory", "")).strip()
            if not memory_text:
                continue

            score = memory.get("score")
            prefix = f"{index}. "
            if score is not None:
                prefix += f"(score={score:.3f}) "
            lines.append(f"{prefix}{memory_text}")

        lines.append(f"</{title}>")
        if len(lines) <= 2:
            return ""
        return "\n".join(lines)

    @classmethod
    async def enable_memory_select_history(
        cls,
        dialog_id: str,
        user_input: str,
        user_id: str,
        top_k: int = 10,
    ) -> dict:
        try:
            cache = await session_context_manager.get_or_build_cache(dialog_id)
            history_text = session_context_manager.format_history_context(cache)

            if not app_settings.memory or not str(user_input).strip():
                return {
                    "history_text": history_text,
                    "memory_text": "",
                    "session_memories": [],
                    "global_memories": [],
                }

            session_limit = min(top_k, app_settings.memory.semantic_session_recall_limit)
            global_limit = min(top_k, app_settings.memory.semantic_global_recall_limit)

            session_search, global_search = await session_context_manager_search(
                user_input=user_input,
                user_id=user_id,
                dialog_id=dialog_id,
                session_limit=session_limit,
                global_limit=global_limit,
            )

            session_memories = session_search.get("results", []) if isinstance(session_search, dict) else []
            global_memories = global_search.get("results", []) if isinstance(global_search, dict) else []

            seen_ids = set()
            deduped_session = []
            for memory in session_memories:
                memory_id = memory.get("id")
                if memory_id and memory_id in seen_ids:
                    continue
                if memory_id:
                    seen_ids.add(memory_id)
                deduped_session.append(memory)

            deduped_global = []
            for memory in global_memories:
                memory_id = memory.get("id")
                if memory_id and memory_id in seen_ids:
                    continue
                if memory_id:
                    seen_ids.add(memory_id)
                deduped_global.append(memory)

            memory_sections = [
                cls._format_memory_results("session_memory_hits", deduped_session, session_limit),
                cls._format_memory_results("long_term_memory_hits", deduped_global, global_limit),
            ]
            memory_text = "\n".join(section for section in memory_sections if section)

            return {
                "history_text": history_text,
                "memory_text": memory_text,
                "session_memories": deduped_session,
                "global_memories": deduped_global,
            }
        except Exception as err:
            raise ValueError(f"Enable memory select history appear error: {err}")

    @classmethod
    async def get_dialog_history(cls, dialog_id: str):
        try:
            results = await HistoryDao.get_dialog_history(dialog_id)
            return [res.to_dict() for res in results]
        except Exception as err:
            raise ValueError(f"Get dialog history is appear error: {err}")

    @classmethod
    async def save_es_documents(cls, index_name, content):
        chunk = ChunkModel(
            chunk_id=uuid4().hex,
            content=content,
            file_id="history_rag",
            knowledge_id=index_name,
            summary="history_rag",
            update_time=get_now_beijing_time(),
            file_name="history_rag",
        )

        await es_client.index_documents(index_name, [chunk])

    @classmethod
    async def save_milvus_documents(cls, collection_name, content):
        chunk = ChunkModel(
            chunk_id=uuid4().hex,
            content=content,
            file_id="history_rag",
            knowledge_id=collection_name,
            update_time=get_now_beijing_time(),
            summary="history_rag",
            file_name="history_rag",
        )

        await milvus_client.insert(collection_name, [chunk])

    @classmethod
    async def save_chat_history(cls, role, content, events, dialog_id, memory_enable: bool = False):
        await cls.create_history(role, content, events, dialog_id)
        await DialogService.update_dialog_time(dialog_id=dialog_id)

        if memory_enable:
            await session_context_manager.append_message(dialog_id, role, content)

    @classmethod
    async def clear_dialog_context(cls, dialog_id: str):
        await session_context_manager.clear_cache(dialog_id)


async def session_context_manager_search(
    user_input: str,
    user_id: str,
    dialog_id: str,
    session_limit: int,
    global_limit: int,
):
    return await asyncio.gather(
        memory_client.search(
            query=user_input,
            user_id=user_id,
            run_id=dialog_id,
            limit=max(1, session_limit),
        ),
        memory_client.search(
            query=user_input,
            user_id=user_id,
            limit=max(1, global_limit),
        ),
    )
