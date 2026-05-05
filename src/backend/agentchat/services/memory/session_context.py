import asyncio
import re
from datetime import datetime
from typing import Any

from loguru import logger

from agentchat.core.models.manager import ModelManager
from agentchat.database.dao.history import HistoryDao
from agentchat.services.redis import redis_client
from agentchat.settings import app_settings
from agentchat.utils.constants import SESSION_CONTEXT_CACHE


class SessionContextManager:
    def __init__(self):
        self._summary_model = None

    def _get_summary_model(self):
        if self._summary_model is None:
            self._summary_model = ModelManager.get_conversation_model()
        return self._summary_model

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_words = len(re.findall(r"[A-Za-z0-9_]+", text))
        punctuation = len(re.findall(r"[^\w\s]", text))
        return chinese_chars + english_words + max(1, punctuation // 2)

    @classmethod
    def estimate_messages_tokens(cls, messages: list[dict[str, str]]) -> int:
        total = 0
        for message in messages:
            total += cls.estimate_tokens(message.get("role", ""))
            total += cls.estimate_tokens(message.get("content", ""))
        return total

    @staticmethod
    def _cache_key(dialog_id: str) -> str:
        return SESSION_CONTEXT_CACHE.format(dialog_id)

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized = []
        for message in messages or []:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if not role or not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def _normalize_cache(self, cache: dict[str, Any] | None) -> dict[str, Any]:
        cache = cache or {}
        return {
            "rolling_summary": str(cache.get("rolling_summary", "") or "").strip(),
            "recent_messages": self._normalize_messages(cache.get("recent_messages", [])),
            "updated_at": cache.get("updated_at") or datetime.utcnow().isoformat(),
        }

    async def get_cache(self, dialog_id: str) -> dict[str, Any] | None:
        if not app_settings.memory or not app_settings.memory.enable_redis_cache:
            return None
        try:
            result = await asyncio.to_thread(redis_client.get, self._cache_key(dialog_id))
            if not result:
                return None
            return self._normalize_cache(result)
        except Exception as err:
            logger.warning(f"Failed to load session context cache for `{dialog_id}`: {err}")
            return None

    async def save_cache(self, dialog_id: str, cache: dict[str, Any]):
        if not app_settings.memory or not app_settings.memory.enable_redis_cache:
            return
        try:
            normalized = self._normalize_cache(cache)
            normalized["updated_at"] = datetime.utcnow().isoformat()
            await asyncio.to_thread(
                redis_client.set,
                self._cache_key(dialog_id),
                normalized,
                app_settings.memory.redis_ttl_seconds,
            )
        except Exception as err:
            logger.warning(f"Failed to save session context cache for `{dialog_id}`: {err}")

    async def clear_cache(self, dialog_id: str):
        if not app_settings.memory or not app_settings.memory.enable_redis_cache:
            return
        try:
            await asyncio.to_thread(redis_client.delete, self._cache_key(dialog_id))
        except Exception as err:
            logger.warning(f"Failed to clear session context cache for `{dialog_id}`: {err}")

    async def summarize_messages(self, existing_summary: str, messages: list[dict[str, str]]) -> str:
        messages = self._normalize_messages(messages)
        if not messages:
            return existing_summary

        conversation_text = []
        for index, message in enumerate(messages, start=1):
            conversation_text.append(f"{index}. role={message['role']}\ncontent={message['content']}")
        conversation_payload = "\n\n".join(conversation_text)

        prompt = f"""
你是一个对话记忆压缩助手，请把旧对话整理成一段可供后续继续工作的滚动摘要。

要求：
1. 保留用户目标、约束条件、已确认事实、进行中的任务、未解决问题、关键偏好。
2. 删除寒暄、重复表达和无关细节。
3. 如果已有历史摘要，请在其基础上合并，不要丢失仍然有效的信息。
4. 输出纯文本摘要，不要使用 markdown 标题，不要添加解释。
5. 控制在 {app_settings.memory.summary_max_tokens} 个 token 对应的紧凑长度内。

已有摘要：
{existing_summary or "无"}

需要压缩的旧对话：
{conversation_payload}
"""
        try:
            model = self._get_summary_model()
            response = await model.ainvoke(prompt)
            return str(getattr(response, "content", "")).strip() or existing_summary
        except Exception as err:
            logger.warning(f"Failed to summarize session context, keep old summary: {err}")
            return existing_summary

    async def compact_cache(self, cache: dict[str, Any]) -> dict[str, Any]:
        cache = self._normalize_cache(cache)
        recent_messages = cache["recent_messages"]
        if not recent_messages:
            return cache

        max_messages = app_settings.memory.max_history_messages
        preserve_messages = min(max_messages, max(2, app_settings.memory.recent_history_pairs * 2))
        token_count = self.estimate_messages_tokens(recent_messages)

        need_compact = (
            len(recent_messages) > max_messages
            or token_count > app_settings.memory.history_compaction_threshold_tokens
        )
        if not need_compact:
            return cache

        overflow_messages = recent_messages[:-preserve_messages] if len(recent_messages) > preserve_messages else []
        if not overflow_messages and len(recent_messages) > 2:
            midpoint = len(recent_messages) // 2
            overflow_messages = recent_messages[:midpoint]
            recent_messages = recent_messages[midpoint:]
        else:
            recent_messages = recent_messages[-preserve_messages:]

        cache["rolling_summary"] = await self.summarize_messages(cache["rolling_summary"], overflow_messages)
        cache["recent_messages"] = recent_messages

        while (
            len(cache["recent_messages"]) > 2
            and self.estimate_messages_tokens(cache["recent_messages"]) > app_settings.memory.max_history_context_tokens
        ):
            overflow_messages = cache["recent_messages"][:2]
            cache["recent_messages"] = cache["recent_messages"][2:]
            cache["rolling_summary"] = await self.summarize_messages(
                cache["rolling_summary"],
                overflow_messages,
            )

        return cache

    async def build_cache_from_history(self, dialog_id: str) -> dict[str, Any]:
        history_records = await HistoryDao.get_dialog_history(dialog_id)
        messages = [
            {"role": record.role, "content": record.content}
            for record in history_records
            if getattr(record, "role", None) and getattr(record, "content", None)
        ]
        cache = {
            "rolling_summary": "",
            "recent_messages": self._normalize_messages(messages),
            "updated_at": datetime.utcnow().isoformat(),
        }
        cache = await self.compact_cache(cache)
        await self.save_cache(dialog_id, cache)
        return cache

    async def get_or_build_cache(self, dialog_id: str) -> dict[str, Any]:
        cache = await self.get_cache(dialog_id)
        if cache:
            return cache
        return await self.build_cache_from_history(dialog_id)

    async def append_message(self, dialog_id: str, role: str, content: str):
        if role not in {"user", "assistant", "system"}:
            return
        content = str(content or "").strip()
        if not content:
            return

        cache = await self.get_or_build_cache(dialog_id)
        cache["recent_messages"].append({"role": role, "content": content})
        cache = await self.compact_cache(cache)
        await self.save_cache(dialog_id, cache)

    @classmethod
    def format_history_context(cls, cache: dict[str, Any]) -> str:
        cache = cache or {}
        rolling_summary = str(cache.get("rolling_summary", "") or "").strip()
        recent_messages = cls._normalize_messages(cache.get("recent_messages", []))
        sections = []

        if rolling_summary:
            sections.append(
                "<conversation_summary>\n"
                f"{rolling_summary}\n"
                "</conversation_summary>"
            )

        if recent_messages:
            sections.append("<recent_history>")
            for index, message in enumerate(recent_messages, start=1):
                sections.append(
                    f"<turn_{index}>\nrole: {message['role']}\ncontent: {message['content']}\n</turn_{index}>"
                )
            sections.append("</recent_history>")
        else:
            sections.append("<recent_history>\nNo recent history.\n</recent_history>")

        return "\n".join(sections)


session_context_manager = SessionContextManager()
