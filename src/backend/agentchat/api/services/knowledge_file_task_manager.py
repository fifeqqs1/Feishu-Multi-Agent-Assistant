import asyncio
from collections.abc import Awaitable

from loguru import logger


class KnowledgeFileTaskManager:
    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._ignore_results: set[str] = set()
        self._lock = asyncio.Lock()

    async def enqueue(self, knowledge_file_id: str, job: Awaitable[None]):
        async with self._lock:
            task = asyncio.create_task(self._run_tracked(knowledge_file_id, job))
            self._tasks[knowledge_file_id] = task
            return task

    async def _run_tracked(self, knowledge_file_id: str, job: Awaitable[None]):
        try:
            await job
        except asyncio.CancelledError:
            logger.info(f"Knowledge file task cancelled: {knowledge_file_id}")
            raise
        except Exception as err:
            logger.exception(f"Knowledge file task crashed: {knowledge_file_id}, error: {err}")
        finally:
            async with self._lock:
                self._tasks.pop(knowledge_file_id, None)
                self._ignore_results.discard(knowledge_file_id)

    async def mark_deleted(self, knowledge_file_id: str) -> bool:
        async with self._lock:
            self._ignore_results.add(knowledge_file_id)
            task = self._tasks.get(knowledge_file_id)
            if task:
                task.cancel()
                return True
            return False

    async def clear_deleted(self, knowledge_file_id: str):
        async with self._lock:
            self._ignore_results.discard(knowledge_file_id)

    async def should_ignore(self, knowledge_file_id: str) -> bool:
        async with self._lock:
            return knowledge_file_id in self._ignore_results


knowledge_file_task_manager = KnowledgeFileTaskManager()
