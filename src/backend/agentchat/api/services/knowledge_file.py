import asyncio
import os
import time
from datetime import datetime
from uuid import uuid4

from loguru import logger

from agentchat.api.services.knowledge_file_task_manager import knowledge_file_task_manager
from agentchat.database.dao.knowledge_file import KnowledgeFileDao
from agentchat.database.models.knowledge_file import Status
from agentchat.database.models.user import AdminUser
from agentchat.services.rag.handler import RagHandler
from agentchat.services.rag.parser import doc_parser
from agentchat.services.storage import storage_client
from agentchat.settings import app_settings
from agentchat.utils.file_utils import get_save_tempfile


class KnowledgeFileService:
    @classmethod
    async def get_knowledge_file(cls, knowledge_id):
        results = await KnowledgeFileDao.select_knowledge_file(knowledge_id)
        return [res.to_dict() for res in results]

    @classmethod
    async def create_record(cls, file_name: str, knowledge_id: str, user_id: str, oss_url: str):
        knowledge_file_id = uuid4().hex
        await KnowledgeFileDao.create_knowledge_file(
            knowledge_file_id=knowledge_file_id,
            file_name=file_name,
            knowledge_id=knowledge_id,
            user_id=user_id,
            oss_url=oss_url,
            file_size_bytes=0,
            status=Status.process,
            parse_mode=app_settings.rag.ocr.parse_mode,
        )
        return knowledge_file_id

    @classmethod
    async def enqueue_parse_job(cls, knowledge_file_id: str):
        knowledge_file = await cls.select_knowledge_file_by_id(knowledge_file_id)
        if not knowledge_file:
            raise ValueError("Knowledge file record not found")

        await knowledge_file_task_manager.enqueue(
            knowledge_file_id,
            cls.run_parse_job(
                knowledge_file_id=knowledge_file.id,
                knowledge_id=knowledge_file.knowledge_id,
                file_name=knowledge_file.file_name,
                oss_url=knowledge_file.oss_url,
            ),
        )

    @classmethod
    async def create_knowledge_file(
        cls,
        file_name: str,
        knowledge_id: str,
        user_id: str,
        oss_url: str,
    ):
        knowledge_file_id = await cls.create_record(file_name, knowledge_id, user_id, oss_url)
        try:
            await cls.enqueue_parse_job(knowledge_file_id)
        except Exception as err:
            await cls.update_parsing_status(
                knowledge_file_id,
                Status.fail,
                error_message=str(err),
                parse_engine="",
                parse_mode=app_settings.rag.ocr.parse_mode,
                finished_at=datetime.utcnow(),
            )
            raise
        return knowledge_file_id

    @classmethod
    async def run_parse_job(
        cls,
        knowledge_file_id: str,
        knowledge_id: str,
        file_name: str,
        oss_url: str,
    ):
        local_file_path = get_save_tempfile(file_name)
        parse_engine = ""
        parse_source = ""
        start_time = time.monotonic()

        try:
            logger.info(f"Start async parsing knowledge file `{file_name}` ({knowledge_file_id})")
            await asyncio.to_thread(storage_client.download_file, oss_url, local_file_path)
            file_size_bytes = os.path.getsize(local_file_path)
            await KnowledgeFileDao.update_knowledge_file(knowledge_file_id, file_size=file_size_bytes)

            parse_start = time.monotonic()
            parse_result = await doc_parser.parse_doc_into_chunks(
                file_id=knowledge_file_id,
                file_path=local_file_path,
                knowledge_id=knowledge_id,
                display_file_name=file_name,
            )
            parse_engine = parse_result.parse_engine
            parse_source = parse_result.parse_source
            parse_duration = time.monotonic() - parse_start

            if await cls.should_ignore_result(knowledge_file_id):
                logger.info(f"Skip indexing deleted knowledge file `{knowledge_file_id}`")
                return

            index_start = time.monotonic()
            await RagHandler.index_milvus_documents(knowledge_id, parse_result.chunks)
            if app_settings.rag.enable_elasticsearch:
                await RagHandler.index_es_documents(knowledge_id, parse_result.chunks)
            index_duration = time.monotonic() - index_start

            await cls.update_parsing_status(
                knowledge_file_id,
                Status.success,
                error_message="",
                parse_engine=parse_engine,
                parse_mode=parse_result.parse_mode,
                finished_at=datetime.utcnow(),
            )
            logger.info(
                f"Knowledge file parsed successfully: file={file_name}, source={parse_source}, "
                f"engine={parse_engine}, parse_seconds={parse_duration:.2f}, index_seconds={index_duration:.2f}, "
                f"total_seconds={time.monotonic() - start_time:.2f}, chunks={len(parse_result.chunks)}"
            )
        except asyncio.CancelledError:
            logger.info(f"Knowledge file parse cancelled: {knowledge_file_id}")
            raise
        except Exception as err:
            if await cls.should_ignore_result(knowledge_file_id):
                logger.info(f"Ignore parsing failure for deleted knowledge file `{knowledge_file_id}`: {err}")
                return

            logger.exception(f"Create Knowledge File Error: {err}")
            await cls.update_parsing_status(
                knowledge_file_id,
                Status.fail,
                error_message=str(err),
                parse_engine=parse_engine or app_settings.rag.ocr.ocr_engine,
                parse_mode=app_settings.rag.ocr.parse_mode,
                finished_at=datetime.utcnow(),
            )
        finally:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

    @classmethod
    async def should_ignore_result(cls, knowledge_file_id: str) -> bool:
        if await knowledge_file_task_manager.should_ignore(knowledge_file_id):
            return True
        knowledge_file = await cls.select_knowledge_file_by_id(knowledge_file_id)
        return knowledge_file is None

    @classmethod
    async def delete_knowledge_file(cls, knowledge_file_id):
        knowledge_file = await cls.select_knowledge_file_by_id(knowledge_file_id)
        if not knowledge_file:
            return

        had_running_task = await knowledge_file_task_manager.mark_deleted(knowledge_file_id)
        await RagHandler.delete_documents_es_milvus(knowledge_file.id, knowledge_file.knowledge_id)
        await KnowledgeFileDao.delete_knowledge_file(knowledge_file_id)

        if not had_running_task:
            await knowledge_file_task_manager.clear_deleted(knowledge_file_id)

    @classmethod
    async def select_knowledge_file_by_id(cls, knowledge_file_id):
        knowledge_file = await KnowledgeFileDao.select_knowledge_file_by_id(knowledge_file_id)
        return knowledge_file

    @classmethod
    async def verify_user_permission(cls, knowledge_file_id, user_id):
        knowledge_file = await cls.select_knowledge_file_by_id(knowledge_file_id)
        if not knowledge_file:
            raise ValueError("知识库文件不存在")
        if user_id not in (AdminUser, knowledge_file.user_id):
            raise ValueError("没有权限访问")

    @classmethod
    async def update_parsing_status(cls, knowledge_file_id, status, **extra_fields):
        return await KnowledgeFileDao.update_parsing_status(knowledge_file_id, status, **extra_fields)
