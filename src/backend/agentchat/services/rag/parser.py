import asyncio
import os
from dataclasses import dataclass

from agentchat.core.models.manager import ModelManager
from agentchat.schema.chunk import ChunkModel
from agentchat.services.rag.doc_parser.artifact import ParsedDocumentArtifact
from agentchat.services.rag.doc_parser.docx import docx_parser
from agentchat.services.rag.doc_parser.excel import excel_to_txt
from agentchat.services.rag.doc_parser.markdown import markdown_parser
from agentchat.services.rag.doc_parser.ocr import ocr_parser
from agentchat.services.rag.doc_parser.other_file import other_file_to_txt
from agentchat.services.rag.doc_parser.pdf import pdf_parser
from agentchat.services.rag.doc_parser.pptx import pptx_parser
from agentchat.services.rag.doc_parser.text import text_parser
from agentchat.settings import app_settings

IMAGE_SUFFIXES = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
TEXT_LIKE_SUFFIXES = {"txt", "json", "html", "htm", "csv"}
EXCEL_SUFFIXES = {"xls", "xlsx"}


@dataclass
class DocumentParseResult:
    chunks: list
    parse_engine: str
    parse_source: str
    parse_mode: str
    page_count: int = 0


class DocParser:
    @classmethod
    async def parse_doc_into_chunks(
        cls,
        file_id,
        file_path,
        knowledge_id,
        display_file_name=None,
        max_concurrent_tasks=5,
    ) -> DocumentParseResult:
        file_suffix = file_path.split(".")[-1].lower()
        parse_engine = "text"
        parse_source = "native_text"
        page_count = 0

        if file_suffix == "md":
            chunks = await markdown_parser.parse_into_chunks(file_id, file_path, knowledge_id)
            parse_engine = "markdown"
            parse_source = "markdown"
        elif file_suffix == "txt":
            chunks = await text_parser.parse_into_chunks(file_id, file_path, knowledge_id)
        elif file_suffix == "docx":
            artifact = await docx_parser.prepare_document(file_path)
            chunks = await cls._parse_artifact(file_id, artifact, knowledge_id)
            parse_engine = artifact.parse_engine
            parse_source = artifact.parse_source
            page_count = artifact.page_count
        elif file_suffix == "pdf":
            artifact = await pdf_parser.prepare_document(file_path)
            chunks = await cls._parse_artifact(file_id, artifact, knowledge_id)
            parse_engine = artifact.parse_engine
            parse_source = artifact.parse_source
            page_count = artifact.page_count
        elif file_suffix == "pptx":
            artifact = await pptx_parser.prepare_document(file_path)
            chunks = await cls._parse_artifact(file_id, artifact, knowledge_id)
            parse_engine = artifact.parse_engine
            parse_source = artifact.parse_source
            page_count = artifact.page_count
        elif file_suffix in IMAGE_SUFFIXES:
            artifact = await asyncio.wait_for(
                ocr_parser.convert_image_to_markdown(file_path),
                timeout=app_settings.rag.ocr.ocr_timeout_seconds,
            )
            chunks = await cls._parse_artifact(file_id, artifact, knowledge_id)
            parse_engine = artifact.parse_engine
            parse_source = artifact.parse_source
            page_count = artifact.page_count
        elif file_suffix in EXCEL_SUFFIXES:
            new_file_path = excel_to_txt(file_path)
            chunks = await text_parser.parse_into_chunks(file_id, new_file_path, knowledge_id)
            parse_engine = "excel_to_text"
            parse_source = "native_text"
        elif file_suffix in TEXT_LIKE_SUFFIXES:
            new_file_path = other_file_to_txt(file_path)
            chunks = await text_parser.parse_into_chunks(file_id, new_file_path, knowledge_id)
            parse_engine = "text_normalizer"
            parse_source = "native_text"
        else:
            raise ValueError(f"Unsupported file suffix for RAG parsing: {file_suffix}")

        cls._apply_chunk_metadata(
            chunks=chunks,
            display_file_name=display_file_name or os.path.basename(file_path),
            parse_source=parse_source,
            page_range=f"1-{page_count}" if page_count else "",
        )

        if app_settings.rag.enable_summary:
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [asyncio.create_task(cls.generate_summary(chunk, semaphore)) for chunk in chunks]
            chunks = await asyncio.gather(*tasks)

        return DocumentParseResult(
            chunks=chunks,
            parse_engine=parse_engine,
            parse_source=parse_source,
            parse_mode=app_settings.rag.ocr.parse_mode,
            page_count=page_count,
        )

    @classmethod
    async def _parse_artifact(cls, file_id: str, artifact: ParsedDocumentArtifact, knowledge_id: str):
        if artifact.parser_kind == "markdown":
            return await markdown_parser.parse_into_chunks(file_id, artifact.parsed_file_path, knowledge_id)
        if artifact.parser_kind == "text":
            return await text_parser.parse_into_chunks(file_id, artifact.parsed_file_path, knowledge_id)
        raise ValueError(f"Unsupported parser kind: {artifact.parser_kind}")

    @classmethod
    def _apply_chunk_metadata(
        cls,
        chunks: list[ChunkModel],
        display_file_name: str,
        parse_source: str,
        page_range: str,
    ):
        for chunk in chunks:
            chunk.file_name = display_file_name
            chunk.parse_source = parse_source
            chunk.page_range = page_range

    @classmethod
    async def generate_summary(cls, chunk: ChunkModel, semaphore):
        async_client = ModelManager.get_conversation_model()

        async with semaphore:
            prompt = f"""
                你是一个专业的摘要生成助手，请根据以下要求为文本生成一段摘要：
                ## 需要总结的文本：
                {chunk.content}
                ## 要求：
                1. 摘要字数控制在 100 字左右。
                2. 摘要中仅包含文字和字母，不得出现链接或其他特殊符号。
                3. 只输出摘要部分，不要输出额外说明。
            """
            response = await async_client.ainvoke(prompt)
            chunk.summary = response.content
            return chunk


doc_parser = DocParser()
