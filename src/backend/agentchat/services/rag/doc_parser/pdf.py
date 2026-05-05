import asyncio
import os
import pathlib
from urllib.parse import urljoin

import aiofiles
import pymupdf4llm
from loguru import logger

from agentchat.services.rag.doc_parser.artifact import ParsedDocumentArtifact
from agentchat.services.rag.doc_parser.markdown import markdown_parser
from agentchat.services.rag.doc_parser.ocr import ocr_parser
from agentchat.services.rewrite.markdown_rewrite import markdown_rewriter
from agentchat.services.storage import storage_client
from agentchat.settings import app_settings
from agentchat.utils.file_utils import (
    generate_unique_filename,
    get_convert_markdown_images_dir,
    get_object_storage_base_path,
)


class PDFParser:
    async def convert_markdown(self, file_path: str) -> str:
        markdown_dir, images_dir = get_convert_markdown_images_dir()
        md_text_words = pymupdf4llm.to_markdown(
            doc=file_path,
            write_images=True,
            image_path=images_dir,
            image_format="png",
            dpi=300,
        )
        markdown_output_path = os.path.join(markdown_dir, generate_unique_filename(file_path, "md"))
        pathlib.Path(markdown_output_path).write_bytes(md_text_words.encode())
        logger.info(f"PDF converted to markdown: {markdown_output_path}")

        file_upload_url_map = await self.upload_folder_to_oss(images_dir)
        await markdown_rewriter.run_rewrite(markdown_output_path, file_upload_url_map)
        await self.upload_file_to_oss(markdown_output_path)
        return markdown_output_path

    async def prepare_document(self, file_path: str) -> ParsedDocumentArtifact:
        page_count = ocr_parser.count_pdf_pages(file_path)
        if app_settings.rag.ocr.enable_ocr and ocr_parser.pdf_requires_ocr(file_path):
            artifact = await asyncio.wait_for(
                ocr_parser.convert_document_to_markdown(file_path),
                timeout=app_settings.rag.ocr.ocr_timeout_seconds,
            )
            artifact.page_count = page_count
            logger.info(f"PDF `{os.path.basename(file_path)}` routed to MinerU OCR")
            return artifact

        markdown_path = await self.convert_markdown(file_path)
        logger.info(f"PDF `{os.path.basename(file_path)}` routed to native markdown extraction")
        return ParsedDocumentArtifact(
            parsed_file_path=markdown_path,
            parser_kind="markdown",
            parse_engine="pymupdf4llm",
            parse_source="native_pdf",
            page_count=page_count,
        )

    async def parse_into_chunks(self, file_id, file_path, knowledge_id):
        artifact = await self.prepare_document(file_path)
        return await markdown_parser.parse_into_chunks(file_id, artifact.parsed_file_path, knowledge_id)

    async def upload_file_to_oss(self, file_path):
        async with aiofiles.open(file_path, "rb") as file:
            file_content = await file.read()
            oss_object_name = get_object_storage_base_path(os.path.basename(file_path))
            sign_url = urljoin(app_settings.storage.active.base_url, oss_object_name)
            storage_client.sign_url_for_get(oss_object_name)
            storage_client.upload_file(oss_object_name, file_content)
            return sign_url

    async def upload_folder_to_oss(self, file_dir):
        if not os.path.exists(file_dir):
            return {}

        file_names = os.listdir(file_dir)
        tasks = [self.upload_file_to_oss(os.path.join(file_dir, file_name)) for file_name in file_names]
        file_upload_url_map = {}
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for file_name, result in zip(file_names, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to upload markdown image `{file_name}`: {result}")
            else:
                file_upload_url_map[file_name] = result
                logger.info(f"Uploaded markdown image `{file_name}`")

        return file_upload_url_map


pdf_parser = PDFParser()
