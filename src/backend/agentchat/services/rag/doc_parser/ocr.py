import os
import shutil
import tempfile
from pathlib import Path

import fitz
from loguru import logger

from agentchat.services.rag.doc_parser.artifact import ParsedDocumentArtifact
from agentchat.settings import app_settings
from agentchat.utils.file_utils import generate_unique_filename, get_markdown_dir


class DocumentOCRService:
    PDF_SUFFIXES = {"pdf"}
    OFFICE_SUFFIXES = {"doc", "docx", "odt", "rtf", "txt", "html", "htm", "ppt", "pptx", "odp"}
    IMAGE_SUFFIXES = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}

    def _load_mineru_dependencies(self):
        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataReader
            from magic_pdf.tools.cli import image_suffixes, ms_office_suffixes, pdf_suffixes
            from magic_pdf.tools.common import do_parse
            from magic_pdf.utils.office_to_pdf import convert_file_to_pdf as mineru_convert_file_to_pdf
        except ImportError as err:
            raise RuntimeError(
                "MinerU OCR dependencies are missing. Please install the magic_pdf package before using OCR."
            ) from err

        return FileBasedDataReader, image_suffixes, ms_office_suffixes, pdf_suffixes, do_parse, mineru_convert_file_to_pdf

    def is_ocr_candidate(self, file_path: str) -> bool:
        suffix = Path(file_path).suffix.lower().lstrip(".")
        return suffix in (self.PDF_SUFFIXES | self.OFFICE_SUFFIXES | self.IMAGE_SUFFIXES)

    def count_pdf_pages(self, file_path: str) -> int:
        with fitz.open(file_path) as document:
            return document.page_count

    def pdf_requires_ocr(self, file_path: str) -> bool:
        threshold = app_settings.rag.ocr.pdf_text_threshold
        max_pages = app_settings.rag.ocr.ocr_max_pages

        with fitz.open(file_path) as document:
            if document.page_count > max_pages:
                raise ValueError(f"PDF page count {document.page_count} exceeds OCR max pages limit {max_pages}")

            page_text_lengths = []
            for page in document:
                text = page.get_text("text") or ""
                page_text_lengths.append(len(text.strip()))

        if not page_text_lengths:
            return True

        non_empty_pages = sum(1 for item in page_text_lengths if item >= threshold)
        average_length = sum(page_text_lengths) / len(page_text_lengths)
        return non_empty_pages == 0 or average_length < threshold

    def _load_document_bytes(self, source_path: Path) -> tuple[bytes, str]:
        (
            FileBasedDataReader,
            image_suffixes,
            ms_office_suffixes,
            pdf_suffixes,
            _,
            mineru_convert_file_to_pdf,
        ) = self._load_mineru_dependencies()
        temp_dir = tempfile.mkdtemp()

        try:
            if source_path.suffix.lower() in ms_office_suffixes:
                mineru_convert_file_to_pdf(str(source_path), temp_dir)
                prepared_file = os.path.join(temp_dir, f"{source_path.stem}.pdf")
            elif source_path.suffix.lower() in image_suffixes:
                with open(str(source_path), "rb") as file:
                    bits = file.read()
                pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
                prepared_file = os.path.join(temp_dir, f"{source_path.stem}.pdf")
                with open(prepared_file, "wb") as file:
                    file.write(pdf_bytes)
            elif source_path.suffix.lower() in pdf_suffixes:
                prepared_file = str(source_path)
            else:
                raise ValueError(f"Unsupported OCR file suffix: {source_path.suffix}")

            disk_reader = FileBasedDataReader(os.path.dirname(prepared_file))
            return disk_reader.read(os.path.basename(prepared_file)), temp_dir
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _pick_markdown_output(self, output_dir: str, stem_name: str) -> str:
        markdown_files = sorted(Path(output_dir).rglob("*.md"))
        if not markdown_files:
            raise ValueError(f"MinerU did not generate markdown output for {stem_name}")

        merged_path = os.path.join(get_markdown_dir(), generate_unique_filename(f"{stem_name}.md"))
        with open(merged_path, "w", encoding="utf-8") as merged_file:
            for markdown_file in markdown_files:
                merged_file.write(markdown_file.read_text(encoding="utf-8"))
                merged_file.write("\n\n")
        return merged_path

    async def convert_document_to_markdown(self, file_path: str) -> ParsedDocumentArtifact:
        _, _, _, _, do_parse, _ = self._load_mineru_dependencies()
        output_dir = tempfile.mkdtemp()
        source_path = Path(file_path)
        lang = None if app_settings.rag.ocr.ocr_lang == "auto" else app_settings.rag.ocr.ocr_lang
        file_bytes, temp_dir = self._load_document_bytes(source_path)

        try:
            do_parse(
                output_dir,
                source_path.stem,
                file_bytes,
                [],
                "auto",
                False,
                start_page_id=0,
                end_page_id=None,
                lang=lang,
            )
            markdown_path = self._pick_markdown_output(output_dir, source_path.stem)
            page_count = 0
            if source_path.suffix.lower().lstrip(".") == "pdf":
                page_count = self.count_pdf_pages(str(source_path))
            logger.info(f"OCR markdown generated for `{source_path.name}` via MinerU")
            return ParsedDocumentArtifact(
                parsed_file_path=markdown_path,
                parser_kind="markdown",
                parse_engine=app_settings.rag.ocr.ocr_engine,
                parse_source="mineru_ocr",
                page_count=page_count,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    async def convert_image_to_markdown(self, file_path: str) -> ParsedDocumentArtifact:
        artifact = await self.convert_document_to_markdown(file_path)
        artifact.page_count = 1
        return artifact


ocr_parser = DocumentOCRService()
