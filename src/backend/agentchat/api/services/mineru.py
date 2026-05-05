from agentchat.services.rag.doc_parser.ocr import ocr_parser


async def convert_pdf_to_markdown(path, output_dir=None, method="auto", lang=None, debug_able=False, start_page_id=0, end_page_id=None):
    artifact = await ocr_parser.convert_document_to_markdown(path)
    return artifact.parsed_file_path
