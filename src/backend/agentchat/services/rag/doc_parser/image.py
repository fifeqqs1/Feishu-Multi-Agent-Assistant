from agentchat.services.rag.doc_parser.ocr import ocr_parser


async def image_to_txt(image_path: str):
    artifact = await ocr_parser.convert_image_to_markdown(image_path)
    return artifact.parsed_file_path
