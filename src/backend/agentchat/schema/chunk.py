class ChunkModel:
    def __init__(
        self,
        chunk_id,
        content,
        file_id,
        file_name,
        update_time,
        knowledge_id,
        summary="",
        parse_source="",
        page_range="",
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.file_id = file_id
        self.file_name = file_name
        self.update_time = update_time
        self.knowledge_id = knowledge_id
        self.summary = summary
        self.parse_source = parse_source
        self.page_range = page_range

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "file_id": self.file_id,
            "file_name": self.file_name,
            "knowledge_id": self.knowledge_id,
            "update_time": self.update_time,
            "summary": self.summary,
            "parse_source": self.parse_source,
            "page_range": self.page_range,
        }
