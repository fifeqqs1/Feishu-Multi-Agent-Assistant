from dataclasses import dataclass


@dataclass
class ParsedDocumentArtifact:
    parsed_file_path: str
    parser_kind: str
    parse_engine: str
    parse_source: str
    page_count: int = 0
    page_range: str = ""
