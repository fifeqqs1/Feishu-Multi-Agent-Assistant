from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    model_name: str = ""
    api_key: str = ""
    base_url: str = ""


class MultiModels(BaseModel):
    class Config:
        extra = "allow"

    reasoning_model: ModelConfig = Field(default_factory=ModelConfig)
    conversation_model: ModelConfig = Field(default_factory=ModelConfig)
    tool_call_model: ModelConfig = Field(default_factory=ModelConfig)
    qwen3_coder: ModelConfig = Field(default_factory=ModelConfig)
    qwen_vl: ModelConfig = Field(default_factory=ModelConfig)
    text2image: ModelConfig = Field(default_factory=ModelConfig)
    embedding: ModelConfig = Field(default_factory=ModelConfig)
    rerank: ModelConfig = Field(default_factory=ModelConfig)


class Tools(BaseModel):
    class Config:
        extra = "allow"

    weather: dict = Field(default_factory=dict)
    tavily: dict = Field(default_factory=dict)
    google: dict = Field(default_factory=dict)
    delivery: dict = Field(default_factory=dict)
    bocha: dict = Field(default_factory=dict)


class OCROptions(BaseModel):
    enable_ocr: bool = Field(default=False)
    ocr_engine: str = Field(default="mineru")
    parse_mode: str = Field(default="async")
    pdf_text_threshold: int = Field(default=80)
    ocr_timeout_seconds: int = Field(default=300)
    ocr_max_pages: int = Field(default=100)
    ocr_lang: str = Field(default="auto")


class Rag(BaseModel):
    class Config:
        extra = "allow"

    enable_elasticsearch: bool = Field(default=False)
    enable_summary: bool = Field(default=False)
    retrival: dict = Field(default_factory=dict)
    split: dict = Field(default_factory=dict)
    elasticsearch: dict = Field(default_factory=dict)
    vector_db: dict = Field(default_factory=dict)
    ocr: OCROptions = Field(default_factory=OCROptions)


class MemoryOptions(BaseModel):
    enable_redis_cache: bool = Field(default=True)
    redis_ttl_seconds: int = Field(default=86400)
    recent_history_pairs: int = Field(default=4)
    max_history_messages: int = Field(default=12)
    history_compaction_threshold_tokens: int = Field(default=1800)
    max_history_context_tokens: int = Field(default=2400)
    summary_max_tokens: int = Field(default=1200)
    semantic_session_recall_limit: int = Field(default=4)
    semantic_global_recall_limit: int = Field(default=4)


class OSSConfig(BaseModel):
    access_key_id: str
    access_key_secret: str
    endpoint: str
    bucket_name: str
    base_url: str


class MinioConfig(BaseModel):
    access_key_id: str
    access_key_secret: str
    endpoint: str
    bucket_name: str
    base_url: str


class StorageConfig(BaseModel):
    mode: Literal["oss", "minio"]
    oss: Optional[OSSConfig] = None
    minio: Optional[MinioConfig] = None

    @model_validator(mode="after")
    def validate_storage(self):
        if self.mode == "oss" and not self.oss:
            raise ValueError("mode=oss requires oss config")
        if self.mode == "minio" and not self.minio:
            raise ValueError("mode=minio requires minio config")
        return self

    @property
    def active(self):
        return self.oss if self.mode == "oss" else self.minio
