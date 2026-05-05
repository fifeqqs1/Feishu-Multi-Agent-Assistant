from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import Column, DateTime, Text, text
from sqlmodel import Field

from agentchat.database.models.base import SQLModelSerializable


class Status:
    fail = "fail"
    process = "process"
    success = "success"


class KnowledgeFileTable(SQLModelSerializable, table=True):
    __tablename__ = "knowledge_file"

    id: str = Field(default=uuid4().hex, description="知识库文件 id", primary_key=True)
    file_name: str = Field(index=True, description="知识库文件名")
    knowledge_id: str = Field(index=True, description="知识库 ID")
    status: str = Field(default=Status.process, description="解析状态")
    user_id: str = Field(index=True, description="用户 ID")
    oss_url: str = Field(default="", description="对象存储中的文件路径")
    file_size: int = Field(default=0, description="文件大小，单位字节")
    error_message: Optional[str] = Field(
        default="",
        sa_column=Column(Text, nullable=True),
        description="解析失败时的错误信息",
    )
    parse_engine: str = Field(default="", description="解析引擎，例如 mineru / pymupdf4llm")
    parse_mode: str = Field(default="sync", description="解析模式，例如 async / sync")
    finished_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime, nullable=True),
        description="解析完成时间",
    )
    update_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
            onupdate=text("CURRENT_TIMESTAMP"),
        ),
        description="修改时间",
    )
    create_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
        description="创建时间",
    )
