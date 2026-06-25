from pydantic import BaseModel
from typing import List, Optional, Dict


class UploadResponse(BaseModel):
    """上传文档响应"""
    message: str
    filename: str
    chunks: int
    status: str = "success"


class ChatRequest(BaseModel):
    """问答请求"""
    question: str
    top_k: int = 3


class Source(BaseModel):
    """答案来源"""
    content: str
    page: int
    source: str
    score: float


class ChatResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: List[Source]
    status: str = "success"


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    filename: str
    chunks: int
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocumentInfo]
    total: int


class ErrorResponse(BaseModel):
    """错误响应"""
    detail: str
    status: str = "error"