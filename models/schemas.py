from pydantic import BaseModel
from typing import List, Optional, Dict


class UploadResponse(BaseModel):
    """上传文档响应"""
    message: str
    filename: str
    chunks: int
    status: str = "success"


class BatchUploadItem(BaseModel):
    """批量上传中单个文件的结果（成功与失败统一结构，前端好渲染）"""
    filename: str
    status: str              # "success" / "failed"
    chunks: int = 0          # 失败时为 0
    message: str = ""        # 失败原因 / 成功提示


class BatchUploadResponse(BaseModel):
    """批量上传响应"""
    total: int               # 提交的文件总数
    succeeded: int           # 成功数
    failed: int              # 失败数
    results: List[BatchUploadItem]  # 逐文件结果


class ChatRequest(BaseModel):
    """
    问答请求

    阶段 2.5 新增两个可选字段（都有默认值，向后兼容）：
        - 不传 -> 行为与从前完全一致（单轮问答、无持久化归属）
        - 传了 -> 阶段 2.6 的 chat 接口据此做对话历史持久化：
                  同一 conversation_id 的多轮问答会被存入 SQLite 并
                  自动作为上下文带入后续提问
    """
    question: str
    top_k: int = 3
    # 用户标识：预留多用户；默认 default 单用户即可跑通
    user_id: Optional[str] = "default"
    # 对话标识：同一会话的多轮问题传同一个 id（前端生成，如 "conv-abc123"）；
    # 为 None 时本轮不关联任何历史会话（每次独立）
    conversation_id: Optional[str] = None


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
    # 阶段 2.5 新增：回显本轮实际归属的对话标识。
    # 请求没传 conversation_id 时，chat 接口（2.6）会自动生成一个并回传，
    # 前端拿它继续下一轮，即实现"自动开启新会话"
    conversation_id: Optional[str] = None


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