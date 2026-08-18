import os

from fastapi import APIRouter, HTTPException

from models.schemas import DocumentListResponse, DocumentInfo
from db.chroma_client import get_chroma_store
from db import repositories as repo
from core.logger import get_logger

logger = get_logger(__name__)


router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    列出所有已上传文档（阶段 2.3：改从 SQLite 台账读取）

    之前做法的缺陷（已废弃）：
        把 ChromaDB 全部 chunk 的 metadata 捞出来再人工聚合反推文档列表，
        查不到上传时间，且文档多了以后低效。
    现在做法：
        直接读 documents 表（含 uploaded_at），一条 SQL 搞定；
        ChromaDB 只负责向量检索，不再兼职做台账。
    """
    # repo.list_documents 返回 [{"id","filename","user_id","chunks","uploaded_at"},...]
    # 按上传时间倒序（最新在前）
    docs = repo.list_documents()

    documents = [
        DocumentInfo(
            id=str(d["filename"]),          # 兼容前端以 filename 作删除键
            filename=d["filename"],
            chunks=d["chunks"],
            uploaded_at=d["uploaded_at"]    # 新增：上传时间（此前 ChromaDB 里查不到）
        )
        for d in docs
    ]

    return DocumentListResponse(
        documents=documents,
        total=len(documents)
    )


@router.delete("/{filename}")
async def delete_document(filename: str):
    """
    删除指定文档：三处同步清理（阶段 2.3 升级）

    清理顺序（失败即中止，保持事务性直觉）：
        1. ChromaDB 向量（检索数据源，最大头）
        2. SQLite 台账行
        3. 本地文件（uploaded_files/ 下的原件）

    说明：
        以 ChromaDB 是否存在该文档的 chunk 作为「文档是否存在」的判据；
        SQLite 台账和本地文件即使个别缺失也容错跳过（幂等删除，不因
        历史遗留脏数据而报错）。
    """
    store = get_chroma_store()
    all_docs = store.get_all()

    # 找到该文件的所有 chunk id
    ids_to_delete = []
    for doc_id, meta in zip(all_docs.get("ids", []), all_docs.get("metadatas", [])):
        if meta and meta.get("filename") == filename:
            ids_to_delete.append(doc_id)

    # 台账也不存在 -> 确认文档从未登记过，返回 404
    db_doc = repo.get_document(filename)
    if not ids_to_delete and db_doc is None:
        logger.warning("删除被拒(不存在) | 文件=%s", filename)
        raise HTTPException(status_code=404, detail=f"文档不存在: {filename}")

    # 1. 删除向量（若存在）
    if ids_to_delete:
        store.delete(ids_to_delete)

    # 2. 删除 SQLite 台账行（返回实际删除行数，0 也容错）
    deleted_rows = repo.delete_document(filename)

    # 3. 删除本地文件（存在才删）
    file_path = os.path.join("uploaded_files", filename)
    file_removed = False
    if os.path.exists(file_path):
        os.remove(file_path)
        file_removed = True

    # 三处清理结果一行日志说清；单边残留时 n 为 0，排查时一目了然
    logger.info(
        "文档已删除 | 文件=%s | 向量=%d条 | 台账=%d行 | 本地文件=%s",
        filename, len(ids_to_delete), deleted_rows, "已删" if file_removed else "本就不存在"
    )

    return {
        "message": f"已删除文档: {filename}",
        "deleted_chunks": len(ids_to_delete),
        "deleted_db_rows": deleted_rows,   # 新增：清理的台账行数（0/1）
        "file_removed": file_removed       # 新增：本地文件是否删除（缺文件时 False）
    }
