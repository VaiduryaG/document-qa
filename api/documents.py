from fastapi import APIRouter, HTTPException
from models.schemas import DocumentListResponse, DocumentInfo
from db.chroma_client import get_chroma_store
import os


router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    列出所有已上传文档
    """
    store = get_chroma_store()
    all_docs = store.get_all()
    
    # 按文件名聚合 chunk
    doc_map = {}
    for meta in all_docs.get("metadatas", []):
        if not meta:
            continue
        filename = meta.get("filename", meta.get("source", "unknown"))
        if filename not in doc_map:
            doc_map[filename] = {
                "chunks": 0,
                "source": meta.get("source", filename)
            }
        doc_map[filename]["chunks"] += 1
    
    documents = [
        DocumentInfo(
            id=filename,
            filename=filename,
            chunks=info["chunks"]
        )
        for filename, info in doc_map.items()
    ]
    
    return DocumentListResponse(
        documents=documents,
        total=len(documents)
    )


@router.delete("/{filename}")
async def delete_document(filename: str):
    """
    删除指定文档及其向量
    """
    store = get_chroma_store()
    all_docs = store.get_all()
    
    # 找到该文件的所有 chunk id
    ids_to_delete = []
    for doc_id, meta in zip(all_docs.get("ids", []), all_docs.get("metadatas", [])):
        if meta and meta.get("filename") == filename:
            ids_to_delete.append(doc_id)
    
    if not ids_to_delete:
        raise HTTPException(status_code=404, detail=f"文档不存在: {filename}")
    
    # 删除向量
    store.delete(ids_to_delete)
    
    # 删除本地文件
    file_path = os.path.join("uploaded_files", filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return {"message": f"已删除文档: {filename}", "deleted_chunks": len(ids_to_delete)}