import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import UploadResponse
from core.parser import parse_document
from core.chunker import chunk_document
from core.embedder_factory import get_embedder
from db.chroma_client import get_chroma_store


router = APIRouter(prefix="/upload", tags=["文档上传"])


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档（PDF/TXT），解析切分后存入向量库
    """
    # 1. 验证文件类型
    allowed_ext = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，仅支持 PDF/TXT"
        )
    
    # 2. 保存文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 3. 解析文档
        pages = parse_document(file_path)
        
        # 4. 切分 chunk
        chunks = chunk_document(pages, chunk_size=500, overlap=50)
        
        # 5. 向量化
        embedder = get_embedder()
        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed(texts)
        
        # 6. 存入 ChromaDB
        store = get_chroma_store()
        
        ids = [f"{file.filename}_chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [{
            "page": c["page"],
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "filename": file.filename
        } for c in chunks]
        
        store.add(
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return UploadResponse(
            message="文档上传成功",
            filename=file.filename,
            chunks=len(chunks)
        )
        
    except Exception as e:
        # 清理失败文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))