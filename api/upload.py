import os
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import UploadResponse, BatchUploadItem, BatchUploadResponse
from core.parser import parse_document
from core.chunker import chunk_document
from core.embedder_factory import get_embedder
from core.hybrid_retriever import get_bm25_index
from db.chroma_client import get_chroma_store
from db import repositories as repo
from core.logger import get_logger

logger = get_logger(__name__)


router = APIRouter(prefix="/upload", tags=["文档上传"])


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档，解析切分后存入向量库。

    阶段 1.3 新增：放开图片类型（JPG/PNG/JPEG），由 core/parser.py 走 OCR 识别。
    阶段 2.4 新增：入库成功后写 SQLite 台账（documents 表），/documents/ 列表由此而来。
    阶段 3.3 新增：关键节点日志 + 异常细化（区分业务拒绝/解析失败/入库失败）。
    支持格式汇总：PDF（含扫描型，OCR 回退）、TXT、Markdown、Word、图片。
    """
    # 1. 验证文件类型
    #    图片类型随阶段 1.2/1.3 加入：parser 已支持整图 OCR
    allowed_ext = {".pdf", ".txt", ".md", ".docx", ".jpg", ".jpeg", ".png"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        logger.warning("上传被拒(格式不支持) | 文件=%s 扩展名=%s", file.filename, ext)
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，仅支持 PDF/TXT/Markdown/Word/图片(JPG/PNG)"
        )

    # 2. 保存文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 3. 解析文档
        #    - 普通文档：parser 抽原生文本
        #    - 扫描型 PDF / 图片：parser 内部触发 OCR 兜底（阶段 1.2）
        pages = parse_document(file_path)

        # 解析为空（如纯图片但未识别到任何文字）则提前拒绝，避免空入库
        if not pages:
            # 业务拒绝也要清理已保存文件，不留脏文件
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.warning("上传被拒(无文本内容) | 文件=%s", file.filename)
            raise HTTPException(
                status_code=400,
                detail="文档解析后无任何文本内容（可能为空白图片或识别失败），未入库"
            )

        # 4. 切分 chunk
        chunks = chunk_document(pages, chunk_size=500, overlap=50)

        # 切分后为空（如文字总量不足 min_chunk_size）同样拒绝入库
        if not chunks:
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.warning("上传被拒(切分后为空) | 文件=%s 页数=%d", file.filename, len(pages))
            raise HTTPException(
                status_code=400,
                detail="文档内容过短，切分后无有效片段，未入库"
            )

        # 5. 向量化
        embedder = get_embedder()
        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed(texts)

        # 6. 存入 ChromaDB
        store = get_chroma_store()

        # 6.1 同名重传 -> 先整删旧向量（幂等入库）
        #     原因：add 对已存在 ID 是静默跳过，若新次切出的块数变少
        #     （如 OCR 结果不同），旧的残块会留在库里污染检索。
        #     先 where 删净同 filename 的全部旧块，再新增本次的。
        old_deleted = store.delete_by_filename(file.filename)
        if old_deleted:
            logger.info("同名重传清理旧向量 | 文件=%s | 清理=%d条", file.filename, old_deleted)

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

        # 6.2 BM25 索引失效（混合检索用，下次检索自动重建）
        get_bm25_index().invalidate()

        # 7. 写 SQLite 台账（阶段 2.4）
        #    登记文件名/片段数/上传时间，供 /documents/ 列表读取。
        #    repo.insert_document 是应用层 upsert：同名文件重传时替换旧台账行，
        #    台账的 chunks 数自动对齐本次实际入库的片段数。
        #    注意写台账失败会让本次上传整体报错（走下方 except 清理文件），
        #    避免「向量入了库、台账没登记」的不一致。
        repo.insert_document(
            filename=file.filename,
            chunks=len(chunks),
            user_id="default"  # 预留多用户；当前单用户固定 default
        )

        # 成功：一行结构化日志，含页数/片段数等关键指标，运营排查有据可查
        logger.info(
            "上传入库成功 | 文件=%s | 页数=%d | 片段数=%d",
            file.filename, len(pages), len(chunks)
        )

        return UploadResponse(
            message="文档上传成功",
            filename=file.filename,
            chunks=len(chunks)
        )

    except HTTPException:
        # 业务校验抛出的 HTTP 异常（如上面的 400）应原样上抛，
        # 否则会被下面的 except Exception 捕获并误包装成 500
        # （具体拒绝原因在上面各自分支已记日志）
        raise
    except Exception as e:
        # 内部错误：清理已保存的文件 + 记完整堆栈。
        # 不再把 str(e) 直接透给前端（可能含内部路径/库报错细节），
        # 全局异常处理器(main.py)会兜底返回统一 500 结构
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.exception("上传失败(内部错误) | 文件=%s | 已清理本地文件", file.filename)
        raise HTTPException(status_code=500, detail="文档处理失败，请检查文件是否损坏或稍后重试")


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    批量上传文档（阶段 4.1 新增）

    与单文件接口的区别：
        - 一次提交多个文件，逐文件独立处理
        - 单个文件失败不影响其他文件（结果按文件隔离返回）
        - 整体 HTTP 恒为 200，成败看每个文件自己的 status 字段；
          这样前端一次拿到全部结果，不会因第一个失败就中断

    实现方式：内部逐个调用单文件接口的处理函数（upload_document），
    捕获其 HTTPException 转成该文件的 failed 结果。
    """
    results: List[BatchUploadItem] = []
    succeeded = 0

    logger.info("批量上传开始 | 文件数=%d", len(files))

    for f in files:
        try:
            # 复用单文件接口：校验/解析/切分/向量化/入库/记账 全链路一致
            resp = await upload_document(f)
            results.append(BatchUploadItem(
                filename=resp.filename,
                status="success",
                chunks=resp.chunks,
                message=resp.message,
            ))
            succeeded += 1
        except HTTPException as e:
            # 该文件被拒（400/500）：记入失败项，继续处理下一个
            results.append(BatchUploadItem(
                filename=f.filename or "(未知文件名)",
                status="failed",
                chunks=0,
                message=str(e.detail),
            ))
        except Exception as e:
            # 理论上 upload_document 已把异常转为 HTTPException，这里防漏网
            logger.exception("批量上传: 未预期异常 | 文件=%s", f.filename)
            results.append(BatchUploadItem(
                filename=f.filename or "(未知文件名)",
                status="failed",
                chunks=0,
                message=f"处理异常: {type(e).__name__}",
            ))

    failed = len(files) - succeeded
    logger.info("批量上传完成 | 总数=%d | 成功=%d | 失败=%d",
                len(files), succeeded, failed)

    return BatchUploadResponse(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )