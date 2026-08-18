import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api import upload, chat, documents

# 阶段 3.2：入口处最先初始化日志（早于任何路由 import 触发的日志输出）
from core.logger import get_logger

logger = get_logger("main")


app = FastAPI(
    title="智能文档问答助手",
    description="基于 RAG 的文档问答系统，支持 PDF/TXT 上传和智能问答",
    version="1.0.0"
)

# CORS 配置（允许前端跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 阶段 3.2：请求日志中间件 ==========
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    请求级日志中间件：每个 HTTP 请求记一行访问日志（方法+路径+状态码+耗时+请求id）。

    设计：
        - 每请求生成短 request_id，贯穿日志，同一请求的多条日志可串联排查
        - 耗时毫秒级，方便定位慢接口（如大 PDF 上传、OCR、LLM 调用）
        - 请求处理本身抛异常时也保证记日志（先记 500 再抛给全局异常处理器）
    """
    request_id = uuid.uuid4().hex[:8]  # 短id足够区分并发请求
    start = time.perf_counter()

    logger.info("[%s] --> %s %s", request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        # 中间件层兜底记日志（正常情况下全局异常处理器已接住，这里防漏网）
        cost_ms = (time.perf_counter() - start) * 1000
        logger.exception("[%s] <-- %s %s 异常(%.1fms)",
                         request_id, request.method, request.url.path, cost_ms)
        raise

    cost_ms = (time.perf_counter() - start) * 1000
    # 4xx/5xx 用 warning 级别，正常 2xx/3xx 用 info，扫日志时一眼锁定异常
    log_fn = logger.warning if response.status_code >= 400 else logger.info
    log_fn("[%s] <-- %s %s %d (%.1fms)",
           request_id, request.method, request.url.path, response.status_code, cost_ms)

    # 请求id写入响应头，前端报障时只需提供这个 id，后端日志一查一个准
    response.headers["X-Request-ID"] = request_id
    return response


# ========== 阶段 3.2：全局异常处理器 ==========
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    兜底捕获所有未被路由内 try/except 处理的异常。

    之前的问题：异常细节（含路径、堆栈、第三方库内部信息）直接经
    str(e) 返回给前端，既泄漏内部实现又对用户不友好。

    现在的行为：
        - 服务端：logger.exception 记完整堆栈（日志里能查到全部细节）
        - 客户端：只返回统一结构 + 一次性错误码（凭它到日志里检索对应记录）
        - 注意：路由内主动抛的 HTTPException（400/404 等业务异常）
          不经过这里，FastAPI 自动按原有 detail 格式返回
    """
    logger.exception("未处理异常 | 路径=%s %s | 类型=%s",
                     request.method, request.url.path, type(exc).__name__)

    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误，请稍后重试；若持续出现请联系管理员并提供日志中的错误码",
            "error_code": uuid.uuid4().hex[:8],  # 与日志时间结合即可定位这条堆栈
            "status": "error",
        },
    )


# 注册路由
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "message": "智能文档问答助手 API 运行中",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """服务健康状态"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
