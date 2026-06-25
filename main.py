from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import upload, chat, documents


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