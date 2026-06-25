from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse, Source
from core.retriever import DocumentRetriever
from core.prompt_builder import build_rag_prompt
from core.llm import get_llm


router = APIRouter(prefix="/chat", tags=["智能问答"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    文档问答接口
    """
    try:
        # 1. 检索相关 chunk
        retriever = DocumentRetriever(top_k=request.top_k)
        chunks = retriever.retrieve(request.question)
        
        # 2. 构建 Prompt
        prompt = build_rag_prompt(request.question, chunks)
        
        # 3. 调用 LLM
        llm = get_llm()
        answer = llm.chat(prompt, temperature=0.3)
        
        # 4. 格式化来源
        sources = [
            Source(
                content=c["content"],
                page=c["page"],
                source=c["source"],
                score=c["score"]
            )
            for c in chunks
        ]
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def chat_get(question: str, top_k: int = 3):
    """
    GET 方式问答（方便浏览器测试）
    """
    return await chat(ChatRequest(question=question, top_k=top_k))