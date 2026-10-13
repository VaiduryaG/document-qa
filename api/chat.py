import uuid

from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse, Source
from core.hybrid_retriever import HybridRetriever
from core.prompt_builder import PromptBuilder
from core.llm import get_llm
from db import repositories as repo
from core.logger import get_logger

logger = get_logger(__name__)


router = APIRouter(prefix="/chat", tags=["智能问答"])

# Prompt 构建器单例（无状态，可安全复用）
_builder = PromptBuilder()

# 多轮对话加载的最大历史条数。
# 太多会撑爆 LLM 上下文且费用高；20 条（约10轮问答）足够维持指代关系
MAX_HISTORY_MESSAGES = 20


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    文档问答接口（阶段 2.6：对话历史持久化）

    流程对比：
        旧：每次问答独立，多轮之间无上下文，历史只在前端 session 里
        新：同一 conversation_id 的多轮问答 ——
            1) 从 SQLite 加载该会话的历史消息（最多 MAX_HISTORY_MESSAGES 条）
            2) 历史拼进 Prompt（build_with_history），LLM 能接上"它/这个"等指代
            3) 本轮 user 问题 + assistant 回答写回 SQLite
            4) 重启服务历史不丢（此前前端一刷新就没了）

    conversation_id 约定：
        请求未传 -> 自动生成新会话 id 并在响应中回传，前端下一轮带上即延续会话
        请求传了 -> 沿用该会话，继续累积上下文

    阶段 3.4 新增：关键节点日志（会话/检索/历史/耗时）+ 异常不泄漏内部细节。
    """
    try:
        # 0. 确定会话标识：没传则生成新的（形如 conv-3f2a...）
        conversation_id = request.conversation_id or f"conv-{uuid.uuid4().hex[:12]}"

        # 确保会话行存在（幂等；标题默认取首条问题前30字，已存在则不覆盖）
        repo.ensure_conversation(
            conversation_id,
            user_id=request.user_id or "default",
            title=request.question[:30],
        )
        # 用户表也顺手登记（幂等），台账完整
        repo.ensure_user(request.user_id or "default")

        # 1. 混合检索相关 chunk（向量语义 + BM25 关键词，RRF 融合，Round 4 优化）
        retriever = HybridRetriever(top_k=request.top_k)
        chunks = retriever.retrieve(request.question)

        # 2. 构建 Prompt
        #    无历史（新会话/首轮）时直接退回单轮 build()，行为与旧版一致
        history = repo.list_messages(conversation_id, limit=MAX_HISTORY_MESSAGES)
        if history:
            # list_messages 返回时间正序，直接转成 LLM 消息格式
            chat_history = [
                {"role": m["role"], "content": m["content"]} for m in history
            ]
            messages = _builder.build_with_history(
                request.question, chunks, chat_history
            )
            # 多轮走 chat_with_history（messages 已含 system/历史/当前轮）
            logger.debug("多轮问答 | 会话=%s | 历史条数=%d | 检索片段=%d",
                         conversation_id, len(chat_history), len(chunks))
            llm = get_llm()
            answer = llm.chat_with_history(messages, temperature=0.3)
        else:
            # 首轮：单轮 Prompt（与旧版完全一致）
            logger.debug("首轮问答 | 会话=%s | 检索片段=%d", conversation_id, len(chunks))
            prompt = _builder.build(request.question, chunks)
            llm = get_llm()
            answer = llm.chat(prompt, temperature=0.3)

        # 3. 本轮消息写回 SQLite（先问后答，保持时间序）
        repo.add_message(conversation_id, "user", request.question)
        repo.add_message(conversation_id, "assistant", answer)

        # 4. 格式化来源（同旧版）
        sources = [
            Source(
                content=c["content"],
                page=c["page"],
                source=c["source"],
                score=c["score"]
            )
            for c in chunks
        ]

        # 成功日志：会话/历史/检索片段数，运营统计与排查都有据可查
        logger.info(
            "问答完成 | 会话=%s | 用户=%s | 历史条数=%d | 检索片段=%d",
            conversation_id, request.user_id, len(history), len(chunks)
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id  # 回传会话 id，前端下一轮带上
        )

    except HTTPException:
        raise
    except Exception as e:
        # 内部错误（检索/LLM调用/数据库等）：记完整堆栈，前端只收通用文案，
        # 不透传 str(e)（可能含 API 报错细节/内部路径）
        logger.exception(
            "问答失败(内部错误) | 会话=%s | 问题=%s... | 类型=%s",
            request.conversation_id or "(新生成)", request.question[:20], type(e).__name__
        )
        raise HTTPException(status_code=500, detail="问答服务暂时不可用，请稍后重试")


@router.get("/")
async def chat_get(question: str, top_k: int = 3):
    """
    GET 方式问答（方便浏览器测试）
    不带会话概念，等价于"每次新会话"的单轮问答
    """
    return await chat(ChatRequest(question=question, top_k=top_k))
