"""
混合检索模块
============
用途：结合「向量语义检索」与「BM25 关键词检索」，解决纯向量检索的短板。

背景（Round 3 诊断结论）：
    纯向量检索对「表格类/精确数字类」文本表征能力弱——
    问题"Pro版的年授权费是多少"与产品规格书 chunk 的余弦相似度仅 0.39，
    排名 #5，Top-4 根本召不回。而 BM25 基于 TF-IDF 关键词匹配，
    对"28万元"这类精确词命中极强，正好补上这个短板。

融合策略：RRF（Reciprocal Rank Fusion，倒数排名融合）
    score(d) = Σ 1 / (k + rank_i(d))，k=60 为经验值
    优点：不需要两路分数归一化（向量分数 0~1，BM25 分数无上界，直接加权会偏），
          只用排名参与计算，对分数分布不敏感，是业界默认融合方案。

流程：
    1. 向量检索：召回 top_n 候选（语义相近）
    2. BM25 检索：召回 top_n 候选（关键词命中）
    3. RRF 融合两路排名，取最终 top_k
"""

import math
import threading
from typing import Dict, List, Optional

import jieba
from rank_bm25 import BM25Okapi

from core.logger import get_logger
from db.chroma_client import ChromaStore, get_chroma_store

logger = get_logger(__name__)

# RRF 常数：排名越靠前贡献越大，60 为论文原推荐值，避免排名 1 与 2 差距过大
RRF_K = 60


def _tokenize(text: str) -> List[str]:
    """
    中文分词（BM25 索引与查询共用）。

    jieba 默认精确模式；转小写让英文术语不区分大小写
    （如 "Pro版" 与 "pro版" 能对上）。
    """
    return [tok.lower() for tok in jieba.lcut(text) if tok.strip()]


class BM25Index:
    """
    BM25 关键词索引（内存态，与 ChromaDB 同步重建）

    设计说明：
        - ChromaDB 不原生支持 BM25，这里把全部 chunk 拉到内存建索引。
          当前规模（几十~几百 chunk）毫秒级完成，无性能问题；
          若未来到万级 chunk，可换成离线重建 + 定时刷新。
        - 线程安全单例：文档增删后调 invalidate() 使缓存失效，
          下次检索自动重建。
    """

    _instance: Optional["BM25Index"] = None
    _lock = threading.Lock()

    def __new__(cls, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._built = False
                cls._instance._corpus_ids = []      # 与 bm25 行号对应的 chunk id
                cls._instance._corpus_meta = []     # 对应的 metadata
                cls._instance._bm25 = None
        return cls._instance

    def _ensure_built(self, store: ChromaStore):
        """索引未建或已失效时，从 ChromaDB 全量拉取重建"""
        if self._built:
            return

        all_docs = store.collection.get(include=["documents", "metadatas"])
        documents = all_docs.get("documents", []) or []
        metadatas = all_docs.get("metadatas", []) or []

        if not documents:
            logger.warning("BM25 索引构建：向量库为空")
            self._bm25 = None
            self._built = True
            return

        tokenized = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus_meta = metadatas
        self._corpus_docs = documents
        self._built = True
        logger.info("BM25 索引构建完成 | chunk 数=%d", len(documents))

    def invalidate(self):
        """文档增删后调用，使索引失效（下次检索时重建）"""
        self._built = False

    def search(self, query: str, top_n: int, store: ChromaStore) -> List[Dict]:
        """
        BM25 关键词检索

        Returns:
            [{"content": ..., "source": ..., "page": ..., "bm25_score": ...}, ...]
        """
        self._ensure_built(store)

        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))

        # 取分数最高的 top_n 个（分数为 0 的不召回）
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_n]:
            if score <= 0:
                break
            meta = self._corpus_meta[idx] or {}
            results.append({
                "content": self._corpus_docs[idx],
                "page": meta.get("page", 1),
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "bm25_score": round(float(score), 4),
            })
        return results


# 模块级单例便捷入口
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """获取 BM25 索引单例"""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


class HybridRetriever:
    """
    混合检索器：向量检索 + BM25 检索 → RRF 融合

    用法与 DocumentRetriever 完全一致（retrieve 接口），可无缝替换。
    """

    def __init__(
        self,
        vector_store: ChromaStore = None,
        embedder=None,
        top_k: int = 4,
        candidate_k: int = 10
    ):
        """
        Args:
            top_k: 最终返回的 chunk 数
            candidate_k: 每路检索的候选数（两路各召回这么多，再融合取 top_k）
        """
        self.embedder = embedder
        self.vector_store = vector_store or get_chroma_store()
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.bm25_index = get_bm25_index()

    def retrieve(self, query: str, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        混合检索

        Args:
            query: 用户问题
            filter_dict: 过滤条件（透传给向量检索）

        Returns:
            RRF 融合排序后的 top_k 个 chunk，字段格式与 DocumentRetriever 一致
        """
        # ---------- 1. 向量语义检索 ----------
        # 延迟导入避免循环依赖（retriever 也 import 本模块的场景）
        from core.retriever import DocumentRetriever

        vec_results = DocumentRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.candidate_k
        ).retrieve(query, filter_dict=filter_dict)

        # ---------- 2. BM25 关键词检索 ----------
        bm25_results = self.bm25_index.search(query, self.candidate_k, self.vector_store)

        # ---------- 3. RRF 融合 ----------
        # key 用 (source, chunk_index)：同源同序号视为同一 chunk
        rrf_scores: Dict[tuple, float] = {}
        chunk_pool: Dict[tuple, Dict] = {}

        for rank, chunk in enumerate(vec_results, start=1):
            key = (chunk["source"], chunk.get("chunk_index", 0))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            chunk_pool.setdefault(key, {**chunk, "vec_rank": rank})

        for rank, chunk in enumerate(bm25_results, start=1):
            key = (chunk["source"], chunk.get("chunk_index", 0))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key in chunk_pool:
                chunk_pool[key]["bm25_rank"] = rank
            else:
                chunk_pool[key] = {**chunk, "bm25_rank": rank}

        # 按融合分排序，取 top_k
        sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for key, score in sorted_keys[:self.top_k]:
            chunk = chunk_pool[key]
            final_results.append({
                "content": chunk["content"],
                "page": chunk["page"],
                "source": chunk["source"],
                "chunk_index": chunk.get("chunk_index", 0),
                "score": round(score * RRF_K, 4),  # 放大便于观察（非真实相似度）
                "vec_rank": chunk.get("vec_rank"),
                "bm25_rank": chunk.get("bm25_rank"),
            })

        logger.debug(
            "混合检索 | 候选: 向量=%d BM25=%d | 融合返回=%d",
            len(vec_results), len(bm25_results), len(final_results)
        )
        return final_results


def hybrid_retrieve(query: str, top_k: int = 4, candidate_k: int = 10) -> List[Dict]:
    """便捷函数"""
    retriever = HybridRetriever(top_k=top_k, candidate_k=candidate_k)
    return retriever.retrieve(query)