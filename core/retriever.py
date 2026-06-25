from typing import List, Dict, Optional
from core.embedder_factory import get_embedder
from db.chroma_client import ChromaStore, get_chroma_store


class DocumentRetriever:
    """
    文档检索器
    流程：问题向量化 → ChromaDB 相似度检索 → 返回 Top-K 相关 chunk
    """
    
    def __init__(
        self,
        embedder=None,
        vector_store: ChromaStore = None,
        top_k: int = 3
    ):
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or get_chroma_store()
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        检索与问题相关的文档片段
        
        Args:
            query: 用户问题
            filter_dict: 过滤条件，如 {"source": "xxx.pdf"}
        
        Returns:
            [
                {
                    "content": "相关文本内容...",
                    "page": 2,
                    "source": "xxx.pdf",
                    "score": 0.85  # 相似度分数（距离转换）
                },
                ...
            ]
        """
        # 1. 问题向量化
        query_embedding = self.embedder.embed_query(query)
        
        # 2. ChromaDB 检索
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=self.top_k,
            filter_dict=filter_dict
        )
        
        # 3. 格式化结果
        retrieved_chunks = []
        
        # ChromaDB 返回的是嵌套列表，需要展开
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            # 距离转相似度分数（余弦距离 → 相似度，越接近1越相关）
            score = 1 - dist
            
            retrieved_chunks.append({
                "content": doc,
                "page": meta.get("page", 1),
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(score, 4)
            })
        
        return retrieved_chunks


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 3,
    filter_dict: Optional[Dict] = None
) -> List[Dict]:
    """便捷函数"""
    retriever = DocumentRetriever(top_k=top_k)
    return retriever.retrieve(query, filter_dict=filter_dict)