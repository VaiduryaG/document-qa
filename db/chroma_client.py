import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os


class ChromaStore:
    """
    ChromaDB 向量存储封装
    支持：添加文档、查询相似度、删除文档、列出所有文档
    """
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """
        初始化 ChromaDB 客户端
        
        Args:
            collection_name: 集合名称，默认 "documents"
            persist_directory: 数据持久化目录，默认当前目录下的 chroma_db
        """
        # 创建持久化客户端（数据会存到磁盘，重启不会丢）
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False  # 关闭匿名数据收集
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        self.collection_name = collection_name
    
    def add(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict]
    ) -> None:
        """
        添加文档到向量库
        
        Args:
            ids: 唯一标识符列表，如 ["doc1_chunk0", "doc1_chunk1"]
            texts: 原始文本列表
            embeddings: 向量列表
            metadatas: 元数据列表，如 [{"page": 1, "source": "xxx.pdf"}, ...]
        """
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 3,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        向量相似度检索
        
        Args:
            query_embedding: 查询向量
            n_results: 返回Top-K结果
            filter_dict: 过滤条件，如 {"source": "xxx.pdf"}
        
        Returns:
            {
                "ids": [["id1", "id2"]],
                "documents": [["text1", "text2"]],
                "metadatas": [[{"page": 1}, {"page": 2}]],
                "distances": [[0.1, 0.2]]
            }
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        return results
    
    def delete(self, ids: List[str]) -> None:
        """删除指定文档"""
        self.collection.delete(ids=ids)

    def delete_by_filename(self, filename: str) -> int:
        """
        按文件名删除该文档的全部向量（where 过滤删除）。

        用途：同名文件重传前先清旧向量。若旧次切出 8 块、新次只切出 5 块，
        只靠 add（同 ID 静默跳过）会残留旧的第 6/7/8 块——先整删再新增才干净。

        返回：
            int：删除前的该文件向量条数（0 表示本来没有）。
        """
        existed = self.collection.get(where={"filename": filename})
        n = len(existed.get("ids", [])) if existed else 0
        if n:
            self.collection.delete(where={"filename": filename})
        return n
    
    def get_all(self) -> Dict:
        """获取集合中所有文档"""
        return self.collection.get()
    
    def count(self) -> int:
        """返回文档总数"""
        return self.collection.count()
    
    def clear(self) -> None:
        """清空整个集合（危险操作！）"""
        all_docs = self.get_all()
        if all_docs["ids"]:
            self.delete(all_docs["ids"])


# 单例模式，方便全局使用
_chroma_store = None

def get_chroma_store(collection_name: str = "documents") -> ChromaStore:
    """获取 ChromaStore 单例"""
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore(collection_name=collection_name)
    return _chroma_store