from zhipuai import ZhipuAI
from typing import List
import os
from dotenv import load_dotenv


# 加载 core/.env 文件
_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_env_path)


class ZhipuEmbedder:
    """
    智谱 AI Embedding API 封装
    模型：embedding-2（向量模型，不是 glm-4-flash）
    """
    
    def __init__(self, api_key: str = None, model: str = "embedding-2"):
        if api_key is None:
            api_key = os.getenv("ZHIPU_API_KEY")
            if not api_key:
                raise ValueError(
                    f"请检查 core/.env 文件是否包含 ZHIPU_API_KEY\n"
                    f"查找路径: {_env_path}"
                )
        
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本转向量"""
        if len(texts) > 64:
            all_embeddings = []
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                all_embeddings.extend([item.embedding for item in response.data])
            return all_embeddings
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """单条查询转向量"""
        return self.embed([text])[0]


# 单例
_embedder = None

def get_embedder(api_key: str = None) -> ZhipuEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ZhipuEmbedder(api_key=api_key)
    return _embedder