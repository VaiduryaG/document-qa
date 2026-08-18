import os
from sentence_transformers import SentenceTransformer
from typing import List

from core.logger import get_logger

logger = get_logger(__name__)


class LocalEmbedder:
    """
    本地嵌入模型（完全离线，无需网络）
    模型：paraphrase-multilingual-MiniLM-L12-v2（支持中文，384维）
    """
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            # 自动定位到项目根目录下的 model_weights 文件夹
            current_file = os.path.abspath(__file__)           # .../core/local_embedder.py
            core_dir = os.path.dirname(current_file)             # .../core
            project_root = os.path.dirname(core_dir)             # .../document-qa
            model_path = os.path.join(
                project_root,
                "model_weights",
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        
        logger.info("加载本地模型: %s", model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"本地模型未找到: {model_path}\n"
                f"请从 https://hf-mirror.com/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 "
                f"下载模型文件放到 {model_path}"
            )
        
        self.model = SentenceTransformer(model_path)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("本地模型加载完成 | 输出维度=%s", self.dimension)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本转向量"""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """单条查询转向量"""
        return self.embed([text])[0]


# 单例
_local_embedder = None

def get_local_embedder() -> LocalEmbedder:
    global _local_embedder
    if _local_embedder is None:
        _local_embedder = LocalEmbedder()
    return _local_embedder