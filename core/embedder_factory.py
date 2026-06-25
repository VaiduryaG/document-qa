import os
from dotenv import load_dotenv


# 加载 core/.env
_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_env_path)


def get_embedder():
    """
    工厂函数：根据 .env 的 EMBEDDER_TYPE 自动选择嵌入模型
    
    配置方式（在 core/.env 里写）：
        EMBEDDER_TYPE=zhipu   -> 智谱 embedding-2（需要额度）
        EMBEDDER_TYPE=local   -> 本地模型（免费，离线）
    
    使用方式：
        from core.embedder_factory import get_embedder
        embedder = get_embedder()  # 自动切换，代码完全不用改
    """
    embedder_type = os.getenv("EMBEDDER_TYPE", "zhipu").lower()
    
    if embedder_type == "local":
        from core.local_embedder import LocalEmbedder
        print("[Embedder工厂] 模式: 本地模型 (paraphrase-multilingual-MiniLM-L12-v2)")
        return LocalEmbedder()
    
    # 默认智谱API
    from core.embedder import ZhipuEmbedder
    print("[Embedder工厂] 模式: 智谱API (embedding-2)")
    return ZhipuEmbedder()