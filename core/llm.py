from zhipuai import ZhipuAI
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv


# 加载 core/.env
_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_env_path)


class ZhipuLLM:
    """
    智谱大模型 API 封装
    模型：glm-4-flash（免费、速度快、适合问答）
    """
    
    def __init__(self, api_key: str = None, model: str = "glm-4-flash"):
        if api_key is None:
            api_key = os.getenv("ZHIPU_API_KEY")
            if not api_key:
                raise ValueError(
                    f"请检查 core/.env 文件是否包含 ZHIPU_API_KEY\n"
                    f"查找路径: {_env_path}"
                )
        
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
    
    def chat(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        单轮对话
        
        Args:
            prompt: 完整的 Prompt 文本
            temperature: 创造性（0-1，越低越保守）
            max_tokens: 最大输出长度
        
        Returns:
            模型生成的文本
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        多轮对话（带历史记录）
        
        Args:
            messages: 消息列表，如 [
                {"role": "system", "content": "你是一个助手"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！"},
                {"role": "user", "content": "新问题"}
            ]
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


# 单例
_llm = None

def get_llm(api_key: str = None) -> ZhipuLLM:
    global _llm
    if _llm is None:
        _llm = ZhipuLLM(api_key=api_key)
    return _llm