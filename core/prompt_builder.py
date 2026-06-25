from typing import List, Dict


class PromptBuilder:
    """
    RAG Prompt 构建器
    将检索到的 chunk 和用户问题拼接成完整的 Prompt
    """
    
    SYSTEM_PROMPT = """你是一个专业的文档问答助手。请基于提供的文档内容回答用户问题。
    
规则：
1. 必须严格基于提供的文档内容回答，不要添加文档外的信息
2. 如果文档中没有相关信息，明确说明"根据提供的文档，无法回答该问题"
3. 回答要简洁准确，必要时可以分点说明
4. 在回答末尾标注信息来源（文档名和页码）"""
    
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
    
    def build(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        """
        构建 RAG Prompt
        
        Args:
            question: 用户问题
            retrieved_chunks: 检索到的 chunk 列表
        
        Returns:
            完整的 Prompt 文本
        """
        if not retrieved_chunks:
            # 没有检索到内容
            return f"""{self.system_prompt}

用户问题：{question}

（未检索到相关文档内容）

请回答：根据提供的文档，无法回答该问题。"""
        
        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                f"[片段{i}] 来源：{chunk['source']} 第{chunk['page']}页\n"
                f"{chunk['content']}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""{self.system_prompt}

文档内容：
{'='*50}
{context}
{'='*50}

用户问题：{question}

请基于上述文档内容回答问题，并在末尾标注来源。"""
        
        return prompt
    
    def build_with_history(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        chat_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        构建带历史记录的多轮对话消息（用于 LLM 的 chat_with_history）
        
        Args:
            question: 当前问题
            retrieved_chunks: 检索结果
            chat_history: 历史消息，如 [
                {"role": "user", "content": "之前的问题"},
                {"role": "assistant", "content": "之前的回答"}
            ]
        
        Returns:
            完整的消息列表，可直接传给 LLM
        """
        # 系统消息
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 历史消息
        messages.extend(chat_history)
        
        # 当前轮次：拼接检索结果 + 问题
        current_prompt = self.build(question, retrieved_chunks)
        messages.append({"role": "user", "content": current_prompt})
        
        return messages


def build_rag_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """便捷函数：单轮对话"""
    builder = PromptBuilder()
    return builder.build(question, retrieved_chunks)