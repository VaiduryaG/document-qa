import re
from typing import List, Dict


class TextChunker:
    """
    文本切分器
    策略：按句子边界切分，保证语义完整，带重叠
    """
    
    def __init__(
        self,
        chunk_size: int = 500,      # 每个 chunk 目标字符数
        overlap: int = 50,          # 相邻 chunk 重叠字符数
        min_chunk_size: int = 100   # 最小 chunk 长度，太短的丢弃
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk(
        self,
        pages: List[Dict[str, any]],
        tables: List[Dict[str, any]] = None
    ) -> List[Dict[str, any]]:
        """
        将解析后的页面切分为 chunk
        
        Args:
            pages: [{"text": "...", "page": 1, "source": "xxx.pdf"}, ...]
            tables: table_parser.parse_tables() 返回的结构化表格列表。
        
        Returns:
            [{"content": "...", "page": 1, "chunk_index": 0, "source": "xxx.pdf"}, ...]
        """
        all_chunks = []
        chunk_index = 0
        tables_by_page = {}
        for table in tables or []:
            tables_by_page.setdefault(int(table.get("page", 1)), []).append(table)
        
        for page in pages:
            page_text = page["text"]
            page_num = page["page"]
            source = page["source"]

            # 表格先按业务行切分，保证“字段-值”关系不被滑动窗口破坏。
            for table in tables_by_page.get(page_num, []):
                for row in table.get("rows", []):
                    row_text = "；".join(
                        f"{key}：{value}"
                        for key, value in row.get("data", {}).items()
                        if str(value).strip()
                    )
                    if not row_text:
                        continue

                    all_chunks.append({
                        "content": f"表格数据。{row_text}",
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "source": source,
                        "is_table": True,
                        "table_index": int(table.get("table_index", 0)),
                        "table_row_index": int(row.get("row_index", 0)),
                        "table_parser": table.get("parser", "ocr_layout"),
                    })
                    chunk_index += 1
            
            # 按句子切分（中文按。！？；，英文按 .!?）
            sentences = self._split_sentences(page_text)
            
            # 滑动窗口组合句子成 chunk
            chunks = self._merge_sentences(sentences)
            
            for chunk_text in chunks:
                if len(chunk_text) >= self.min_chunk_size:
                    all_chunks.append({
                        "content": chunk_text,
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "source": source
                    })
                    chunk_index += 1
        
        return all_chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """按句子边界切分"""
        # 中文标点 + 英文标点，保留标点
        pattern = r'(?<=[。！？；.!?])\s*'
        sentences = re.split(pattern, text)
        # 过滤空句子和纯空白
        return [s.strip() for s in sentences if s.strip()]
    
    def _merge_sentences(self, sentences: List[str]) -> List[str]:
        """将句子组合成目标长度的 chunk，带重叠"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 如果当前句子本身超长，直接作为一个 chunk
            if len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                chunks.append(sentence.strip())
                current_chunk = ""
                continue
            
            # 尝试加入当前 chunk
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                # 当前 chunk 满了，保存
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 新 chunk 带上重叠（取上一个 chunk 末尾的 overlap 字符）
                if current_chunk and self.overlap > 0:
                    overlap_text = current_chunk[-self.overlap:]
                    current_chunk = overlap_text + sentence
                else:
                    current_chunk = sentence
        
        # 最后一个 chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


def chunk_document(
    pages: List[Dict[str, any]],
    tables: List[Dict[str, any]] = None,
    **kwargs
) -> List[Dict[str, any]]:
    """便捷函数"""
    chunker = TextChunker(**kwargs)
    return chunker.chunk(pages, tables=tables)
