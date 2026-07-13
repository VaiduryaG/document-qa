import os
import re
from typing import List, Dict, Optional
from pathlib import Path


class DocumentParser:
    """
    文档解析器
    支持：PDF、TXT
    返回：带页码/行号的结构化文本块
    """
    
    def parse(self, file_path: str) -> List[Dict[str, any]]:
        """
        根据文件扩展名自动选择解析器
        
        Returns:
            [{"text": "...", "page": 1, "source": "xxx.pdf"}, ...]
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext == ".md":
            return self._parse_md(file_path)
        elif ext == ".docx":
            return self._parse_docx(file_path)
        elif ext == ".txt":
            return self._parse_txt(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持 PDF/TXT/Markdown/Word")
    
    def _parse_pdf(self, file_path: str) -> List[Dict[str, any]]:
        """解析 PDF，按页返回"""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("请安装 pdfplumber: pip install pdfplumber")
        
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "text": text.strip(),
                        "page": i,
                        "source": os.path.basename(file_path)
                    })
        return pages

    def _parse_md(self, file_path: str) -> List[Dict[str, any]]:
        """解析 Markdown"""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # 去掉Markdown标记，保留纯文本
        text = re.sub(r'[#*`_\[\]\(\)!]', '', text)
        return [{
            "text": text.strip(),
            "page": 1,
            "source": os.path.basename(file_path)
        }]

    def _parse_docx(self, file_path: str) -> List[Dict[str, any]]:
        """解析 Word"""
        try:
            import docx
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")
        
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        
        return [{
            "text": text.strip(),
            "page": 1,
            "source": os.path.basename(file_path)
        }]
    
    def _parse_txt(self, file_path: str) -> List[Dict[str, any]]:
        """解析 TXT，按行返回（模拟单页）"""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # 清理多余空行
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        return [{
            "text": text.strip(),
            "page": 1,  # TXT 没有页码，统一为1
            "source": os.path.basename(file_path)
        }]


def parse_document(file_path: str) -> List[Dict[str, any]]:
    """便捷函数"""
    parser = DocumentParser()
    return parser.parse(file_path)