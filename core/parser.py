import os
import re
from typing import List, Dict, Optional
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """
    文档解析器
    支持：PDF（含扫描型）、TXT、Markdown、Word、图片（JPG/PNG）
    返回：带页码/来源的结构化文本块

    OCR 回退策略（阶段 1.2 新增）：
        对于 PDF，先用 pdfplumber 直接抽文字（快、无损）；
        若某页抽不到文字（典型如扫描型 PDF / 纯图像页），则用 PyMuPDF
        把该页渲染成图像，再交给 RapidOCR 识别，作为兜底补充。
        这样既保留原生文本页的高质量，又覆盖了扫描页。
    """

    # 扫描页 OCR 兜底：单页文本字符数低于此阈值视为「抽文字失败」，触发 OCR
    # PDF 中一页通常至少几十字符，低于此值很可能是空白图像页或扫描页
    OCR_MIN_TEXT_LEN = 5
    # 渲染 PDF 页面的 DPI：150 兼顾清晰度与速度（300 更清晰但慢 4 倍）
    OCR_RENDER_DPI = 150

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
        elif ext in (".jpg", ".jpeg", ".png"):
            # 阶段 1.2 新增：纯图片文件直接走 OCR
            return self._parse_image(file_path)
        else:
            raise ValueError(
                f"不支持的文件格式: {ext}，仅支持 PDF/TXT/Markdown/Word/图片(JPG/PNG)"
            )

    def _parse_pdf(self, file_path: str) -> List[Dict[str, any]]:
        """
        解析 PDF，按页返回。

        流程（每页）：
            1. pdfplumber.extract_text() 抽取原生文本（数字化 PDF 直接成功）。
            2. 若文本过短（< OCR_MIN_TEXT_LEN），判定为扫描/图像页，
               用 PyMuPDF 渲染成图像后调 OCR 兜底识别。
            3. 若 OCR 也无结果，跳过该页（避免空页入库）。
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("请安装 pdfplumber: pip install pdfplumber")

        pages = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                text = text.strip() if text else ""

                # 抽不到足够文字 → 疑似扫描页，走 OCR 兜底
                if len(text) < self.OCR_MIN_TEXT_LEN:
                    ocr_text = self._ocr_pdf_page(file_path, i)
                    if ocr_text:
                        text = ocr_text
                    else:
                        # OCR 也没识别出内容，跳过该页，不留空块
                        continue

                pages.append({
                    "text": text,
                    "page": i,
                    "source": os.path.basename(file_path)
                })
        return pages

    def _ocr_pdf_page(self, file_path: str, page_num: int) -> str:
        """
        用 PyMuPDF 把 PDF 指定页渲染成 PNG 图像，再交给 RapidOCR 识别。

        参数：
            file_path : PDF 文件路径
            page_num  : 页码（1-based，与 pdfplumber enumerate 的 i 一致）

        返回：
            str：识别出的文本；失败或无文字返回 ""。

        说明：
            PyMuPDF 的页码是 0-based，这里传入前减 1 转换。
            get_pixmap(dpi=...) 渲染页面为图像，tobytes("png") 得到 PNG 字节流，
            RapidOCR 可直接吃 bytes。
        """
        try:
            import pymupdf
        except ImportError:
            raise ImportError("请安装 pymupdf: pip install pymupdf")

        try:
            # 延迟导入 OCR 引擎（仅在真正需要 OCR 时加载，避免普通 PDF 白白初始化模型）
            from core.ocr import get_ocr_engine

            doc = pymupdf.open(file_path)
            # PyMuPDF 页码 0-based，page_num 是 1-based
            page = doc[page_num - 1]
            # 渲染：matrix 可由 dpi 推导，这里用 dpi 参数更直观
            pix = page.get_pixmap(dpi=self.OCR_RENDER_DPI)
            png_bytes = pix.tobytes("png")
            doc.close()

            # OCR 引擎接收 bytes 输入，返回拼接文本
            return get_ocr_engine().extract(png_bytes)
        except Exception as e:
            # OCR 单页失败不应中断整篇解析，记日志（含堆栈）后返回空串，该页被跳过
            logger.exception("PDF 第%s页 OCR 失败，该页跳过 | 文件=%s", page_num, file_path)
            return ""

    def _parse_image(self, file_path: str) -> List[Dict[str, any]]:
        """
        解析纯图片文件（JPG/PNG）——直接整图 OCR。

        图片无「页」概念，统一记为第 1 页。
        """
        from core.ocr import get_ocr_engine

        text = get_ocr_engine().extract(file_path)
        text = text.strip() if text else ""

        if not text:
            # 图片无任何可识别文字，返回空列表（不上传空内容）
            return []

        return [{
            "text": text,
            "page": 1,
            "source": os.path.basename(file_path)
        }]

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
