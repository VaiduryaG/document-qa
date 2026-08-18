"""
OCR 引擎封装模块
================
用途：对「扫描型 PDF 渲染出的图像」和「纯图片文件」做文字识别，
      补齐 pdfplumber 无法处理图像型内容的短板。

引擎：RapidOCR（onnxruntime 后端）
      - 纯 pip 安装，自带 ONNX 模型权重，无需额外下载、无需联网。
      - 支持中英文混合识别。
      - 内部为「检测(det) → 方向分类(cls) → 识别(rec)」三段式 PaddleOCR 管线。

设计：单例模式，避免重复加载三个 ONNX 模型，节省内存与初始化时间。
"""

import threading
from typing import Optional, Union

import numpy as np
from rapidocr_onnxruntime import RapidOCR

from core.logger import get_logger

logger = get_logger(__name__)


class OCREngine:
    """
    OCR 引擎封装（线程安全单例）

    RapidOCR 初始化参数说明（通过 kwargs 传入，对应 config.yaml 的配置项）：
    -------------------------------------------------------------------
    以下参数均映射到底层 ONNX 推理会话，影响识别速度与精度：

    全局参数（Global）：
        - text_score        : 文本检测置信度阈值（0~1）。低于此值的文本框丢弃，
                              调高→更严格(漏检少假阳)，调低→更宽松(多召回多噪声)。
        - min_height        : 最小文本框高度(像素)。低于此高度的框视为噪声丢弃，
                              用于过滤小噪点。
        - max_side_len      : 输入图像最长边限制(像素)。超过则等比缩放，
                              控制显存/内存占用，越大越精细但越慢。
        - use_det/use_cls/use_rec : 是否启用 检测/分类/识别 三个阶段。
                              纯文字识别通常三者全开。

    检测参数（Det，前缀 det_）：
        - det_model_path    : 文本「检测」模型路径。检测模型负责在图上框出文字区域
                              （输出若干四点多边形框）。默认 ch_PP-OCRv4_det_infer.onnx。
        - det_thresh        : 像素级二值化阈值，用于生成检测热力图。
        - det_box_thresh    : 框级置信度阈值，单个候选框低于此值丢弃。
        - det_unclip_ratio  : 框扩展系数。检测框通常偏紧，需向外扩展以包含完整文字，
                              1.6 为经验默认值，调大→框更松。
        - det_limit_side_len: 检测阶段输入图边长限制，736 为默认。

    分类参数（Cls，前缀 cls_）：
        - cls_model_path    : 文本「方向分类」模型路径。判断文字框是 0° 还是 180°
                              （ upside-down），用于自动旋转纠正倒置文本。
                              默认 ch_ppocr_mobile_v2.0_cls_infer.onnx。
        - cls_thresh        : 分类置信度阈值，0.9 表示只有非常确信倒置才翻转。

    识别参数（Rec，前缀 rec_）：
        - rec_model_path    : 文本「识别」模型路径。把每个检测框内的图像转为文字字符串。
                              默认 ch_PP-OCRv4_rec_infer.onnx。
        - rec_img_shape     : 识别阶段输入图尺寸 [C,H,W]，统一缩放到此尺寸再送入网络。

    线程数（性能调优）：
        - intra_op_num_threads : 单个算子内部并行线程数，-1 表示按 CPU 自动选择。
        - inter_op_num_threads : 算子之间并行线程数，-1 表示自动。
    """

    _instance: Optional["OCREngine"] = None
    _lock = threading.Lock()  # 保证多线程下只初始化一次

    def __new__(cls, **kwargs):
        """线程安全的单例创建：首次调用时加载模型，后续直接返回同一实例。"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_engine(**kwargs)  # 仅首次创建时初始化
        return cls._instance

    def _init_engine(self, **kwargs):
        """
        实际初始化 RapidOCR 引擎（仅单例首次创建时执行一次）。

        kwargs 可覆盖默认配置，例如：
            OCREngine(text_score=0.6, det_unclip_ratio=1.8)
        未传入的参数使用 RapidOCR 自带默认值（见模块文档与 config.yaml）。
        """
        # RapidOCR(config_path=None, **kwargs):
        #   - config_path=None：使用内置默认配置 config.yaml
        #   - **kwargs：按前缀规则覆盖配置（det_/cls_/rec_/全局）
        self._ocr = RapidOCR(
            **kwargs
        )

    def extract(self, image: Union[bytes, np.ndarray, str]) -> str:
        """
        对一张图像执行 OCR，返回拼接后的纯文本。

        参数：
            image : 支持三种输入
                    - bytes   ：原始图片字节（如 PDF 渲染后的 PNG bytes）
                    - np.ndarray：已解码的图像数组（BGR/RGB 均可，内部会处理）
                    - str     ：图片文件路径

        返回：
            str：识别出的全部文本，各文本框按从上到下、从左到右顺序用换行拼接。
                 识别失败或图像无文字时返回空字符串 ""。

        说明：
            RapidOCR.__call__ 返回 (result, elapse)：
              - result : 形如 [[box, text, score], ...] 的列表，box 为四点坐标，
                         text 为该框识别文字，score 为置信度。
              - elapse : 各阶段耗时统计，此处不使用。
            当图像无法解码或无任何文本框时 result 为 None。
        """
        try:
            result, _elapse = self._ocr(image)
        except Exception:
            # 解码失败等异常不应中断整个解析流程，记日志（含堆栈）后返回空串
            logger.exception("OCR 识别异常，返回空文本 | 输入类型=%s", type(image).__name__)
            return ""

        # 无文本框 / 全程未识别到内容
        if not result:
            return ""

        # result 每项: [box, text, score]，取 text 拼接
        lines = [item[1] for item in result if item and len(item) >= 2]
        return "\n".join(lines).strip()


# ========== 单例便捷入口 ==========
_ocr_engine: Optional[OCREngine] = None


def get_ocr_engine(**kwargs) -> OCREngine:
    """
    获取 OCR 引擎单例（线程安全）。

    首次调用时加载三个 ONNX 模型（约耗时 1~2 秒）；
    之后调用直接返回已加载实例，零开销。

    kwargs 仅在首次调用时生效，后续调用传入也会被忽略（单例已创建）。
    """
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine(**kwargs)
    return _ocr_engine
