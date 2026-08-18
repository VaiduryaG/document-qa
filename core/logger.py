"""
日志配置模块（阶段 3.1）
==========================
用途：统一全项目的日志输出，替代散落各处的 print()：
      - 控制台：开发时直接看（彩色可读格式）
      - 文件    ：按大小轮转，持久留痕，排查问题/审计用

设计：
    - 标准 logging 模块 + RotatingFileHandler（标准库，零依赖）
    - root logger 只配置一次（幂等：重复调用 init_logging 不会叠加 handler）
    - 各模块用 get_logger(__name__) 拿各自命名空间 logger，
      日志行自动带上模块名，方便定位来源
    - 级别可通过 .env 的 LOG_LEVEL 调（DEBUG/INFO/WARNING/ERROR），
      不动代码即可切换详细程度

日志格式说明：
    "2026-08-17 11:30:01 | INFO     | core.parser:42 | PDF 第3页触发OCR回退"
     时间              级别          模块:行号          消息
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from dotenv import load_dotenv

# 加载项目根目录 .env（读取 LOG_DIR / LOG_LEVEL）
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_env_path)

# 项目根目录（core/ 的上一级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 默认日志目录与文件名
_DEFAULT_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
_LOG_FILENAME = "app.log"

# 统一日志格式：时间 | 级别(左对齐8) | 模块:行号 | 消息
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 轮转策略：单文件 5MB，最多保留 3 个备份（约 20MB 上限，防撑爆磁盘）
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# 模块级标记：是否已完成初始化（防重复配置）
_initialized = False


def init_logging() -> logging.Logger:
    """
    初始化 root logger（幂等，全项目只需也只会生效一次）。

    做了什么：
        1. 读 .env：LOG_DIR（默认 <项目根>/logs）、LOG_LEVEL（默认 INFO）
        2. 给 root logger 挂两个 handler：
           - StreamHandler  -> 控制台（跟随全局级别）
           - RotatingFileHandler -> logs/app.log（5MB 轮转，留3个备份）
        3. 降噪第三方库：uvicorn/chromadb 等内部日志太啰嗦，压到 WARNING

    返回：
        配置好的 root logger。
    """
    global _initialized
    if _initialized:
        # 幂等：重复调用直接返回，避免 handler 叠加导致日志重复输出
        return logging.getLogger()

    # ---- 读配置 ----
    log_dir = os.getenv("LOG_DIR", _DEFAULT_LOG_DIR)
    # 相对路径锚定项目根，不受启动目录影响
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(_PROJECT_ROOT, log_dir)
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)  # 非法值兜底 INFO

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, _LOG_FILENAME)

    # ---- 配置 root logger ----
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 控制台 handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # 文件 handler（轮转）
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",  # 中文日志不乱码
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # ---- 第三方库降噪 ----
    # 这些库内部 INFO 很啰嗦（每次请求/每步操作都打），压到 WARNING：
    # 我们只关心它们报的警告和错误
    for noisy in ("uvicorn", "uvicorn.access", "chromadb", "httpx", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True
    root.info("日志系统初始化完成 | 级别=%s | 文件=%s", level_name, log_path)
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取命名空间 logger（各模块的统一入口）。

    用法（推荐在各模块顶部）：
        from core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("...")
        logger.error("...", exc_info=True)  # 自动附带异常堆栈

    说明：
        - 首次调用会自动触发 init_logging（保证任何入口进来都有日志配置）
        - name 传 __name__ 时，日志行自带模块路径（如 core.parser:42），
          一眼看出日志来自哪个文件哪一行
    """
    if not _initialized:
        init_logging()
    return logging.getLogger(name)
