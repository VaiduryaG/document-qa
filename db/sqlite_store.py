"""
SQLite 持久化存储模块（阶段 2.1）
====================================
用途：用原生 sqlite3（标准库，零依赖）持久化 ChromaDB 管不了的结构化数据：
      - 文档元数据（文件名、片段数、上传时间）-- 现在散落在 ChromaDB chunk
        metadata 里靠聚合反推，且查不到"上传时间"这类信息
      - 用户（轻量 user_id 预留，不做登录鉴权）
      - 对话（conversation）与消息（message）-- 聊天记录持久化，
        重启服务后多轮对话不丢失

设计：
    - 单文件数据库（默认 app.db），连接单例，全项目共用一个连接
    - 建表语句集中在 SCHEMA，首次调用自动建表（幂等，IF NOT EXISTS）
    - 开启 WAL 模式：读写并发更友好，Streamlit + FastAPI 同时访问不易锁库
    - 外键约束显式开启（SQLite 默认关闭），保证引用一致性

与 ChromaDB 的分工（不替代、只协作）：
    - ChromaDB：向量 + chunk 文本，负责"语义检索"
    - SQLite  ：用户/文档元数据/对话记录，负责"结构化台账"
"""

import os
import sqlite3
import threading
from typing import Optional

from dotenv import load_dotenv

# 加载项目根目录 .env（读取 DB_PATH 配置）
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_env_path)

# 默认数据库文件：项目根目录下的 app.db
_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.db"
)


# ========== 建表语句（幂等：IF NOT EXISTS，可反复执行） ==========
SCHEMA = """
-- 用户表：轻量预留，不做登录鉴权
-- 一个 user_id 对应一个使用者；默认可以用 "default" 单用户跑通
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,  -- 内部自增主键
    user_id    TEXT    NOT NULL UNIQUE,            -- 业务用户标识（前端传入，如 "default"）
    created_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))  -- 创建时间
);

-- 文档元数据表：每次上传成功后登记一行
-- 与 ChromaDB 的关系：Chroma 里存的是该文档切分后的 chunk 向量，
-- 这里只存"台账信息"（谁传的、什么时候传的、切了几片）
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,          -- 文件名（与 ChromaDB metadata.filename 一致，作为关联键）
    user_id     TEXT    NOT NULL DEFAULT 'default',  -- 上传者（预留多用户）
    chunks      INTEGER NOT NULL DEFAULT 0,         -- 切分出的片段数
    uploaded_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))  -- 上传时间
);

-- 对话表：一轮会话一个 conversation（同一用户可开多个对话）
CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL UNIQUE,        -- 业务对话标识（前端传入或自动生成）
    user_id         TEXT    NOT NULL DEFAULT 'default',
    title           TEXT    NOT NULL DEFAULT '',    -- 对话标题（可先用首条问题截断）
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 消息表：对话内的每一条 user/assistant 消息
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,               -- 所属对话（关联 conversations.conversation_id）
    role            TEXT    NOT NULL CHECK(role IN ('user','assistant')),  -- 角色
    content         TEXT    NOT NULL,               -- 消息内容
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 消息表辅助索引：按对话取时间有序消息是高频查询（阶段 2.6 加载历史用）
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
-- 文档表辅助索引：按用户列文档是高频查询
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
"""


class SQLiteStore:
    """
    SQLite 连接封装（线程安全单例）

    说明：
        - 整个项目共用一个连接实例（get_sqlite_store() 获取）
        - check_same_thread=False 允许跨线程使用；
          配合类内 threading.Lock 保证写操作串行化，避免并发写冲突
        - 每次操作用完的 cursor 自动释放；连接进程退出时自动关闭
    """

    _instance: Optional["SQLiteStore"] = None
    _lock = threading.Lock()  # 双重检查锁，保证多线程下只建一个实例

    def __new__(cls, db_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_store(db_path)
        return cls._instance

    def _init_store(self, db_path: Optional[str] = None):
        """
        实际初始化（仅单例首次创建时执行一次）。

        参数：
            db_path : 数据库文件路径。None 时依次取：
                      1) .env 里的 DB_PATH
                      2) 默认 <项目根>/app.db
        """
        if db_path is None:
            db_path = os.getenv("DB_PATH", _DEFAULT_DB_PATH)
        # 相对路径统一锚定到项目根，避免受「从哪个目录启动」影响
        if not os.path.isabs(db_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(project_root, db_path)

        self.db_path = db_path
        self._write_lock = threading.Lock()  # 写操作串行化锁

        # 建父目录（如 DB_PATH 指向子目录）
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        # check_same_thread=False：FastAPI 线程池 + Streamlit 都会跨线程访问
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # 查询结果可按列名取值（dict 风格）
        # 显式开启外键约束（SQLite 默认关闭）
        self.conn.execute("PRAGMA foreign_keys = ON")
        # WAL 模式：写不阻塞读，多进程/多线程并发更稳
        self.conn.execute("PRAGMA journal_mode = WAL")

        # 首次运行自动建表（幂等）
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------- 基础执行入口（数据访问层在阶段 2.2 封装） ----------
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        执行写操作（INSERT/UPDATE/DELETE/DDL），线程安全。

        加锁原因：SQLite 单写者模型，两个线程同时写同一连接会抛
        "cannot start a transaction within a transaction"。
        """
        with self._write_lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list:
        """
        执行读操作（SELECT），返回 sqlite3.Row 列表（可按列名访问）。
        读操作不加写锁（WAL 模式下读写互不阻塞）。
        """
        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows

    def close(self):
        """关闭连接（通常无需手动调用，进程退出自动释放）。"""
        with self._write_lock:
            self.conn.close()


# ========== 单例便捷入口 ==========
_sqlite_store: Optional[SQLiteStore] = None


def get_sqlite_store(db_path: Optional[str] = None) -> SQLiteStore:
    """
    获取 SQLiteStore 单例。

    首次调用时建库建表；之后调用直接返回已有实例
    （db_path 参数仅在首次调用时生效）。
    """
    global _sqlite_store
    if _sqlite_store is None:
        _sqlite_store = SQLiteStore(db_path)
    return _sqlite_store
