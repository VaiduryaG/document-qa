"""
数据访问层（阶段 2.2）
======================
用途：把 SQLite 四张表（users/documents/conversations/messages）的读写
      封装成函数，供后续阶段的 API 层调用（阶段 2.3~2.6）：
        - 阶段 2.3 api/documents.py  -> list_documents / delete_document
        - 阶段 2.4 api/upload.py     -> insert_document
        - 阶段 2.6 api/chat.py       -> ensure_conversation / add_message / list_messages

设计约定：
    - 返回普通 dict（由 sqlite3.Row 转换），调用方不需要感知数据库细节
    - 「确保存在」类操作（ensure_user/ensure_conversation）用 INSERT OR IGNORE
      实现幂等：调多少次都不会产生重复行
    - insert_document 是应用层 upsert：同文件名重复上传时替换旧行，
      保持「一个文件一条台账」
    - 查询类函数不修改数据；写操作底层已加线程锁（见 sqlite_store.execute）
"""

from typing import List, Dict, Optional

from db.sqlite_store import get_sqlite_store


# ================= 用户 =================

def ensure_user(user_id: str) -> None:
    """
    确保用户存在：不存在则创建，已存在则什么都不做（幂等）。

    参数：
        user_id : 业务用户标识，如 "default"、后续多用户时的 "alice" 等
    """
    store = get_sqlite_store()
    # users.user_id 有 UNIQUE 约束，INSERT OR IGNORE 靠它去重
    store.execute(
        "INSERT OR IGNORE INTO users (user_id) VALUES (?)",
        (user_id,)
    )


# ================= 文档元数据 =================

def insert_document(filename: str, chunks: int, user_id: str = "default") -> int:
    """
    登记一篇文档的元数据（上传成功后调用）。

    upsert 语义：同一文件名重复上传时，先删旧行再插新行——
    保持「一个文件一条台账」，chunks 数和上传时间取最新值。

    参数：
        filename : 文件名（与 ChromaDB metadata.filename 一致，作关联键）
        chunks   : 本次切分出的片段数
        user_id  : 上传者

    返回：
        int：新登记行的自增 id
    """
    store = get_sqlite_store()
    # documents.filename 没建唯一约束（2.1 已定稿的 schema），这里应用层保证去重
    store.execute(
        "DELETE FROM documents WHERE filename = ?",
        (filename,)
    )
    cur = store.execute(
        "INSERT INTO documents (filename, user_id, chunks) VALUES (?, ?, ?)",
        (filename, user_id, chunks)
    )
    return cur.lastrowid


def list_documents(user_id: Optional[str] = None) -> List[Dict]:
    """
    列出文档元数据，最近上传的排前面。

    参数：
        user_id : 传 None 列所有用户的文档；传具体值只列该用户的

    返回：
        [{"id":1, "filename":"a.pdf", "user_id":"default",
          "chunks":6, "uploaded_at":"2026-08-17 11:14:33"}, ...]
    """
    store = get_sqlite_store()
    base_sql = (
        "SELECT id, filename, user_id, chunks, uploaded_at FROM documents"
    )
    if user_id is None:
        rows = store.query(
            base_sql + " ORDER BY uploaded_at DESC, id DESC"
        )
    else:
        rows = store.query(
            base_sql + " WHERE user_id = ? ORDER BY uploaded_at DESC, id DESC",
            (user_id,)
        )
    return [dict(r) for r in rows]


def get_document(filename: str) -> Optional[Dict]:
    """
    按文件名查单篇文档元数据。

    返回：
        dict：文档行；不存在返回 None。
    """
    store = get_sqlite_store()
    rows = store.query(
        "SELECT id, filename, user_id, chunks, uploaded_at "
        "FROM documents WHERE filename = ? LIMIT 1",
        (filename,)
    )
    return dict(rows[0]) if rows else None


def delete_document(filename: str) -> int:
    """
    删除一篇文档的登记行。

    返回：
        int：实际删除的行数（0 表示本来就不存在，调用方可据此判断）。
    """
    store = get_sqlite_store()
    cur = store.execute(
        "DELETE FROM documents WHERE filename = ?",
        (filename,)
    )
    return cur.rowcount


# ================= 对话 =================

def ensure_conversation(
    conversation_id: str,
    user_id: str = "default",
    title: str = ""
) -> None:
    """
    确保对话存在：不存在则创建（带标题），已存在则什么都不做（幂等）。

    参数：
        conversation_id : 业务对话标识（前端生成或传入，如 "conv-abc123"）
        user_id         : 归属用户
        title           : 对话标题（建议用首条问题截断，如问题前 30 字）
    """
    store = get_sqlite_store()
    # conversations.conversation_id 有 UNIQUE 约束，OR IGNORE 幂等
    store.execute(
        "INSERT OR IGNORE INTO conversations (conversation_id, user_id, title) "
        "VALUES (?, ?, ?)",
        (conversation_id, user_id, title)
    )


def list_conversations(user_id: Optional[str] = None) -> List[Dict]:
    """
    列出对话，最新创建的排前面（供后续前端做「历史会话列表」）。
    user_id 为 None 时列全部。
    """
    store = get_sqlite_store()
    base_sql = (
        "SELECT id, conversation_id, user_id, title, created_at FROM conversations"
    )
    if user_id is None:
        rows = store.query(base_sql + " ORDER BY created_at DESC, id DESC")
    else:
        rows = store.query(
            base_sql + " WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,)
        )
    return [dict(r) for r in rows]


# ================= 消息 =================

def add_message(conversation_id: str, role: str, content: str) -> int:
    """
    向对话追加一条消息（阶段 2.6 每轮问答后调两次：存 user 问题 + assistant 回答）。

    参数：
        conversation_id : 所属对话
        role            : "user" 或 "assistant"（表级 CHECK 约束，其他值会抛异常）
        content         : 消息内容

    返回：
        int：消息自增 id。

    说明：
        内部自动 ensure_conversation 兜底——即使调用方忘了先建对话，
        也不会产生孤儿消息（防止历史丢失）。
    """
    store = get_sqlite_store()
    # 兜底：对话不存在时先建（已存在则为空操作）
    ensure_conversation(conversation_id)
    cur = store.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content)
    )
    return cur.lastrowid


def list_messages(conversation_id: str, limit: Optional[int] = None) -> List[Dict]:
    """
    取对话内的消息，按时间正序返回（ oldest 在前），可直接拼接为 LLM 历史记录。

    参数：
        conversation_id : 对话标识
        limit           : 只取「最近 N 条」。注意返回时仍按时间正序排列——
                          先按 id 倒序取最新 N 条，再反转回正序，方便直接喂给
                          prompt_builder 的 chat_history。

    返回：
        [{"id":1, "conversation_id":"c1", "role":"user",
          "content":"...", "created_at":"..."}, ...]
        对话不存在或无消息时返回空列表。
    """
    store = get_sqlite_store()

    if limit is None:
        # 不限条数：直接按 id 升序（id 自增即插入顺序，天然时间序）
        rows = store.query(
            "SELECT id, conversation_id, role, content, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,)
        )
    else:
        # 限条数：子查询取最近 N 条（id 倒序 + LIMIT），外层再转回正序
        rows = store.query(
            "SELECT id, conversation_id, role, content, created_at FROM ("
            "  SELECT * FROM messages WHERE conversation_id = ? "
            "  ORDER BY id DESC LIMIT ?"
            ") ORDER BY id ASC",
            (conversation_id, limit)
        )
    return [dict(r) for r in rows]
