# 快速启动
uvicorn main:app --reload --port 8000
streamlit run frontend/app.py
# 文件结构
document-qa/
├── main.py                 # FastAPI入口
├── api/
│   ├── __init__.py         # 包标识
│   ├── upload.py           # 上传文档接口
│   ├── chat.py             # 问答接口
│   └── documents.py        # 文档管理接口
├── core/
│   ├── .env                # API_KEY
│   ├── parser.py           # 文档解析（PDF/TXT）
│   ├── chunker.py          # 文本切分
│   ├── embedder.py         # 调用智谱Embedding API
│   ├── local_embedder.py   # 本地嵌入模型（Sentence-Transformers）
│   ├── embedder_factory.py # 嵌入模型统一入口（支持切换）
│   ├── llm.py              # 调用智谱大模型API
│   ├── ocr.py              # OCR 引擎封装（RapidOCR，扫描型PDF/图片识别）
│   ├── logger.py           # 日志配置（控制台+轮转文件双通道，阶段3）
│   ├── retriever.py        # ChromaDB检索
│   └── prompt_builder.py   # Prompt拼接
├── db/
│   └── chroma_client.py    # ChromaDB封装
├── frontend/               # Streamlit前端界面
│   ├── __init__.py
│   └── app.py              # 前端应用入口
├── models/
│   └── schemas.py          # Pydantic模型
├── model_weights/          # 本地模型文件（离线嵌入）
│   └── paraphrase-multilingual-MiniLM-L12-v2/
│       ├── 1_Pooling/           
│       │   └── config.json      
│       ├── config.json
│       ├── model.safetensors
│       ├── modules.json
│       ├── sentencepiece.bpe.model
│       ├── tokenizer.json
│       └── ...
├── static/                 # 前端文件（如果用HTML）
├── uploaded_files/         # 上传文档存储
├── requirements.txt
└── README.md

代码开发流程
# 模块1：基础设施
| 组件                   | 文件                         | 功能                            |
| -------------------- | -------------------------- | ----------------------------- |
| **ChromaDB 封装**      | `db/chroma_client.py`      | 向量存储、检索、过滤、删除                 |
| **SQLite 封装**        | `db/sqlite_store.py`       | 用户/文档元数据/对话记录持久化（原生sqlite3，WAL模式） |
| **智谱 Embedding API** | `core/embedder.py`         | 调用智谱 embedding-2（备用）          |
| **本地嵌入模型**           | `core/local_embedder.py`   | 离线加载 Sentence-Transformers 模型 |
| **统一工厂**             | `core/embedder_factory.py` | 一行配置切换智谱/本地模型                 |
| **大模型封装**            | `core/llm.py`              | 调用智谱 glm-4-flash
| **OCR 引擎**             | `core/ocr.py`             | 封装 RapidOCR，识别扫描型PDF/图片中的文字
## 设计亮点
1. 统一抽象层：get_embedder() 工厂函数，调用方完全不用关心底层是 API 还是本地模型
2. 热切换配置：改 .env 里的 EMBEDDER_TYPE 即可切换，零代码改动
3. 完全离线支持：本地模型加载后无需网络，适合内网部署
4. 单例模式：避免重复加载模型，节省内存              |


# 模块2：文档处理
| 组件        | 文件                | 功能                     |
| --------- | ----------------- | ---------------------- |
| **文档解析器** | `core/parser.py`  | 自动识别 PDF(含扫描型)/TXT/Word/Markdown/图片，按页返回结构化文本 |
| **文本切分器** | `core/chunker.py` | 按句子边界切分，带重叠，保留页码来源     |
## 设计亮点
1. 按语义切分：不是暴力固定长度，而是按句子边界（。！？.!?）切分，保证每个 chunk 语义完整
2. 滑动窗口重叠：相邻 chunk 有 overlap 字符重叠，避免关键信息被切在边界上
3. 元数据追踪：每个 chunk 自带 page、chunk_index、source，后续溯源展示直接能用
4. 可配置参数：chunk_size、overlap、min_chunk_size 可调，适配不同文档类型
5. OCR 智能回退：PDF 每页先抽原生文本（快、无损），抽不到才渲染成图走 OCR 兜底，普通 PDF 零额外开销
6. 双引擎协作：PyMuPDF 负责页面渲染（150 DPI），RapidOCR 负责文字识别，延迟加载只在需要时初始化模型


# 模块3：检索流程
| 组件             | 文件                       | 功能                                 |
| -------------- | ------------------------ | ---------------------------------- |
| **文档检索器**      | `core/retriever.py`      | 问题向量化 → ChromaDB 检索 → 格式化 Top-K 结果 |
| **Prompt 构建器** | `core/prompt_builder.py` | 检索结果 + 用户问题 → 完整 RAG Prompt        |
## 设计亮点
1. 检索结果格式化：自动将 ChromaDB 的原始输出（嵌套列表）转换为结构化字典，包含 content、page、source、score
2. 相似度分数转换：将余弦距离 dist 转为 1 - dist 的相似度分数，直观展示相关性
3. Prompt 工程：
    ·系统角色设定（"专业文档问答助手"）
    ·严格约束（禁止编造、必须标注来源）
    ·防幻觉机制（无检索结果时明确告知无法回答）
4. 支持多轮对话：build_with_history() 方法可拼接历史记录，为后续聊天功能预留


# 模块4：API接口
| 组件              | 文件                  | 功能                                     |
| --------------- | ------------------- | -------------------------------------- |
| **Pydantic 模型** | `models/schemas.py` | 统一请求/响应数据结构（ChatRequest 含 user_id/conversation_id 可选字段），自动校验和文档生成                  |
| **上传接口**        | `api/upload.py`     | 单文件 /upload/ 与批量 /upload/batch 双接口：批量一次多文件、逐文件成败隔离；解析(含OCR回退) → 切分 → 向量化 → 入库 → 写SQLite台账   |
| **问答接口**        | `api/chat.py`       | 接收问题 → 加载会话历史(SQLite) → 检索 → 拼多轮Prompt → 调 LLM → 消息落库 → 返回答案+来源+会话id |
| **文档管理**        | `api/documents.py`  | 列出已上传文档（SQLite台账含上传时间）、删除文档（向量+台账+本地文件三处同步清理）              |
| **FastAPI 入口**  | `main.py`           | 注册路由、CORS、健康检查、Swagger 文档、请求日志中间件、全局异常处理器              |
## 设计亮点
1. 自动 API 文档：Pydantic + FastAPI 自动生成 Swagger UI（/docs），无需手写接口文档
2. 统一响应格式：所有接口返回结构化 JSON，成功/失败格式一致，前端好处理
3. 完整上传链路：单接口覆盖"文件保存 → 解析 → 切分 → 嵌入 → 存储"全流程
4. 空内容拒绝：纯图片 OCR 无文字、解析结果为空时返回 400 拒绝入库，避免空数据污染向量库
5. 异常分级：业务校验异常(400)原样上抛，内部异常(500)自动清理已保存文件，避免脏文件残留
6. 清理一致性：删除文档时三处同步清理——Chroma向量、SQLite台账行、本地文件；单边残留容错删除不报错，全部不存在才 404
7. CORS 配置：开箱即用支持前端跨域，为块5 Streamlit/前端页面做准备
8. 台账化列表：文档列表从 SQLite 直读（含上传时间），不再聚合 ChromaDB 全量 chunk 反推
9. 上传即记账：入库成功自动写 SQLite 台账，同名重传 upsert 只保留最新记录；写台账失败则整单失败回滚文件，杜绝「向量入库、台账漏登」的不一致；上传被拒（空白/过短）不产生台账行
10. 对话历史持久化：同一 conversation_id 的多轮问答自动加载历史拼进 Prompt（上限20条防撑爆上下文），"它/这个"等指代能接上上文；不传 id 自动开新会话并回传 id；重启服务历史不丢
11. 兼容旧调用：不带会话字段的请求行为与从前完全一致（首轮单轮 Prompt），前端逐步接入即可


# 模块5：前端界面
| 组件               | 文件                | 功能           |
| ---------------- | ----------------- | ------------ |
| **Streamlit 前端** | `frontend/app.py` | 完整的 Web 交互界面（多选/拖拽批量上传） |
## 设计亮点
1. 批量上传：一次选择/拖入多个文件，逐文件显示成功/失败结果，失败原因直接可见
2. 会话历史持久化：st.session_state.messages 保存多轮对话，页面刷新不丢失
2. 来源可追溯：每个 AI 回复附带 📎 查看来源，展示原始 chunk 内容、页码、相似度分数
3. 实时反馈：上传、检索、生成均有 spinner 加载提示，用户体验流畅
4. 错误处理：后端断开、上传失败、问答异常均有友好提示
5. 响应式布局：侧边栏管理 + 主界面聊天，信息结构清晰

# 模块6：OCR 能力（图片 + 扫描型 PDF）
| 组件        | 文件          | 功能                                                     |
| --------- | ----------- | ------------------------------------------------------ |
| **OCR 引擎** | `core/ocr.py` | 封装 RapidOCR（onnxruntime），对扫描型PDF渲染图/纯图片做文字识别，单例模式复用引擎 |
## 设计亮点
1. 零外部依赖：RapidOCR 自带 ONNX 模型权重，pip 安装即可用，无需联网下载模型
2. 三段式管线：检测(det)→方向分类(cls)→识别(rec)，自动纠正倒置文本，支持中英文混合
3. 单例 + 线程安全：三个 ONNX 模型只加载一次，多线程并发下也安全
4. 纯新增模块：本阶段不动任何现有代码，OCR 能力作为独立组件，后续阶段再接入 parser

# 模块7：SQLite 持久化（阶段2 进行中）
| 组件           | 文件                  | 功能                                                     |
| ------------ | ------------------- | ------------------------------------------------------ |
| **SQLite 封装** | `db/sqlite_store.py` | 原生 sqlite3 连接单例，自动建表（users/documents/conversations/messages），WAL并发 |
| **数据访问层**   | `db/repositories.py` | 四张表的 CRUD 函数封装（幂等 ensure / upsert / 正序消息查询），供 API 层调用 |
## 设计亮点
1. 与 ChromaDB 分工协作：向量检索归 Chroma，用户/元数据/对话台账归 SQLite，互不替代
2. 零依赖：Python 标准库 sqlite3，无需 pip 安装，单文件即整库（app.db）
3. WAL 模式 + 写锁：读写不互相阻塞，FastAPI/Streamlit 并发访问安全
4. 幂等建表：IF NOT EXISTS，首次运行自动建 4 表 2 索引，重复初始化无副作用
5. 轻量用户预留：user_id 字段预留多用户，默认 default 单用户即可跑通，不做登录鉴权
6. 数据完整性：外键约束显式开启，messages.role 加 CHECK 约束，时间戳自动填充
7. 业务级幂等：ensure_user/ensure_conversation 重复调用无副作用；insert_document 应用层 upsert，同文件重传只保留最新台账
8. 消息查询人性化：list_messages 天然时间正序、limit 取最近N条仍保持正序，可直接拼接 LLM 历史记录；add_message 自动兜底建对话，杜绝孤儿消息

# 模块8：异常处理与日志（阶段3 完成）
| 组件           | 文件              | 功能                                                     |
| ------------ | --------------- | ------------------------------------------------------ |
| **日志配置**     | `core/logger.py` | 标准 logging 双通道（控制台+5MB轮转文件），幂等初始化，模块名:行号定位 |
| **请求日志/全局异常** | `main.py`       | 每请求记 request_id/状态码/耗时日志；未捕获异常统一500结构不泄漏内部细节 |
| **业务层日志**    | `api/*.py` `core/*.py` | 全量 print->logger 替换；上传/问答/删除关键节点结构化日志；业务拒绝记WARNING、内部错误记堆栈且不泄漏细节 |
## 设计亮点
1. 零依赖：标准库 logging + RotatingFileHandler，无需 pip 安装
2. 双通道：控制台供开发实时看，logs/app.log 持久留痕（5MB x 3备份，约20MB上限）
3. 幂等初始化：重复调 init_logging 不叠加 handler，日志不会重复打印
4. 精准定位：日志行自带 模块名:行号（如 core.parser:42），一眼看出来源
5. 异常留痕：exc_info=True 自动附完整堆栈，排查问题不用复现
6. 环境可调：.env 里 LOG_LEVEL=DEBUG/INFO/WARNING 一改即生效，不动代码
7. 三方降噪：uvicorn/chromadb 等内部 INFO 压到 WARNING，日志信噪比高
8. 请求可追踪：每请求生成 request_id 贯穿日志并回传 X-Request-ID 响应头，报障凭id秒查
9. 4xx记WARNING/2xx记INFO：扫日志时一眼锁定异常请求；耗时毫秒级定位慢接口
10. 异常分级隔离：业务异常(HTTPException 400/404)原样返回detail；未捕获异常统一友好500+error_code，完整堆栈只进日志不进响应，既不泄漏内部实现又可凭error_code溯源
11. 关键节点留痕：上传(成功/格式拒/空内容拒/内部错)、问答(会话/历史/片段数/异常)、删除(向量/台账/文件三处明细)全链路结构化日志，排查与运营统计有据可查
12. 批量上传隔离：/upload/batch 一次多文件，单文件失败不影响其他；整体恒200成败看逐文件status；失败文件不登记台账、无本地残留
13. 重传幂等：同名文件重传前先整删旧向量再新增，杜绝旧残块污染检索（add对已有ID是静默跳过，块数变少时残块会留存）
14. 前端多选拖拽：file_uploader 开启 accept_multiple_files，自带拖拽区可一次拖入多个文件，结果逐文件展示；上传完成自动清空选择队列（换key重建组件），另配🧹手动清空按钮

