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
| **智谱 Embedding API** | `core/embedder.py`         | 调用智谱 embedding-2（备用）          |
| **本地嵌入模型**           | `core/local_embedder.py`   | 离线加载 Sentence-Transformers 模型 |
| **统一工厂**             | `core/embedder_factory.py` | 一行配置切换智谱/本地模型                 |
| **大模型封装**            | `core/llm.py`              | 调用智谱 glm-4-flash
# 设计亮点
1. 统一抽象层：get_embedder() 工厂函数，调用方完全不用关心底层是 API 还是本地模型
2. 热切换配置：改 .env 里的 EMBEDDER_TYPE 即可切换，零代码改动
3. 完全离线支持：本地模型加载后无需网络，适合内网部署
4. 单例模式：避免重复加载模型，节省内存              |


# 模块2：文档处理
| 组件        | 文件                | 功能                     |
| --------- | ----------------- | ---------------------- |
| **文档解析器** | `core/parser.py`  | 自动识别 PDF/TXT/Word/Markdown，按页返回结构化文本 |
| **文本切分器** | `core/chunker.py` | 按句子边界切分，带重叠，保留页码来源     |
# 设计亮点
1. 按语义切分：不是暴力固定长度，而是按句子边界（。！？.!?）切分，保证每个 chunk 语义完整
2. 滑动窗口重叠：相邻 chunk 有 overlap 字符重叠，避免关键信息被切在边界上
3. 元数据追踪：每个 chunk 自带 page、chunk_index、source，后续溯源展示直接能用
4. 可配置参数：chunk_size、overlap、min_chunk_size 可调，适配不同文档类型


# 模块3：检索流程
| 组件             | 文件                       | 功能                                 |
| -------------- | ------------------------ | ---------------------------------- |
| **文档检索器**      | `core/retriever.py`      | 问题向量化 → ChromaDB 检索 → 格式化 Top-K 结果 |
| **Prompt 构建器** | `core/prompt_builder.py` | 检索结果 + 用户问题 → 完整 RAG Prompt        |
# 设计亮点
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
| **Pydantic 模型** | `models/schemas.py` | 统一请求/响应数据结构，自动校验和文档生成                  |
| **上传接口**        | `api/upload.py`     | 接收 PDF/TXT → 保存 → 解析 → 切分 → 向量化 → 入库   |
| **问答接口**        | `api/chat.py`       | 接收问题 → 检索 → 拼 Prompt → 调 LLM → 返回答案+来源 |
| **文档管理**        | `api/documents.py`  | 列出已上传文档、删除文档（同步清理向量+本地文件）              |
| **FastAPI 入口**  | `main.py`           | 注册路由、CORS、健康检查、Swagger 文档              |
# 设计亮点
1. 自动 API 文档：Pydantic + FastAPI 自动生成 Swagger UI（/docs），无需手写接口文档
2. 统一响应格式：所有接口返回结构化 JSON，成功/失败格式一致，前端好处理
3. 完整上传链路：单接口覆盖"文件保存 → 解析 → 切分 → 嵌入 → 存储"全流程
4. 清理一致性：删除文档时同时删向量库记录和本地文件，避免脏数据
5. CORS 配置：开箱即用支持前端跨域，为块5 Streamlit/前端页面做准备


# 模块5：前端界面
| 组件               | 文件                | 功能           |
| ---------------- | ----------------- | ------------ |
| **Streamlit 前端** | `frontend/app.py` | 完整的 Web 交互界面 |
# 设计亮点
1. 会话历史持久化：st.session_state.messages 保存多轮对话，页面刷新不丢失
2. 来源可追溯：每个 AI 回复附带 📎 查看来源，展示原始 chunk 内容、页码、相似度分数
3. 实时反馈：上传、检索、生成均有 spinner 加载提示，用户体验流畅
4. 错误处理：后端断开、上传失败、问答异常均有友好提示
5. 响应式布局：侧边栏管理 + 主界面聊天，信息结构清晰

