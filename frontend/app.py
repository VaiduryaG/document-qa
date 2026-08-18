"""
Streamlit 前端界面
运行方式：streamlit run frontend/app.py
"""

import streamlit as st
import requests
import os


# ========== 配置 ==========
API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload/"             # 单文件上传（保留兼容）
UPLOAD_BATCH_ENDPOINT = f"{API_BASE_URL}/upload/batch"  # 批量上传（多选/拖拽）
CHAT_ENDPOINT = f"{API_BASE_URL}/chat/"
DOCS_ENDPOINT = f"{API_BASE_URL}/documents/"


# ========== 页面配置 ==========
st.set_page_config(
    page_title="智能文档问答助手",
    page_icon="📚",
    layout="wide"
)

st.title("📚 智能文档问答助手")
st.caption("基于 RAG 技术，上传文档即可智能问答")


# ========== 侧边栏：文档管理 ==========
with st.sidebar:
    st.header("📁 文档管理")

    # 「清空选择」状态：key 变化时 Streamlit 会重建组件，等于清空已选文件。
    # 上传完成后 _update_uploader_key +1，下一帧 file_uploader 是全新空队列，
    # 避免「已解析文件还留在选择框里，下次不注意重复上传」的问题
    if "_uploader_key" not in st.session_state:
        st.session_state._uploader_key = 0

    # 上传文件
    # accept_multiple_files=True 开启多选；st.file_uploader 本身自带
    # 拖拽投放区（Drag and drop），多选开启后即可一次拖入多个文件
    uploaded_files = st.file_uploader(
        "上传文档（可多选 / 拖拽到此处）",
        type=["pdf", "txt", "md", "docx", "jpg", "jpeg", "png"],
        accept_multiple_files=True,   # 关键：允许多文件（同时支持拖入多个）
        help="支持 PDF（含扫描件）/TXT/Markdown/Word/图片，可一次选择或拖入多个文件",
        key=f"uploader_{st.session_state._uploader_key}",  # 换 key = 清空选择
    )

    if uploaded_files:
        st.caption(f"已选择 {len(uploaded_files)} 个文件：")
        for f in uploaded_files:
            st.text(f"📄 {f.name}")

        # 手动清空按钮（不upload也要能主动清掉选错的文件）
        clear_clicked = st.button("🧹 清空选择", use_container_width=True)
        if clear_clicked:
            st.session_state._uploader_key += 1
            st.rerun()

        if st.button("📤 上传并解析", use_container_width=True):
            # 组装多文件表单：files 字段传 (name, bytes) 列表
            multipart = [
                ("files", (f.name, f.getvalue())) for f in uploaded_files
            ]
            with st.spinner(f"正在解析 {len(uploaded_files)} 个文件（扫描件OCR较慢，请耐心等待）..."):
                try:
                    response = requests.post(
                        UPLOAD_BATCH_ENDPOINT, files=multipart, timeout=300
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # 逐文件展示结果（成功/失败隔离）
                        for item in data["results"]:
                            if item["status"] == "success":
                                st.success(f"✅ {item['filename']}（{item['chunks']} 个片段）")
                            else:
                                st.error(f"❌ {item['filename']}：{item['message']}")
                        st.info(f"批量上传完成：成功 {data['succeeded']} / 失败 {data['failed']} / 共 {data['total']}")
                        # 上传流程结束即清空选择队列：
                        # - 成功的文件已在库里，留着只会造成误重传
                        # - 失败的文件重试前往往需要先处理（换格式等），也应重新选择
                        st.session_state._uploader_key += 1
                        st.rerun()
                    else:
                        st.error(f"批量上传失败: {response.json().get('detail', '未知错误')}")
                except Exception as e:
                    st.error(f"请求失败: {str(e)}")
    
    st.divider()
    
    # 文档列表
    st.subheader("已上传文档")
    try:
        resp = requests.get(DOCS_ENDPOINT, timeout=10)
        if resp.status_code == 200:
            docs = resp.json().get("documents", [])
            if not docs:
                st.info("暂无文档，请先上传")
            else:
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"📄 {doc['filename']}")
                        st.caption(f"{doc['chunks']} 个片段")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['id']}"):
                            try:
                                del_resp = requests.delete(
                                    f"{DOCS_ENDPOINT}{doc['filename']}",
                                    timeout=10
                                )
                                if del_resp.status_code == 200:
                                    st.success("已删除")
                                    st.rerun()
                                else:
                                    st.error("删除失败")
                            except Exception as e:
                                st.error(f"删除出错: {str(e)}")
        else:
            st.warning("无法获取文档列表")
    except Exception as e:
        st.error(f"连接后端失败: {str(e)}\n\n请确保 FastAPI 服务已启动：\n`python main.py`")


# ========== 主界面：问答 ==========
st.header("💬 智能问答")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 显示来源（如果是 AI 回复）
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 查看来源"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(
                        f"**来源 {i}** | `{src['source']}` 第{src['page']}页 "
                        f"(相关度: {src['score']:.2f})"
                    )
                    st.info(src["content"][:300] + "...")

# 用户输入
if question := st.chat_input("请输入你的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)
    
    # AI 回复
    with st.chat_message("assistant"):
        with st.spinner("正在检索文档并生成回答..."):
            try:
                response = requests.post(
                    CHAT_ENDPOINT,
                    json={"question": question, "top_k": 3},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    # 显示答案
                    st.markdown(answer)
                    
                    # 显示来源（可折叠）
                    if sources:
                        with st.expander("📎 查看来源"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(
                                    f"**来源 {i}** | `{src['source']}` 第{src['page']}页 "
                                    f"(相关度: {src['score']:.2f})"
                                )
                                st.info(src["content"][:300] + "...")
                    
                    # 保存到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = response.json().get("detail", "请求失败")
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}"
                    })
                    
            except Exception as e:
                st.error(f"请求出错: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ 连接失败: {str(e)}"
                })


# ========== 底部操作 ==========
st.divider()
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("🔄 刷新文档列表", use_container_width=True):
        st.rerun()