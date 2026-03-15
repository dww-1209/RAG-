import streamlit as st 
from rag import rag_service 
import config_data as config
from file_history_store import get_history
from langchain_core.messages import HumanMessage,AIMessage

#set title
st.title("智能客服")
st.divider() # 分隔符

if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"你好，有什么可以帮助你"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = rag_service()

if "session_id" not in st.session_state:
    st.session_state.session_id= 'user_001'

if "session_config" not in st.session_state:
    st.session_state.session_config = {"configurable":{
        "session_id":st.session_state.session_id
    }}

# 定义从文件加载消息到 session_state 的函数
def load_messages_from_history(session_id):
    """从文件历史中加载消息，转换为页面显示格式，存入 st.session_state.message"""
    history = get_history(session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            continue  # 忽略 system 等类型
        messages.append({"role": role, "content": msg.content})
    # 如果历史为空，添加一条默认欢迎语
    if not messages:
        messages = [{"role": "assistant", "content": "你好，有什么可以帮助你"}]
    st.session_state["message"] = messages

with st.sidebar:
    st.header("会话管理")
    new_session_id = st.text_input("会话 ID", value=st.session_state.session_id)

    #创建一个复选框，提示“清除历史记录（开始新对话）”
    clear_history = st.checkbox("清除历史记录（开始新对话）")

    #创建一个按钮，标签为“加载/切换会话”。
    #当用户点击按钮时，内部的代码块才会执行。
    if st.button("加载/切换会话"):
        # 更新会话 ID
        st.session_state.session_id = new_session_id
        # 如果勾选了清除，则清空该会话的历史文件
        if clear_history:
            history = get_history(st.session_state.session_id)
            history.clear()
        # 更新 session_config
        st.session_state.session_config = {"configurable": {"session_id": st.session_state.session_id}}
        # 重新从文件加载消息到界面
        load_messages_from_history(st.session_state.session_id)
        st.rerun()  # 立即刷新页面显示新会话

for message in st.session_state["message"]:
    st.chat_message(name=message["role"]).write(message["content"])
#在页面最下方提供用户输入
prompt = st.chat_input()

if prompt :
    #在页面输出用户提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})
    with st.spinner("AI思考中……"):
        result_stream = st.session_state["rag"].chain.stream({"input":prompt},st.session_state)
        with st.chat_message('assistant'):
            full_response = st.write_stream(result_stream)
        #保存助手回复
        st.session_state["message"].append({"role":"assistant","content":full_response})