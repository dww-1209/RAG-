#如果想封装历史记录，除了自行维护历史消息外，也可以借助Langchain内置的历史记录附加功能
#Langchian提供了History类来管理对话历史，可以自动将历史消息附加到当前输入中，方便模型生成更连贯的回复
#基于RunnbleWithMessageHistory在原有链的基础上创建带有历史记录功能的新链
#基于InMemoryChatMessageHistory为历史记录提供内存存储，适合简单的对话场景


import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
load_dotenv()

def print_prompt(prompt):
    print("="*20,prompt.to_string(),"="*20)
    return prompt

#获取指定会话历史记录的函数
chat_history_store = {} #存放多个绘画ID所对应的历史会话记录
def get_chat_history(conversation_id):
    """
    函数传入为会话ID（字符串类型）
    函数要求返回BaseChatMessageHistory的实例，包含该会话的历史消息记录
    函数内部会根据会话ID检查chat_history_store字典中是否存在
    InMemoryChatMessageHistory是官方自带的基于内存存放历史纪录的类
    """
    if conversation_id not in chat_history_store:

        chat_history_store[conversation_id] = InMemoryChatMessageHistory()

    return chat_history_store[conversation_id]

# model = ChatOpenAI(
#     model="qwen3-max-2026-01-23",
#     openai_api_key=os.getenv("DASHSCOPE_API_KEY"),  # 注意：这里调用 os.getenv 获取真实值
#     openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     temperature=0.5
# )

model = ChatTongyi(
    model="qwen3-max-2026-01-23",
    temperature=0.5
)
str_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个AI助手,根据会话历史回应用户问题"),
        MessagesPlaceholder("history"), #占位符，表示历史消息会被注入到这里
        ("human", "{input}")
    ]
)
base_chain = prompt |print_prompt | model | str_parser

#通过RunnableWithMessageHistory获取一个新的带有历史记录的chain
conversation_chain = RunnableWithMessageHistory(
    runnable=base_chain,               #被附加历史消息的Runnable，通常是chain
    get_session_history=get_chat_history,  #获取指定会话历史记录的函数，默认为None，表示获取当前会话的历史记录
    input_messages_key="input",            #声明用户输入消息在模板中的占位符名称，默认为"input"
    history_messages_key="history"         #声明历史消息在模板中的占位符名称，默认为"history"
)

if __name__ == "__main__":
    conversation_id = "conversation_1" #会话ID，可以是任意字符串，唯一标识一个对话会话
    # 会话配置：双层字典结构
    # 外层 {"configurable": ...} 是LangChain的标准config格式，configurable是固定键名，用于存放可配置参数
    # 内层 {"session_id": conversation_id} 是具体的可配置项，session_id会被传递给get_session_history函数
    session_config = {"configurable":{"session_id": conversation_id}}
    for i in range(4):
        user_input = input("用户输入：")
        response = conversation_chain.invoke({"input": user_input},config=session_config)
        print("AI回复：", response)