import os
import json
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import message_to_dict,messages_from_dict
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()


# message_to_dict函数,将当个消息对象（Basemessage类） -> 字典格式
# messages_from_dict函数,将字典格式的消息转换为消息对象

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id,storage_path):
        self.session_id = session_id
        self.storage_path = storage_path

        self.file_path = os.path.join(self.storage_path, self.session_id + ".json")

        #确定文件是否存在
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)


    def add_message(self, message):
        # 将消息转换为字典格式
        message_dict = message_to_dict(message)
        
        # 读取现有的历史消息
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                existing_messages = json.load(f)
        else:
            existing_messages = []

        # 添加新消息
        existing_messages.append(message_dict)

        # 将更新后的消息列表写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(existing_messages, f)

    #获取消息记录
    @property     #@property装饰器将messages方法转换为属性，使得可以通过history.messages来访问消息记录，而不需要调用history.messages()方法
    def messages(self):
        #当前文件内容：list[字典]
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_dict_list = json.load(f)   #返回值是list[字典]
                return messages_from_dict(messages_dict_list)  #将list[字典]转换为list[消息对象]
            
        except FileNotFoundError:
            return []
        
    def clear(self):
        #清空历史记录
        if os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)  #写入一个空列表，表示清空历史记录
                print(f"历史记录已清空，文件路径：{self.file_path},会话ID：{self.session_id}")

# def print_prompt(prompt):
#     print("="*20,prompt.to_string(),"="*20)

choice= input("是否清空历史记录？(y/n):")

try:    
    if choice.lower() == 'y':
        A = FileChatMessageHistory(session_id="conversation_1", storage_path="RAG_learning/chat_histories")
        A.clear()  #清空历史记录
    if choice.lower() == 'n':
        print("历史记录未清空，将继续使用现有历史记录进行对话。")

except Exception as e:
    print(f"对话记录清除失败，本次回复将加载历史信息，请输入正确的 y 或 n")

#获取指定会话历史记录的函数
chat_history_store = {} #存放多个ID所对应的历史会话记录
def get_chat_history(conversation_id):
    return FileChatMessageHistory(session_id=conversation_id, storage_path="RAG_learning/chat_histories")


model = ChatTongyi(
    model="qwen3-max-2026-01-23",
    temperature=0.5
)
str_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个友好的AI助手,用户将会向你提问一些问题，结合历史信息进行回答"),
        #("system", "你是一个AI coding助手,用户将会向你提问一些算法相关问题，结合历史信息进行回答，尽可能给出详细的解答和代码示例。"),
        MessagesPlaceholder("history"), #占位符，表示历史消息会被注入到这里
        ("human", "{input}")
    ]
)
base_chain = prompt | model | str_parser

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
    print("请输入quit或按回车键以结束对话。")

    while True:
        print("AI回复：您好，很高兴为您服务，有什么可以帮您的吗")
        user_input = input("用户输入：")
        if user_input.lower() == "quit" or user_input == "":
            print("对话结束，祝您生活愉快，再见！")
            break
        # response = conversation_chain.invoke({"input": user_input},config=session_config)
        # print("AI回复：", response)
        for chunk in conversation_chain.stream(input={"input": user_input},config=session_config):
            print(chunk,end="",flush=True)

