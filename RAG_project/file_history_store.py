import os
import json
import config_data as config
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import messages_from_dict,message_to_dict

def get_history(session_id):
    return FileChatMessageHistory(session_id,storage_path=config.chat_history_path)

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
