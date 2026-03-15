import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#md5
md5_path = os.path.join(BASE_DIR,"md5.text")

#Chroma
collection_name = "rag"
persist_directory = os.path.join(BASE_DIR, "chroma_db")

#spliter
chunk_size = 1000       #每个分段允许最大长度为1000个字符
chunk_overlap = 100     #两个分段之间重叠的字符最大允许100
separators = ["\n\n","\n",".","!","?","。","！","？"," ",""]
max_split_char_number = 20


#相似度检索的阈值K
similarity_threshold = 1  #检索文档返回的数量

#模型选择
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max-2026-01-23"

#会话记录地址
chat_history_path = os.path.join(BASE_DIR,"chat_history")

#session_id
session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }