from vectors_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as  config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory,RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history

def format_document(docs:list[Document]):

    # 文档拼接格式说明：
    # - 文档之间加换行（\n\n）以明确分隔，防止信息混淆，帮助LLM识别独立片段。
    # - 附带元数据（如来源、时间）提供背景信息，增强回答可信度、可追溯性，便于调试。

    if not docs :
        return "无相关资料"
    
    formatted_str = ""
    for doc in docs :
        formatted_str+=f"文档片段：{doc.page_content}\n 文档元数据：{doc.metadata}\n\n"

    return formatted_str


class rag_service(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system","以我提供的已知资料为主，"
                 "简介和专业的回答用户问题，参考资料{context}\n如果{context}没有给出相关资料"
                 "请有礼貌的表达不知道，不要理会用户的提问"),
                 ("system","并且我提供用户的历史会话记录，会话记录如下："),
                 MessagesPlaceholder("history"),
                ("user","请回答用户的提问：{input}")
            ]
        ) 

        self.chat_model = ChatTongyi(
            model=config.chat_model_name,
            temperature = 0.5
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行chain"""
        retriever = self.vector_service.get_retriever() #retriever 返回检索结果


        chain = (
            RunnablePassthrough.assign(
                context = RunnableLambda(lambda x: x['input']) |retriever|format_document
            )
        ) | self.prompt_template |self.chat_model |StrOutputParser()


        #创建增强链，记录会话
        converstion_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        return converstion_chain

if __name__ =="__main__":
    #session_id 配置
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }
    
    res = rag_service().chain.invoke({"input":"我体重180斤，尺码推荐"},session_config)
    print(res)