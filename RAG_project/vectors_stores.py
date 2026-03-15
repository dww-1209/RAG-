from langchain_chroma import Chroma
import config_data as config
from dotenv import load_dotenv
load_dotenv()


class VectorStoreService(object):
    def __init__(self,embedding):
        """
        param: embedding : 嵌入模型的传入
        """
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})
    
if __name__ =="__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    service = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4"))
    
    # 获取所有文档
    all_data = service.vector_store.get()
    print(f"总文档数：{len(all_data['ids'])}")
    for i, (doc_id, text, meta) in enumerate(zip(all_data['ids'], all_data['documents'], all_data['metadatas'])):
        print(f"\n--- 文档 {i} ---")
        print(f"ID: {doc_id}")
        print(f"内容前50字: {text[:50]}...")
        print(f"元数据: {meta}")