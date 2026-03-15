"""
创建向量存储，基于向量存储完成：
*存入向量 --> 先将文本转换为向量，再将向量存入向量数据库中 add_documents
*删除向量 --> 根据向量的id删除向量数据库中的向量 delete
*向量检索 --> 根据查询文本转换为向量，再根据向量在向量数据库中进行相似度检索，返回相关的文本信息 similarity_search
"""

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
load_dotenv()  # 加载环境变量，确保DASHSCOPE_API_KEY

# 创建向量存储对象，指定使用Chroma作为向量数据库，DashScopeEmbeddings作为文本向量化工具
vectorstore = Chroma(
    collection_name="my_collection",  #向量集合名称
    embedding_function=DashScopeEmbeddings(),  #文本向量化工具
    persist_directory="RAG_learning/vectorstore"  #向量数据库数据存储路径
)

# 加载CSV文件，提取文本内容并转换为Document对象
loader = CSVLoader(
    file_path="RAG_learning/data/stu.csv",  #CSV文件路径
    encoding="utf-8",  #文件编码格式
    source_column="name",  #CSV文件中包含文本内容的列名
    csv_args={"delimiter": ","}  #CSV文件的分隔符，默认为逗号
)

documents = loader.load()  #加载CSV文件，返回Document对象列表

vectorstore.add_documents(
    documents=documents,  #要存入向量数据库的Document,类型：list[Document]
    ids=["id"+str(i) for i in range(1,len(documents)+1)]  #每个Document对象对应的唯一标识符列表，如果为None，则会自动生成唯一ID
)

#删除，传入要删除的向量ID列表
vectorstore.delete(ids=["id1", "id2"])

results = vectorstore.similarity_search(
    query="郑怡芸",  #查询文本
    k=1  #返回最相似的前k个结果
)

print(results)