from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

model = ChatTongyi(
    model="qwen3-max-2026-01-23",
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简介和专业的回答用户问题，参考资料{context}.如果无法从参考资料中找到相关信息，请直接说不知道，不要编造答案."),
        ("user", "用户提问：{input}"),
    ]
)

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

#准备一下资料（向量库的数据）
#add_texts (传入一个list[str])
vector_store.add_texts(["减肥就是要少吃多练","在减肥期间吃东西很重要，清淡少油控制卡路里摄入并运动起来","跑步是很好的运动哦"])
input_text = "怎么减肥"

#检索向量
result = vector_store.similarity_search(input_text,k=2)

reference_text = '['
for doc in result:
    reference_text += doc.page_content
reference_text+="]"  

def print_prompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt

# chain = prompt | model | StrOutputParser()
# res = chain.invoke({"query":input_text,"context":reference_text})
# print(res)

retriever = vector_store.as_retriever(search_kwargs={"k":2})

def format_func(docs:list[Document]):
    if not docs :
        return "无相关参考资料"
    
    format_str = "["
    for doc in docs :
        format_str+=doc.page_content
    format_str+="]"
    return format_str

#chain
chain =(
    {"input":RunnablePassthrough(),"context":retriever|format_func} | prompt |print_prompt| model |StrOutputParser()
)

res = chain.invoke(input_text)
print(res)
"""
retriever:
    -输入：用户的提问       str
    -输出：向量库的检索结果 list[Document]
prompt:
    -输入：用户的提问+向量库的检索结果 dict
    -输出：完整的提示词             PromptValue

"""