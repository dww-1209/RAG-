#StrOutputParser 是Langchain 内置的简单字符串解析器
#可以将AIMessage中的文本内容直接提取出来，作为字符串返回，符合模型invoke方法要求
#是runnable接口的子类（可以加入链）

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

parser = StrOutputParser()
LLM = ChatGroq(model="llama-3.1-8b-instant",temperature=0.5)
prompt = PromptTemplate.from_template(
    "我邻居姓{last_name}刚生了一个{gender}孩，仅告诉名字无需其他内容"
)

chain = prompt | LLM | parser |LLM | parser

res = chain.invoke(input={"last_name":"王","gender":"女"})

print(res)