#我们可以自己编写lambda匿名函数来完成自定义逻辑的数据转换，更自由
#RunnableLambda类是LangChain内置的，将普通函数转换为Runnable接口实例，方便自定义还能输加入Chain中使用

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

str_parser = StrOutputParser()
LLM = ChatGroq(model="llama-3.1-8b-instant",temperature=0.5)

my_func = RunnableLambda(lambda ai_meg: {"name": ai_meg.content})

first_prompt = PromptTemplate.from_template(
    "我邻居姓{last_name},刚生了一个{gender}孩,请起一个名字. 仅告知名字，不需要其他信息"
)

seconde_prompt = PromptTemplate.from_template(
    "姓名{name}，请你解析这个名字的含义"
)

chain = first_prompt | LLM | my_func | seconde_prompt | LLM | str_parser

for chunk in chain.stream(input={"last_name":"李","gender":"女"}):
    print(chunk,end="",flush=True)