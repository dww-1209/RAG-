# LangChain 中的绝大多数核心组件都继承了 Runnable的抽象基类
# chain = prompt | model
#chain变量是RunnableSequence的实例，RunnableSequence会自动将上一个组件的输出作为下一个组件的输入，实现数据的自动化流转和协同工作.

from langchain_groq import  ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

load_dotenv()

LLM = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

prompt = PromptTemplate.from_template("你是一个ai助手")

chain = prompt | LLM | prompt | LLM
print(type(chain))